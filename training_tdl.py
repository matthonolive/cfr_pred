import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import time

import numpy as np
import polars as pl

from scipy.ndimage import gaussian_filter, generic_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# mlink
from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor
from mlink.geometry import generate_wall_map, walls_to_mesh
from mlink.scene import Scene
from mlink.channel_tdl import RtCfg, subcarrier_frequencies_centered, compute_tdl_batch


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    out_dir: str = "runs/tau_rms_3072_200k_merged"

    # scene / grids
    frequency_hz: float = 5.21e9
    img_hw: tuple[int, int] = (64, 64)
    K_slices: int = 4
    z_step: float = 1.0
    z_margin: float = 0.5
    floor_h: float = 0.0
    ceil_min: float = 8.0
    ceil_max: float = 20.0
    scale: float = 0.625

    # TX grid
    tx_origin_xy: tuple[float, float] = (1.75, 1.75)
    tx_z: float = 2.4
    tx_spacing_xy: float = 12.0
    tx_shape: tuple[int, int, int] = (1, 5, 5)

    # OFDM
    fft_size: int = 3072
    subcarrier_spacing_hz: float = 78_125.0

    # label generation
    rx_batch: int = 256
    no_path_wb_db: float = 199.5  # compute_tdl_batch uses 200 dB sentinel
    rt: RtCfg = field(default_factory=lambda: RtCfg(
        max_depth=5,
        samples_per_src=200_000,
        diffuse_reflection=True,
        diffraction=False,
    ))

    # features
    dataset_features: list[str] = field(default_factory=lambda: [
        "binary_walls", "electrical_distance", "cost", "height_cond"
    ])

    model_features: list[str] = field(default_factory=lambda: [
        "binary_walls", "electrical_distance", "cost", "height_cond"
    ])


    # excess delay smoothing
    smooth_kind: str = "median"   # "median" or "gaussian" or "none"
    smooth_median_size: int = 3
    smooth_gauss_sigma: float = 1.0

    # dataset size
    num_scenes: int = 20
    train_frac: float = 0.8
    seed: int = 3077

    # training
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 2e-4
    epochs: int = 30
    base: int = 32
    groups: int = 8
    dropout: float = 0.1
    grad_clip: float = 1.0

    ex_loss_thresh_ns: float = 0.5     # or 1.0
    tau_loss_thresh_ns: float = 0.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    tau_target: str = "raw"          # "raw" or "log10"
    tau_log_eps_ns: float = 1e-3       # avoids log(0); 1e-3 ns is tiny
    tau_cap_ns: float = 50.0           # soft prior: discourage > this
    tau_phys_loss_w: float = 0.0      # add a small physical-space loss term
    tau_cap_w: float = 0.0            # hinge penalty weight

    tau_paths_rel_db: float = 30.0

cfg = CFG()


# ----------------------------
# Utilities
# ----------------------------

def default_material_db(freq: float) -> pl.DataFrame:
    return pl.DataFrame(
        data={
            "id": [0],
            "frequency": [freq],
            "permittivity": [4.0],
            "permeability": [1.0],
            "conductivity": [0.01],
            "transmission_loss_vertical": [10.0],
            "transmission_loss_horizontal": [20.0],
            "reflection_loss": [9.0],
            "diffraction_loss_min": [8.0],
            "diffraction_loss_max": [15.0],
            "diffraction_loss": [5.0],
            "name": ["0"],
            "thickness": [0.1],
        }
    )


def masked_gaussian_2d(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores invalid pixels (mask==0).

    img: (H,W) float32
    mask: (H,W) bool or {0,1}
    """
    if sigma <= 0:
        return img.astype(np.float32)
    m = mask.astype(np.float32)
    num = gaussian_filter(img * m, sigma=sigma, mode="nearest")
    den = gaussian_filter(m, sigma=sigma, mode="nearest")
    out = np.zeros_like(img, dtype=np.float32)
    good = den > 1e-6
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def masked_median_2d(img: np.ndarray, mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter that ignores invalid pixels (mask==0) using generic_filter.

    For 64x64 maps this is cheap compared to RT label generation.
    """
    if size <= 1:
        return img.astype(np.float32)

    work = img.astype(np.float32).copy()
    work[~mask] = np.nan

    def nanmed(w):
        return np.nanmedian(w)

    out = generic_filter(work, nanmed, size=size, mode="nearest")
    out = out.astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def smooth_map_stack(x_map: np.ndarray, wb_map: np.ndarray) -> np.ndarray:
    """x_map: (K,H,W), wb_map: (K,H,W). Smooths each slice using valid mask from wb."""
    K, H, W = x_map.shape
    out = np.zeros_like(x_map, dtype=np.float32)
    for k in range(K):
        mask = wb_map[k] < cfg.no_path_wb_db
        if cfg.smooth_kind == "none":
            out[k] = x_map[k]
        elif cfg.smooth_kind == "gaussian":
            out[k] = masked_gaussian_2d(x_map[k], mask, cfg.smooth_gauss_sigma)
        elif cfg.smooth_kind == "median":
            out[k] = masked_median_2d(x_map[k], mask, cfg.smooth_median_size)
        else:
            raise ValueError("smooth_kind must be none/gaussian/median")
    return out

def tau_rms_from_taps(taps: np.ndarray, df_hz: float) -> np.ndarray:
    """
    taps: (B,L) complex, assumed normalized to unit total power (sum |taps|^2 ~ 1)
    df_hz: subcarrier spacing
    Returns tau_rms in seconds, shape (B,)
    """
    B, L = taps.shape
    Ts = 1.0 / (float(L) * float(df_hz))          # seconds, FFT-time resolution
    t = (np.arange(L, dtype=np.float32) * Ts)     # (L,)
    p = (np.abs(taps) ** 2).astype(np.float32)    # (B,L)

    psum = np.sum(p, axis=1, keepdims=True)
    p = p / np.maximum(psum, 1e-12)

    mu = p @ t
    mu2 = p @ (t * t)
    var = np.maximum(mu2 - mu * mu, 0.0)
    return np.sqrt(var).astype(np.float32)

def infer_feature_channel_counts(scene, freq_hz, features):
    counts = {}
    for f in features:
        x = build_feature_tensor(scene, freq_hz, requested=[f]).astype(np.float32)
        counts[f] = int(x.shape[1])  # channels per slice for this feature
    return counts

def build_keep_idx(dataset_features, model_features, K, feat_counts):
    # offsets within one slice
    offsets = {}
    off = 0
    for f in dataset_features:
        offsets[f] = (off, off + feat_counts[f])
        off += feat_counts[f]
    c_full = off  # true channels per slice

    keep_in_slice = []
    for f in model_features:
        a, b = offsets[f]
        keep_in_slice.extend(range(a, b))

    keep = []
    for k in range(K):
        base = k * c_full
        keep.extend([base + i for i in keep_in_slice])

    return np.asarray(keep, dtype=np.int64), c_full


def make_scene(rng: np.random.Generator) -> Scene:
    H, W = cfg.img_hw
    ceiling_h = float(rng.uniform(cfg.ceil_min, cfg.ceil_max))

    mesh = walls_to_mesh(
        generate_wall_map(
            (H, W),
            min_wall_length=8,
            min_door_length=4,
            max_partitions=24,
            rng=rng,
        ),
        floor_height=cfg.floor_h,
        ceiling_height=ceiling_h,
    ).apply_scale(cfg.scale)

    usable = max(ceiling_h - cfg.floor_h - 2 * cfg.z_margin, 1e-3)
    total_span = (cfg.K_slices - 1) * cfg.z_step
    z_step = usable / max(cfg.K_slices - 1, 1) if total_span > usable else cfg.z_step

    z_start = cfg.floor_h + cfg.z_margin
    z_end = (ceiling_h - cfg.z_margin) - total_span
    z0 = z_start if z_end < z_start else float(rng.uniform(z_start, z_end))

    rx_grid = AntennaGrid(
        origin=cfg.scale * np.asarray([0.0, 0.0, z0], dtype=np.float32),
        deltas=cfg.scale * np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, z_step]], dtype=np.float32),
        shape=(cfg.K_slices, H, W),
    )

    tx_grid = AntennaGrid(
        origin=cfg.scale * np.asarray([cfg.tx_origin_xy[0], cfg.tx_origin_xy[1], cfg.tx_z], dtype=np.float32),
        deltas=cfg.scale * np.asarray([[cfg.tx_spacing_xy, 0, 0], [0, cfg.tx_spacing_xy, 0], [0, 0, 1]], dtype=np.float32),
        shape=cfg.tx_shape,
    )

    antenna_db = AntennaDatabase.from_grid(tx_grid, rx_grid)
    mat_db = default_material_db(cfg.frequency_hz)
    face2material = {k: 0 for k in range(mesh.faces.shape[0])}
    return Scene(mesh=mesh, material_database=mat_db, face2material=face2material, antenna_database=antenna_db)


def _to_sionna_geometry(scene: Scene, freq: float):
    if hasattr(scene, "to_sionna_geometry"):
        return scene.to_sionna_geometry(freq)
    return scene.to_sionna(freq)


def compute_labels_for_scene(scene: Scene) -> np.ndarray:
    """Returns y of shape (num_tx, 3, K, H, W) float32.

    Channels:
      0: wb_loss_db
      1: excess_delay_ns (smoothed)
      2: tau_rms_ns (smoothed)
    """
    rx_grid = scene.antenna_database.rx_grid
    assert rx_grid is not None
    K, H_img, W_img = rx_grid.shape

    tx_coords = scene.antenna_database.tx_coords
    rx_coords = scene.antenna_database.rx_coords  # (K*H*W,3)
    P = rx_coords.shape[0]

    N = int(cfg.fft_size)
    y = np.zeros((tx_coords.shape[0], 3, K, H_img, W_img), dtype=np.float32)

    freqs = subcarrier_frequencies_centered(cfg.fft_size, cfg.subcarrier_spacing_hz)
    si = _to_sionna_geometry(scene, cfg.frequency_hz)

    for t, tx in enumerate(tx_coords):
        wb_all  = np.zeros((P,), dtype=np.float32)
        ex_all  = np.zeros((P,), dtype=np.float32)
        tau_all = np.zeros((P,), dtype=np.float32)

        for i0 in range(0, P, cfg.rx_batch):
            i1 = min(i0 + cfg.rx_batch, P)

            wb_db, ex_s, taps, tau_rms_s = compute_tdl_batch(
                si_scene=si,
                tx_xyz=tx,
                rx_xyz=rx_coords[i0:i1],
                frequencies_hz=freqs,
                L_taps=N,              # still fine to keep taps for other uses
                rt=cfg.rt,
                return_tau_rms=True,
            )

            wb_all[i0:i1] = wb_db
            ex_all[i0:i1] = ex_s * 1e9

            good = wb_db < cfg.no_path_wb_db
            if np.any(good):
                idx_g = np.nonzero(good)[0]
                tau_all[i0 + idx_g] = tau_rms_s[good] * 1e9

        wb_map  = wb_all.reshape(K, H_img, W_img)
        ex_map  = ex_all.reshape(K, H_img, W_img)
        tau_map = tau_all.reshape(K, H_img, W_img)

        ex_sm  = smooth_map_stack(ex_map, wb_map)
        tau_sm = smooth_map_stack(tau_map, wb_map)

        ex_sm  = np.maximum(ex_sm, 0.0)
        tau_sm = np.maximum(tau_sm, 0.0)

        y[t, 0] = wb_map
        y[t, 1] = ex_sm
        y[t, 2] = tau_sm

        print(f"  tx {t+1}/{tx_coords.shape[0]} labels done")

    return y


def compute_norm_stats(
    x_mm: np.memmap,
    y_mm: np.memmap,
    max_samples: int = 512,
    seed: int = 0,
    keep_idx: np.ndarray | list[int] | None = None,
):
    """
    Computes per-channel mean/std over a random subset of samples.

    If keep_idx is provided, stats are computed ONLY for x[:, keep_idx, :, :],
    and returned x_mean/x_std match that reduced channel count.
    y stats are always computed on all y channels (no keep_idx here).
    """
    rng = np.random.default_rng(seed)
    n_total = int(x_mm.shape[0])
    take = min(max_samples, n_total)
    idx = rng.choice(n_total, size=take, replace=False)

    x = np.array(x_mm[idx], dtype=np.float32)  # (S, Cx, H, W)
    y = np.array(y_mm[idx], dtype=np.float32)  # (S, Cy, H, W)

    y = apply_y_transform_np(y, K=cfg.K_slices, y_ch=3)

    if keep_idx is not None:
        keep_idx = np.asarray(keep_idx, dtype=np.int64)
        x = x[:, keep_idx, :, :]  # (S, Cx_keep, H, W)

    x_mean = torch.from_numpy(x.mean(axis=(0, 2, 3))).view(-1, 1, 1)
    x_std  = torch.from_numpy(x.std(axis=(0, 2, 3))).view(-1, 1, 1)
    y_mean = torch.from_numpy(y.mean(axis=(0, 2, 3))).view(-1, 1, 1)
    y_std  = torch.from_numpy(y.std(axis=(0, 2, 3))).view(-1, 1, 1)

    x_std = torch.clamp(x_std, min=1e-6)
    y_std = torch.clamp(y_std, min=1e-6)

    out = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    if keep_idx is not None:
        out["keep_idx"] = torch.from_numpy(keep_idx)
    return out


class MemmapIndexDataset(Dataset):
    def __init__(
        self,
        x_mm,
        y_mm,
        indices,
        stats,
        no_path_wb_db: float,
        K: int,
        y_ch: int,
        H: int,
        W: int,
        ex_loss_thresh_ns: float,
        tau_loss_thresh_ns: float,
        keep_idx: np.ndarray | list[int] | None = None,
    ):
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.indices = indices.astype(np.int64)
        self.stats = stats

        self.no_path_wb_db = float(no_path_wb_db)
        self.K = int(K)
        self.y_ch = int(y_ch)
        self.H = int(H)
        self.W = int(W)

        self.ex_loss_thresh_ns = float(ex_loss_thresh_ns)
        self.tau_loss_thresh_ns = float(tau_loss_thresh_ns)

        self.keep_idx = None
        if keep_idx is not None:
            self.keep_idx = np.asarray(keep_idx, dtype=np.int64)

        # sanity: if keep_idx is used, stats must match kept channels
        if self.keep_idx is not None:
            if int(self.stats["x_mean"].shape[0]) != int(self.keep_idx.shape[0]):
                raise ValueError(
                    f"stats['x_mean'] has {int(self.stats['x_mean'].shape[0])} channels "
                    f"but keep_idx has {int(self.keep_idx.shape[0])}. "
                    "Recompute norm stats with the same keep_idx."
                )

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])

        # Read full sample from memmap
        x_np = np.array(self.x_mm[j], dtype=np.float32)  # (Cx, H, W) where Cx = K*c_in (stacked)
        y_np = np.array(self.y_mm[j], dtype=np.float32)  # (Cy, H, W) where Cy = K*y_ch

        # Apply keep_idx BEFORE normalization if requested
        if self.keep_idx is not None:
            x_np = x_np[self.keep_idx, :, :]  # (Cx_keep, H, W)

        x = torch.from_numpy(x_np)
        y_phys = y_np.astype(np.float32, copy=False)
        y3_phys = torch.from_numpy(y_phys).view(self.K, self.y_ch, self.H, self.W)
        wb  = y3_phys[:, 0]
        ex  = y3_phys[:, 1]
        tau = y3_phys[:, 2]

        mask_path = (wb < self.no_path_wb_db)
        mask_ex   = mask_path & (ex  >= self.ex_loss_thresh_ns)
        # IMPORTANT: supervise tau everywhere on-path unless you really want tail-only
        mask_tau  = mask_path & (tau >= self.tau_loss_thresh_ns)

        # --- now transform tau channel for training target ---
        y_tf = y_phys.copy()
        # y_tf is (K*y_ch,H,W); expand to (1, Cy, H, W) for reuse of helper
        y_tf_ = y_tf[None, ...]
        apply_y_transform_np(y_tf_, K=self.K, y_ch=self.y_ch)
        y_tf = y_tf_[0]

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_tf)

        # normalize
        x = (x - self.stats["x_mean"]) / self.stats["x_std"]
        y = (y - self.stats["y_mean"]) / self.stats["y_std"]

        return x, y, mask_path.float(), mask_ex.float(), mask_tau.float()
    

def _valid_groups(ch: int, groups: int) -> int:
    g = min(groups, ch)
    while ch % g != 0:
        g -= 1
    return max(g, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g1 = _valid_groups(in_ch, groups)
        g2 = _valid_groups(out_ch, groups)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


class UNet3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 32, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.e1 = nn.Sequential(
            ResBlock(in_ch, base, groups=groups, dropout=dropout),
            ResBlock(base, base, groups=groups, dropout=dropout),
        )
        self.p1 = nn.MaxPool2d(2)

        self.e2 = nn.Sequential(
            ResBlock(base, base*2, groups=groups, dropout=dropout),
            ResBlock(base*2, base*2, groups=groups, dropout=dropout),
        )
        self.p2 = nn.MaxPool2d(2)

        self.e3 = nn.Sequential(
            ResBlock(base*2, base*4, groups=groups, dropout=dropout),
            ResBlock(base*4, base*4, groups=groups, dropout=dropout),
        )
        self.p3 = nn.MaxPool2d(2)

        self.mid = nn.Sequential(
            ResBlock(base*4, base*8, groups=groups, dropout=dropout),
            ResBlock(base*8, base*8, groups=groups, dropout=dropout),
        )

        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = nn.Sequential(
            ResBlock(base*8, base*4, groups=groups, dropout=0.0),
            ResBlock(base*4, base*4, groups=groups, dropout=0.0),
        )

        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = nn.Sequential(
            ResBlock(base*4, base*2, groups=groups, dropout=0.0),
            ResBlock(base*2, base*2, groups=groups, dropout=0.0),
        )

        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = nn.Sequential(
            ResBlock(base*2, base, groups=groups, dropout=0.0),
            ResBlock(base, base, groups=groups, dropout=0.0),
        )

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        m = self.mid(self.p3(e3))

        d3 = self.u3(m)
        d3 = self.d3(torch.cat([d3, e3], dim=1))

        d2 = self.u2(d3)
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.u1(d2)
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


@torch.no_grad()
def report_metrics(model, dl, stats_dev, y_ch: int, H: int, W: int,
                   tag="val", ex_nmse_thresh_ns=0.5, tau_nmse_thresh_ns=100.0,
                   max_batches=None):
    model.eval()

    # counts
    sum_path = 0.0
    sum_ex_tail = 0.0
    sum_tau_tail = 0.0
    sum_nopath = 0.0

    # WB
    sum_wb_mae = 0.0
    wb_nmse_num = 0.0
    wb_nmse_den = 0.0

    # EX (all-path)
    sum_ex_mae_path = 0.0
    sum_ex_mse_path = 0.0

    # EX (tail)
    sum_ex_mae_tail = 0.0
    sum_ex_mse_tail = 0.0
    ex_nmse_num = 0.0
    ex_nmse_den = 0.0

    # EX leakage on no-path
    sum_ex_abs_nopath = 0.0

    # TAU (all-path)
    sum_tau_mae_path = 0.0
    sum_tau_mse_path = 0.0

    # TAU (tail)
    sum_tau_mae_tail = 0.0
    sum_tau_mse_tail = 0.0
    tau_nmse_num = 0.0
    tau_nmse_den = 0.0

    # TAU leakage on no-path
    sum_tau_abs_nopath = 0.0

    n_batches = 0

    for xb, yb, m_path, m_ex, m_tau in dl:
        xb = xb.to(cfg.device, non_blocking=True)
        yb = yb.to(cfg.device, non_blocking=True)
        m_path = m_path.to(cfg.device, non_blocking=True)  # (B,K,H,W)
        m_ex   = m_ex.to(cfg.device, non_blocking=True)
        m_tau  = m_tau.to(cfg.device, non_blocking=True)

        pred = model(xb)

        # unnormalize to physical units
        y_mean = stats_dev["y_mean"]
        y_std  = stats_dev["y_std"]
        pred_p = pred * y_std + y_mean
        tgt_p  = yb   * y_std + y_mean

        B = pred_p.shape[0]
        K = cfg.K_slices

        pred_p = pred_p.view(B, K, y_ch, H, W)
        tgt_p  = tgt_p.view(B, K, y_ch, H, W)

        wb_hat = pred_p[:, :, 0]
        wb_tgt = tgt_p[:, :, 0]
        ex_hat = pred_p[:, :, 1]
        ex_tgt = tgt_p[:, :, 1]
        tau_hat_tgt = pred_p[:, :, 2]
        tau_tgt_tgt = tgt_p[:, :, 2]

        if cfg.tau_target == "log10":
            tau_hat = torch.pow(10.0, tau_hat_tgt) - cfg.tau_log_eps_ns
            tau_tgt = torch.pow(10.0, tau_tgt_tgt) - cfg.tau_log_eps_ns
            tau_hat = torch.clamp(tau_hat, min=0.0)
            tau_tgt = torch.clamp(tau_tgt, min=0.0)
        else:
            tau_hat = tau_hat_tgt
            tau_tgt = tau_tgt_tgt


        # masks
        mp = m_path
        me = m_ex
        mt = m_tau
        mn = (1.0 - mp)  # no-path

        # counts
        sum_path += mp.sum().clamp_min(0.0).item()
        sum_ex_tail += me.sum().clamp_min(0.0).item()
        sum_tau_tail += mt.sum().clamp_min(0.0).item()
        sum_nopath += mn.sum().clamp_min(0.0).item()

        # ---------------- WB on path ----------------
        sum_wb_mae += ((wb_hat - wb_tgt).abs() * mp).sum().item()

        g_hat = torch.pow(10.0, -wb_hat / 10.0)
        g_tgt = torch.pow(10.0, -wb_tgt / 10.0)
        wb_nmse_num += (((g_hat - g_tgt) ** 2) * mp).sum().item()
        wb_nmse_den += (((g_tgt) ** 2) * mp).sum().clamp_min(1e-12).item()

        # ---------------- EX ----------------
        ex_err = ex_hat - ex_tgt

        # all-path
        sum_ex_mae_path += (ex_err.abs() * mp).sum().item()
        sum_ex_mse_path += ((ex_err ** 2) * mp).sum().item()

        # tail (your m_ex)
        sum_ex_mae_tail += (ex_err.abs() * me).sum().item()
        sum_ex_mse_tail += ((ex_err ** 2) * me).sum().item()

        # NMSE on tail (also use explicit threshold to be safe)
        ex_mask = ((ex_tgt >= ex_nmse_thresh_ns).to(ex_tgt.dtype) * me)
        ex_nmse_num += ((ex_err ** 2) * ex_mask).sum().item()
        ex_nmse_den += ((ex_tgt ** 2) * ex_mask).sum().clamp_min(1e-12).item()

        # leakage on no-path
        sum_ex_abs_nopath += (ex_hat.abs() * mn).sum().item()

        # ---------------- TAU ----------------
        tau_err = tau_hat - tau_tgt

        # all-path
        sum_tau_mae_path += (tau_err.abs() * mp).sum().item()
        sum_tau_mse_path += ((tau_err ** 2) * mp).sum().item()

        # tail
        sum_tau_mae_tail += (tau_err.abs() * mt).sum().item()
        sum_tau_mse_tail += ((tau_err ** 2) * mt).sum().item()

        tau_mask = ((tau_tgt >= tau_nmse_thresh_ns).to(tau_tgt.dtype) * mt)
        tau_nmse_num += ((tau_err ** 2) * tau_mask).sum().item()
        tau_nmse_den += ((tau_tgt ** 2) * tau_mask).sum().clamp_min(1e-12).item()

        sum_tau_abs_nopath += (tau_hat.abs() * mn).sum().item()

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    # finalize helpers
    def safe_div(a, b): return a / max(b, 1e-12)

    wb_mae = safe_div(sum_wb_mae, sum_path)
    wb_nmse = safe_div(wb_nmse_num, wb_nmse_den)
    wb_nmse_db = 10.0 * math.log10(max(wb_nmse, 1e-12))

    ex_mae_path = safe_div(sum_ex_mae_path, sum_path)
    ex_rmse_path = math.sqrt(safe_div(sum_ex_mse_path, sum_path))

    ex_mae_tail = safe_div(sum_ex_mae_tail, sum_ex_tail)
    ex_rmse_tail = math.sqrt(safe_div(sum_ex_mse_tail, sum_ex_tail))
    ex_nmse = safe_div(ex_nmse_num, ex_nmse_den)
    ex_nmse_db = 10.0 * math.log10(max(ex_nmse, 1e-12))

    ex_leak = safe_div(sum_ex_abs_nopath, sum_nopath)

    tau_mae_path = safe_div(sum_tau_mae_path, sum_path)
    tau_rmse_path = math.sqrt(safe_div(sum_tau_mse_path, sum_path))

    tau_mae_tail = safe_div(sum_tau_mae_tail, sum_tau_tail)
    tau_rmse_tail = math.sqrt(safe_div(sum_tau_mse_tail, sum_tau_tail))
    tau_nmse = safe_div(tau_nmse_num, tau_nmse_den)
    tau_nmse_db = 10.0 * math.log10(max(tau_nmse, 1e-12))

    tau_leak = safe_div(sum_tau_abs_nopath, sum_nopath)

    print(
        f"  [{tag}] wb_MAE(path)={wb_mae:.3f} dB | wb_NMSE={wb_nmse:.4f} ({wb_nmse_db:.1f} dB)\n"
        f"        ex_MAE(path)={ex_mae_path:.3f} ns | ex_RMSE(path)={ex_rmse_path:.3f} ns | "
        f"ex_MAE(tail)={ex_mae_tail:.3f} ns | ex_RMSE(tail)={ex_rmse_tail:.3f} ns | "
        f"ex_NMSE(tail)={ex_nmse:.4f} ({ex_nmse_db:.1f} dB) | ex_leak(no-path)={ex_leak:.3f} ns\n"
        f"        tau_MAE(path)={tau_mae_path:.3f} ns | tau_RMSE(path)={tau_rmse_path:.3f} ns | "
        f"tau_MAE(tail)={tau_mae_tail:.3f} ns | tau_RMSE(tail)={tau_rmse_tail:.3f} ns | "
        f"tau_NMSE(tail)={tau_nmse:.4f} ({tau_nmse_db:.1f} dB) | tau_leak(no-path)={tau_leak:.3f} ns"
    )


### TAU RMS Helpers ###

def tau_rms_from_paths(a: np.ndarray, tau_s: np.ndarray, rel_db: float = 30.0) -> np.ndarray:
    """
    a:     (B, P) complex path coefficients
    tau_s: (B, P) path delays in seconds (absolute)
    returns: (B,) tau_rms in seconds, computed on excess delays (tau - tau_min)
    """
    a = np.asarray(a)
    tau_s = np.asarray(tau_s)

    B = a.shape[0]
    a = a.reshape(B, -1)
    tau_s = tau_s.reshape(B, -1)

    # power weights
    p = (np.abs(a) ** 2).astype(np.float64)

    # valid paths: finite delay, non-negative, positive power
    valid = np.isfinite(tau_s) & (tau_s >= 0.0) & np.isfinite(p) & (p > 0.0)

    out = np.zeros((B,), dtype=np.float64)

    for b in range(B):
        vb = valid[b]
        if not np.any(vb):
            continue

        tb = tau_s[b, vb].astype(np.float64)
        pb = p[b, vb].astype(np.float64)

        # drop weak paths (relative to strongest remaining)
        if rel_db is not None and rel_db > 0:
            pmax = np.max(pb)
            pb = np.where(pb >= pmax * (10.0 ** (-rel_db / 10.0)), pb, 0.0)

        psum = np.sum(pb)
        if psum <= 0:
            continue

        # IMPORTANT: excess delay axis
        t0 = np.min(tb)
        te = tb - t0

        w = pb / psum
        mu  = np.sum(w * te)
        mu2 = np.sum(w * te * te)
        var = max(mu2 - mu * mu, 0.0)
        out[b] = np.sqrt(var)

    return out.astype(np.float32)


def tau_to_target(tau_ns: np.ndarray) -> np.ndarray:
    """Physical tau(ns) -> training target for tau channel."""
    tau_ns = np.maximum(tau_ns, 0.0).astype(np.float32)
    if cfg.tau_target == "raw":
        return tau_ns
    elif cfg.tau_target == "log10":
        return np.log10(tau_ns + cfg.tau_log_eps_ns).astype(np.float32)
    else:
        raise ValueError("cfg.tau_target must be 'raw' or 'log10'")

def tau_from_target(tau_tgt: np.ndarray) -> np.ndarray:
    """Training target -> physical tau(ns)."""
    tau_tgt = tau_tgt.astype(np.float32)
    if cfg.tau_target == "raw":
        return np.maximum(tau_tgt, 0.0)
    elif cfg.tau_target == "log10":
        return np.maximum((10.0 ** tau_tgt) - cfg.tau_log_eps_ns, 0.0)
    else:
        raise ValueError("cfg.tau_target must be 'raw' or 'log10'")

def apply_y_transform_np(y: np.ndarray, K: int, y_ch: int) -> np.ndarray:
    """
    Apply τ transform IN PLACE on numpy y.
    y: (..., K*y_ch, H, W) float32
    """
    if cfg.tau_target == "raw":
        return y
    # tau channel is ch=2 in each slice
    for k in range(K):
        tau_idx = k * y_ch + 2
        y[:, tau_idx, :, :] = tau_to_target(y[:, tau_idx, :, :])
    return y

def main():
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TF memory growth (avoid TF grabbing VRAM before torch)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass

    rng = np.random.default_rng(cfg.seed)

    # infer shapes
    tmp = make_scene(rng)
    x_tmp = build_feature_tensor(tmp, cfg.frequency_hz, requested=cfg.dataset_features).astype(np.float32)
    c_in = int(x_tmp.shape[1])
    num_tx = int(tmp.antenna_database.tx_coords.shape[0])
    K, H, W = tmp.antenna_database.rx_grid.shape

    y_ch = 3

    total_samples = cfg.num_scenes * num_tx

    x_path = out_dir / "x.dat"
    y_path = out_dir / "y.dat"
    meta_path = out_dir / "meta.json"
    stats_path = out_dir / "norm_stats.npz"
    state_path = out_dir / "model_state.pt"
    jit_path = out_dir / "model.pt"

    feat_counts = infer_feature_channel_counts(tmp, cfg.frequency_hz, cfg.dataset_features)
    keep_idx, c_full = build_keep_idx(cfg.dataset_features, cfg.model_features, K, feat_counts)

    print("x exists?", x_path.exists(), "y exists?", y_path.exists())

    # Build dataset if missing
    if not (x_path.exists() and y_path.exists() and meta_path.exists()):
        print("Building memmap dataset...")
        x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(total_samples, c_in*K, H, W))
        y_mm = np.memmap(y_path, dtype="float16", mode="w+", shape=(total_samples, y_ch*K, H, W))

        # write meta early so you can peek while generating
        meta = dict(
            total_samples=int(total_samples),
            H=int(H), W=int(W), K=int(K), num_tx=int(num_tx),
            c_in=int(c_in),
            y_ch=int(y_ch),
            fft_size=int(cfg.fft_size),
            subcarrier_spacing_hz=float(cfg.subcarrier_spacing_hz),
            smooth_kind=str(cfg.smooth_kind),
            smooth_median_size=int(cfg.smooth_median_size),
            smooth_gauss_sigma=float(cfg.smooth_gauss_sigma),
            frequency_hz=float(cfg.frequency_hz),
            x_dtype="float32", y_dtype="float16",
            y_channels=["wb_loss_db", "excess_delay_ns_sm", "tau_rms_ns_sm"],
            keep_idx = keep_idx.tolist(),
            tau_target=str(cfg.tau_target),
            tau_log_eps_ns=float(cfg.tau_log_eps_ns),
            tau_cap_ns=float(cfg.tau_cap_ns),
        )
        meta_path.write_text(json.dumps(meta, indent=2))
        print("Wrote meta (early):", meta_path)

        idx = 0
        for s in range(cfg.num_scenes):
            scene = make_scene(rng)

            x = build_feature_tensor(scene, cfg.frequency_hz, requested=cfg.dataset_features).astype(np.float32)
            y = compute_labels_for_scene(scene)

            # x: (num_tx, c_in, K, H, W)  -> (num_tx, K, c_in, H, W) -> (num_tx, K*c_in, H, W)
            x_stack = x.transpose(0, 2, 1, 3, 4).reshape(num_tx, K * c_in, H, W)

            # y: (num_tx, y_ch, K, H, W) -> (num_tx, K, y_ch, H, W) -> (num_tx, K*y_ch, H, W)
            y_stack = y.transpose(0, 2, 1, 3, 4).reshape(num_tx, K * y_ch, H, W)

            n = num_tx
            x_mm[idx:idx+n] = x_stack
            y_mm[idx:idx+n] = y_stack.astype(np.float16)
            idx += n
            x_mm.flush(); y_mm.flush()
            print(f"[scene {s+1:03d}/{cfg.num_scenes}] wrote {n} samples (total {idx})")

    # reopen memmaps
    meta = json.loads(meta_path.read_text())
    total_samples = int(meta["total_samples"])
    H = int(meta["H"]); W = int(meta["W"])
    K = int(meta["K"]); c_in = int(meta["c_in"]); y_ch = int(meta["y_ch"])

    # sanity:
    assert c_full == c_in, (c_full, c_in)  # these should match
    in_ch = int(keep_idx.size)

    # keep_idx = None
    # in_ch = c_in * K

    x_mm = np.memmap(x_path, dtype="float32", mode="r", shape=(total_samples, c_in*K, H, W))
    y_mm = np.memmap(y_path, dtype="float16", mode="r", shape=(total_samples, y_ch*K, H, W))

    # quick sanity stats
    samp = np.random.default_rng(cfg.seed + 1).choice(total_samples, size=min(256, total_samples), replace=False)
    wb = np.array(y_mm[samp, 0], dtype=np.float32)
    ex = np.array(y_mm[samp, 1], dtype=np.float32)
    tr = np.array(y_mm[samp, 2], dtype=np.float32)
    print(f"wb_loss(dB): mean={wb.mean():.2f} std={wb.std():.2f} min={wb.min():.2f} max={wb.max():.2f}")
    print(f"excess_delay_sm(ns): mean={ex.mean():.2f} std={ex.std():.2f} min={ex.min():.2f} max={ex.max():.2f}")
    print(f"tau_rms_sm(ns): mean={tr.mean():.2f} std={tr.std():.2f} min={tr.min():.2f} max={tr.max():.2f}")

    samp = np.random.default_rng(cfg.seed + 2).choice(total_samples, size=min(128, total_samples), replace=False)
    y_s = np.array(y_mm[samp], dtype=np.float32)  # (S, K*y_ch, H, W)

    K = int(meta["K"]); y_ch = int(meta["y_ch"])
    wb_s  = y_s[:, 0::y_ch, :, :]      # (S,K,H,W)
    tau_s = y_s[:, 2::y_ch, :, :]      # (S,K,H,W)

    m_path = (wb_s < cfg.no_path_wb_db)
    tau_path = tau_s[m_path]
    if tau_path.size > 0:
        qs = np.percentile(tau_path, [1, 5, 10, 50, 90, 95, 99]).astype(np.float32)
        print("tau_rms(ns) on PATH pixels quantiles [1,5,10,50,90,95,99]%:", qs)
        print("tau_rms(ns) path min/max:", float(tau_path.min()), float(tau_path.max()))
        print("tau_rms(ns) path frac <=10ns:", float(np.mean(tau_path <= 10.0)))
        print("tau_rms(ns) path frac <=50ns:", float(np.mean(tau_path <= 50.0)))
    else:
        print("WARNING: no path pixels found in tau sanity sample")

    # norm stats
    stats = compute_norm_stats(x_mm, y_mm, keep_idx=keep_idx, max_samples=512, seed=cfg.seed)
    np.savez(
        stats_path,
        x_mean=stats["x_mean"].numpy(), x_std=stats["x_std"].numpy(),
        y_mean=stats["y_mean"].numpy(), y_std=stats["y_std"].numpy(),
        keep_idx=keep_idx,   
    )
    print("Saved stats:", stats_path)

    stats_dev = {k: v.to(cfg.device) for k, v in stats.items()}
    y_mean = stats_dev["y_mean"]
    y_std  = stats_dev["y_std"]

    # split train and validation by scene
    rng = np.random.default_rng(cfg.seed)

    samples_per_scene = num_tx   # for 2.5D
    scene_ids = rng.permutation(cfg.num_scenes)

    n_train_scenes = int(round(cfg.train_frac * cfg.num_scenes))
    train_scenes = np.sort(scene_ids[:n_train_scenes])
    val_scenes   = np.sort(scene_ids[n_train_scenes:])

    def scene_to_indices(s):
        base = s * samples_per_scene
        return np.arange(base, base + samples_per_scene, dtype=np.int64)

    train_idx = np.concatenate([scene_to_indices(s) for s in train_scenes])
    val_idx   = np.concatenate([scene_to_indices(s) for s in val_scenes])

    (out_dir/"split.json").write_text(json.dumps({
    "train_scenes": train_scenes.tolist(),
    "val_scenes": val_scenes.tolist(),
    "samples_per_scene": int(samples_per_scene),
    "seed": int(cfg.seed),
    }, indent=2))

    train_ds = MemmapIndexDataset(x_mm, y_mm, train_idx, stats, cfg.no_path_wb_db, K, y_ch, H, W, cfg.ex_loss_thresh_ns, cfg.tau_loss_thresh_ns, keep_idx=keep_idx)
    val_ds   = MemmapIndexDataset(x_mm, y_mm, val_idx,   stats, cfg.no_path_wb_db, K, y_ch, H, W, cfg.ex_loss_thresh_ns, cfg.tau_loss_thresh_ns, keep_idx=keep_idx)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # model
    model = UNet3(in_ch=in_ch, out_ch=y_ch*K, base=cfg.base, groups=cfg.groups, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

    def loss_fn(pred, tgt, m_path, m_ex, m_tau, y_mean, y_std):
        """
        pred, tgt: (B, K*Y, H, W) normalized
        masks:     (B, K,   H, W) float {0,1}
        y_mean/y_std: (K*Y,1,1) tensors on device
        """
        B, _, H, W = pred.shape
        K = cfg.K_slices
        Y = y_ch  # 3

        pred3 = pred.view(B, K, Y, H, W)
        tgt3  = tgt.view(B, K, Y, H, W)

        mp = m_path.unsqueeze(2)  # (B,K,1,H,W)
        me = m_ex.unsqueeze(2)
        mt = m_tau.unsqueeze(2)

        # ---- WB: compute NMSE in linear gain space using PHYSICAL wb (dB) ----
        # reshape stats to (1,K,Y,1,1)
        y_mean3 = y_mean.view(1, K, Y, 1, 1)
        y_std3  = y_std.view(1, K, Y, 1, 1)

        wb_hat = pred3[:, :, 0:1] * y_std3[:, :, 0:1] + y_mean3[:, :, 0:1]  # dB
        wb_tgt = tgt3[:, :, 0:1]  * y_std3[:, :, 0:1] + y_mean3[:, :, 0:1]  # dB

        g_hat = torch.pow(10.0, -wb_hat / 10.0)
        g_tgt = torch.pow(10.0, -wb_tgt / 10.0)

        num = ((g_hat - g_tgt) ** 2 * mp).sum()
        den = ((g_tgt ** 2) * mp).sum().clamp_min(1e-12)
        l_wb_gain_nmse = num / den

        # (optional) also keep a small dB loss for stability
        def masked_smooth_l1(a, b, m):
            denom = m.sum().clamp_min(1.0)
            return F.smooth_l1_loss(a*m, b*m, reduction="sum") / denom

        l_wb_db  = masked_smooth_l1(pred3[:, :, 0:1], tgt3[:, :, 0:1], mp)

        # ---- Delay heads (can stay normalized; masking was computed in physical space) ----
        l_ex  = masked_smooth_l1(pred3[:, :, 1:2], tgt3[:, :, 1:2], me)
        tau_mask = mp

        # log/target-space loss (stable)
        l_tau_tgt = masked_smooth_l1(pred3[:, :, 2:3], tgt3[:, :, 2:3], tau_mask)

        # Optional: small physical-space loss + soft cap prior (only if using log target)
        l_tau_phys = torch.tensor(0.0, device=pred.device)
        l_tau_cap  = torch.tensor(0.0, device=pred.device)

        if cfg.tau_target == "log10":
            # unnormalize tau target to log10(ns)
            tau_hat_log = pred3[:, :, 2:3] * y_std3[:, :, 2:3] + y_mean3[:, :, 2:3]
            tau_tgt_log = tgt3[:, :, 2:3]  * y_std3[:, :, 2:3] + y_mean3[:, :, 2:3]

            tau_hat_ns = torch.pow(10.0, tau_hat_log) - cfg.tau_log_eps_ns
            tau_tgt_ns = torch.pow(10.0, tau_tgt_log) - cfg.tau_log_eps_ns
            tau_hat_ns = torch.clamp(tau_hat_ns, min=0.0)
            tau_tgt_ns = torch.clamp(tau_tgt_ns, min=0.0)

            l_tau_phys = masked_smooth_l1(tau_hat_ns, tau_tgt_ns, tau_mask)

            # hinge penalty discouraging huge tau
            denom = tau_mask.sum().clamp_min(1.0)
            l_tau_cap = ((F.relu(tau_hat_ns - cfg.tau_cap_ns) ** 2) * tau_mask).sum() / denom

        #l_tau = l_tau_tgt + cfg.tau_phys_loss_w * l_tau_phys + cfg.tau_cap_w * l_tau_cap
        l_tau = l_tau_tgt

        # combine
        l_wb = 0.5 * l_wb_db + 0.5 * l_wb_gain_nmse   # tweak weights as you like
        return 1.4*l_wb + 0.5*l_ex + 0.8*l_tau


    # training loop
    best_val = float("inf")
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        for xb, yb, m_path, m_ex, m_tau in train_dl:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)
            m_path = m_path.to(cfg.device, non_blocking=True)
            m_ex = m_ex.to(cfg.device, non_blocking=True)
            m_tau = m_tau.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device.startswith("cuda"))):
                pred = model(xb)
                loss = loss_fn(pred, yb, m_path, m_ex, m_tau, y_mean, y_std)

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()

        tr_loss /= max(len(train_dl), 1)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb, m_path, m_ex, m_tau in val_dl:
                xb = xb.to(cfg.device, non_blocking=True)
                yb = yb.to(cfg.device, non_blocking=True)
                m_path = m_path.to(cfg.device, non_blocking=True)
                m_ex = m_ex.to(cfg.device, non_blocking=True)
                m_tau = m_tau.to(cfg.device, non_blocking=True)
                pred = model(xb)
                va_loss += loss_fn(pred, yb, m_path, m_ex, m_tau, y_mean, y_std).item()

        va_loss /= max(len(val_dl), 1)

        dt = time.time() - t0
        print(f"ep {ep:03d}  train={tr_loss:.4f}  val={va_loss:.4f}  ({dt:.1f}s)")

        report_metrics(model, val_dl, stats_dev, y_ch, H, W, tag="val")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), state_path)
            print("  saved best state:", state_path)

            # TorchScript export (state dict + scripted)
            example = torch.randn(1, in_ch, H, W, device=cfg.device)
            try:
                scripted = torch.jit.trace(model, example)
                scripted.save(str(jit_path))
                print("  saved TorchScript:", jit_path)
            except Exception as e:
                print("  TorchScript export failed:", repr(e))

    print("Done.")


if __name__ == "__main__":
    main()
