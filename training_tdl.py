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
    out_dir: str = "runs/wb_ex_delay"

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
    fft_size: int = 512
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
    requested_features: list[str] = field(default_factory=lambda: [
        "binary_walls", "electrical_distance", "cost", "height_cond"
    ])


    # excess delay smoothing
    smooth_kind: str = "median"   # "median" or "gaussian" or "none"
    smooth_median_size: int = 3
    smooth_gauss_sigma: float = 1.0

    # dataset size
    num_scenes: int = 20
    train_frac: float = 0.9
    seed: int = 0

    # training
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 2e-4
    epochs: int = 20
    base: int = 32
    groups: int = 8
    dropout: float = 0.1
    grad_clip: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

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

            wb_db, ex_s, taps = compute_tdl_batch(
                si_scene=si,
                tx_xyz=tx,
                rx_xyz=rx_coords[i0:i1],
                frequencies_hz=freqs,
                L_taps=N,
                rt=cfg.rt,
            )

            wb_all[i0:i1] = wb_db
            ex_all[i0:i1] = ex_s * 1e9  # ns

            good = wb_db < cfg.no_path_wb_db
            if np.any(good):
                tau_rms_s = tau_rms_from_taps(taps[good], cfg.subcarrier_spacing_hz)
                idx_g = np.nonzero(good)[0]
                tau_all[i0 + idx_g] = tau_rms_s * 1e9  # ns

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

def compute_norm_stats(x_mm: np.memmap, y_mm: np.memmap, max_samples: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_total = x_mm.shape[0]
    take = min(max_samples, n_total)
    idx = rng.choice(n_total, size=take, replace=False)

    x = np.array(x_mm[idx], dtype=np.float32)  # (S,C,H,W)
    y = np.array(y_mm[idx], dtype=np.float32)  # (S,Y,H,W)

    x_mean = torch.from_numpy(x.mean(axis=(0, 2, 3))).view(-1, 1, 1)
    x_std  = torch.from_numpy(x.std(axis=(0, 2, 3))).view(-1, 1, 1)
    y_mean = torch.from_numpy(y.mean(axis=(0, 2, 3))).view(-1, 1, 1)
    y_std  = torch.from_numpy(y.std(axis=(0, 2, 3))).view(-1, 1, 1)

    x_std = torch.clamp(x_std, min=1e-6)
    y_std = torch.clamp(y_std, min=1e-6)
    return dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


class MemmapIndexDataset(Dataset):
    def __init__(self, x_mm: np.memmap, y_mm: np.memmap, indices: np.ndarray, stats: dict, no_path_wb_db: float):
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.indices = indices.astype(np.int64)
        self.stats = stats
        self.no_path_wb_db = float(no_path_wb_db)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        x = torch.from_numpy(np.array(self.x_mm[j], dtype=np.float32))
        y = torch.from_numpy(np.array(self.y_mm[j], dtype=np.float32))

        mask = (y[0] < self.no_path_wb_db).to(torch.float32)

        x = (x - self.stats["x_mean"]) / self.stats["x_std"]
        y = (y - self.stats["y_mean"]) / self.stats["y_std"]
        return x, y, mask


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
def report_metrics(model, dl, stats_dev, tag="val",
                   ex_nmse_thresh_ns=0.5, tau_nmse_thresh_ns=5.0):
    model.eval()

    sum_mask = 0.0

    sum_wb_mae = 0.0
    wb_nmse_num = 0.0
    wb_nmse_den = 0.0

    sum_ex_mae = 0.0
    sum_ex_mse = 0.0
    ex_nmse_num = 0.0
    ex_nmse_den = 0.0

    sum_tau_mae = 0.0
    sum_tau_mse = 0.0
    tau_nmse_num = 0.0
    tau_nmse_den = 0.0

    for xb, yb, mb in dl:
        xb = xb.to(cfg.device, non_blocking=True)
        yb = yb.to(cfg.device, non_blocking=True)
        mb = mb.to(cfg.device, non_blocking=True)
        m = mb.unsqueeze(1)

        pred = model(xb)

        y_mean = stats_dev["y_mean"]
        y_std  = stats_dev["y_std"]
        pred_p = pred * y_std + y_mean
        tgt_p  = yb   * y_std + y_mean

        denom = m.sum().clamp_min(1.0)
        sum_mask += denom.item()

        # wb
        wb_err = (pred_p[:,0:1] - tgt_p[:,0:1]).abs()
        sum_wb_mae += (wb_err * m).sum().item()

        g_hat = torch.pow(10.0, -pred_p[:,0:1] / 10.0)
        g_tgt = torch.pow(10.0, -tgt_p[:,0:1]  / 10.0)
        wb_nmse_num += (((g_hat - g_tgt)**2) * m).sum().item()
        wb_nmse_den += ((g_tgt**2) * m).sum().clamp_min(1e-12).item()

        # excess delay
        ex_err = pred_p[:,1:2] - tgt_p[:,1:2]
        sum_ex_mae += (ex_err.abs() * m).sum().item()
        sum_ex_mse += ((ex_err**2) * m).sum().item()

        ex_mask = (tgt_p[:,1:2] >= ex_nmse_thresh_ns).to(tgt_p.dtype) * m
        ex_nmse_num += ((ex_err**2) * ex_mask).sum().item()
        ex_nmse_den += ((tgt_p[:,1:2]**2) * ex_mask).sum().clamp_min(1e-12).item()

        # tau_rms
        tau_err = pred_p[:,2:3] - tgt_p[:,2:3]
        sum_tau_mae += (tau_err.abs() * m).sum().item()
        sum_tau_mse += ((tau_err**2) * m).sum().item()

        tau_mask = (tgt_p[:,2:3] >= tau_nmse_thresh_ns).to(tgt_p.dtype) * m
        tau_nmse_num += ((tau_err**2) * tau_mask).sum().item()
        tau_nmse_den += ((tgt_p[:,2:3]**2) * tau_mask).sum().clamp_min(1e-12).item()

    wb_mae = sum_wb_mae / max(sum_mask, 1e-12)
    wb_nmse = wb_nmse_num / max(wb_nmse_den, 1e-12)
    wb_nmse_db = 10.0 * math.log10(max(wb_nmse, 1e-12))

    ex_mae = sum_ex_mae / max(sum_mask, 1e-12)
    ex_rmse = math.sqrt(sum_ex_mse / max(sum_mask, 1e-12))
    ex_nmse = ex_nmse_num / max(ex_nmse_den, 1e-12)
    ex_nmse_db = 10.0 * math.log10(max(ex_nmse, 1e-12))

    tau_mae = sum_tau_mae / max(sum_mask, 1e-12)
    tau_rmse = math.sqrt(sum_tau_mse / max(sum_mask, 1e-12))
    tau_nmse = tau_nmse_num / max(tau_nmse_den, 1e-12)
    tau_nmse_db = 10.0 * math.log10(max(tau_nmse, 1e-12))

    print(
        f"  [{tag}] wb_MAE={wb_mae:.3f} dB | wb_NMSE={wb_nmse:.4f} ({wb_nmse_db:.1f} dB) | "
        f"ex_MAE={ex_mae:.3f} ns | ex_RMSE={ex_rmse:.3f} ns | ex_NMSE={ex_nmse:.4f} ({ex_nmse_db:.1f} dB) | "
        f"tauRMS_MAE={tau_mae:.3f} ns | tauRMS_RMSE={tau_rmse:.3f} ns | tauRMS_NMSE={tau_nmse:.4f} ({tau_nmse_db:.1f} dB)"
    )


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
    x_tmp = build_feature_tensor(tmp, cfg.frequency_hz, requested=cfg.requested_features).astype(np.float32)
    c_in = int(x_tmp.shape[1])
    num_tx = int(tmp.antenna_database.tx_coords.shape[0])
    K, H, W = tmp.antenna_database.rx_grid.shape

    y_ch = 3

    total_samples = cfg.num_scenes * num_tx * K

    x_path = out_dir / "x.dat"
    y_path = out_dir / "y.dat"
    meta_path = out_dir / "meta.json"
    stats_path = out_dir / "norm_stats.npz"
    state_path = out_dir / "model_state.pt"
    jit_path = out_dir / "model.pt"

    print("x exists?", x_path.exists(), "y exists?", y_path.exists())

    # Build dataset if missing
    if not (x_path.exists() and y_path.exists() and meta_path.exists()):
        print("Building memmap dataset...")
        x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(total_samples, c_in, H, W))
        y_mm = np.memmap(y_path, dtype="float16", mode="w+", shape=(total_samples, y_ch, H, W))

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
            y_channels=["wb_loss_db", "excess_delay_ns_sm", "tau_rms_ns_sm"]
        )
        meta_path.write_text(json.dumps(meta, indent=2))
        print("Wrote meta (early):", meta_path)

        idx = 0
        for s in range(cfg.num_scenes):
            scene = make_scene(rng)

            x = build_feature_tensor(scene, cfg.frequency_hz, requested=cfg.requested_features).astype(np.float32)
            y = compute_labels_for_scene(scene)

            x_flat = x.transpose(0, 2, 1, 3, 4).reshape(num_tx * K, c_in, H, W)
            y_flat = y.transpose(0, 2, 1, 3, 4).reshape(num_tx * K, y_ch, H, W)

            n = x_flat.shape[0]
            x_mm[idx:idx+n] = x_flat
            y_mm[idx:idx+n] = y_flat.astype(np.float16)
            idx += n
            x_mm.flush(); y_mm.flush()
            print(f"[scene {s+1:03d}/{cfg.num_scenes}] wrote {n} samples (total {idx})")

    # reopen memmaps
    x_mm = np.memmap(x_path, dtype="float32", mode="r", shape=(total_samples, c_in, H, W))
    y_mm = np.memmap(y_path, dtype="float16", mode="r", shape=(total_samples, y_ch, H, W))

    # quick sanity stats
    samp = np.random.default_rng(cfg.seed + 1).choice(total_samples, size=min(256, total_samples), replace=False)
    wb = np.array(y_mm[samp, 0], dtype=np.float32)
    ex = np.array(y_mm[samp, 1], dtype=np.float32)
    tr = np.array(y_mm[samp, 2], dtype=np.float32)
    print(f"wb_loss(dB): mean={wb.mean():.2f} std={wb.std():.2f} min={wb.min():.2f} max={wb.max():.2f}")
    print(f"excess_delay_sm(ns): mean={ex.mean():.2f} std={ex.std():.2f} min={ex.min():.2f} max={ex.max():.2f}")
    print(f"tau_rms_sm(ns): mean={tr.mean():.2f} std={tr.std():.2f} min={tr.min():.2f} max={tr.max():.2f}")

    # norm stats
    stats = compute_norm_stats(x_mm, y_mm, max_samples=512, seed=cfg.seed)
    np.savez(
        stats_path,
        x_mean=stats["x_mean"].numpy(), x_std=stats["x_std"].numpy(),
        y_mean=stats["y_mean"].numpy(), y_std=stats["y_std"].numpy(),
    )
    print("Saved stats:", stats_path)

    stats_dev = {k: v.to(cfg.device) for k, v in stats.items()}

    # split indices
    perm = np.random.default_rng(cfg.seed).permutation(total_samples)
    n_train = int(cfg.train_frac * total_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = MemmapIndexDataset(x_mm, y_mm, train_idx, stats, cfg.no_path_wb_db)
    val_ds   = MemmapIndexDataset(x_mm, y_mm, val_idx,   stats, cfg.no_path_wb_db)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # model
    model = UNet3(in_ch=c_in, out_ch=y_ch, base=cfg.base, groups=cfg.groups, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

    def loss_fn(pred, tgt, mask):
        m = mask.unsqueeze(1)
        denom = m.sum().clamp_min(1.0)

        l_wb  = F.smooth_l1_loss(pred[:,0:1]*m, tgt[:,0:1]*m, reduction="sum") / denom
        l_ex  = F.smooth_l1_loss(pred[:,1:2]*m, tgt[:,1:2]*m, reduction="sum") / denom
        l_tau = F.smooth_l1_loss(pred[:,2:3]*m, tgt[:,2:3]*m, reduction="sum") / denom

        return 1.0*l_wb + 0.5*l_ex + 0.25*l_tau


    # training loop
    best_val = float("inf")
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        for xb, yb, mb in train_dl:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)
            mb = mb.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device.startswith("cuda"))):
                pred = model(xb)
                loss = loss_fn(pred, yb, mb, ep)

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
            for xb, yb, mb in val_dl:
                xb = xb.to(cfg.device, non_blocking=True)
                yb = yb.to(cfg.device, non_blocking=True)
                mb = mb.to(cfg.device, non_blocking=True)
                pred = model(xb)
                va_loss += loss_fn(pred, yb, mb, ep).item()

        va_loss /= max(len(val_dl), 1)

        dt = time.time() - t0
        print(f"ep {ep:03d}  train={tr_loss:.4f}  val={va_loss:.4f}  ({dt:.1f}s)")

        report_metrics(model, val_dl, stats_dev, tag="val")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), state_path)
            print("  saved best state:", state_path)

            # TorchScript export (state dict + scripted)
            example = torch.randn(1, c_in, H, W, device=cfg.device)
            try:
                scripted = torch.jit.trace(model, example)
                scripted.save(str(jit_path))
                print("  saved TorchScript:", jit_path)
            except Exception as e:
                print("  TorchScript export failed:", repr(e))

    print("Done.")


if __name__ == "__main__":
    main()
