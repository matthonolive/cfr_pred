import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import math
import shutil
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
from mlink.feature import build_feature_tensor, REGISTRY, Specification
from mlink.geometry import generate_wall_map, walls_to_mesh
from mlink.scene import Scene
from mlink.channel_tdl import RtCfg, subcarrier_frequencies_centered, compute_tdl_batch

from trimesh.intersections import mesh_plane

C0 = 299_792_458.0


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    # fine-tune output / base run
    out_dir: str = "runs/delta_3072_offpatch_finetune"
    base_run: str = "runs/delta_3072_10int"

    # local patch fed to the U-Net (must match pretrained run)
    img_hw: tuple[int, int] = (64, 64)
    K_slices: int = 4
    z_step: float = 1.0
    z_margin: float = 0.5
    floor_h: float = 0.0
    scale: float = 0.625

    # larger scenes from which we sample off-patch crops
    full_h_min: int = 96
    full_h_max: int = 160
    full_w_min: int = 96
    full_w_max: int = 160
    ceil_min: float = 8.0
    ceil_max: float = 20.0

    # off-patch TX sampling
    tx_z: float = 2.4
    patches_per_scene: int = 6
    min_tx_patch_gap_cells: int = 1
    max_tx_patch_offset_cells: int = 48

    # OFDM / RT
    frequency_hz: float = 5.21e9
    fft_size: int = 3072
    subcarrier_spacing_hz: float = 78_125.0
    rx_batch: int = 256
    no_path_wb_db: float = 199.5
    rt: RtCfg = field(default_factory=lambda: RtCfg(
        max_depth=10,
        samples_per_src=1_000_000,
        diffuse_reflection=True,
        diffraction=True,
        edge_diffraction=True,
        diffraction_lit_region=True,
    ))

    # features
    dataset_features: list[str] = field(default_factory=lambda: [
        "binary_walls", "electrical_distance", "cost", "height_cond"
    ])
    model_features: list[str] = field(default_factory=lambda: [
        "binary_walls", "electrical_distance", "cost", "height_cond"
    ])

    # smoothing
    smooth_kind: str = "median"
    smooth_median_size: int = 3
    smooth_gauss_sigma: float = 1.0

    # dataset size
    num_scenes: int = 40
    train_frac: float = 0.8
    seed: int = 20001999

    # model/training (must match the old architecture)
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 5e-5
    epochs: int = 10
    base: int = 48
    groups: int = 8
    dropout: float = 0.1
    grad_clip: float = 1.0
    weight_decay: float = 1e-4

    ex_loss_thresh_ns: float = 0.5
    tau_loss_thresh_ns: float = 0.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    tau_target: str = "raw"  # or "log10"
    tau_log_eps_ns: float = 1e-3
    tau_cap_ns: float = 50.0
    tau_phys_loss_w: float = 0.0
    tau_cap_w: float = 0.0

    hard_thr_db: float = 8.0
    hard_gain: float = 8.0
    hard_soft_db: float = 2.0
    delta_clip_lo: float = -30.0
    delta_clip_hi: float = 80.0


cfg = CFG()


# ----------------------------
# Feature patch: safer binary_walls for local off-patch crops
# ----------------------------
def binary_walls_offpatch(scene: Scene, frequency: float) -> np.ndarray:
    mesh = scene.mesh
    z_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    rx_grid = scene.antenna_database.rx_grid
    if rx_grid is None:
        raise Exception("Receivers must be initialized with a grid!")

    K, H, W = rx_grid.shape

    def rasterize_line(line_xyz: np.ndarray) -> np.ndarray:
        src = np.asarray(rx_grid.xyz2ijk(line_xyz[0, :]), dtype=np.float32)
        dst = np.asarray(rx_grid.xyz2ijk(line_xyz[1, :]), dtype=np.float32)
        n = int(max(np.max(np.abs(dst - src)), 1)) + 1
        pts = np.linspace(src, dst, num=n, endpoint=True).astype(np.int32)

        keep = (
            (pts[:, 0] >= 0) & (pts[:, 0] < H) &
            (pts[:, 1] >= 0) & (pts[:, 1] < W)
        )
        pts = pts[keep]
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        return pts[:, :2]

    wall_tensor_lst = []
    for k in range(K):
        plane_origin = rx_grid.origin + k * rx_grid.deltas[2]
        lines = mesh_plane(
            mesh,
            plane_normal=z_normal,
            plane_origin=plane_origin,
            return_faces=False,
        )

        walls = np.zeros((H, W), dtype=np.float32)
        if lines is not None and len(lines) > 0:
            for line in lines:
                ij = rasterize_line(np.asarray(line))
                if ij.shape[0] > 0:
                    walls[ij[:, 0], ij[:, 1]] = 1.0

        wall_tensor_lst.append(walls)

    wall_tensor = np.stack(wall_tensor_lst, axis=0).astype(np.float32)  # (K,H,W)
    wall_maps = wall_tensor[None, None, :, :, :]
    wall_maps = np.repeat(wall_maps, repeats=scene.antenna_database.tx_coords.shape[0], axis=0)
    return wall_maps


REGISTRY["binary_walls"] = Specification(
    name="binary_walls",
    requires=(),
    fn=binary_walls_offpatch,
)


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


def fspl_db(d_m: np.ndarray, fc_hz: float) -> np.ndarray:
    d = np.maximum(d_m, 1e-3)
    lam = C0 / float(fc_hz)
    return (20.0 * np.log10(4.0 * np.pi * d / lam)).astype(np.float32)


def tau_to_target(tau_ns: np.ndarray) -> np.ndarray:
    tau_ns = np.maximum(tau_ns, 0.0).astype(np.float32)
    if cfg.tau_target == "raw":
        return tau_ns
    if cfg.tau_target == "log10":
        return np.log10(tau_ns + cfg.tau_log_eps_ns).astype(np.float32)
    raise ValueError("cfg.tau_target must be 'raw' or 'log10'")


def apply_y_transform_np(y: np.ndarray, K: int, y_ch: int) -> np.ndarray:
    if cfg.tau_target == "raw":
        return y
    for k in range(K):
        tau_idx = k * y_ch + 2
        y[:, tau_idx, :, :] = tau_to_target(y[:, tau_idx, :, :])
    return y


def infer_feature_channel_counts(scene, freq_hz, features):
    counts = {}
    for f in features:
        x = build_feature_tensor(scene, freq_hz, requested=[f]).astype(np.float32)
        counts[f] = int(x.shape[1])
    return counts


def build_keep_idx(dataset_features, model_features, K, feat_counts):
    offsets = {}
    off = 0
    for f in dataset_features:
        offsets[f] = (off, off + feat_counts[f])
        off += feat_counts[f]
    c_full = off

    keep_in_slice = []
    for f in model_features:
        a, b = offsets[f]
        keep_in_slice.extend(range(a, b))

    keep = []
    for k in range(K):
        base = k * c_full
        keep.extend([base + i for i in keep_in_slice])

    return np.asarray(keep, dtype=np.int64), c_full


def _to_sionna_geometry(scene: Scene, freq: float):
    if hasattr(scene, "to_sionna_geometry"):
        return scene.to_sionna_geometry(freq)
    return scene.to_sionna(freq)


# ----------------------------
# Full-scene + patch sampling
# ----------------------------
def make_full_scene(rng: np.random.Generator):
    full_H = int(rng.integers(cfg.full_h_min, cfg.full_h_max + 1))
    full_W = int(rng.integers(cfg.full_w_min, cfg.full_w_max + 1))

    walls_2d = generate_wall_map(
        (full_H, full_W),
        min_wall_length=8,
        min_door_length=4,
        max_partitions=24,
        rng=rng,
    )

    ceiling_h_units = float(rng.uniform(cfg.ceil_min, cfg.ceil_max))

    mesh = walls_to_mesh(
        walls_2d,
        floor_height=cfg.floor_h,
        ceiling_height=ceiling_h_units,
    ).apply_scale(cfg.scale)

    usable = max(ceiling_h_units - cfg.floor_h - 2 * cfg.z_margin, 1e-3)
    total_span = (cfg.K_slices - 1) * cfg.z_step
    z_step_units = usable / max(cfg.K_slices - 1, 1) if total_span > usable else cfg.z_step

    z_start = cfg.floor_h + cfg.z_margin
    z_end = (ceiling_h_units - cfg.z_margin) - total_span
    z0_units = z_start if z_end < z_start else float(rng.uniform(z_start, z_end))

    mat_db = default_material_db(cfg.frequency_hz)
    face2material = {k: 0 for k in range(mesh.faces.shape[0])}
    empty = np.empty((0, 3), dtype=np.float32)

    base_scene = Scene(
        mesh=mesh,
        material_database=mat_db,
        face2material=face2material,
        antenna_database=AntennaDatabase(empty, empty, None, None),
    )

    full_meta = {
        "full_H": full_H,
        "full_W": full_W,
        "z0_m": float(cfg.scale * z0_units),
        "z_step_m": float(cfg.scale * z_step_units),
        "ceiling_h_m": float(cfg.scale * ceiling_h_units),
    }
    return base_scene, walls_2d, full_meta


def sample_free_tx_xyz(walls_2d: np.ndarray, ceiling_h_m: float, rng: np.random.Generator) -> np.ndarray:
    free = np.argwhere(np.asarray(walls_2d) == 0)
    if free.shape[0] == 0:
        raise RuntimeError("No free cells available for TX sampling.")

    rc = free[rng.integers(0, free.shape[0])]
    x_m = (float(rc[0]) + 0.5) * cfg.scale
    y_m = (float(rc[1]) + 0.5) * cfg.scale

    z_lo = cfg.scale * cfg.floor_h + cfg.scale * cfg.z_margin
    z_hi = ceiling_h_m - cfg.scale * cfg.z_margin
    z_m = np.clip(cfg.scale * cfg.tx_z, z_lo, z_hi)

    return np.asarray([x_m, y_m, float(z_m)], dtype=np.float32)


def sample_patch_top_left(
    full_H: int,
    full_W: int,
    patch_H: int,
    patch_W: int,
    tx_xyz_m: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    tx_i = int(np.floor(tx_xyz_m[0] / cfg.scale))
    tx_j = int(np.floor(tx_xyz_m[1] / cfg.scale))

    i_max = full_H - patch_H
    j_max = full_W - patch_W
    if i_max < 0 or j_max < 0:
        raise RuntimeError("Patch larger than full scene.")

    def dist_to_patch(i0: int, j0: int) -> float:
        if i0 <= tx_i < i0 + patch_H:
            di = 0
        else:
            di = min(abs(tx_i - i0), abs(tx_i - (i0 + patch_H - 1)))
        if j0 <= tx_j < j0 + patch_W:
            dj = 0
        else:
            dj = min(abs(tx_j - j0), abs(tx_j - (j0 + patch_W - 1)))
        return float(np.hypot(di, dj))

    for _ in range(4000):
        i0 = int(rng.integers(0, i_max + 1))
        j0 = int(rng.integers(0, j_max + 1))

        inside = (i0 <= tx_i < i0 + patch_H) and (j0 <= tx_j < j0 + patch_W)
        if inside:
            continue

        d = dist_to_patch(i0, j0)
        if d < cfg.min_tx_patch_gap_cells:
            continue
        if d > cfg.max_tx_patch_offset_cells:
            continue

        return i0, j0

    raise RuntimeError("Failed to sample an off-patch crop for this TX.")


def make_patch_scene_from_full(base_scene: Scene, tx_xyz_m: np.ndarray, full_meta: dict, i0: int, j0: int) -> Scene:
    patch_H, patch_W = cfg.img_hw

    origin = np.asarray([
        i0 * cfg.scale,
        j0 * cfg.scale,
        full_meta["z0_m"],
    ], dtype=np.float32)

    rx_grid = AntennaGrid(
        origin=origin,
        deltas=np.asarray([
            [cfg.scale, 0.0, 0.0],
            [0.0, cfg.scale, 0.0],
            [0.0, 0.0, full_meta["z_step_m"]],
        ], dtype=np.float32),
        shape=(cfg.K_slices, patch_H, patch_W),
    )

    k, i, j = np.meshgrid(
        np.arange(cfg.K_slices),
        np.arange(patch_H),
        np.arange(patch_W),
        indexing="ij",
    )
    rx_coords = rx_grid.ijk2xyz(i, j, k).reshape(-1, 3).astype(np.float32)

    adb = AntennaDatabase(
        tx_xyz_m.reshape(1, 3).astype(np.float32),
        rx_coords,
        None,
        rx_grid,
    )
    return replace(base_scene, antenna_database=adb)


# ----------------------------
# Labels for a patch scene
# ----------------------------
def compute_labels_for_scene(scene: Scene) -> np.ndarray:
    """
    Returns y of shape (num_tx, 4, K, H, W) float32.

    Channels:
      0: delta_pl_db
      1: excess_delay_ns (smoothed)
      2: tau_rms_ns (smoothed)
      3: wb_loss_db
    """
    rx_grid = scene.antenna_database.rx_grid
    assert rx_grid is not None
    K, H_img, W_img = rx_grid.shape

    tx_coords = scene.antenna_database.tx_coords
    rx_coords = scene.antenna_database.rx_coords
    P = rx_coords.shape[0]

    N = int(cfg.fft_size)
    y = np.zeros((tx_coords.shape[0], 4, K, H_img, W_img), dtype=np.float32)

    freqs = subcarrier_frequencies_centered(cfg.fft_size, cfg.subcarrier_spacing_hz)
    si = _to_sionna_geometry(scene, cfg.frequency_hz)

    for t, tx in enumerate(tx_coords):
        wb_all = np.zeros((P,), dtype=np.float32)
        ex_all = np.zeros((P,), dtype=np.float32)
        tau_all = np.zeros((P,), dtype=np.float32)

        for i0 in range(0, P, cfg.rx_batch):
            i1 = min(i0 + cfg.rx_batch, P)

            wb_db, ex_s, taps, tau_rms_s = compute_tdl_batch(
                si_scene=si,
                tx_xyz=tx,
                rx_xyz=rx_coords[i0:i1],
                frequencies_hz=freqs,
                L_taps=N,
                rt=cfg.rt,
                return_tau_rms=True,
            )

            wb_all[i0:i1] = wb_db
            ex_all[i0:i1] = ex_s * 1e9

            good = wb_db < cfg.no_path_wb_db
            if np.any(good):
                idx_g = np.nonzero(good)[0]
                tau_all[i0 + idx_g] = tau_rms_s[good] * 1e9

        wb_map = wb_all.reshape(K, H_img, W_img)
        ex_map = ex_all.reshape(K, H_img, W_img)
        tau_map = tau_all.reshape(K, H_img, W_img)

        ex_sm = smooth_map_stack(ex_map, wb_map)
        tau_sm = smooth_map_stack(tau_map, wb_map)

        ex_sm = np.maximum(ex_sm, 0.0)
        tau_sm = np.maximum(tau_sm, 0.0)

        d_m = np.linalg.norm(rx_coords - tx[None, :], axis=1).reshape(K, H_img, W_img).astype(np.float32)
        fspl = fspl_db(d_m, cfg.frequency_hz)

        valid = wb_map < cfg.no_path_wb_db
        delta = (wb_map - fspl).astype(np.float32)
        delta = np.clip(delta, cfg.delta_clip_lo, cfg.delta_clip_hi)

        delta[~valid] = 0.0
        ex_sm[~valid] = 0.0
        tau_sm[~valid] = 0.0

        y[t, 0] = delta
        y[t, 1] = ex_sm
        y[t, 2] = tau_sm
        y[t, 3] = wb_map

        print(f"  tx {t+1}/{tx_coords.shape[0]} labels done")

    return y


# ----------------------------
# Dataset / stats
# ----------------------------
def to_c11(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None, None]
    return a


def load_base_stats(base_run: Path):
    stats_npz = np.load(base_run / "norm_stats.npz")
    out = {
        "x_mean": torch.from_numpy(to_c11(stats_npz["x_mean"])).float(),
        "x_std": torch.from_numpy(to_c11(stats_npz["x_std"])).float().clamp_min(1e-6),
        "y_mean": torch.from_numpy(to_c11(stats_npz["y_mean"])).float(),
        "y_std": torch.from_numpy(to_c11(stats_npz["y_std"])).float().clamp_min(1e-6),
    }
    if "keep_idx" in stats_npz.files:
        ki = np.asarray(stats_npz["keep_idx"], dtype=np.int64)
        if ki.size > 0:
            out["keep_idx"] = torch.from_numpy(ki)
    return out, stats_npz


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
        augment: bool = False
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

        self.augment = augment

        self.ex_loss_thresh_ns = float(ex_loss_thresh_ns)
        self.tau_loss_thresh_ns = float(tau_loss_thresh_ns)

        self.keep_idx = None
        if keep_idx is not None:
            self.keep_idx = np.asarray(keep_idx, dtype=np.int64)

        if self.keep_idx is not None:
            if int(self.stats["x_mean"].shape[0]) != int(self.keep_idx.shape[0]):
                raise ValueError(
                    f"stats['x_mean'] has {int(self.stats['x_mean'].shape[0])} channels but keep_idx has {int(self.keep_idx.shape[0])}."
                )

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])

        x_np = np.array(self.x_mm[j], dtype=np.float32)
        y_np = np.array(self.y_mm[j], dtype=np.float32)

        if self.keep_idx is not None:
            x_np = x_np[self.keep_idx, :, :]

        y_phys = y_np.astype(np.float32, copy=False)
        y3_phys = torch.from_numpy(y_phys).view(self.K, self.y_ch, self.H, self.W)

        ex = y3_phys[:, 1]
        tau = y3_phys[:, 2]
        wb = y3_phys[:, 3]

        mask_path = wb < self.no_path_wb_db
        mask_ex = mask_path & (ex >= self.ex_loss_thresh_ns)
        mask_tau = mask_path & (tau >= self.tau_loss_thresh_ns)

        y_tf = y_phys.copy()
        y_tf_ = y_tf[None, ...]
        apply_y_transform_np(y_tf_, K=self.K, y_ch=self.y_ch)
        y_tf = y_tf_[0]

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_tf)

        x = (x - self.stats["x_mean"]) / self.stats["x_std"]
        y = (y - self.stats["y_mean"]) / self.stats["y_std"]

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, dims=(-2, -1))
                y = torch.rot90(y, k, dims=(-2, -1))
                mask_path = torch.rot90(mask_path, k, dims=(-2, -1))
                mask_ex = torch.rot90(mask_ex, k, dims=(-2, -1))
                mask_tau = torch.rot90(mask_tau, k, dims=(-2, -1))
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])
                mask_path = torch.flip(mask_path, dims=[-1])
                mask_ex = torch.flip(mask_ex, dims=[-1])
                mask_tau = torch.flip(mask_tau, dims=[-1])

        return x, y, mask_path.float(), mask_ex.float(), mask_tau.float()


# ----------------------------
# Model
# ----------------------------
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

class SelfAttention2d(nn.Module):
    """Lightweight self-attention for small spatial resolutions."""
    def __init__(self, ch: int):
        super().__init__()
        g = _valid_groups(ch, 8)
        self.norm = nn.GroupNorm(g, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-1, -2) @ k * self.scale).softmax(dim=-1)
        out = (v @ attn).reshape(B, C, H, W)
        return x + self.proj(out)

class UNet3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 32, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.e1 = nn.Sequential(
            ResBlock(in_ch, base, groups=groups, dropout=dropout),
            ResBlock(base, base, groups=groups, dropout=dropout),
        )
        self.p1 = nn.MaxPool2d(2)

        self.e2 = nn.Sequential(
            ResBlock(base, base * 2, groups=groups, dropout=dropout),
            ResBlock(base * 2, base * 2, groups=groups, dropout=dropout),
        )
        self.p2 = nn.MaxPool2d(2)

        self.e3 = nn.Sequential(
            ResBlock(base * 2, base * 4, groups=groups, dropout=dropout),
            ResBlock(base * 4, base * 4, groups=groups, dropout=dropout),
        )
        self.p3 = nn.MaxPool2d(2)

        self.mid = nn.Sequential(
            ResBlock(base * 4, base * 8, groups=groups, dropout=dropout),
            SelfAttention2d(base * 8),
            ResBlock(base * 8, base * 8, groups=groups, dropout=dropout),
        )

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = nn.Sequential(
            ResBlock(base * 8, base * 4, groups=groups, dropout=0.0),
            ResBlock(base * 4, base * 4, groups=groups, dropout=0.0),
        )

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = nn.Sequential(
            ResBlock(base * 4, base * 2, groups=groups, dropout=0.0),
            ResBlock(base * 2, base * 2, groups=groups, dropout=0.0),
        )

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = nn.Sequential(
            ResBlock(base * 2, base, groups=groups, dropout=0.0),
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


# ----------------------------
# Metrics / loss
# ----------------------------
@torch.no_grad()
def report_metrics(model, dl, stats_dev, y_ch: int, H: int, W: int, tag="val", max_batches=None):
    model.eval()

    sum_path = 0.0
    sum_wb_mae = 0.0
    sum_delta_mae = 0.0
    sum_ex_mae_path = 0.0
    sum_tau_mae_path = 0.0
    wb_nmse_num = 0.0
    wb_nmse_den = 0.0

    n_batches = 0
    for xb, yb, m_path, m_ex, m_tau in dl:
        xb = xb.to(cfg.device, non_blocking=True)
        yb = yb.to(cfg.device, non_blocking=True)
        m_path = m_path.to(cfg.device, non_blocking=True)

        pred = model(xb)

        y_mean = stats_dev["y_mean"]
        y_std = stats_dev["y_std"]
        pred_p = pred * y_std + y_mean
        tgt_p = yb * y_std + y_mean

        B = pred_p.shape[0]
        K = cfg.K_slices
        pred_p = pred_p.view(B, K, y_ch, H, W)
        tgt_p = tgt_p.view(B, K, y_ch, H, W)

        delta_hat = pred_p[:, :, 0]
        delta_tgt = tgt_p[:, :, 0]
        ex_hat = pred_p[:, :, 1]
        ex_tgt = tgt_p[:, :, 1]
        tau_hat = pred_p[:, :, 2]
        tau_tgt = tgt_p[:, :, 2]
        wb_tgt = tgt_p[:, :, 3]
        wb_hat = wb_tgt + (delta_hat - delta_tgt)

        mp = m_path

        sum_path += mp.sum().clamp_min(0.0).item()
        sum_delta_mae += ((delta_hat - delta_tgt).abs() * mp).sum().item()
        sum_ex_mae_path += ((ex_hat - ex_tgt).abs() * mp).sum().item()
        sum_tau_mae_path += ((tau_hat - tau_tgt).abs() * mp).sum().item()
        sum_wb_mae += ((wb_hat - wb_tgt).abs() * mp).sum().item()

        g_hat = torch.pow(10.0, torch.clamp(-wb_hat / 10.0, min=-20.0, max=20.0))
        g_tgt = torch.pow(10.0, torch.clamp(-wb_tgt / 10.0, min=-20.0, max=20.0))
        wb_nmse_num += (((g_hat - g_tgt) ** 2) * mp).sum().item()
        wb_nmse_den += (((g_tgt) ** 2) * mp).sum().clamp_min(1e-12).item()

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    def safe_div(a, b):
        return a / max(b, 1e-12)

    wb_mae = safe_div(sum_wb_mae, sum_path)
    delta_mae = safe_div(sum_delta_mae, sum_path)
    ex_mae = safe_div(sum_ex_mae_path, sum_path)
    tau_mae = safe_div(sum_tau_mae_path, sum_path)
    wb_nmse = safe_div(wb_nmse_num, wb_nmse_den)
    wb_nmse_db = 10.0 * math.log10(max(wb_nmse, 1e-12))

    print(
        f"  [{tag}] delta_MAE(path)={delta_mae:.3f} dB | "
        f"wb_MAE(path)={wb_mae:.3f} dB | wb_NMSE={wb_nmse:.4f} ({wb_nmse_db:.1f} dB) | "
        f"ex_MAE(path)={ex_mae:.3f} ns | tau_MAE(path)={tau_mae:.3f} ns"
    )


def loss_fn(pred, tgt, m_path, m_ex, m_tau, y_mean, y_std, y_ch: int):
    B, _, H, W = pred.shape
    K = cfg.K_slices
    Y = y_ch

    pred3 = pred.view(B, K, Y, H, W)
    tgt3 = tgt.view(B, K, Y, H, W)

    mp = m_path.unsqueeze(2)
    me = m_ex.unsqueeze(2)
    mt = m_tau.unsqueeze(2)

    y_mean3 = y_mean.view(1, K, Y, 1, 1)
    y_std3 = y_std.view(1, K, Y, 1, 1)

    def masked_smooth_l1(a, b, m):
        denom = m.sum().clamp_min(1.0)
        return F.smooth_l1_loss(a * m, b * m, reduction="sum") / denom

    delta_hat = pred3[:, :, 0:1] * y_std3[:, :, 0:1] + y_mean3[:, :, 0:1]
    delta_tgt = tgt3[:, :, 0:1] * y_std3[:, :, 0:1] + y_mean3[:, :, 0:1]

    w_hard = 1.0 + cfg.hard_gain * torch.sigmoid((delta_tgt - cfg.hard_thr_db) / cfg.hard_soft_db)
    w = w_hard * mp
    denom = w.sum().clamp_min(1.0)
    l_delta_db = (F.smooth_l1_loss(delta_hat, delta_tgt, reduction="none") * w).sum() / denom

    wb_tgt = tgt3[:, :, 3:4] * y_std3[:, :, 3:4] + y_mean3[:, :, 3:4]
    wb_hat = wb_tgt + (delta_hat - delta_tgt)

    g_hat = torch.pow(10.0, torch.clamp(-wb_hat / 10.0, min=-20.0, max=20.0))
    g_tgt = torch.pow(10.0, torch.clamp(-wb_tgt / 10.0, min=-20.0, max=20.0))
    l_wb_gain_nmse = ((((g_hat - g_tgt) ** 2) * mp).sum() /
                      (((g_tgt) ** 2) * mp).sum().clamp_min(1e-12))

    l_wb_db = (F.smooth_l1_loss(wb_hat, wb_tgt, reduction="none") * mp).sum() / mp.sum().clamp_min(1.0)

    l_ex = masked_smooth_l1(pred3[:, :, 1:2], tgt3[:, :, 1:2], me)
    l_tau = masked_smooth_l1(pred3[:, :, 2:3], tgt3[:, :, 2:3], mt if mt.sum() > 0 else mp)

    return (
        1.0 * l_delta_db
        + 0.5 * l_wb_gain_nmse
        + 0.1 * l_wb_db
        + 0.5 * l_ex
        + 0.8 * l_tau
    )


# ----------------------------
# Main
# ----------------------------
def main():
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_run = Path(cfg.base_run)
    if not base_run.exists():
        raise FileNotFoundError(f"Base run not found: {base_run}")
    if not (base_run / "model_state.pt").exists():
        raise FileNotFoundError(
            f"Need {base_run/'model_state.pt'} for fine-tuning. Training from only model.pt is not supported here."
        )
    if not (base_run / "norm_stats.npz").exists():
        raise FileNotFoundError(f"Need {base_run/'norm_stats.npz'}")
    if not (base_run / "meta.json").exists():
        raise FileNotFoundError(f"Need {base_run/'meta.json'}")

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
    base_meta = json.loads((base_run / "meta.json").read_text())

    # Build a temporary patch scene to infer feature counts/shapes.
    tmp_base, tmp_walls, tmp_meta = make_full_scene(rng)
    tmp_tx = sample_free_tx_xyz(tmp_walls, tmp_meta["ceiling_h_m"], rng)
    tmp_i0, tmp_j0 = sample_patch_top_left(
        tmp_meta["full_H"], tmp_meta["full_W"],
        cfg.img_hw[0], cfg.img_hw[1],
        tmp_tx, rng,
    )
    tmp_patch = make_patch_scene_from_full(tmp_base, tmp_tx, tmp_meta, tmp_i0, tmp_j0)

    x_tmp = build_feature_tensor(tmp_patch, cfg.frequency_hz, requested=cfg.dataset_features).astype(np.float32)
    c_in = int(x_tmp.shape[1])
    K, H, W = tmp_patch.antenna_database.rx_grid.shape
    y_ch = 4

    if int(base_meta.get("H", H)) != H or int(base_meta.get("W", W)) != W or int(base_meta.get("K", K)) != K:
        raise RuntimeError(
            f"Patch/grid mismatch versus base run: base (H,W,K)=({base_meta.get('H')},{base_meta.get('W')},{base_meta.get('K')}) "
            f"but fine-tune is ({H},{W},{K})."
        )

    feat_counts = infer_feature_channel_counts(tmp_patch, cfg.frequency_hz, cfg.dataset_features)
    keep_idx_calc, c_full = build_keep_idx(cfg.dataset_features, cfg.model_features, K, feat_counts)
    assert c_full == c_in, (c_full, c_in)

    base_stats, _ = load_base_stats(base_run)
    keep_idx = None
    if "keep_idx" in base_stats:
        keep_idx = np.asarray(base_stats["keep_idx"].numpy(), dtype=np.int64)
    in_ch = int(base_stats["x_mean"].shape[0])

    expected_full = c_in * K
    if keep_idx is None and in_ch != expected_full:
        raise RuntimeError(
            f"Base stats expect {in_ch} input channels but current features build {expected_full}."
        )
    if keep_idx is not None and in_ch != keep_idx.shape[0]:
        raise RuntimeError("Base keep_idx and base x_mean disagree.")
    if keep_idx is not None and keep_idx_calc.shape == keep_idx.shape and not np.array_equal(keep_idx_calc, keep_idx):
        print("[warn] Base keep_idx differs from current calculated keep_idx; using base keep_idx for compatibility.")

    total_samples = cfg.num_scenes * cfg.patches_per_scene
    samples_per_scene = cfg.patches_per_scene

    x_path = out_dir / "x.dat"
    y_path = out_dir / "y.dat"
    meta_path = out_dir / "meta.json"
    state_path = out_dir / "model_state.pt"
    jit_path = out_dir / "model.pt"
    stats_path = out_dir / "norm_stats.npz"

    if not (x_path.exists() and y_path.exists() and meta_path.exists()):
        print("Building off-patch fine-tune dataset...")
        x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(total_samples, c_in * K, H, W))
        y_mm = np.memmap(y_path, dtype="float16", mode="w+", shape=(total_samples, y_ch * K, H, W))

        meta = dict(
            total_samples=int(total_samples),
            H=int(H), W=int(W), K=int(K),
            c_in=int(c_in),
            y_ch=int(y_ch),
            fft_size=int(cfg.fft_size),
            subcarrier_spacing_hz=float(cfg.subcarrier_spacing_hz),
            smooth_kind=str(cfg.smooth_kind),
            smooth_median_size=int(cfg.smooth_median_size),
            smooth_gauss_sigma=float(cfg.smooth_gauss_sigma),
            frequency_hz=float(cfg.frequency_hz),
            x_dtype="float32", y_dtype="float16",
            y_channels=["delta_pl_db", "excess_delay_ns_sm", "tau_rms_ns_sm", "wb_loss_db"],
            tau_target=str(cfg.tau_target),
            tau_log_eps_ns=float(cfg.tau_log_eps_ns),
            tau_cap_ns=float(cfg.tau_cap_ns),
            keep_idx=(None if keep_idx is None else keep_idx.tolist()),
            dataset_features=list(cfg.dataset_features),
            model_features=list(cfg.model_features),
            scale_m=float(cfg.scale),
            base_run=str(base_run),
            patches_per_scene=int(cfg.patches_per_scene),
            full_h_min=int(cfg.full_h_min),
            full_h_max=int(cfg.full_h_max),
            full_w_min=int(cfg.full_w_min),
            full_w_max=int(cfg.full_w_max),
            min_tx_patch_gap_cells=int(cfg.min_tx_patch_gap_cells),
            max_tx_patch_offset_cells=int(cfg.max_tx_patch_offset_cells),
        )
        meta_path.write_text(json.dumps(meta, indent=2))

        idx = 0
        for s in range(cfg.num_scenes):
            full_scene, walls_2d, full_meta = make_full_scene(rng)
            

            for _ in range(cfg.patches_per_scene):

                tx_xyz = sample_free_tx_xyz(walls_2d, full_meta["ceiling_h_m"], rng)
                i0, j0 = sample_patch_top_left(
                    full_meta["full_H"], full_meta["full_W"], H, W, tx_xyz, rng
                )
                patch_scene = make_patch_scene_from_full(full_scene, tx_xyz, full_meta, i0, j0)

                x = build_feature_tensor(patch_scene, cfg.frequency_hz, requested=cfg.dataset_features).astype(np.float32)
                y = compute_labels_for_scene(patch_scene)

                x_stack = x.transpose(0, 2, 1, 3, 4).reshape(1, K * c_in, H, W)
                y_stack = y.transpose(0, 2, 1, 3, 4).reshape(1, K * y_ch, H, W)

                x_mm[idx] = x_stack[0]
                y_mm[idx] = y_stack[0].astype(np.float16)
                idx += 1

            x_mm.flush()
            y_mm.flush()
            print(f"[scene {s+1:03d}/{cfg.num_scenes}] wrote {cfg.patches_per_scene} off-patch samples (total {idx})")

    meta = json.loads(meta_path.read_text())
    total_samples = int(meta["total_samples"])
    H = int(meta["H"])
    W = int(meta["W"])
    K = int(meta["K"])
    c_in = int(meta["c_in"])
    y_ch = int(meta["y_ch"])

    x_mm = np.memmap(x_path, dtype="float32", mode="r", shape=(total_samples, c_in * K, H, W))
    y_mm = np.memmap(y_path, dtype="float16", mode="r", shape=(total_samples, y_ch * K, H, W))

    if not stats_path.exists():
        shutil.copy2(base_run / "norm_stats.npz", stats_path)

    stats = base_stats
    stats_dev = {k: v.to(cfg.device) for k, v in stats.items()}
    y_mean = stats_dev["y_mean"]
    y_std = stats_dev["y_std"]

    rng = np.random.default_rng(cfg.seed)
    scene_ids = rng.permutation(cfg.num_scenes)
    n_train_scenes = int(round(cfg.train_frac * cfg.num_scenes))
    train_scenes = np.sort(scene_ids[:n_train_scenes])
    val_scenes = np.sort(scene_ids[n_train_scenes:])

    def scene_to_indices(s):
        base = s * samples_per_scene
        return np.arange(base, base + samples_per_scene, dtype=np.int64)

    train_idx = np.concatenate([scene_to_indices(s) for s in train_scenes])
    val_idx = np.concatenate([scene_to_indices(s) for s in val_scenes])

    (out_dir / "split.json").write_text(json.dumps({
        "train_scenes": train_scenes.tolist(),
        "val_scenes": val_scenes.tolist(),
        "samples_per_scene": int(samples_per_scene),
        "seed": int(cfg.seed),
    }, indent=2))

    train_ds = MemmapIndexDataset(
        x_mm, y_mm, train_idx, stats, cfg.no_path_wb_db, K, y_ch, H, W,
        cfg.ex_loss_thresh_ns, cfg.tau_loss_thresh_ns, keep_idx=keep_idx, augment = True
    )
    val_ds = MemmapIndexDataset(
        x_mm, y_mm, val_idx, stats, cfg.no_path_wb_db, K, y_ch, H, W,
        cfg.ex_loss_thresh_ns, cfg.tau_loss_thresh_ns, keep_idx=keep_idx,
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    model = UNet3(in_ch=in_ch, out_ch=y_ch * K, base=cfg.base, groups=cfg.groups, dropout=cfg.dropout).to(cfg.device)

    ckpt = torch.load(base_run / "model_state.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("Loaded pretrained weights from:", base_run / "model_state.pt")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    if len(unexpected) > 0:
        raise RuntimeError("Unexpected keys while loading the base checkpoint.")

    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

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
            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device.startswith("cuda")), dtype=torch.bfloat16):
                pred = model(xb)
                loss = loss_fn(pred, yb, m_path, m_ex, m_tau, y_mean, y_std, y_ch)

            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue

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
                va_loss += loss_fn(pred, yb, m_path, m_ex, m_tau, y_mean, y_std, y_ch).item()
        va_loss /= max(len(val_dl), 1)

        dt = time.time() - t0
        print(f"ep {ep:03d}  train={tr_loss:.4f}  val={va_loss:.4f}  ({dt:.1f}s)")
        report_metrics(model, val_dl, stats_dev, y_ch, H, W, tag="val")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), state_path)
            print("  saved best state:", state_path)
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
