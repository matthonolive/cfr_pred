"""
train_residual_cost.py
==================================================================
U-Net surrogate retargeted to BEAT the analytic cost model on the
downstream network statistics (linear-power / SINR), not just on
hard-case delta MAE.

Why this is different from train_delta_tau.py
------------------------------------------------------------------
Your inference pipeline turns predictions into network stats almost
entirely through wb_db (the CFR is power-normalized; the Rician K-factor
comes from a separate LOS ray test; tau only shapes in-band ripple).
So "cost beats the U-Net" means the U-Net's wb is, in aggregate, worse
than the analytic cost model -- especially on LOS / light-NLOS links,
which dominate throughput and where cost is essentially exact.

This script addresses that directly:

1. RESIDUAL-OVER-COST. The net predicts r = wb_RT - wb_cost, and we
   reconstruct wb = wb_cost + r. Worst case r->0 recovers cost exactly,
   so the model cannot do worse than cost on average; it spends capacity
   only on the diffraction/reflection gains cost misses.
   (wb_cost = -cost; the "cost" feature is stored as negative path loss.)

2. LOS EXACTNESS. Where num_obstructions == 0, r is gated to 0, so wb
   equals wb_cost (= FSPL) exactly. The net can only perturb NLOS.

3. LINEAR-POWER OBJECTIVE. Primary loss is linear-power NMSE on the
   reconstructed wb, which inherently weights high-power links. The dB
   term is small and the aggressive hard-weighting is gone.

4. COVERAGE HEAD. A sigmoid path-existence head so the net can DROP
   no-path links (emit the sentinel) instead of hallucinating power.

5. CHECKPOINT on wb_nmse_db (the deployment-aligned metric).

6. DIAGNOSTIC. Every epoch prints U-Net vs cost wb error, stratified by
   LOS / NLOS, so you can see directly whether (and where) the net beats
   cost. If it never beats cost in the LOS stratum, the network statistic
   will keep favoring cost regardless of headline MAE.

Heads (model output, 3 per slice): [ r, tau, coverage_logit ]
On-disk store (4 per slice):        [ wb_RT, tau, wb_cost, num_obstr ]

REQUIRED inference change (ns3unet_spectrum.sample_heads, U-Net path)
------------------------------------------------------------------
    maps = model(x)                      # (K*3,H,W) -> (K,3,H,W)
    r        = maps[:, 0]                 # residual (normalized -> denorm with r_mean/r_std)
    tau_rms  = tau_from_target(maps[:,1]) # ns
    cov_p    = sigmoid(maps[:, 2])        # path-existence prob
    wb_cost  = -cost_at_rx                # from the cost feature (already computed)
    nobs     = num_obstr_at_rx            # from the num_obstructions feature
    if los_gate and nobs < 0.5: r = 0.0
    wb_db    = wb_cost + r
    delta_db = wb_db - fspl_db(d, fc)     # downstream still does wb = fspl + delta
    if cov_p < 0.5: emit no-path sentinel
  Set y_ch = 3 in the loader; r/tau/cov live at indices 0/1/2.
  r_mean/r_std/tau_mean/tau_std are saved in norm_stats.npz (scalars).

CONFIDENCE / VERIFY
------------------------------------------------------------------
Model / loss / metrics / training loop are the confident parts. The data
generators (esp. off-patch) are reassembled from your scripts and were
NOT run against mlink/sionna here -- check the  # >>> VERIFY  spots once.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass, field, replace
from pathlib import Path
import copy
import json
import time

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter, generic_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR

from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor, REGISTRY, Specification
from mlink.geometry import generate_wall_map, walls_to_mesh
from mlink.scene import Scene
from mlink.channel_tdl import RtCfg, subcarrier_frequencies_centered, compute_tdl_batch

C0 = 299_792_458.0

# On-disk store layout (physical units, fp16)
S_WBRT, S_TAU, S_WBCOST, S_NOBS = 0, 1, 2, 3
Y_STORE = 4
# Model output layout
P_R, P_TAU, P_COV = 0, 1, 2
PRED_CH = 3


# ==================================================================
# Config
# ==================================================================
@dataclass
class CFG:
    out_dir: str = "runs/shannon"

    # scene / grid
    frequency_hz: float = 5.21e9
    img_hw: tuple[int, int] = (64, 64)
    K_slices: int = 4
    z_step: float = 1.0
    z_margin: float = 0.5
    floor_h: float = 0.0
    ceil_min: float = 8.0
    ceil_max: float = 20.0
    scale: float = 0.625

    # wall density
    min_wall_length: int = 8
    min_door_length: int = 4
    max_partitions: int = 24

    # in-patch TX grid
    tx_origin_xy: tuple[float, float] = (1.75, 1.75)
    tx_z: float = 2.4
    tx_spacing_xy: float = 12.0
    tx_shape: tuple[int, int, int] = (1, 5, 5)

    # off-patch full scenes
    full_h_min: int = 96
    full_h_max: int = 160
    full_w_min: int = 96
    full_w_max: int = 160
    off_ceil_min: float = 8.0
    off_ceil_max: float = 20.0
    patches_per_scene: int = 6
    min_tx_patch_gap_cells: int = 1
    max_tx_patch_offset_cells: int = 48

    # OFDM / RT
    fft_size: int = 3072
    subcarrier_spacing_hz: float = 78_125.0
    rx_batch: int = 256
    no_path_wb_db: float = 199.5
    rt: RtCfg = field(default_factory=lambda: RtCfg(
        max_depth=10, samples_per_src=1_000_000, diffuse_reflection=True,
        diffraction=True, edge_diffraction=True, diffraction_lit_region=True))

    # features (cost + num_obstructions are BOTH required here)
    dataset_features: list[str] = field(default_factory=lambda: [
        "binary_walls", "electrical_distance", "cost", "num_obstructions", "height_cond"])
    model_features: list[str] = field(default_factory=lambda: [
        "binary_walls", "electrical_distance", "cost", "num_obstructions", "height_cond"])

    # smoothing (tau labels)
    smooth_kind: str = "median"
    smooth_median_size: int = 3
    smooth_gauss_sigma: float = 1.0

    # dataset sizes -- match deployment link distribution; do NOT oversample NLOS
    num_inpatch_scenes: int = 300
    num_offpatch_scenes: int = 120
    train_frac: float = 0.8
    seed: int = 20001999
    offpatch_oversample: float = 1.0    # 1.0 = natural mix (recommended for this objective)

    # training
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 2e-4
    epochs: int = 50
    base: int = 48
    groups: int = 8
    dropout: float = 0.1
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # targets
    tau_target: str = "raw"             # "raw" | "log10"
    tau_log_eps_ns: float = 1e-3
    tau_loss_thresh_ns: float = 0.0
    r_clip_db: float = 60.0             # clip residual target for stability

    # residual / LOS
    los_gate_residual: bool = True      # force r=0 where num_obstructions==0
    nobs_los_thresh: float = 0.5

    # loss weights (linear power is PRIMARY)
    w_power: float = 1.0                # linear-power NMSE on reconstructed wb
    w_db: float = 0.2                   # modest dB smooth-L1 (gives gradient in deep NLOS)
    w_tau: float = 0.5
    w_cov: float = 0.3                  # BCE on coverage
    # optional mild NLOS emphasis on the dB term only (0 = off)
    db_nobs_gain: float = 0.0

    # EMA + checkpoint
    ema_decay: float = 0.999
    ckpt_metric: str = "wb_nmse_db"     # lower is better


cfg = CFG()


# ==================================================================
# Utilities
# ==================================================================
def fspl_db(d_m, fc_hz):
    d = np.maximum(d_m, 1e-3)
    lam = C0 / float(fc_hz)
    return (20.0 * np.log10(4.0 * np.pi * d / lam)).astype(np.float32)


def default_material_db(freq):
    return pl.DataFrame(data={
        "id": [0], "frequency": [freq], "permittivity": [4.0], "permeability": [1.0],
        "conductivity": [0.01], "transmission_loss_vertical": [10.0],
        "transmission_loss_horizontal": [20.0], "reflection_loss": [9.0],
        "diffraction_loss_min": [8.0], "diffraction_loss_max": [15.0],
        "diffraction_loss": [5.0], "name": ["0"], "thickness": [0.1]})


def masked_gaussian_2d(img, mask, sigma):
    if sigma <= 0:
        return img.astype(np.float32)
    m = mask.astype(np.float32)
    num = gaussian_filter(img * m, sigma=sigma, mode="nearest")
    den = gaussian_filter(m, sigma=sigma, mode="nearest")
    out = np.zeros_like(img, dtype=np.float32)
    good = den > 1e-6
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def masked_median_2d(img, mask, size=3):
    if size <= 1:
        return img.astype(np.float32)
    work = img.astype(np.float32).copy()
    work[~mask] = np.nan
    out = generic_filter(work, np.nanmedian, size=size, mode="nearest").astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def smooth_map_stack(x_map, wb_map):
    valid = wb_map < cfg.no_path_wb_db
    out = np.empty_like(x_map, dtype=np.float32)
    for k in range(x_map.shape[0]):
        if cfg.smooth_kind == "median":
            out[k] = masked_median_2d(x_map[k], valid[k], cfg.smooth_median_size)
        elif cfg.smooth_kind == "gaussian":
            out[k] = masked_gaussian_2d(x_map[k], valid[k], cfg.smooth_gauss_sigma)
        else:
            out[k] = x_map[k].astype(np.float32)
    return out


def tau_to_target(t):
    t = np.maximum(t, 0.0).astype(np.float32)
    if cfg.tau_target == "raw":
        return t
    return np.log10(t + cfg.tau_log_eps_ns).astype(np.float32)


def tau_from_target_t(t):
    """Torch version for metrics."""
    if cfg.tau_target == "raw":
        return torch.clamp(t, min=0.0)
    return torch.clamp(torch.pow(10.0, t) - cfg.tau_log_eps_ns, min=0.0)


def infer_feature_channel_counts(scene, freq, features):
    return {f: int(build_feature_tensor(scene, freq, requested=[f]).shape[1]) for f in features}


def feature_slice_offset(dataset_features, feat_counts, name):
    off = 0
    for f in dataset_features:
        if f == name:
            return off
        off += feat_counts[f]
    raise KeyError(name)


def build_keep_idx(dataset_features, model_features, K, feat_counts):
    offsets, off = {}, 0
    for f in dataset_features:
        offsets[f] = (off, off + feat_counts[f]); off += feat_counts[f]
    c_full = off
    keep_in_slice = []
    for f in model_features:
        a, b = offsets[f]; keep_in_slice.extend(range(a, b))
    keep = []
    for k in range(K):
        keep.extend([k * c_full + i for i in keep_in_slice])
    return np.asarray(keep, dtype=np.int64), c_full


def _to_sionna_geometry(scene, freq):
    if hasattr(scene, "to_sionna_geometry"):
        return scene.to_sionna_geometry(freq)
    return scene.to_sionna(freq)


# ==================================================================
# Scene generation (in-patch + off-patch)  -- same as merged script
# ==================================================================
def make_scene(rng):
    H, W = cfg.img_hw
    ceiling_h = float(rng.uniform(cfg.ceil_min, cfg.ceil_max))
    mesh = walls_to_mesh(
        generate_wall_map((H, W), min_wall_length=cfg.min_wall_length,
                          min_door_length=cfg.min_door_length,
                          max_partitions=cfg.max_partitions, rng=rng),
        floor_height=cfg.floor_h, ceiling_height=ceiling_h).apply_scale(cfg.scale)
    usable = max(ceiling_h - cfg.floor_h - 2 * cfg.z_margin, 1e-3)
    total_span = (cfg.K_slices - 1) * cfg.z_step
    z_step = usable / max(cfg.K_slices - 1, 1) if total_span > usable else cfg.z_step
    z_start = cfg.floor_h + cfg.z_margin
    z_end = (ceiling_h - cfg.z_margin) - total_span
    z0 = z_start if z_end < z_start else float(rng.uniform(z_start, z_end))
    rx_grid = AntennaGrid(origin=cfg.scale * np.asarray([0.0, 0.0, z0], np.float32),
                          deltas=cfg.scale * np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, z_step]], np.float32),
                          shape=(cfg.K_slices, H, W))
    tx_grid = AntennaGrid(origin=cfg.scale * np.asarray([cfg.tx_origin_xy[0], cfg.tx_origin_xy[1], cfg.tx_z], np.float32),
                          deltas=cfg.scale * np.asarray([[cfg.tx_spacing_xy, 0, 0], [0, cfg.tx_spacing_xy, 0], [0, 0, 1]], np.float32),
                          shape=cfg.tx_shape)
    adb = AntennaDatabase.from_grid(tx_grid, rx_grid)
    return Scene(mesh=mesh, material_database=default_material_db(cfg.frequency_hz),
                 face2material={k: 0 for k in range(mesh.faces.shape[0])}, antenna_database=adb)


def make_full_scene(rng):
    # >>> VERIFY: prefer your existing offpatch_finetune full-scene generator if present.
    H = int(rng.integers(cfg.full_h_min, cfg.full_h_max + 1))
    W = int(rng.integers(cfg.full_w_min, cfg.full_w_max + 1))
    ceiling_h = float(rng.uniform(cfg.off_ceil_min, cfg.off_ceil_max))
    mesh = walls_to_mesh(
        generate_wall_map((H, W), min_wall_length=cfg.min_wall_length,
                          min_door_length=cfg.min_door_length,
                          max_partitions=cfg.max_partitions, rng=rng),
        floor_height=cfg.floor_h, ceiling_height=ceiling_h).apply_scale(cfg.scale)
    usable = max(ceiling_h - cfg.floor_h - 2 * cfg.z_margin, 1e-3)
    total_span = (cfg.K_slices - 1) * cfg.z_step
    z_step = usable / max(cfg.K_slices - 1, 1) if total_span > usable else cfg.z_step
    z_start = cfg.floor_h + cfg.z_margin
    z_end = (ceiling_h - cfg.z_margin) - total_span
    z0 = z_start if z_end < z_start else float(rng.uniform(z_start, z_end))
    placeholder = AntennaDatabase.from_grid(
        AntennaGrid(origin=np.zeros(3, np.float32),
                    deltas=cfg.scale * np.asarray([[cfg.tx_spacing_xy, 0, 0], [0, cfg.tx_spacing_xy, 0], [0, 0, 1]], np.float32),
                    shape=(1, 1, 1)),
        AntennaGrid(origin=cfg.scale * np.asarray([0.0, 0.0, z0], np.float32),
                    deltas=cfg.scale * np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, z_step]], np.float32),
                    shape=(cfg.K_slices, cfg.img_hw[0], cfg.img_hw[1])))
    base = Scene(mesh=mesh, material_database=default_material_db(cfg.frequency_hz),
                 face2material={k: 0 for k in range(mesh.faces.shape[0])}, antenna_database=placeholder)
    return base, {"H_full": H, "W_full": W, "z0_m": float(cfg.scale * z0), "z_step_m": float(cfg.scale * z_step)}


def sample_offpatch_crop(rng, tx_i, tx_j, full_H, full_W):
    pH, pW = cfg.img_hw
    i_max, j_max = full_H - pH, full_W - pW
    if i_max < 0 or j_max < 0:
        raise RuntimeError("Full scene smaller than patch.")

    def dist(i0, j0):
        di = 0 if i0 <= tx_i < i0 + pH else min(abs(tx_i - i0), abs(tx_i - (i0 + pH - 1)))
        dj = 0 if j0 <= tx_j < j0 + pW else min(abs(tx_j - j0), abs(tx_j - (j0 + pW - 1)))
        return float(np.hypot(di, dj))

    for _ in range(4000):
        i0 = int(rng.integers(0, i_max + 1)); j0 = int(rng.integers(0, j_max + 1))
        if (i0 <= tx_i < i0 + pH) and (j0 <= tx_j < j0 + pW):
            continue
        d = dist(i0, j0)
        if d < cfg.min_tx_patch_gap_cells or d > cfg.max_tx_patch_offset_cells:
            continue
        return i0, j0
    raise RuntimeError("Failed to sample an off-patch crop.")


def make_patch_scene_from_full(base_scene, tx_xyz_m, full_meta, i0, j0):
    pH, pW = cfg.img_hw
    rx_grid = AntennaGrid(
        origin=np.asarray([i0 * cfg.scale, j0 * cfg.scale, full_meta["z0_m"]], np.float32),
        deltas=np.asarray([[cfg.scale, 0, 0], [0, cfg.scale, 0], [0, 0, full_meta["z_step_m"]]], np.float32),
        shape=(cfg.K_slices, pH, pW))
    k, i, j = np.meshgrid(np.arange(cfg.K_slices), np.arange(pH), np.arange(pW), indexing="ij")
    rx_coords = rx_grid.ijk2xyz(i, j, k).reshape(-1, 3).astype(np.float32)
    adb = AntennaDatabase(tx_xyz_m.reshape(1, 3).astype(np.float32), rx_coords, None, rx_grid)
    return replace(base_scene, antenna_database=adb)


# ==================================================================
# RT labels (wb_RT, tau)  -- delta/clip happens later as residual
# ==================================================================
def compute_rt_labels(scene):
    rx_grid = scene.antenna_database.rx_grid
    K, H, W = rx_grid.shape
    tx_coords = scene.antenna_database.tx_coords
    rx_coords = scene.antenna_database.rx_coords
    P = rx_coords.shape[0]
    N = int(cfg.fft_size)
    freqs = subcarrier_frequencies_centered(cfg.fft_size, cfg.subcarrier_spacing_hz)
    si = _to_sionna_geometry(scene, cfg.frequency_hz)

    wb_out = np.full((tx_coords.shape[0], K, H, W), cfg.no_path_wb_db, np.float32)
    tau_out = np.zeros((tx_coords.shape[0], K, H, W), np.float32)
    for t, tx in enumerate(tx_coords):
        wb_all = np.full((P,), cfg.no_path_wb_db, np.float32)
        tau_all = np.zeros((P,), np.float32)
        for i0 in range(0, P, cfg.rx_batch):
            i1 = min(i0 + cfg.rx_batch, P)
            wb_db, ex_s, taps, tau_rms_s = compute_tdl_batch(
                si_scene=si, tx_xyz=tx, rx_xyz=rx_coords[i0:i1], frequencies_hz=freqs,
                L_taps=N, rt=cfg.rt, return_tau_rms=True)
            wb_all[i0:i1] = wb_db
            good = wb_db < cfg.no_path_wb_db
            if np.any(good):
                tau_all[i0 + np.nonzero(good)[0]] = tau_rms_s[good] * 1e9
        wb_map = wb_all.reshape(K, H, W)
        tau_map = np.maximum(smooth_map_stack(tau_all.reshape(K, H, W), wb_map), 0.0)
        tau_map[wb_map >= cfg.no_path_wb_db] = 0.0
        wb_out[t] = wb_map
        tau_out[t] = tau_map
        print(f"    tx {t+1}/{tx_coords.shape[0]} labels done", flush=True)
    return wb_out, tau_out


# ==================================================================
# Store builders
# ==================================================================
def _store_paths(out_dir, kind, feat_tag):
    d = out_dir / f"store_{kind}_{feat_tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d / "x.dat", d / "y.dat", d / "meta.json"


def _assemble_y(scene, x_prescale, cost_off, nobs_off, num_samples, K, H, W):
    """y store per sample: [wb_RT, tau, wb_cost, nobs]."""
    wb_rt, tau = compute_rt_labels(scene)            # (S,K,H,W)
    wb_cost = -x_prescale[:, cost_off]               # (S,K,H,W)  (cost is negative loss)
    nobs = x_prescale[:, nobs_off]                   # (S,K,H,W)
    y = np.zeros((num_samples, Y_STORE, K, H, W), np.float32)
    y[:, S_WBRT] = wb_rt
    y[:, S_TAU] = tau
    y[:, S_WBCOST] = wb_cost
    y[:, S_NOBS] = nobs
    return y


def build_inpatch_store(out_dir, rng, c_in, num_tx, K, H, W, feat_tag, cost_off, nobs_off):
    xp, yp, mp = _store_paths(out_dir, "inpatch", feat_tag)
    total = cfg.num_inpatch_scenes * num_tx
    if xp.exists() and yp.exists() and mp.exists():
        return xp, yp, json.loads(mp.read_text())
    x_mm = np.memmap(xp, dtype="float32", mode="w+", shape=(total, c_in * K, H, W))
    y_mm = np.memmap(yp, dtype="float16", mode="w+", shape=(total, Y_STORE * K, H, W))
    idx = 0
    for s in range(cfg.num_inpatch_scenes):
        scene = make_scene(rng)
        x = build_feature_tensor(scene, cfg.frequency_hz, requested=cfg.dataset_features).astype(np.float32)
        y = _assemble_y(scene, x, cost_off, nobs_off, num_tx, K, H, W)
        x_mm[idx:idx + num_tx] = x.transpose(0, 2, 1, 3, 4).reshape(num_tx, K * c_in, H, W)
        y_mm[idx:idx + num_tx] = y.transpose(0, 2, 1, 3, 4).reshape(num_tx, K * Y_STORE, H, W).astype(np.float16)
        idx += num_tx; x_mm.flush(); y_mm.flush()
        print(f"[inpatch {s+1:03d}/{cfg.num_inpatch_scenes}] total {idx}")
    meta = {"total_samples": int(total), "samples_per_scene": int(num_tx),
            "num_scenes": int(cfg.num_inpatch_scenes)}
    mp.write_text(json.dumps(meta, indent=2))
    return xp, yp, meta


def build_offpatch_store(out_dir, rng, c_in, K, H, W, feat_tag, cost_off, nobs_off):
    xp, yp, mp = _store_paths(out_dir, "offpatch", feat_tag)
    sps = cfg.patches_per_scene
    total = cfg.num_offpatch_scenes * sps
    if xp.exists() and yp.exists() and mp.exists():
        return xp, yp, json.loads(mp.read_text())
    x_mm = np.memmap(xp, dtype="float32", mode="w+", shape=(total, c_in * K, H, W))
    y_mm = np.memmap(yp, dtype="float16", mode="w+", shape=(total, Y_STORE * K, H, W))
    # >>> VERIFY: register binary_walls_offpatch for crops if you have it.
    idx = 0
    for s in range(cfg.num_offpatch_scenes):
        base, fm = make_full_scene(rng)
        Hf, Wf = fm["H_full"], fm["W_full"]
        tx_i = int(rng.integers(0, Hf)); tx_j = int(rng.integers(0, Wf))
        tx_xyz = np.asarray([tx_i * cfg.scale, tx_j * cfg.scale, cfg.tx_z * cfg.scale], np.float32)  # >>> VERIFY tx_z scaling
        for _p in range(sps):
            i0, j0 = sample_offpatch_crop(rng, tx_i, tx_j, Hf, Wf)
            scene = make_patch_scene_from_full(base, tx_xyz, fm, i0, j0)
            x = build_feature_tensor(scene, cfg.frequency_hz, requested=cfg.dataset_features).astype(np.float32)
            y = _assemble_y(scene, x, cost_off, nobs_off, 1, K, H, W)
            x_mm[idx:idx + 1] = x.transpose(0, 2, 1, 3, 4).reshape(1, K * c_in, H, W)
            y_mm[idx:idx + 1] = y.transpose(0, 2, 1, 3, 4).reshape(1, K * Y_STORE, H, W).astype(np.float16)
            idx += 1
        x_mm.flush(); y_mm.flush()
        print(f"[offpatch {s+1:03d}/{cfg.num_offpatch_scenes}] total {idx}")
    meta = {"total_samples": int(total), "samples_per_scene": int(sps),
            "num_scenes": int(cfg.num_offpatch_scenes)}
    mp.write_text(json.dumps(meta, indent=2))
    return xp, yp, meta


# ==================================================================
# Norm stats (masked for r/tau; per-channel for x)
# ==================================================================
def compute_norm_stats(mmaps, c_in, K, H, W, keep_idx, max_samples=512, seed=0):
    rng = np.random.default_rng(seed)
    xs, r_vals, tau_vals = [], [], []
    per = max(1, max_samples // len(mmaps))
    for (x_mm, y_mm) in mmaps:
        n = x_mm.shape[0]
        sel = rng.choice(n, size=min(per, n), replace=False)
        x = np.array(x_mm[sel], np.float32)
        y = np.array(y_mm[sel], np.float32).reshape(len(sel), K, Y_STORE, H, W)
        xs.append(x[:, keep_idx, :, :])
        wb_rt = y[:, :, S_WBRT]; wb_cost = y[:, :, S_WBCOST]; tau = y[:, :, S_TAU]
        valid = wb_rt < cfg.no_path_wb_db
        r = np.clip(wb_rt - wb_cost, -cfg.r_clip_db, cfg.r_clip_db)
        r_vals.append(r[valid])
        tau_vals.append(tau_to_target(tau[valid]))
    x = np.concatenate(xs, 0)
    r = np.concatenate(r_vals, 0); tau = np.concatenate(tau_vals, 0)
    return {
        "x_mean": torch.from_numpy(x.mean((0, 2, 3))).view(-1, 1, 1).float(),
        "x_std": torch.clamp(torch.from_numpy(x.std((0, 2, 3))).view(-1, 1, 1).float(), min=1e-6),
        "r_mean": torch.tensor(float(r.mean())), "r_std": torch.tensor(max(float(r.std()), 1e-6)),
        "tau_mean": torch.tensor(float(tau.mean())), "tau_std": torch.tensor(max(float(tau.std()), 1e-6)),
        "keep_idx": torch.from_numpy(np.asarray(keep_idx, np.int64)),
    }


# ==================================================================
# Dataset
# ==================================================================
def _aug(tensors, k, flip):
    out = []
    for t in tensors:
        if k > 0:
            t = torch.rot90(t, k, (-2, -1))
        if flip:
            t = torch.flip(t, [-1])
        out.append(t)
    return out


class ResidualDataset(Dataset):
    def __init__(self, x_mm, y_mm, indices, stats, K, H, W, keep_idx, augment=False):
        self.x_mm, self.y_mm = x_mm, y_mm
        self.indices = indices.astype(np.int64)
        self.K, self.H, self.W = int(K), int(H), int(W)
        self.keep_idx = np.asarray(keep_idx, np.int64)
        self.augment = augment
        self.x_mean = stats["x_mean"]; self.x_std = stats["x_std"]
        self.r_mean = float(stats["r_mean"]); self.r_std = float(stats["r_std"])
        self.tau_mean = float(stats["tau_mean"]); self.tau_std = float(stats["tau_std"])

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        j = int(self.indices[i])
        x = np.array(self.x_mm[j], np.float32)[self.keep_idx, :, :]
        y = np.array(self.y_mm[j], np.float32).reshape(self.K, Y_STORE, self.H, self.W)
        wb_rt = torch.from_numpy(y[:, S_WBRT])
        tau = torch.from_numpy(y[:, S_TAU])
        wb_cost = torch.from_numpy(y[:, S_WBCOST])
        nobs = torch.from_numpy(y[:, S_NOBS])

        m_path = (wb_rt < cfg.no_path_wb_db).float()
        m_tau = (m_path.bool() & (tau >= cfg.tau_loss_thresh_ns)).float()
        cov = m_path.clone()

        r = torch.clamp(wb_rt - wb_cost, -cfg.r_clip_db, cfg.r_clip_db)
        r_norm = (r - self.r_mean) / self.r_std
        tau_t = torch.from_numpy(tau_to_target(tau.numpy()))
        tau_norm = (tau_t - self.tau_mean) / self.tau_std
        r_norm = r_norm * m_path                     # zero where invalid (masked in loss too)
        tau_norm = tau_norm * m_tau

        x = (torch.from_numpy(x) - self.x_mean) / self.x_std

        if self.augment:
            k = int(torch.randint(0, 4, (1,)).item())
            flip = torch.rand(1).item() < 0.5
            x, r_norm, tau_norm, cov, wb_cost, wb_rt, nobs, m_path, m_tau = _aug(
                [x, r_norm, tau_norm, cov, wb_cost, wb_rt, nobs, m_path, m_tau], k, flip)

        return {"x": x, "r": r_norm, "tau": tau_norm, "cov": cov,
                "wb_cost": wb_cost, "wb_rt": wb_rt, "nobs": nobs,
                "m_path": m_path, "m_tau": m_tau}


# ==================================================================
# Model
# ==================================================================
def _vg(ch, g):
    g = min(g, ch)
    while ch % g != 0:
        g -= 1
    return max(g, 1)


class ResBlock(nn.Module):
    def __init__(self, ic, oc, g=8, p=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(_vg(ic, g), ic); self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.n2 = nn.GroupNorm(_vg(oc, g), oc); self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.drop = nn.Dropout2d(p) if p > 0 else nn.Identity()
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(_vg(ch, 8), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1); self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-1, -2) @ k * self.scale).softmax(-1)
        return x + self.proj((v @ attn).reshape(B, C, H, W))


class UNet3(nn.Module):
    def __init__(self, in_ch, out_ch, base=32, groups=8, dropout=0.1):
        super().__init__()
        self.e1 = nn.Sequential(ResBlock(in_ch, base, groups, dropout), ResBlock(base, base, groups, dropout))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(ResBlock(base, base * 2, groups, dropout), ResBlock(base * 2, base * 2, groups, dropout))
        self.p2 = nn.MaxPool2d(2)
        self.e3 = nn.Sequential(ResBlock(base * 2, base * 4, groups, dropout), ResBlock(base * 4, base * 4, groups, dropout))
        self.p3 = nn.MaxPool2d(2)
        self.mid = nn.Sequential(ResBlock(base * 4, base * 8, groups, dropout), SelfAttention2d(base * 8),
                                 ResBlock(base * 8, base * 8, groups, dropout))
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = nn.Sequential(ResBlock(base * 8, base * 4, groups, 0.0), ResBlock(base * 4, base * 4, groups, 0.0))
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = nn.Sequential(ResBlock(base * 4, base * 2, groups, 0.0), ResBlock(base * 2, base * 2, groups, 0.0))
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = nn.Sequential(ResBlock(base * 2, base, groups, 0.0), ResBlock(base, base, groups, 0.0))
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p1(e1)); e3 = self.e3(self.p2(e2))
        m = self.mid(self.p3(e3))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


# ==================================================================
# Reconstruction + loss
# ==================================================================
def reconstruct_wb(pred3, batch, stats):
    """wb_hat = wb_cost + r_hat (r gated to 0 in LOS). Returns wb_hat (B,K,H,W)."""
    r_hat = pred3[:, :, P_R] * stats["r_std"] + stats["r_mean"]
    if cfg.los_gate_residual:
        gate = (batch["nobs"] >= cfg.nobs_los_thresh).float()
        r_hat = r_hat * gate
    return batch["wb_cost"] + r_hat


def loss_fn(pred, batch, stats):
    B, _, H, W = pred.shape
    K = cfg.K_slices
    p3 = pred.view(B, K, PRED_CH, H, W)
    mp = batch["m_path"]
    mt = batch["m_tau"]

    wb_hat = reconstruct_wb(p3, batch, stats)
    wb_rt = batch["wb_rt"]

    # primary: linear-power NMSE (high-power links dominate naturally)
    g_hat = torch.pow(10.0, torch.clamp(-wb_hat / 10.0, -20.0, 20.0))
    g_tgt = torch.pow(10.0, torch.clamp(-wb_rt / 10.0, -20.0, 20.0))
    l_power = (((g_hat - g_tgt) ** 2) * mp).sum() / (((g_tgt) ** 2) * mp).sum().clamp_min(1e-12)

    # modest dB term (gradient where power ~ 0, i.e. deep NLOS)
    w = mp
    if cfg.db_nobs_gain > 0:
        w = mp * (1.0 + cfg.db_nobs_gain * (batch["nobs"] >= cfg.nobs_los_thresh).float())
    l_db = (F.smooth_l1_loss(wb_hat, wb_rt, reduction="none") * w).sum() / w.sum().clamp_min(1.0)

    # tau in normalized target space
    tau_hat = p3[:, :, P_TAU]
    l_tau = (F.smooth_l1_loss(tau_hat * mt, batch["tau"] * mt, reduction="sum") / mt.sum().clamp_min(1.0))

    # coverage (path existence) over ALL pixels
    l_cov = F.binary_cross_entropy_with_logits(p3[:, :, P_COV], batch["cov"])

    return cfg.w_power * l_power + cfg.w_db * l_db + cfg.w_tau * l_tau + cfg.w_cov * l_cov


# ==================================================================
# Metrics  (U-Net vs cost, stratified LOS/NLOS)
# ==================================================================
@torch.no_grad()
def report_metrics(model, dl, stats, tag="val"):
    model.eval()
    K = cfg.K_slices
    acc = dict(num=0.0, den=0.0, mae=0.0, n=0.0,
               mae_los=0.0, n_los=0.0, mae_nlos=0.0, n_nlos=0.0,
               c_num=0.0, c_den=0.0, c_mae=0.0,
               c_mae_los=0.0, c_mae_nlos=0.0,
               tau=0.0, ntau=0.0, cov_ok=0.0, cov_n=0.0)
    for batch in dl:
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        pred = model(batch["x"])
        B = pred.shape[0]
        p3 = pred.view(B, K, PRED_CH, batch["x"].shape[-2], batch["x"].shape[-1])
        wb_hat = reconstruct_wb(p3, batch, stats)
        wb_rt = batch["wb_rt"]; wb_cost = batch["wb_cost"]
        mp = batch["m_path"]
        los = (batch["nobs"] < cfg.nobs_los_thresh).float() * mp
        nlos = (batch["nobs"] >= cfg.nobs_los_thresh).float() * mp

        g_hat = torch.pow(10.0, torch.clamp(-wb_hat / 10.0, -20.0, 20.0))
        g_cost = torch.pow(10.0, torch.clamp(-wb_cost / 10.0, -20.0, 20.0))
        g_tgt = torch.pow(10.0, torch.clamp(-wb_rt / 10.0, -20.0, 20.0))

        acc["num"] += (((g_hat - g_tgt) ** 2) * mp).sum().item()
        acc["c_num"] += (((g_cost - g_tgt) ** 2) * mp).sum().item()
        acc["den"] += (((g_tgt) ** 2) * mp).sum().item()

        e = (wb_hat - wb_rt).abs(); ec = (wb_cost - wb_rt).abs()
        acc["mae"] += (e * mp).sum().item(); acc["n"] += mp.sum().item()
        acc["c_mae"] += (ec * mp).sum().item()
        acc["mae_los"] += (e * los).sum().item(); acc["n_los"] += los.sum().item()
        acc["mae_nlos"] += (e * nlos).sum().item(); acc["n_nlos"] += nlos.sum().item()
        acc["c_mae_los"] += (ec * los).sum().item()
        acc["c_mae_nlos"] += (ec * nlos).sum().item()

        tau_hat = tau_from_target_t(p3[:, :, P_TAU] * stats["tau_std"] + stats["tau_mean"])
        tau_tgt = tau_from_target_t(batch["tau"] * stats["tau_std"] + stats["tau_mean"])
        mt = batch["m_tau"]
        acc["tau"] += ((tau_hat - tau_tgt).abs() * mt).sum().item(); acc["ntau"] += mt.sum().item()

        cov_pred = (torch.sigmoid(p3[:, :, P_COV]) >= 0.5).float()
        acc["cov_ok"] += (cov_pred == batch["cov"]).float().sum().item()
        acc["cov_n"] += batch["cov"].numel()

    sd = lambda a, b: a / b if b > 0 else 0.0
    nmse = sd(acc["num"], acc["den"]); cnmse = sd(acc["c_num"], acc["den"])
    m = {
        "wb_nmse_db": 10 * np.log10(max(nmse, 1e-12)),
        "cost_nmse_db": 10 * np.log10(max(cnmse, 1e-12)),
        "wb_mae_db": sd(acc["mae"], acc["n"]),
        "cost_mae_db": sd(acc["c_mae"], acc["n"]),
        "wb_mae_los": sd(acc["mae_los"], acc["n_los"]),
        "cost_mae_los": sd(acc["c_mae_los"], acc["n_los"]),
        "wb_mae_nlos": sd(acc["mae_nlos"], acc["n_nlos"]),
        "cost_mae_nlos": sd(acc["c_mae_nlos"], acc["n_nlos"]),
        "tau_mae_ns": sd(acc["tau"], acc["ntau"]),
        "cov_acc": sd(acc["cov_ok"], acc["cov_n"]),
    }
    print(f"  [{tag}] NMSE  unet={m['wb_nmse_db']:.2f}dB  cost={m['cost_nmse_db']:.2f}dB"
          f"  | MAE all unet={m['wb_mae_db']:.2f} cost={m['cost_mae_db']:.2f}"
          f"  | LOS unet={m['wb_mae_los']:.2f} cost={m['cost_mae_los']:.2f}"
          f"  | NLOS unet={m['wb_mae_nlos']:.2f} cost={m['cost_mae_nlos']:.2f}"
          f"  | tauMAE={m['tau_mae_ns']:.2f}ns covAcc={m['cov_acc']:.3f}")
    return m


# ==================================================================
# Main
# ==================================================================
def scene_split(meta, rng):
    sp, ns = meta["samples_per_scene"], meta["num_scenes"]
    order = rng.permutation(ns)
    n_tr = int(round(cfg.train_frac * ns))
    to_idx = lambda s: np.arange(s * sp, s * sp + sp, dtype=np.int64)
    tr = np.concatenate([to_idx(s) for s in np.sort(order[:n_tr])]) if n_tr else np.array([], np.int64)
    va = np.concatenate([to_idx(s) for s in np.sort(order[n_tr:])]) if n_tr < ns else np.array([], np.int64)
    return tr, va


def main():
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    feat_tag = "-".join(cfg.dataset_features)
    try:
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass

    rng = np.random.default_rng(cfg.seed)
    probe = make_scene(rng)
    x_probe = build_feature_tensor(probe, cfg.frequency_hz, requested=cfg.dataset_features).astype(np.float32)
    c_in = int(x_probe.shape[1])
    num_tx = int(probe.antenna_database.tx_coords.shape[0])
    K, H, W = probe.antenna_database.rx_grid.shape
    feat_counts = infer_feature_channel_counts(probe, cfg.frequency_hz, cfg.dataset_features)
    keep_idx, _ = build_keep_idx(cfg.dataset_features, cfg.model_features, K, feat_counts)
    cost_off = feature_slice_offset(cfg.dataset_features, feat_counts, "cost")
    nobs_off = feature_slice_offset(cfg.dataset_features, feat_counts, "num_obstructions")

    xin, yin, min_meta = build_inpatch_store(out_dir, rng, c_in, num_tx, K, H, W, feat_tag, cost_off, nobs_off)
    xoff, yoff, moff = build_offpatch_store(out_dir, rng, c_in, K, H, W, feat_tag, cost_off, nobs_off)

    xin_mm = np.memmap(xin, "float32", "r", shape=(min_meta["total_samples"], c_in * K, H, W))
    yin_mm = np.memmap(yin, "float16", "r", shape=(min_meta["total_samples"], Y_STORE * K, H, W))
    xoff_mm = np.memmap(xoff, "float32", "r", shape=(moff["total_samples"], c_in * K, H, W))
    yoff_mm = np.memmap(yoff, "float16", "r", shape=(moff["total_samples"], Y_STORE * K, H, W))

    stats = compute_norm_stats([(xin_mm, yin_mm), (xoff_mm, yoff_mm)], c_in, K, H, W, keep_idx, seed=cfg.seed)
    np.savez(out_dir / "norm_stats.npz",
             x_mean=stats["x_mean"].numpy(), x_std=stats["x_std"].numpy(),
             r_mean=float(stats["r_mean"]), r_std=float(stats["r_std"]),
             tau_mean=float(stats["tau_mean"]), tau_std=float(stats["tau_std"]),
             keep_idx=keep_idx)
    stats_dev = {k: (v.to(cfg.device) if torch.is_tensor(v) else v) for k, v in stats.items()}

    tr_in, va_in = scene_split(min_meta, np.random.default_rng(cfg.seed + 1))
    tr_off, va_off = scene_split(moff, np.random.default_rng(cfg.seed + 2))
    dtr_in = ResidualDataset(xin_mm, yin_mm, tr_in, stats, K, H, W, keep_idx, augment=True)
    dtr_off = ResidualDataset(xoff_mm, yoff_mm, tr_off, stats, K, H, W, keep_idx, augment=True)
    dva_in = ResidualDataset(xin_mm, yin_mm, va_in, stats, K, H, W, keep_idx)
    dva_off = ResidualDataset(xoff_mm, yoff_mm, va_off, stats, K, H, W, keep_idx)
    train_ds = ConcatDataset([dtr_in, dtr_off]); val_ds = ConcatDataset([dva_in, dva_off])

    weights = np.concatenate([np.ones(len(dtr_in)),
                              cfg.offpatch_oversample * np.ones(len(dtr_off))]).astype(np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    in_ch = len(keep_idx)
    model = UNet3(in_ch, PRED_CH * K, base=cfg.base, groups=cfg.groups, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))
    sched = OneCycleLR(opt, max_lr=cfg.lr * 3, epochs=cfg.epochs, steps_per_epoch=len(train_dl), pct_start=0.1)
    ema = EMA(model, cfg.ema_decay)

    (out_dir / "meta.json").write_text(json.dumps({
        "H": H, "W": W, "K": K, "c_in": in_ch, "y_ch": PRED_CH,
        "y_channels": ["residual_db", "tau_rms_ns", "coverage_logit"],
        "parametrization": "residual_over_cost",
        "los_gate_residual": cfg.los_gate_residual, "nobs_los_thresh": cfg.nobs_los_thresh,
        "dataset_features": cfg.dataset_features, "model_features": cfg.model_features,
        "tau_target": cfg.tau_target, "tau_log_eps_ns": cfg.tau_log_eps_ns,
        "scale_m": cfg.scale, "frequency_hz": cfg.frequency_hz, "no_path_wb_db": cfg.no_path_wb_db,
    }, indent=2))

    best = float("inf")
    history = []
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time(); model.train(); tr_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device.startswith("cuda")), dtype=torch.bfloat16):
                loss = loss_fn(model(batch["x"]), batch, stats_dev)
            if not torch.isfinite(loss):
                sched.step(); continue
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            tr_loss += loss.item()
        tr_loss /= max(len(train_dl), 1)

        metrics = report_metrics(ema.shadow, val_dl, stats_dev, tag="val(ema)")
        score = metrics[cfg.ckpt_metric]
        history.append({"epoch": ep, "train_loss": tr_loss, **metrics})
        pl.DataFrame(history).write_csv(out_dir / "history.csv")
        print(f"ep {ep:03d} train={tr_loss:.4f} {cfg.ckpt_metric}={score:.3f} ({time.time()-t0:.1f}s)")

        if score < best:
            best = score
            torch.save(model.state_dict(), out_dir / "model_state.pt")
            torch.save(ema.shadow.state_dict(), out_dir / "model_state_ema.pt")
            try:
                ex = torch.randn(1, in_ch, H, W, device=cfg.device)
                torch.jit.trace(ema.shadow, ex).save(str(out_dir / "model.pt"))
            except Exception as e:
                print("  TorchScript export failed:", repr(e))
            print(f"  saved best (EMA) {cfg.ckpt_metric}={best:.3f}  "
                  f"(cost baseline NMSE={metrics['cost_nmse_db']:.2f}dB)")

    print("Done. best", cfg.ckpt_metric, "=", best)


if __name__ == "__main__":
    main()