#!/usr/bin/env python3
"""
Reciprocity test for wideband loss (WB loss / path loss) for:
  1) Your U-Net surrogate (TorchScript run_dir with model.pt + norm_stats.npz + meta.json)
  2) Sionna RT (PathSolver) on the same set of point-to-point links

IMPORTANT FIX (vs previous script):
  - The U-Net WB-loss head is NOT reliably supervised in your training setup.
  - This script uses the SUPERVISED delta_pl_db head (channel "delta_pl_db")
    and reconstructs WB loss as:  wb_hat = FSPL(d) + delta_hat

Outputs:
  - out_dir/summary.csv : per-scene aggregate reciprocity stats for UNet and Sionna
  - out_dir/links.csv   : per-link (i<j) reciprocity deltas for UNet and Sionna
  - optional plots (histograms) if --plots is enabled
  - optional per-scene visualization if --viz is enabled

Examples (XML scenes):
  python reciprocity_test.py \
    --unet_run runs/delta_3072_interimdiff_merged \
    --xml_glob "suites/mc_200/**/scene.xml" \
    --num_scenes 10 --num_nodes 16 \
    --out_dir reciprocity_out --plots --viz \
    --samples_per_src 200000

Examples (random scenes, training-style):
  python reciprocity_test.py \
    --unet_run runs/delta_3072_interimdiff_merged \
    --random_scenes 5 --num_nodes 16 \
    --out_dir reciprocity_out --plots --viz
"""

# IMPORTANT: Sionna import first (avoids crash in many setups)
import sionna.rt  # noqa: F401

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import tensorflow as tf
import mitsuba as mi

from sionna.rt import load_scene, PlanarArray, PathSolver, Transmitter, Receiver

# mlink
from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor
from mlink.geometry import generate_wall_map, walls_to_mesh
from mlink.scene import Scene as MlinkScene

C0 = 299_792_458.0


# ----------------------------
# Small utilities
# ----------------------------
def set_tf_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass


def fspl_db(d_m: np.ndarray, fc_hz: float) -> np.ndarray:
    """FSPL = 20 log10(4π d / λ). Matches your training helper (with clamp)."""
    d = np.maximum(d_m, 1e-3).astype(np.float32)  # avoid log(0)
    lam = C0 / float(fc_hz)
    return (20.0 * np.log10(4.0 * np.pi * d / lam)).astype(np.float32)


def default_material_db(freq: float):
    # matches your training script defaults
    import polars as pl
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


def subcarrier_frequencies_centered(fft_size: int, subcarrier_spacing_hz: float) -> np.ndarray:
    N = int(fft_size)
    df = float(subcarrier_spacing_hz)
    k = np.arange(N, dtype=np.float64)
    return (k - (N // 2)) * df


def grid_coords(rx_grid: AntennaGrid) -> np.ndarray:
    """
    Match AntennaDatabase.from_grid() ordering:
      shape (K,H,W)
      k,i,j = meshgrid(K,H,W) with indexing='ij'
      xyz = ijk2xyz(i,j,k)
    Returns (K*H*W,3) float32
    """
    K, H, W = rx_grid.shape
    k, i, j = np.meshgrid(
        np.arange(K, dtype=np.int32),
        np.arange(H, dtype=np.int32),
        np.arange(W, dtype=np.int32),
        indexing="ij",
    )
    xyz = rx_grid.ijk2xyz(i, j, k).reshape(-1, 3).astype(np.float32)
    return xyz


def node_xyz_from_kij(rx_grid: AntennaGrid, kij: np.ndarray) -> np.ndarray:
    kij = np.asarray(kij, dtype=np.int32)
    k = kij[:, 0]
    i = kij[:, 1]
    j = kij[:, 2]
    xyz = rx_grid.ijk2xyz(i, j, k).astype(np.float32)
    return np.asarray(xyz, dtype=np.float32)


def _clear_radio_nodes(scene: sionna.rt.Scene):
    for name in list(scene.transmitters.keys()):
        scene.remove(name)
    for name in list(scene.receivers.keys()):
        scene.remove(name)


def force_isotropic_arrays(scene: sionna.rt.Scene):
    """Make TX/RX arrays single-element isotropic so reciprocity should hold (up to MC variance)."""
    try:
        iso = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        scene.tx_array = iso
        scene.rx_array = iso
    except Exception:
        pass


def _coerce_cfr_to_rx_by_freq(h: np.ndarray, B: int) -> np.ndarray:
    """Coerce CFR output into a shape with receiver axis first and frequency last."""
    h = np.asarray(h)
    if h.ndim == 1:
        return h.reshape(1, -1)

    rx_axis = None
    for ax in range(h.ndim):
        if h.shape[ax] == B:
            rx_axis = ax
            break

    if rx_axis is None:
        if B == 1:
            return h.reshape(1, -1)
        raise ValueError(f"Could not find receiver axis of length B={B} in CFR shape {h.shape}")

    if rx_axis != 0:
        h = np.moveaxis(h, rx_axis, 0)

    return h


def sionna_wb_loss_batch(
    scene: sionna.rt.Scene,
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    max_depth: int,
    samples_per_src: int,
    los: bool = True,
    specular_reflection: bool = True,
    diffuse_reflection: bool = True,
    refraction: bool = True,
    synthetic_array: bool = False,
    diffraction: bool = True,
    edge_diffraction: bool = True,
    diffraction_lit_region: bool = True,
    no_path_db: float = 200.0,
) -> np.ndarray:
    """
    Compute WB loss (dB) from mean(|H(f)|^2) for one TX to many RX.
    Returns: (B,) float32
    """
    tx_xyz = np.asarray(tx_xyz, dtype=np.float32).reshape(3)
    rx_xyz = np.asarray(rx_xyz, dtype=np.float32).reshape(-1, 3)
    B = rx_xyz.shape[0]

    _clear_radio_nodes(scene)

    scene.add(Transmitter(name="tx", position=mi.Point3f(tx_xyz)))
    for i in range(B):
        scene.add(Receiver(name=f"rx{i:05d}", position=mi.Point3f(rx_xyz[i])))

    p_solver = PathSolver()
    paths = p_solver(
        scene=scene,
        max_depth=int(max_depth),
        samples_per_src=int(samples_per_src),
        los=bool(los),
        specular_reflection=bool(specular_reflection),
        diffuse_reflection=bool(diffuse_reflection),
        refraction=bool(refraction),
        synthetic_array=bool(synthetic_array),
        diffraction=bool(diffraction),
        edge_diffraction=bool(edge_diffraction),
        diffraction_lit_region=bool(diffraction_lit_region),
    )

    h = paths.cfr(
        frequencies=freqs_hz,
        sampling_frequency=1.0,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy",
    )
    h = np.squeeze(h)
    h = _coerce_cfr_to_rx_by_freq(h, B=B)

    if h.ndim == 2:
        power = np.mean(np.abs(h) ** 2, axis=1)
    else:
        B2 = h.shape[0]
        Nf = h.shape[-1]
        h2 = h.reshape(B2, -1, Nf)
        power = np.mean(np.abs(h2) ** 2, axis=(1, 2))

    wb_loss_db = (-10.0 * np.log10(np.maximum(power, 1e-12))).astype(np.float32)
    bad = ~np.isfinite(wb_loss_db)
    if np.any(bad):
        wb_loss_db[bad] = float(no_path_db)
    return wb_loss_db


def _to_sionna_geometry(mscene: MlinkScene, freq_hz: float):
    """Compatibility: some versions have to_sionna_geometry, some only to_sionna."""
    if hasattr(mscene, "to_sionna_geometry"):
        return mscene.to_sionna_geometry(freq_hz)
    return mscene.to_sionna(freq_hz)


# ----------------------------
# UNet wrapper (delta -> WB reconstruction)
# ----------------------------
class UNetDeltaWB:
    def __init__(self, run_dir: str, device: str = "cuda"):
        run = Path(run_dir)
        self.run = run
        self.meta = json.loads((run / "meta.json").read_text())
        stats = np.load(run / "norm_stats.npz")

        dev = device
        if dev != "cpu" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = torch.device(dev)

        self.model = torch.jit.load(str(run / "model.pt"), map_location=self.device).eval()

        def to_c11(a):
            a = np.asarray(a)
            if a.ndim == 1:
                a = a[:, None, None]
            return a

        self.x_mean = torch.from_numpy(to_c11(stats["x_mean"])).float().to(self.device)
        self.x_std  = torch.from_numpy(to_c11(stats["x_std"])).float().to(self.device).clamp_min(1e-6)
        self.y_mean = torch.from_numpy(to_c11(stats["y_mean"])).float().to(self.device)
        self.y_std  = torch.from_numpy(to_c11(stats["y_std"])).float().to(self.device).clamp_min(1e-6)

        self.keep_idx = None
        if "keep_idx" in stats.files:
            ki = stats["keep_idx"]
            if ki is not None and np.size(ki) > 0:
                self.keep_idx = np.asarray(ki, dtype=np.int64)

        # Grid defaults (meta.json from training may not store scale/z_step)
        self.H = int(self.meta.get("H", 64))
        self.W = int(self.meta.get("W", 64))
        self.K = int(self.meta.get("K", 4))

        # In your training: scale = 0.625 meters/cell; z_step = 1 cell -> z_step_m = 0.625 m
        self.scale_m = float(self.meta.get("scale", 0.625))
        z_step_cells = float(self.meta.get("z_step", 1.0))
        self.z_step_m = float(self.scale_m * z_step_cells)

        self.no_path_wb = float(self.meta.get("no_path_wb_db", 199.5))

        self.dataset_features = list(self.meta.get(
            "dataset_features",
            ["binary_walls", "electrical_distance", "cost", "height_cond"]
        ))

        self.y_channels = list(self.meta.get(
            "y_channels",
            ["delta_pl_db", "excess_delay_ns_sm", "tau_rms_ns_sm", "wb_loss_db"]
        ))
        self.y_ch = int(len(self.y_channels))

        # FIX: we use DELTA head, not wb head
        if "delta_pl_db" in self.y_channels:
            self.delta_idx = int(self.y_channels.index("delta_pl_db"))
        else:
            self.delta_idx = 0

    def make_bbox_grid(self, bbox, origin_xy_mode: str, z_margin_m: float) -> Tuple[AntennaGrid, np.ndarray]:
        if origin_xy_mode == "zero":
            x0, y0 = 0.0, 0.0
        else:
            x0 = float(bbox.min.x)
            y0 = float(bbox.min.y)

        z_min = float(bbox.min.z)
        z_max = float(bbox.max.z)
        total_span = (self.K - 1) * self.z_step_m

        z0 = z_min + z_margin_m
        if z0 + total_span > (z_max - z_margin_m):
            z0 = max(z_min, (z_max - z_margin_m) - total_span)

        rx_grid = AntennaGrid(
            origin=np.array([x0, y0, z0], dtype=np.float32),
            deltas=np.asarray(
                [[self.scale_m, 0.0, 0.0],
                 [0.0, self.scale_m, 0.0],
                 [0.0, 0.0, self.z_step_m]], dtype=np.float32),
            shape=(self.K, self.H, self.W),
        )
        rx_coords = grid_coords(rx_grid)
        return rx_grid, rx_coords

    @staticmethod
    def _sanitize_features(x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Replace NaN/Inf produced by feature extraction (e.g., log10(0) at tx pixel)."""
        x = np.asarray(x, dtype=np.float32)
        bad = ~np.isfinite(x)
        n_bad = int(bad.sum())
        if n_bad > 0:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, n_bad

    @torch.no_grad()
    def predict_delta_maps(
        self,
        base_mlink_scene: MlinkScene,
        rx_grid: AntennaGrid,
        rx_coords: np.ndarray,
        tx_xyz_list: np.ndarray,
        fc_hz: float,
        batch_size: int = 4,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Returns delta_maps: (T, K, H, W) float32 in dB (delta_pl_db head)
        """
        tx_xyz_list = np.asarray(tx_xyz_list, dtype=np.float32).reshape(-1, 3)
        T = tx_xyz_list.shape[0]

        adb = AntennaDatabase(tx_xyz_list, rx_coords, None, rx_grid)
        scene = replace(base_mlink_scene, antenna_database=adb)

        x = build_feature_tensor(scene, fc_hz, requested=self.dataset_features).astype(np.float32)
        c_in = int(x.shape[1])

        # x: (T, c_in, K, H, W) -> (T, K*c_in, H, W)
        x_stack = x.transpose(0, 2, 1, 3, 4).reshape(T, self.K * c_in, self.H, self.W)

        if self.keep_idx is not None:
            x_stack = x_stack[:, self.keep_idx, :, :]

        x_stack, n_bad = self._sanitize_features(x_stack)
        if verbose and n_bad > 0:
            print(f"[UNet] sanitized {n_bad} non-finite feature values (NaN/Inf)")

        x_t = torch.from_numpy(x_stack).to(self.device)
        x_t = (x_t - self.x_mean) / self.x_std

        preds = []
        for i0 in range(0, T, batch_size):
            i1 = min(i0 + batch_size, T)
            y_n = self.model(x_t[i0:i1])  # (B, K*y_ch, H, W)
            y_p = y_n * self.y_std + self.y_mean
            preds.append(y_p.detach().float().cpu().numpy())

        pred = np.concatenate(preds, axis=0)  # (T, K*y_ch, H, W)
        pred = pred.reshape(T, self.K, self.y_ch, self.H, self.W)

        delta_maps = pred[:, :, self.delta_idx, :, :].astype(np.float32)  # (T,K,H,W)
        return delta_maps

    def reconstruct_wb_from_delta(
        self,
        delta_maps_tkhw: np.ndarray,
        rx_coords: np.ndarray,
        tx_xyz_list: np.ndarray,
        fc_hz: float,
    ) -> np.ndarray:
        """
        delta_maps_tkhw: (T,K,H,W) in dB
        Returns wb_hat_maps: (T,K,H,W) in dB where wb_hat = fspl(d) + delta_hat
        """
        delta = np.asarray(delta_maps_tkhw, dtype=np.float32)
        tx = np.asarray(tx_xyz_list, dtype=np.float32).reshape(-1, 3)

        K, H, W = self.K, self.H, self.W
        rx = np.asarray(rx_coords, dtype=np.float32).reshape(K, H, W, 3)  # (K,H,W,3)

        diff = rx[None, ...] - tx[:, None, None, None, :]  # (T,K,H,W,3)
        d = np.linalg.norm(diff, axis=-1).astype(np.float32)

        fspl = fspl_db(d, fc_hz)  # (T,K,H,W)
        wb_hat = fspl + delta
        wb_hat = np.nan_to_num(wb_hat, nan=200.0, posinf=200.0, neginf=200.0).astype(np.float32)
        return wb_hat

    @staticmethod
    def wb_matrix_from_maps(wb_maps_tkhw: np.ndarray, nodes_kij: np.ndarray) -> np.ndarray:
        """
        wb_maps_tkhw: (T, K, H, W) for each TX index t
        nodes_kij: (T, 3) int [k,i,j] for each node index
        Returns: (T, T) wb_loss_db
        """
        T, K, H, W = wb_maps_tkhw.shape
        out = np.zeros((T, T), dtype=np.float32)
        for i in range(T):
            for j in range(T):
                k, ii, jj = nodes_kij[j]
                out[i, j] = wb_maps_tkhw[i, k, ii, jj]
        return out


# ----------------------------
# Random scene generation (optional)
# ----------------------------
def make_random_mlink_scene(
    rng: np.random.Generator,
    *,
    frequency_hz: float,
    img_hw: Tuple[int, int] = (64, 64),
    K_slices: int = 4,
    z_step_cells: float = 1.0,
    z_margin_cells: float = 0.5,
    floor_h: float = 0.0,
    ceil_min: float = 8.0,
    ceil_max: float = 20.0,
    scale: float = 0.625,
) -> MlinkScene:
    H, W = img_hw
    ceiling_h = float(rng.uniform(ceil_min, ceil_max))

    mesh = walls_to_mesh(
        generate_wall_map(
            (H, W),
            min_wall_length=8,
            min_door_length=4,
            max_partitions=24,
            rng=rng,
        ),
        floor_height=floor_h,
        ceiling_height=ceiling_h,
    ).apply_scale(scale)

    usable = max(ceiling_h - floor_h - 2 * z_margin_cells, 1e-3)
    total_span = (K_slices - 1) * z_step_cells
    z_step = usable / max(K_slices - 1, 1) if total_span > usable else z_step_cells

    z_start = floor_h + z_margin_cells
    z_end = (ceiling_h - z_margin_cells) - total_span
    z0 = z_start if z_end < z_start else float(rng.uniform(z_start, z_end))

    rx_grid = AntennaGrid(
        origin=scale * np.asarray([0.0, 0.0, z0], dtype=np.float32),
        deltas=scale * np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, z_step]], dtype=np.float32),
        shape=(K_slices, H, W),
    )
    rx_coords = grid_coords(rx_grid)

    # dummy tx (overridden later)
    tx_coords = np.asarray([[scale, scale, scale * (z0 + 1.0)]], dtype=np.float32)

    adb = AntennaDatabase(tx_coords, rx_coords, None, rx_grid)
    mat_db = default_material_db(frequency_hz)
    face2material = {k: 0 for k in range(mesh.faces.shape[0])}
    return MlinkScene(mesh=mesh, material_database=mat_db, face2material=face2material, antenna_database=adb)


# ----------------------------
# Reciprocity stats + IO
# ----------------------------
def reciprocity_stats(mat: np.ndarray, no_path_db: float) -> Dict[str, float]:
    """
    mat: (N,N) WB loss in dB
    Uses i<j pairs.
    """
    mat = np.asarray(mat, dtype=np.float32)
    N = mat.shape[0]

    n_pairs = N * (N - 1) // 2
    if n_pairs == 0:
        return dict(n_pairs=0, n_valid=0, frac_valid=0.0,
                    mean_abs=np.nan, median_abs=np.nan, p95_abs=np.nan, max_abs=np.nan,
                    n_asym_nopath=0)

    abs_diffs = []
    n_valid = 0
    n_asym_nopath = 0

    for i in range(N):
        for j in range(i + 1, N):
            a = float(mat[i, j])
            b = float(mat[j, i])

            a_ok = (math.isfinite(a) and a < no_path_db)
            b_ok = (math.isfinite(b) and b < no_path_db)

            if a_ok and b_ok:
                n_valid += 1
                abs_diffs.append(abs(a - b))
            elif a_ok != b_ok:
                n_asym_nopath += 1

    if n_valid > 0:
        ad = np.array(abs_diffs, dtype=np.float32)
        out = dict(
            n_pairs=int(n_pairs),
            n_valid=int(n_valid),
            frac_valid=float(n_valid / max(n_pairs, 1)),
            mean_abs=float(ad.mean()),
            median_abs=float(np.median(ad)),
            p95_abs=float(np.percentile(ad, 95)),
            max_abs=float(ad.max()),
            n_asym_nopath=int(n_asym_nopath),
        )
    else:
        out = dict(
            n_pairs=int(n_pairs),
            n_valid=0,
            frac_valid=0.0,
            mean_abs=np.nan,
            median_abs=np.nan,
            p95_abs=np.nan,
            max_abs=np.nan,
            n_asym_nopath=int(n_asym_nopath),
        )
    return out


def write_rows_csv(path: Path, fieldnames: List[str], rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_hist(abs_diffs: np.ndarray, title: str, out_png: Path):
    import matplotlib.pyplot as plt
    abs_diffs = np.asarray(abs_diffs, dtype=np.float32)
    plt.figure()
    plt.hist(abs_diffs, bins=30)
    plt.xlabel("|L(i→j) - L(j→i)| (dB)")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_scene_viz(
    *,
    out_dir: Path,
    scene_id: str,
    mscene,
    si_scene,
    rx_grid,
    rx_coords,
    nodes_kij: np.ndarray,
    nodes_xyz: np.ndarray,
    unet: "UNetDeltaWB | None",
    fc_hz: float,
    freqs: np.ndarray,
    max_depth: int,
    viz_samples_per_src: int,
    viz_rx_batch: int,
    viz_tx: int,
    viz_k: int,
    no_path_db: float,
):
    import matplotlib.pyplot as plt

    K, H, W = rx_grid.shape
    viz_tx = int(np.clip(viz_tx, 0, nodes_xyz.shape[0]-1))
    viz_k = int(np.clip(viz_k, 0, K-1))

    tx_xyz = nodes_xyz[viz_tx]
    tx_kij = nodes_kij[viz_tx]
    tx_k, tx_i, tx_j = map(int, tx_kij.tolist())

    # --- binary_walls map (slice viz_k) ---
    adb = AntennaDatabase(np.asarray([tx_xyz], np.float32), rx_coords, None, rx_grid)
    mscene2 = replace(mscene, antenna_database=adb)

    x_bw = build_feature_tensor(mscene2, fc_hz, requested=["binary_walls"]).astype(np.float32)
    bw = x_bw[0]  # (C,K,H,W)
    if bw.shape[0] > 1:
        bw2 = bw.sum(axis=0)
    else:
        bw2 = bw[0]
    bw2 = bw2[viz_k]  # (H,W)

    # --- UNet maps: delta + reconstructed wb ---
    unet_delta = None
    unet_wb = None
    if unet is not None:
        delta_maps = unet.predict_delta_maps(
            base_mlink_scene=mscene,
            rx_grid=rx_grid,
            rx_coords=rx_coords,
            tx_xyz_list=np.asarray([tx_xyz], np.float32),
            fc_hz=fc_hz,
            batch_size=1,
            verbose=True,
        )  # (1,K,H,W)
        wb_maps = unet.reconstruct_wb_from_delta(delta_maps, rx_coords, np.asarray([tx_xyz], np.float32), fc_hz)
        unet_delta = delta_maps[0, viz_k]
        unet_wb = wb_maps[0, viz_k]

    # --- Sionna wb_loss map (only slice viz_k to keep it cheap) ---
    i0 = viz_k * H * W
    i1 = (viz_k + 1) * H * W
    rx_slice = rx_coords[i0:i1]  # (H*W,3)

    sionna_map = np.zeros((H*W,), dtype=np.float32)
    for j0 in range(0, rx_slice.shape[0], viz_rx_batch):
        j1 = min(j0 + viz_rx_batch, rx_slice.shape[0])
        wb = sionna_wb_loss_batch(
            si_scene,
            tx_xyz=tx_xyz,
            rx_xyz=rx_slice[j0:j1],
            freqs_hz=freqs,
            max_depth=max_depth,
            samples_per_src=viz_samples_per_src,
            no_path_db=max(no_path_db, 200.0),
        )
        sionna_map[j0:j1] = wb
    sionna_map = sionna_map.reshape(H, W)

    # --- plot (4 panels) ---
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    png = plots_dir / f"{scene_id}_k{viz_k}_tx{viz_tx}.png"

    fig, axs = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)

    axs[0].imshow(bw2, origin="lower")
    axs[0].set_title(f"binary_walls (k={viz_k})")
    axs[0].scatter([tx_j], [tx_i], s=30)

    if unet_delta is not None:
        im1 = axs[1].imshow(unet_delta, origin="lower")
        axs[1].set_title("UNet delta_pl (dB)")
        axs[1].scatter([tx_j], [tx_i], s=30)
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    else:
        axs[1].axis("off")
        axs[1].set_title("UNet skipped")

    if unet_wb is not None:
        im2 = axs[2].imshow(unet_wb, origin="lower")
        axs[2].set_title("UNet wb_hat = FSPL + delta (dB)")
        axs[2].scatter([tx_j], [tx_i], s=30)
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    else:
        axs[2].axis("off")
        axs[2].set_title("UNet skipped")

    im3 = axs[3].imshow(sionna_map, origin="lower")
    axs[3].set_title(f"Sionna wb_loss (dB)\n(samples_per_src={viz_samples_per_src})")
    axs[3].scatter([tx_j], [tx_i], s=30)
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{scene_id} | TX node {viz_tx} @ (k,i,j)=({tx_k},{tx_i},{tx_j})", y=1.03)
    fig.savefig(png, dpi=160)
    plt.close(fig)

    if unet_wb is not None:
        print(f"[viz] UNet wb_hat: finite={np.isfinite(unet_wb).mean()*100:.1f}% "
              f"min={np.nanmin(unet_wb):.2f} max={np.nanmax(unet_wb):.2f}")
    print(f"[viz] Sionna wb_map: finite={np.isfinite(sionna_map).mean()*100:.1f}% "
          f"min={np.nanmin(sionna_map):.2f} max={np.nanmax(sionna_map):.2f}")
    print(f"[viz] wrote {png}")


def wall_occupancy_khw(mscene: MlinkScene, rx_grid: AntennaGrid, rx_coords: np.ndarray, fc_hz: float) -> np.ndarray:
    """
    Returns occ[k,i,j] True where there's a wall on that voxel.
    Uses the existing binary_walls feature which returns (tx,1,K,H,W).:contentReference[oaicite:1]{index=1}
    """
    # build_feature_tensor requires an AntennaDatabase with rx_grid + tx_coords
    # tx location doesn't matter for binary_walls, but tx_coords must exist
    dummy_tx = np.asarray([rx_grid.origin], dtype=np.float32)  # shape (1,3)

    adb = AntennaDatabase(dummy_tx.astype(np.float32), rx_coords.astype(np.float32), None, rx_grid)
    mscene2 = replace(mscene, antenna_database=adb)

    bw = build_feature_tensor(mscene2, fc_hz, requested=["binary_walls"]).astype(np.float32)
    # bw shape: (tx, C, K, H, W). In your codebase C is effectively 1 for binary_walls.:contentReference[oaicite:2]{index=2}
    bw0 = bw[0]                      # (C,K,H,W)
    occ = (bw0.sum(axis=0) > 0.5)    # (K,H,W) bool
    return occ


def dilate_xy_per_slice(occ_khw: np.ndarray, r: int) -> np.ndarray:
    """
    Inflate walls by r cells (in x/y only, per k-slice) so nodes aren't placed *on* walls.
    """
    if r <= 0:
        return occ_khw

    try:
        from scipy.ndimage import binary_dilation
        out = np.zeros_like(occ_khw, dtype=bool)
        structure = np.ones((2*r + 1, 2*r + 1), dtype=bool)
        for k in range(occ_khw.shape[0]):
            out[k] = binary_dilation(occ_khw[k], structure=structure)
        return out
    except Exception:
        # If scipy isn't available, just fall back to no dilation
        print("[warn] scipy not available; skipping wall dilation")
        return occ_khw


def sample_nodes_from_free(
    rng: np.random.Generator,
    free_khw: np.ndarray,
    n: int,
    *,
    margin_ij: int = 1,
) -> np.ndarray:
    """
    Sample (k,i,j) only from free cells.
    """
    K, H, W = free_khw.shape
    free = np.argwhere(free_khw)  # rows: [k,i,j]

    # keep away from image boundary like you already do with `margin`
    ok = (
        (free[:, 1] >= margin_ij) & (free[:, 1] < H - margin_ij) &
        (free[:, 2] >= margin_ij) & (free[:, 2] < W - margin_ij)
    )
    free = free[ok]

    if free.shape[0] == 0:
        raise RuntimeError("No free cells found on this grid (after margins).")

    replace = free.shape[0] < n
    idx = rng.choice(free.shape[0], size=n, replace=replace)
    return free[idx].astype(np.int32)

def aggregate_reciprocity_from_link_rows(link_rows: List[Dict], prefix: str, no_path_db: float) -> Dict[str, float]:
    """
    Pooled (micro-averaged) reciprocity stats across *all scenes* using per-link rows.

    prefix: "unet" or "sionna"
    Expects columns like:
      f"{prefix}_ij_db", f"{prefix}_ji_db"
    """
    abs_diffs = []
    n_pairs = 0
    n_valid = 0
    n_asym_nopath = 0

    for r in link_rows:
        a_s = r.get(f"{prefix}_ij_db", "")
        b_s = r.get(f"{prefix}_ji_db", "")
        if a_s == "" or b_s == "":
            continue  # model was skipped or missing

        try:
            a = float(a_s)
            b = float(b_s)
        except Exception:
            continue

        n_pairs += 1
        a_ok = (math.isfinite(a) and a < no_path_db)
        b_ok = (math.isfinite(b) and b < no_path_db)

        if a_ok and b_ok:
            n_valid += 1
            abs_diffs.append(abs(a - b))
        elif a_ok != b_ok:
            n_asym_nopath += 1

    if n_pairs == 0:
        return dict(
            n_pairs=0, n_valid=0, frac_valid=0.0,
            mean_abs=np.nan, median_abs=np.nan, p95_abs=np.nan, max_abs=np.nan,
            n_asym_nopath=0,
        )

    if n_valid == 0:
        return dict(
            n_pairs=int(n_pairs), n_valid=0, frac_valid=0.0,
            mean_abs=np.nan, median_abs=np.nan, p95_abs=np.nan, max_abs=np.nan,
            n_asym_nopath=int(n_asym_nopath),
        )

    ad = np.asarray(abs_diffs, dtype=np.float32)
    return dict(
        n_pairs=int(n_pairs),
        n_valid=int(n_valid),
        frac_valid=float(n_valid / max(n_pairs, 1)),
        mean_abs=float(ad.mean()),
        median_abs=float(np.median(ad)),
        p95_abs=float(np.percentile(ad, 95)),
        max_abs=float(ad.max()),
        n_asym_nopath=int(n_asym_nopath),
    )

def main():
    set_tf_memory_growth()

    ap = argparse.ArgumentParser()
    ap.add_argument("--unet_run", type=str, required=True, help="Run dir with model.pt + norm_stats.npz + meta.json")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--xml_glob", type=str, default=None, help='Glob for scene.xml files, e.g. "suites/**/scene.xml"')
    src.add_argument("--random_scenes", type=int, default=0, help="Number of random-generated scenes to test")

    ap.add_argument("--num_scenes", type=int, default=10, help="Max number of scenes to evaluate")
    ap.add_argument("--num_nodes", type=int, default=16, help="Number of test points (nodes) per scene")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--origin_xy_mode", type=str, default="bbox_min", choices=["bbox_min", "zero"])
    ap.add_argument("--z_margin_m", type=float, default=0.3125, help="Z margin in meters for bbox-grid mode")
    ap.add_argument("--grid_mode", type=str, default="bbox", choices=["bbox", "training"],
                    help="bbox: build grid from scene bbox; training: use training rx_grid (random only)")

    # Sionna RT params
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--samples_per_src", type=int, default=1_000_000)
    ap.add_argument("--no_path_db", type=float, default=199.5)

    ap.add_argument("--skip_unet", action="store_true")
    ap.add_argument("--skip_sionna", action="store_true")

    ap.add_argument("--out_dir", type=str, default="reciprocity_out")
    ap.add_argument("--plots", action="store_true")

    ap.add_argument("--viz", action="store_true", help="Save per-scene maps plot (walls, delta, wb_hat, sionna)")
    ap.add_argument("--viz_tx", type=int, default=0, help="Which node index to use as TX for visualization")
    ap.add_argument("--viz_k", type=int, default=0, help="Which height slice k to visualize")
    ap.add_argument("--viz_samples_per_src", type=int, default=50000, help="Sionna samples_per_src used for viz maps")
    ap.add_argument("--viz_rx_batch", type=int, default=512, help="Receiver batch size for Sionna viz")

    ap.add_argument("--wall_clearance_cells", type=int, default=1,
                help="Inflate wall mask by this many cells in x/y (per k slice). 0 disables.")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    unet = None
    if not args.skip_unet:
        unet = UNetDeltaWB(args.unet_run, device=args.device)

    # OFDM/frequency settings: use meta if present
    meta = (unet.meta if unet is not None else json.loads((Path(args.unet_run) / "meta.json").read_text()))
    fc_hz = float(meta.get("frequency_hz", 5.21e9))
    fft_size = int(meta.get("fft_size", 3072))
    df_hz = float(meta.get("subcarrier_spacing_hz", 78_125.0))
    freqs = subcarrier_frequencies_centered(fft_size, df_hz)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []
    link_rows: List[Dict] = []

    # build scene list
    scene_items: List[Tuple[str, str | None]] = []
    if args.xml_glob:
        import glob as _glob
        paths = sorted(_glob.glob(args.xml_glob, recursive=True))
        paths = paths[: args.num_scenes]
        for p in paths:
            pp = Path(p)
            scene_id = f"{pp.parent.name}_{pp.stem}"
            scene_items.append((scene_id, p))
    else:
        for s in range(min(args.random_scenes, args.num_scenes)):
            scene_items.append((f"random_{s:03d}", None))

    if len(scene_items) == 0:
        raise SystemExit("No scenes to evaluate. Check --xml_glob or --random_scenes.")

    for scene_id, scene_path in scene_items:
        print(f"\n=== Scene: {scene_id} ===")

        # --------------------
        # Load / build scene
        # --------------------
        if scene_path is None:
            mscene = make_random_mlink_scene(rng, frequency_hz=fc_hz)
            si_scene = _to_sionna_geometry(mscene, fc_hz)
            force_isotropic_arrays(si_scene)
            bbox = si_scene.mi_scene.bbox()

            if args.grid_mode == "training":
                rx_grid = mscene.antenna_database.rx_grid
                rx_coords = mscene.antenna_database.rx_coords
            else:
                assert unet is not None
                rx_grid, rx_coords = unet.make_bbox_grid(bbox, args.origin_xy_mode, args.z_margin_m)
        else:
            si_scene = load_scene(str(scene_path))
            force_isotropic_arrays(si_scene)
            bbox = si_scene.mi_scene.bbox()

            mscene = MlinkScene.from_sionna(si_scene)

            assert unet is not None
            rx_grid, rx_coords = unet.make_bbox_grid(bbox, args.origin_xy_mode, args.z_margin_m)

        # set carrier
        try:
            si_scene.frequency = fc_hz
        except Exception:
            pass

        # --------------------
        # Sample N node points on the grid (integer indices)
        # --------------------
        N = int(args.num_nodes)
        K, H, W = rx_grid.shape

        margin = 1

        # 1) build occupancy once
        occ_khw = wall_occupancy_khw(mscene, rx_grid, rx_coords, fc_hz)

        # 2) inflate for clearance (recommended)
        occ_khw = dilate_xy_per_slice(occ_khw, args.wall_clearance_cells)

        # 3) sample from free cells only
        free_khw = ~occ_khw
        nodes_kij = sample_nodes_from_free(rng, free_khw, N, margin_ij=margin)
        nodes_xyz = node_xyz_from_kij(rx_grid, nodes_kij)

        # (optional sanity check)
        if np.any(occ_khw[nodes_kij[:,0], nodes_kij[:,1], nodes_kij[:,2]]):
            raise RuntimeError("Bug: sampled a node on a wall cell")
        

        # --------------------
        # UNet WB matrix (FIXED: delta -> wb_hat)
        # --------------------
        wb_unet = None
        if not args.skip_unet:
            delta_maps = unet.predict_delta_maps(
                base_mlink_scene=mscene,
                rx_grid=rx_grid,
                rx_coords=rx_coords,
                tx_xyz_list=nodes_xyz,
                fc_hz=fc_hz,
                batch_size=4,
                verbose=True,
            )  # (N,K,H,W)
            wb_maps = unet.reconstruct_wb_from_delta(delta_maps, rx_coords, nodes_xyz, fc_hz)  # (N,K,H,W)
            wb_unet = unet.wb_matrix_from_maps(wb_maps, nodes_kij)

        # --------------------
        # Sionna WB matrix
        # --------------------
        wb_sionna = None
        if not args.skip_sionna:
            wb_sionna = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                np.random.seed(args.seed + 1000 + i)
                tf.random.set_seed(args.seed + 1000 + i)

                wb_vec = sionna_wb_loss_batch(
                    si_scene,
                    tx_xyz=nodes_xyz[i],
                    rx_xyz=nodes_xyz,
                    freqs_hz=freqs,
                    max_depth=args.max_depth,
                    samples_per_src=args.samples_per_src,
                    no_path_db=max(args.no_path_db, 200.0),
                )
                wb_sionna[i, :] = wb_vec

        # --------------------
        # Stats + printing
        # --------------------
        if wb_unet is not None:
            st_u = reciprocity_stats(wb_unet, no_path_db=args.no_path_db)
            print(f"UNet  : valid_pairs={st_u['n_valid']}/{st_u['n_pairs']} ({st_u['frac_valid']*100:.1f}%) "
                  f"| mean|Δ|={st_u['mean_abs']:.3f} dB | p95|Δ|={st_u['p95_abs']:.3f} dB | max|Δ|={st_u['max_abs']:.3f} dB "
                  f"| asym_noPath={st_u['n_asym_nopath']}")
            summary_rows.append(dict(scene=scene_id, model="unet", **st_u))

        if wb_sionna is not None:
            st_s = reciprocity_stats(wb_sionna, no_path_db=args.no_path_db)
            print(f"Sionna: valid_pairs={st_s['n_valid']}/{st_s['n_pairs']} ({st_s['frac_valid']*100:.1f}%) "
                  f"| mean|Δ|={st_s['mean_abs']:.3f} dB | p95|Δ|={st_s['p95_abs']:.3f} dB | max|Δ|={st_s['max_abs']:.3f} dB "
                  f"| asym_noPath={st_s['n_asym_nopath']}")
            summary_rows.append(dict(scene=scene_id, model="sionna", **st_s))

        # --------------------
        # Per-link rows (i<j)
        # --------------------
        for i in range(N):
            for j in range(i + 1, N):
                row = dict(
                    scene=scene_id,
                    i=i, j=j,
                    pi_xyz=",".join([f"{v:.3f}" for v in nodes_xyz[i]]),
                    pj_xyz=",".join([f"{v:.3f}" for v in nodes_xyz[j]]),
                )

                if wb_unet is not None:
                    u_ij = float(wb_unet[i, j]); u_ji = float(wb_unet[j, i])
                    u_ok = (math.isfinite(u_ij) and u_ij < args.no_path_db and math.isfinite(u_ji) and u_ji < args.no_path_db)
                    row.update(dict(unet_ij_db=u_ij, unet_ji_db=u_ji, unet_absdiff_db=abs(u_ij - u_ji), unet_valid=int(u_ok)))
                else:
                    row.update(dict(unet_ij_db="", unet_ji_db="", unet_absdiff_db="", unet_valid=""))

                if wb_sionna is not None:
                    s_ij = float(wb_sionna[i, j]); s_ji = float(wb_sionna[j, i])
                    s_ok = (math.isfinite(s_ij) and s_ij < args.no_path_db and math.isfinite(s_ji) and s_ji < args.no_path_db)
                    row.update(dict(sionna_ij_db=s_ij, sionna_ji_db=s_ji, sionna_absdiff_db=abs(s_ij - s_ji), sionna_valid=int(s_ok)))
                else:
                    row.update(dict(sionna_ij_db="", sionna_ji_db="", sionna_absdiff_db="", sionna_valid=""))

                link_rows.append(row)

        # --------------------
        # Plots (optional)
        # --------------------
        if args.plots:
            if wb_unet is not None:
                u_abs = []
                for i in range(N):
                    for j in range(i + 1, N):
                        a = wb_unet[i, j]; b = wb_unet[j, i]
                        if (a < args.no_path_db) and (b < args.no_path_db) and np.isfinite(a) and np.isfinite(b):
                            u_abs.append(abs(a - b))
                if len(u_abs) > 0:
                    plot_hist(np.array(u_abs), f"{scene_id} - UNet reciprocity |Δ|", out_dir / "plots" / f"{scene_id}_unet_hist.png")

            if wb_sionna is not None:
                s_abs = []
                for i in range(N):
                    for j in range(i + 1, N):
                        a = wb_sionna[i, j]; b = wb_sionna[j, i]
                        if (a < args.no_path_db) and (b < args.no_path_db) and np.isfinite(a) and np.isfinite(b):
                            s_abs.append(abs(a - b))
                if len(s_abs) > 0:
                    plot_hist(np.array(s_abs), f"{scene_id} - Sionna reciprocity |Δ|", out_dir / "plots" / f"{scene_id}_sionna_hist.png")

        if args.viz:
            save_scene_viz(
                out_dir=out_dir,
                scene_id=scene_id,
                mscene=mscene,
                si_scene=si_scene,
                rx_grid=rx_grid,
                rx_coords=rx_coords,
                nodes_kij=nodes_kij,
                nodes_xyz=nodes_xyz,
                unet=unet,
                fc_hz=fc_hz,
                freqs=freqs,
                max_depth=args.max_depth,
                viz_samples_per_src=args.viz_samples_per_src,
                viz_rx_batch=args.viz_rx_batch,
                viz_tx=args.viz_tx,
                viz_k=args.viz_k,
                no_path_db=args.no_path_db,
            )

    # --------------------
    # Aggregate (ALL scenes)
    # --------------------
    if not args.skip_unet:
        agg_u = aggregate_reciprocity_from_link_rows(link_rows, "unet", args.no_path_db)
        print(f"\nAGG UNet  : valid_pairs={agg_u['n_valid']}/{agg_u['n_pairs']} ({agg_u['frac_valid']*100:.1f}%) "
            f"| mean|Δ|={agg_u['mean_abs']:.3f} dB | p95|Δ|={agg_u['p95_abs']:.3f} dB | max|Δ|={agg_u['max_abs']:.3f} dB "
            f"| asym_noPath={agg_u['n_asym_nopath']}")
        summary_rows.append(dict(scene="ALL", model="unet", **agg_u))

    if not args.skip_sionna:
        agg_s = aggregate_reciprocity_from_link_rows(link_rows, "sionna", args.no_path_db)
        print(f"AGG Sionna: valid_pairs={agg_s['n_valid']}/{agg_s['n_pairs']} ({agg_s['frac_valid']*100:.1f}%) "
            f"| mean|Δ|={agg_s['mean_abs']:.3f} dB | p95|Δ|={agg_s['p95_abs']:.3f} dB | max|Δ|={agg_s['max_abs']:.3f} dB "
            f"| asym_noPath={agg_s['n_asym_nopath']}")
        summary_rows.append(dict(scene="ALL", model="sionna", **agg_s))

    # --------------------
    # Save outputs
    # --------------------
    summary_path = out_dir / "summary.csv"
    links_path = out_dir / "links.csv"

    write_rows_csv(
        summary_path,
        fieldnames=["scene", "model", "n_pairs", "n_valid", "frac_valid", "mean_abs", "median_abs", "p95_abs", "max_abs", "n_asym_nopath"],
        rows=summary_rows,
    )
    write_rows_csv(
        links_path,
        fieldnames=["scene", "i", "j", "pi_xyz", "pj_xyz",
                    "unet_ij_db", "unet_ji_db", "unet_absdiff_db", "unet_valid",
                    "sionna_ij_db", "sionna_ji_db", "sionna_absdiff_db", "sionna_valid"],
        rows=link_rows,
    )

    print("\nWrote:")
    print(" ", summary_path)
    print(" ", links_path)
    if args.plots or args.viz:
        print(" ", out_dir / "plots")


if __name__ == "__main__":
    main()