# training_tdl.py
import os
from pyexpat import model
os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import math
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange



from mlink.antenna import AntennaGrid, AntennaDatabase 
from mlink.feature import build_feature_tensor 
from mlink.geometry import generate_wall_map, walls_to_mesh 
from mlink.scene import Scene

from mlink.channel_tdl import (
    RtCfg,
    subcarrier_frequencies_centered,
    compute_tdl_batch,
)


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    out_dir: str = "runs/tdl"
    frequency_hz: float = 5.21e9  # match ns3 WiFi run if desired

    img_hw: tuple[int, int] = (64, 64)
    K_slices: int = 4
    z_step: float = 1.0
    z_margin: float = 0.5

    floor_h: float = 0.0
    ceil_min: float = 8.0
    ceil_max: float = 20.0

    scale: float = 0.625

    tx_origin_xy: tuple[float, float] = (1.75, 1.75)
    tx_z: float = 2.4
    tx_spacing_xy: float = 12.0
    tx_shape: tuple[int, int, int] = (1, 5, 5)

    # OFDM grid / TDL
    fft_size: int = 512           # START SMALL. 3072 later.
    subcarrier_spacing_hz: float = 78_125.0
    L_taps: int = 16              # START SMALL. 32/64 later.

    # Ray tracing label generation
    rx_batch: int = 256
    rt: RtCfg = field(default_factory=lambda: RtCfg(
        max_depth=5,
        samples_per_src=200_000,
        diffuse_reflection=True,
        diffraction=False,
    ))

    diag_full_L: int = 512 
    diag_tail_L: int = 4      
    diag_print_every_tx: int = 1

    requested_features: list[str] = field(default_factory=lambda: ["binary_walls", "electrical_distance", "cost", "height_cond"])

    # dataset size
    num_scenes: int = 20
    train_frac: float = 0.9

    # training
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 2e-4
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

cfg = CFG()


def default_material_db(freq: float) -> pl.DataFrame:
    # keep consistent with your current pipeline
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


def make_scene(rng: np.random.Generator) -> Scene:
    H, W = cfg.img_hw
    ceiling_h = float(rng.uniform(cfg.ceil_min, cfg.ceil_max))
    mesh = walls_to_mesh(
        generate_wall_map((H, W), min_wall_length=8, min_door_length=4, max_partitions=24, rng=rng),
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
        deltas=cfg.scale * np.asarray([[1,0,0],[0,1,0],[0,0,z_step]], dtype=np.float32),
        shape=(cfg.K_slices, H, W),
    )

    tx_grid = AntennaGrid(
        origin=cfg.scale * np.asarray([cfg.tx_origin_xy[0], cfg.tx_origin_xy[1], cfg.tx_z], dtype=np.float32),
        deltas=cfg.scale * np.asarray([[cfg.tx_spacing_xy,0,0],[0,cfg.tx_spacing_xy,0],[0,0,1]], dtype=np.float32),
        shape=cfg.tx_shape,
    )

    antenna_db = AntennaDatabase.from_grid(tx_grid, rx_grid)
    mat_db = default_material_db(cfg.frequency_hz)
    face2material = {k: 0 for k in range(mesh.faces.shape[0])}

    return Scene(mesh=mesh, material_database=mat_db, face2material=face2material, antenna_database=antenna_db)


def compute_labels_for_scene(scene: Scene):
    rx_grid = scene.antenna_database.rx_grid
    assert rx_grid is not None
    K, H, W = rx_grid.shape
    tx_coords = scene.antenna_database.tx_coords
    rx_coords = scene.antenna_database.rx_coords

    L = cfg.L_taps
    L_full = int(min(cfg.diag_full_L, cfg.fft_size))
    tail_L = int(min(cfg.diag_tail_L, L))
    y_ch = 2 + 2 * L
    y = np.zeros((tx_coords.shape[0], y_ch, K, H, W), dtype=np.float32)

    freqs = subcarrier_frequencies_centered(cfg.fft_size, cfg.subcarrier_spacing_hz)
    si = scene.to_sionna_geometry(cfg.frequency_hz)

    # collect truncation stats across all tx/rx in this scene
    eta_all = []       # fraction of energy captured by first L taps
    tailfrac_all = []  # fraction of (stored) energy sitting in last tail_L taps

    for t, tx in enumerate(tx_coords):
        wb_all = np.zeros((rx_coords.shape[0],), dtype=np.float32)
        ex_all = np.zeros((rx_coords.shape[0],), dtype=np.float32)
        taps_all = np.zeros((rx_coords.shape[0], L), dtype=np.complex64)

        B = cfg.rx_batch
        for i0 in range(0, rx_coords.shape[0], B):
            i1 = min(i0 + B, rx_coords.shape[0])

            wb, ex, taps_full = compute_tdl_batch(
                si_scene=si,
                tx_xyz=tx,
                rx_xyz=rx_coords[i0:i1],
                frequencies_hz=freqs,
                L_taps=L_full,   # <-- request longer taps for diagnostics
                rt=cfg.rt,
            )  # taps_full: (batch, L_full) complex

            wb_all[i0:i1] = wb
            ex_all[i0:i1] = ex

            # truncation metrics
            p_full = np.sum(np.abs(taps_full)**2, axis=-1) + 1e-12
            p_head = np.sum(np.abs(taps_full[:, :L])**2, axis=-1)
            eta = p_head / p_full                      # 1.0 means “no truncation”
            eta_all.append(eta)

            if tail_L > 0:
                p_tail = np.sum(np.abs(taps_full[:, L-tail_L:L])**2, axis=-1)
                tailfrac = p_tail / (p_head + 1e-12)    # large => energy “piled up” at end of stored window
                tailfrac_all.append(tailfrac)

            # store only first L taps as labels
            taps_all[i0:i1] = taps_full[:, :L].astype(np.complex64)

        wb_map = wb_all.reshape(K, H, W)
        ex_map = ex_all.reshape(K, H, W)
        tr_map = np.real(taps_all).reshape(K, H, W, L)
        ti_map = np.imag(taps_all).reshape(K, H, W, L)

        y[t, 0, :, :, :] = wb_map
        y[t, 1, :, :, :] = ex_map
        y[t, 2:2+L, :, :, :] = np.transpose(tr_map, (3,0,1,2))
        y[t, 2+L:2+2*L, :, :, :] = np.transpose(ti_map, (3,0,1,2))

        if cfg.diag_print_every_tx:
            eta_t = np.concatenate(eta_all[-(rx_coords.shape[0]//B + 1):], axis=0)
            print(f"tx {t+1}: eta(p50/p90/p99)={np.percentile(eta_t,[50,90,99])}")

    # scene-level print
    eta_all = np.concatenate(eta_all, axis=0)
    if len(tailfrac_all):
        tailfrac_all = np.concatenate(tailfrac_all, axis=0)
    else:
        tailfrac_all = None

    print(
        f"[tap diag] L={L}/{L_full} "
        f"eta(p10/p50/p90/p99)={np.percentile(eta_all,[10,50,90,99])} "
        f"mean_eta={eta_all.mean():.4f} "
        + (f"tailfrac(p90/p99)={np.percentile(tailfrac_all,[90,99])}" if tailfrac_all is not None else "")
    )

    return y


def compute_norm_stats(x_mm: np.memmap, y_mm: np.memmap, max_samples: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    N = x_mm.shape[0]
    take = min(max_samples, N)
    idx = rng.choice(N, size=take, replace=False)

    x = torch.from_numpy(np.array(x_mm[idx], dtype=np.float32))  # (take,4,H,W)
    y = torch.from_numpy(np.array(y_mm[idx], dtype=np.float32))  # (take,Yo,H,W)

    x_mean = x.mean(dim=(0,2,3)).view(x.shape[1], 1, 1)
    x_std  = x.std(dim=(0,2,3)).clamp_min(1e-6).view(x.shape[1], 1, 1)

    y_mean = y.mean(dim=(0,2,3)).view(y.shape[1], 1, 1)
    y_std  = y.std(dim=(0,2,3)).clamp_min(1e-6).view(y.shape[1], 1, 1)

    return dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


class MemmapDataset(Dataset):
    def __init__(self, x_mm, y_mm, stats):
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.stats = stats

    def __len__(self): return self.x_mm.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.x_mm[i], dtype=np.float32))
        y = torch.from_numpy(np.array(self.y_mm[i], dtype=np.float32))

        x = (x - self.stats["x_mean"]) / self.stats["x_std"]
        y = (y - self.stats["y_mean"]) / self.stats["y_std"]
        return x, y


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)

        self.mid = nn.Sequential(nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.ReLU())

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        m  = self.mid(self.pool2(e2))
        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)
    
def _valid_groups(ch: int, groups: int) -> int:
    g = min(groups, ch)
    while ch % g != 0:
        g -= 1
    return max(g, 1)

###Alternative Model 
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        g1 = _valid_groups(in_ch, groups)
        g2 = _valid_groups(out_ch, groups)

        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.drop(h)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)
    
class UNet3(nn.Module):
    """
    3-level U-Net: 64->32->16->8 bottleneck and back.
    Good default for 64x64 maps.
    """
    def __init__(self, in_ch, out_ch, base=64, groups=8, dropout=0.0):
        super().__init__()

        # encoder
        self.e1 = nn.Sequential(
            ResBlock(in_ch, base, groups=groups, dropout=0.0),
            ResBlock(base, base, groups=groups, dropout=0.0),
        )
        self.p1 = nn.MaxPool2d(2)  # 64->32

        self.e2 = nn.Sequential(
            ResBlock(base, base*2, groups=groups, dropout=0.0),
            ResBlock(base*2, base*2, groups=groups, dropout=0.0),
        )
        self.p2 = nn.MaxPool2d(2)  # 32->16

        self.e3 = nn.Sequential(
            ResBlock(base*2, base*4, groups=groups, dropout=0.0),
            ResBlock(base*4, base*4, groups=groups, dropout=0.0),
        )
        self.p3 = nn.MaxPool2d(2)  # 16->8

        # bottleneck
        self.mid = nn.Sequential(
            ResBlock(base*4, base*8, groups=groups, dropout=dropout),
            ResBlock(base*8, base*8, groups=groups, dropout=dropout),
        )

        # decoder
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)  # 8->16
        self.d3 = nn.Sequential(
            ResBlock(base*8, base*4, groups=groups, dropout=0.0),
            ResBlock(base*4, base*4, groups=groups, dropout=0.0),
        )

        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)  # 16->32
        self.d2 = nn.Sequential(
            ResBlock(base*4, base*2, groups=groups, dropout=0.0),
            ResBlock(base*2, base*2, groups=groups, dropout=0.0),
        )

        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)    # 32->64
        self.d1 = nn.Sequential(
            ResBlock(base*2, base, groups=groups, dropout=0.0),
            ResBlock(base, base, groups=groups, dropout=0.0),
        )

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        m  = self.mid(self.p3(e3))

        d3 = self.u3(m)
        d3 = self.d3(torch.cat([d3, e3], dim=1))

        d2 = self.u2(d3)
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.u1(d2)
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


def _unnorm_y(y_norm: torch.Tensor, stats_dev: dict) -> torch.Tensor:
    # y_norm: (B, y_ch, H, W)
    return y_norm * stats_dev["y_std"] + stats_dev["y_mean"]

def cfr_from_taps(y_phys: torch.Tensor, L_taps: int, fft_size: int) -> torch.Tensor:
    """
    y_phys: (B, y_ch, H, W) in physical (unnormalized) units
    Returns: H_cfr (B, N, H, W) complex64/complex32 in *centered* (fftshift) order.

    This matches the label construction you used:
      taps = ifft(ifftshift(H_norm))  =>  H_norm = fftshift(fft(taps))
    """
    B, y_ch, H, W = y_phys.shape
    L = int(L_taps)
    N = int(fft_size)

    tr = y_phys[:, 2:2+L, :, :]            # (B, L, H, W)
    ti = y_phys[:, 2+L:2+2*L, :, :]        # (B, L, H, W)
    taps = torch.complex(tr, ti)           # (B, L, H, W)

    # zero-pad taps to length N along the tap dimension
    if L < N:
        pad = torch.zeros((B, N - L, H, W), device=y_phys.device, dtype=taps.dtype)
        tapsN = torch.cat([taps, pad], dim=1)   # (B, N, H, W)
    else:
        tapsN = taps[:, :N, :, :]               # (B, N, H, W)

    # FFT to frequency domain (unshifted), then shift to centered ordering
    H_unshift = torch.fft.fft(tapsN, dim=1)                 # (B, N, H, W)
    H_centered = torch.fft.fftshift(H_unshift, dim=1)       # (B, N, H, W)
    return H_centered

@torch.no_grad()
def cfr_metrics_on_batch(pred_norm: torch.Tensor, tgt_norm: torch.Tensor, stats_dev: dict,
                         L_taps: int, fft_size: int, max_pixels: int = 2048):
    """
    Computes CFR NMSE (complex) and magnitude MAE on a random subset of pixels.
    Returns:
      nmse (float), nmse_db (float), mag_mae (float)
    """
    device = pred_norm.device
    pred_phys = _unnorm_y(pred_norm, stats_dev)
    tgt_phys  = _unnorm_y(tgt_norm,  stats_dev)

    H_pred = cfr_from_taps(pred_phys, L_taps=L_taps, fft_size=fft_size)  # (B,N,H,W)
    H_true = cfr_from_taps(tgt_phys,  L_taps=L_taps, fft_size=fft_size)

    B, N, H, W = H_pred.shape
    P = H * W

    # sample a subset of spatial pixels to keep this cheap
    take = min(int(max_pixels), P)
    idx = torch.randperm(P, device=device)[:take]  # (take,)
    # flatten spatial dims, gather
    Hp = H_pred.reshape(B, N, P)[:, :, idx]  # (B,N,take)
    Ht = H_true.reshape(B, N, P)[:, :, idx]

    err = Hp - Ht
    nmse = (err.abs()**2).mean() / (Ht.abs()**2).mean().clamp_min(1e-12)
    nmse = float(nmse.item())
    nmse_db = 10.0 * math.log10(max(nmse, 1e-12))

    mag_mae = float((Hp.abs() - Ht.abs()).abs().mean().item())

    return nmse, nmse_db, mag_mae

def taps_complex_from_y(y, L):
    tr = y[:, 2:2+L]
    ti = y[:, 2+L:2+2*L]
    return torch.complex(tr, ti)  # (B,L,H,W)

def cfr_from_taps_complex(taps, fft_size):
    # taps: (B,L,H,W) complex
    B,L,H,W = taps.shape
    N = int(fft_size)
    if L < N:
        pad = torch.zeros((B, N-L, H, W), device=taps.device, dtype=taps.dtype)
        tapsN = torch.cat([taps, pad], dim=1)
    else:
        tapsN = taps[:, :N]
    H_unshift = torch.fft.fft(tapsN, dim=1)
    H_centered = torch.fft.fftshift(H_unshift, dim=1)
    return H_centered  # (B,N,H,W)

def cfr_nmse_loss(pred, tgt, L, fft_size, max_pixels=2048):
    # pred/tgt are NORMALIZED (same space you train in)
    taps_p = taps_complex_from_y(pred, L)
    taps_t = taps_complex_from_y(tgt,  L)

    Hp = cfr_from_taps_complex(taps_p, fft_size)
    Ht = cfr_from_taps_complex(taps_t, fft_size)

    B,N,H,W = Hp.shape
    P = H*W
    take = min(max_pixels, P)
    idx = torch.randperm(P, device=pred.device)[:take]

    Hp = Hp.reshape(B, N, P)[:, :, idx]
    Ht = Ht.reshape(B, N, P)[:, :, idx]

    num = (Hp - Ht).abs().pow(2).mean()
    den = Ht.abs().pow(2).mean().clamp_min(1e-12)
    return num / den

def weighted_tap_loss(pred, tgt, L, eps=1e-6):
    pr = pred[:, 2:2+L]; pi = pred[:, 2+L:2+2*L]
    tr = tgt[:,  2:2+L]; ti = tgt[:,  2+L:2+2*L]

    # per-tap power weight from target
    w = (tr*tr + ti*ti).detach()  # (B,L,H,W)
    w = w / (w.mean() + eps)

    err2 = (pr-tr)**2 + (pi-ti)**2
    return (w * err2).mean()

def weighted_tap_loss_phys(pred_norm, tgt_norm, stats_dev, L, eps=1e-6):
    m = stats_dev["y_mean"][2:2+2*L]
    s = stats_dev["y_std"][2:2+2*L]

    pred_t = pred_norm[:, 2:2+2*L] * s + m
    tgt_t  = tgt_norm[:,  2:2+2*L] * s + m

    pr = pred_t[:, 0:L]; pi = pred_t[:, L:2*L]
    tr = tgt_t[:,  0:L]; ti = tgt_t[:,  L:2*L]

    w = (tr*tr + ti*ti).detach()
    w = w / (w.mean() + eps)

    err2 = (pr-tr)**2 + (pi-ti)**2
    return (w * err2).mean()

def cfr_nmse_loss_phys(pred_norm, tgt_norm, stats_dev, L, fft_size, max_pixels=2048):
    # Unnormalize tap channels only
    m = stats_dev["y_mean"][2:2+2*L]  # (2L,1,1)
    s = stats_dev["y_std"][2:2+2*L]   # (2L,1,1)

    pred_t = pred_norm[:, 2:2+2*L] * s + m
    tgt_t  = tgt_norm[:,  2:2+2*L] * s + m

    pr = pred_t[:, 0:L]; pi = pred_t[:, L:2*L]
    tr = tgt_t[:,  0:L]; ti = tgt_t[:,  L:2*L]

    taps_p = torch.complex(pr, pi)  # (B,L,H,W)
    taps_t = torch.complex(tr, ti)

    Hp = cfr_from_taps_complex(taps_p, fft_size)
    Ht = cfr_from_taps_complex(taps_t, fft_size)

    B, N, H, W = Hp.shape
    P = H * W
    take = min(max_pixels, P)
    idx = torch.randperm(P, device=pred_norm.device)[:take]

    Hp = Hp.reshape(B, N, P)[:, :, idx]
    Ht = Ht.reshape(B, N, P)[:, :, idx]

    num = (Hp - Ht).abs().pow(2).mean()
    den = Ht.abs().pow(2).mean().clamp_min(1e-12)
    return num / den

def estimate_truncation_fraction_for_scene(scene: Scene, tx_xyz: np.ndarray, num_rx_samples: int = 256):
    rx = scene.antenna_database.rx_coords
    if rx.shape[0] <= num_rx_samples:
        rx_s = rx
    else:
        ridx = np.random.default_rng(0).choice(rx.shape[0], size=num_rx_samples, replace=False)
        rx_s = rx[ridx]

    si = scene.to_sionna_geometry(cfg.frequency_hz)

    freqs = subcarrier_frequencies_centered(cfg.fft_size, cfg.subcarrier_spacing_hz)

    # ask for FULL taps length N (diagnostic only!)
    wb, ex, taps_full = compute_tdl_batch(
        si_scene=si,
        tx_xyz=tx_xyz,
        rx_xyz=rx_s,
        frequencies_hz=freqs,
        L_taps=cfg.fft_size,     # <-- full length
        rt=cfg.rt,
    )  # taps_full: (num_rx_samples, N) complex

    p_all  = np.sum(np.abs(taps_full)**2, axis=-1)
    p_head = np.sum(np.abs(taps_full[:, :cfg.L_taps])**2, axis=-1)
    eta = p_head / (p_all + 1e-12)
    return eta

def nmse_best_complex_scalar(Hp: torch.Tensor, Ht: torch.Tensor, eps: float = 1e-12, phase_only: bool = False):
    """
    Hp, Ht: complex tensors shaped (B, N, P) (or any (B, ...)).
    Finds per-sample best complex scalar alpha (B,) minimizing ||alpha*Hp - Ht||^2.
    Returns: nmse (float), alpha (B,) complex
    """
    # Flatten everything except batch
    B = Hp.shape[0]
    Hp_f = Hp.reshape(B, -1)
    Ht_f = Ht.reshape(B, -1)

    num = torch.sum(torch.conj(Hp_f) * Ht_f, dim=1)                     # (B,)
    den = torch.sum(torch.abs(Hp_f) ** 2, dim=1).clamp_min(eps)         # (B,)
    alpha = num / den                                                   # (B,)

    if phase_only:
        alpha = alpha / alpha.abs().clamp_min(eps)

    Hp_rot = Hp_f * alpha[:, None]
    err = torch.sum(torch.abs(Hp_rot - Ht_f) ** 2, dim=1)               # (B,)
    sig = torch.sum(torch.abs(Ht_f) ** 2, dim=1).clamp_min(eps)         # (B,)
    nmse = (err / sig).mean()

    return float(nmse.item()), alpha


@torch.no_grad()
def cfr_metrics_with_rotation(pred_norm: torch.Tensor, tgt_norm: torch.Tensor, stats_dev: dict,
                              L_taps: int, fft_size: int, max_pixels: int = 2048):
    """
    Returns raw_nmse, raw_nmse_db, rot_nmse, rot_nmse_db, phase_nmse, phase_nmse_db, |H|_mae, alpha_stats
    """
    pred_phys = _unnorm_y(pred_norm, stats_dev)
    tgt_phys  = _unnorm_y(tgt_norm,  stats_dev)

    H_pred = cfr_from_taps(pred_phys, L_taps=L_taps, fft_size=fft_size)  # (B,N,H,W)
    H_true = cfr_from_taps(tgt_phys,  L_taps=L_taps, fft_size=fft_size)

    B, N, H, W = H_pred.shape
    P = H * W

    take = min(int(max_pixels), P)
    idx = torch.randperm(P, device=pred_norm.device)[:take]

    Hp = H_pred.reshape(B, N, P)[:, :, idx]   # (B,N,take)
    Ht = H_true.reshape(B, N, P)[:, :, idx]

    # raw NMSE (per-sample averaged)
    err = torch.sum(torch.abs(Hp - Ht) ** 2, dim=(1,2))
    sig = torch.sum(torch.abs(Ht) ** 2, dim=(1,2)).clamp_min(1e-12)
    raw_nmse = (err / sig).mean().item()
    raw_nmse_db = 10.0 * math.log10(max(raw_nmse, 1e-12))

    # best complex scalar
    rot_nmse, alpha = nmse_best_complex_scalar(Hp, Ht, phase_only=False)
    rot_nmse_db = 10.0 * math.log10(max(rot_nmse, 1e-12))

    # best phase-only scalar
    ph_nmse, alpha_ph = nmse_best_complex_scalar(Hp, Ht, phase_only=True)
    ph_nmse_db = 10.0 * math.log10(max(ph_nmse, 1e-12))

    mag_mae = float((Hp.abs() - Ht.abs()).abs().mean().item())

    # summarize alpha (magnitude + angle)
    alpha_mag = alpha.abs()
    alpha_ang = torch.angle(alpha) * (180.0 / math.pi)

    alpha_stats = dict(
        mag_mean=float(alpha_mag.mean().item()),
        mag_p10=float(alpha_mag.quantile(0.10).item()) if alpha_mag.numel() > 1 else float(alpha_mag.item()),
        mag_p90=float(alpha_mag.quantile(0.90).item()) if alpha_mag.numel() > 1 else float(alpha_mag.item()),
        ang_mean_deg=float(alpha_ang.mean().item()),
        ang_std_deg=float(alpha_ang.std().item()) if alpha_ang.numel() > 1 else 0.0,
    )

    return raw_nmse, raw_nmse_db, rot_nmse, rot_nmse_db, ph_nmse, ph_nmse_db, mag_mae, alpha_stats

def main():
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- figure out shapes once ----
    rng = np.random.default_rng(0)
    tmp = make_scene(rng)
    x_tmp = build_feature_tensor(tmp, cfg.frequency_hz, requested=cfg.requested_features).astype(np.float32)
    c_in = x_tmp.shape[1]
    num_tx = tmp.antenna_database.tx_coords.shape[0]
    K, H, W = tmp.antenna_database.rx_grid.shape
    y_ch = 2 + 2 * cfg.L_taps
    total_samples = cfg.num_scenes * num_tx * K

    x_path = out_dir / "x.dat"
    y_path = out_dir / "y.dat"
    meta_path = out_dir / "meta.json"

    print("x exists?", x_path.exists(), "y exists?", y_path.exists())
    if x_path.exists() and y_path.exists():
        print("Using cached memmap dataset. Delete runs/tdl/x.dat and y.dat to rebuild labels.")

    # ---- build memmaps (with no_path_frac logging) ----
    if not x_path.exists() or not y_path.exists():
        x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(total_samples, c_in, H, W))
        y_mm = np.memmap(y_path, dtype="float16", mode="w+", shape=(total_samples, y_ch, H, W))

        idx = 0
        for s in range(cfg.num_scenes):
            scene = make_scene(rng)

            x_ft = build_feature_tensor(scene, cfg.frequency_hz, requested=cfg.requested_features).astype(np.float32)
            y_ft = compute_labels_for_scene(scene).astype(np.float32)  # (tx,y_ch,K,H,W)

            # report how often PathSolver produced "no path" defaults ---
            wb = y_ft[:, 0, :, :, :]  # (tx,K,H,W) wb_loss_db in physical units
            no_path_frac = float(np.mean(wb >= 199.0))
            print(f"[scene {s+1:03d}/{cfg.num_scenes}] no_path_frac={no_path_frac:.3f}")

            # flatten (tx,K) -> samples
            x_flat = rearrange(x_ft, "tx c k h w -> (tx k) c h w")
            y_flat = rearrange(y_ft, "tx c k h w -> (tx k) c h w")

            n = x_flat.shape[0]
            x_mm[idx:idx + n] = x_flat
            y_mm[idx:idx + n] = y_flat.astype(np.float16)
            idx += n
            x_mm.flush(); y_mm.flush()
            print(f"[scene {s+1:03d}/{cfg.num_scenes}] wrote {n} samples (total {idx})")

        meta = dict(
            total_samples=total_samples, H=H, W=W, K=K, num_tx=num_tx,
            y_ch=y_ch,
            fft_size=cfg.fft_size, subcarrier_spacing_hz=cfg.subcarrier_spacing_hz, L_taps=cfg.L_taps,
            x_dtype="float32", y_dtype="float16",
        )
        meta_path.write_text(json.dumps(meta, indent=2))

    # ---- open memmaps (read) ----
    x_mm = np.memmap(x_path, dtype="float32", mode="r", shape=(total_samples, c_in, H, W))
    y_mm = np.memmap(y_path, dtype="float16", mode="r", shape=(total_samples, y_ch, H, W))

    rng = np.random.default_rng(0)
    idx = rng.choice(total_samples, size=min(512, total_samples), replace=False)

    wb = np.array(y_mm[idx, 0], dtype=np.float32)
    ex = np.array(y_mm[idx, 1], dtype=np.float32)
    taps = np.array(y_mm[idx, 2:], dtype=np.float32)

    print(f"wb: mean={wb.mean():.3f} std={wb.std():.3f} min={wb.min():.3f} max={wb.max():.3f}")
    print(f"ex: mean={ex.mean():.3e} std={ex.std():.3e} min={ex.min():.3e} max={ex.max():.3e}")
    print(f"taps: mean={taps.mean():.3e} std={taps.std():.3e} min={taps.min():.3e} max={taps.max():.3e}")

    no_path_frac = float(np.mean(wb >= 199.0))
    print("no_path_frac (wb>=199):", no_path_frac)


    # ---- norm stats ----
    stats = compute_norm_stats(x_mm, y_mm, max_samples=512)
    np.savez(
        out_dir / "norm_stats.npz",
        x_mean=stats["x_mean"].numpy(), x_std=stats["x_std"].numpy(),
        y_mean=stats["y_mean"].numpy(), y_std=stats["y_std"].numpy()
    )

    # convenience: keep a device copy for un-normalizing metrics
    stats_dev = {k: v.to(cfg.device) for k, v in stats.items()}

    # ---- split ----
    perm = np.random.default_rng(0).permutation(total_samples)
    n_train = int(cfg.train_frac * total_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    x_train = x_mm[train_idx]
    y_train = y_mm[train_idx]
    x_val = x_mm[val_idx]
    y_val = y_mm[val_idx]

    train_ds = MemmapDataset(x_train, y_train, stats)
    val_ds = MemmapDataset(x_val, y_val, stats)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # ---- model ----
    model = UNet3(in_ch=c_in, out_ch=y_ch, base=64, groups=8, dropout=0.1).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    L = cfg.L_taps

    # ---- loss + NEW logging helpers ----
    def loss_fn(pred, tgt):
        l_wb = torch.mean(torch.abs(pred[:, 0:1] - tgt[:, 0:1]))
        l_ex = torch.mean(torch.abs(pred[:, 1:2] - tgt[:, 1:2]))
        l_t = weighted_tap_loss_phys(pred, tgt, stats_dev, L=cfg.L_taps)
        l_cfr = cfr_nmse_loss_phys(pred, tgt, stats_dev, L=cfg.L_taps, fft_size=cfg.fft_size, max_pixels=2048)
        return 0.2 * l_wb + 0.2 * l_ex + 0.5 * l_t + 1.0* l_cfr

    @torch.no_grad()
    def loss_parts(pred, tgt):
        l_wb = torch.mean(torch.abs(pred[:, 0:1] - tgt[:, 0:1])).item()
        l_ex = torch.mean(torch.abs(pred[:, 1:2] - tgt[:, 1:2])).item()
        l_t = torch.mean((pred[:, 2:2 + 2 * L] - tgt[:, 2:2 + 2 * L]) ** 2).item()
        return l_wb, l_ex, l_t

    @torch.no_grad()
    def unnorm_metrics(pred_norm, tgt_norm):
        # pred_norm/tgt_norm: normalized outputs (B,y_ch,H,W)
        pred = pred_norm * stats_dev["y_std"] + stats_dev["y_mean"]
        tgt = tgt_norm * stats_dev["y_std"] + stats_dev["y_mean"]

        wb_mae_db = torch.mean(torch.abs(pred[:, 0] - tgt[:, 0])).item()
        ex_mae_ns = (torch.mean(torch.abs(pred[:, 1] - tgt[:, 1])) * 1e9).item()  # if seconds
        taps_mse = torch.mean((pred[:, 2:2 + 2 * L] - tgt[:, 2:2 + 2 * L]) ** 2).item()
        return wb_mae_db, ex_mae_ns, taps_mse

    # ---- AMP ----
    amp_enabled = cfg.amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- training loop with NEW prints ----
    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0

        for x, y in train_dl:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                p = model(x)
                loss = loss_fn(p, y)

            

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += float(loss.detach().cpu())

        tr_loss /= max(len(train_dl), 1)

        # ---- validation + component breakdown + unnorm metrics ----
        model.eval()
        va_loss = 0.0

        # grab the first val batch for “detailed” metrics
        first_parts = None
        first_metrics = None

        with torch.no_grad():
            for j, (x, y) in enumerate(val_dl):
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                p = model(x)

                va_loss += float(loss_fn(p, y).detach().cpu())

                if j == 0:
                    first_parts = loss_parts(p, y)
                    first_metrics = unnorm_metrics(p, y)
                    nmse, nmse_db, mag_mae = cfr_metrics_on_batch(
                        pred_norm=p,
                        tgt_norm=y,
                        stats_dev=stats_dev,
                        L_taps=cfg.L_taps,
                        fft_size=cfg.fft_size,
                        max_pixels=2048,
                    )
                    print(f"  CFR: NMSE={nmse:.3e} ({nmse_db:.2f} dB), |H| MAE={mag_mae:.4f}")


        va_loss /= max(len(val_dl), 1)

        if first_parts is None:
            print(f"epoch {epoch+1:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        else:
            l_wb, l_ex, l_t = first_parts
            wb_mae_db, ex_mae_ns, taps_mse = first_metrics
            print(
                f"epoch {epoch+1:03d} | train {tr_loss:.4f} | val {va_loss:.4f} "
                f"| parts(wb={l_wb:.4f}, ex={l_ex:.4f}, taps={l_t:.4f}) "
                f"| unnorm(wb_MAE={wb_mae_db:.2f} dB, ex_MAE={ex_mae_ns:.2f} ns, taps_MSE={taps_mse:.4e})"
            )

    # ---- save ----
    torch.save(model.state_dict(), out_dir / "model_state.pt")
    model.eval()
    example = torch.randn(1, c_in, H, W, device=cfg.device)
    traced = torch.jit.trace(model, example)
    traced.save(str(out_dir / "model.pt"))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
