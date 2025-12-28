# mlink/channel_tdl.py
from __future__ import annotations

import math
import numpy as np
import mitsuba as mi
from dataclasses import dataclass
from typing import Tuple

import sionna.rt
from sionna.rt import Transmitter, Receiver, PathSolver

from mlink.constants import FREE_SPACE_CONSTS 


def subcarrier_frequencies_centered(fft_size: int, subcarrier_spacing_hz: float) -> np.ndarray:
    """
    Returns centered baseband subcarrier frequencies in Hz, ordered from negative to positive:
      f[k] = (k - N/2)*Δf,  k=0..N-1  (for even N)
    This ordering matches an FFTSHIFTed CFR vector.
    """
    N = int(fft_size)
    df = float(subcarrier_spacing_hz)
    k = np.arange(N, dtype=np.float32)
    return (k - (N // 2)) * df


def _clear_radio_nodes(scene: sionna.rt.Scene):
    # remove previously placed transmitters/receivers (if any)
    # scene.transmitters/receivers are dict-like in many Sionna versions
    for name in list(scene.transmitters.keys()):
        scene.remove(name)
    for name in list(scene.receivers.keys()):
        scene.remove(name)


@dataclass
class RtCfg:
    max_depth: int = 5
    samples_per_src: int = 10**6
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = True
    synthetic_array: bool = False
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = False


def compute_tdl_batch(
    si_scene: sionna.rt.Scene,
    tx_xyz: np.ndarray,                 # (3,)
    rx_xyz: np.ndarray,                 # (B,3)
    frequencies_hz: np.ndarray,         # (N,)
    L_taps: int,
    rt: RtCfg,
    cir_sampling_frequency_hz: float = 1e9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      wb_loss_db : (B,) float32
      excess_delay_s : (B,) float32
      taps_norm : (B, L_taps) complex64   (unit-power normalized)
    """
    tx_xyz = np.asarray(tx_xyz, dtype=np.float32).reshape(3)
    rx_xyz = np.asarray(rx_xyz, dtype=np.float32).reshape(-1, 3)
    B = rx_xyz.shape[0]
    N = frequencies_hz.shape[0]
    L = int(L_taps)

    _clear_radio_nodes(si_scene)

    # Place one TX
    si_scene.add(Transmitter(name="tx", position=mi.Point3f(tx_xyz)))

    # Place receivers
    for i in range(B):
        si_scene.add(Receiver(name=f"rx{i:05d}", position=mi.Point3f(rx_xyz[i])))

    # Run path solver
    p_solver = PathSolver()
    paths = p_solver(
        scene=si_scene,
        max_depth=rt.max_depth,
        samples_per_src=rt.samples_per_src,
        los=rt.los,
        specular_reflection=rt.specular_reflection,
        diffuse_reflection=rt.diffuse_reflection,
        refraction=rt.refraction,
        synthetic_array=rt.synthetic_array,
        diffraction=rt.diffraction,
        edge_diffraction=rt.edge_diffraction,
        diffraction_lit_region=rt.diffraction_lit_region,
    )

    # --- First arrival delay from CIR (absolute, NOT normalized) ---
    # a,tau shapes: [num_rx, rx_ant, num_tx, tx_ant, num_paths, num_time_steps]
    a, tau = paths.cir(
        sampling_frequency=cir_sampling_frequency_hz,
        normalize_delays=False,
        out_type="numpy",
    )
    tau = np.squeeze(tau)  # typically (B, P)
    if tau.ndim == 1:      # if P==1 edge case
        tau = tau[:, None]

    tau_min = np.full((B,), np.nan, dtype=np.float32)
    for i in range(B):
        t = tau[i]
        t = t[np.isfinite(t) & (t >= 0)]
        tau_min[i] = float(t.min()) if t.size > 0 else np.nan

    # --- CFR (delay-normalized, NOT power-normalized) ---
    # h_raw shape typically: [B, rx_ant, tx, tx_ant, ofdm_symbols, N]
    h_raw = paths.cfr(
        frequencies=frequencies_hz,
        sampling_frequency=1.0,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy",
    )
    h_raw = np.squeeze(h_raw)  # -> (B, N) in common 1x1 cases
    if h_raw.ndim == 1:
        h_raw = h_raw[None, :]

    # wb_loss from mean(|H|^2)
    power = np.mean(np.abs(h_raw) ** 2, axis=1).astype(np.float32)  # (B,)
    power = np.maximum(power, 1e-12)
    wb_loss_db = (-10.0 * np.log10(power)).astype(np.float32)

    # Normalize CFR to unit power (matches your server’s semantics) :contentReference[oaicite:4]{index=4}
    h_norm = h_raw / np.sqrt(power)[:, None]

    # Convert CFR -> taps on fixed grid
    # Our frequencies are "centered", so CFR is in FFTSHIFT order; undo that before IFFT.
    h_unshift = np.fft.ifftshift(h_norm, axes=1)
    taps = np.fft.ifft(h_unshift, axis=1)  # (B,N) complex
    taps = taps[:, :L].astype(np.complex64)

    # Compute excess delay relative to free-space d/c
    d = np.linalg.norm(rx_xyz - tx_xyz[None, :], axis=1).astype(np.float32)
    tau_fs = (d / float(FREE_SPACE_CONSTS.c)).astype(np.float32)
    excess = (tau_min - tau_fs).astype(np.float32)

    # Handle "no path" cases: set something safe
    bad = ~np.isfinite(excess)
    if np.any(bad):
        excess[bad] = 0.0
        wb_loss_db[bad] = 200.0
        taps[bad, :] = 0.0

    return wb_loss_db, excess, taps
