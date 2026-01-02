# mlink/channel_tdl.py
from __future__ import annotations

import math
import numpy as np
import mitsuba as mi
from dataclasses import dataclass
from typing import Tuple, Optional, Union

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

C0 = 299_792_458.0 # speed of light in m/s


def _as_complex(x) -> np.ndarray:
    """
    Sionna RT with out_type='numpy' returns (re, im) tuples for CIR/CFR coeffs. :contentReference[oaicite:1]{index=1}
    Convert to complex ndarray.
    """
    if isinstance(x, (tuple, list)) and len(x) == 2:
        re, im = x
        return np.asarray(re, dtype=np.float32) + 1j * np.asarray(im, dtype=np.float32)
    # Some versions may already return complex
    return np.asarray(x)


def _squeeze_to_BxP(x: np.ndarray, B: int) -> np.ndarray:
    """
    Squeeze singleton dims and reshape to (B, P).
    Works for both CIR (… , num_paths, …) and tau (… , num_paths).
    """
    x = np.asarray(x)
    x = np.squeeze(x)

    if x.size == 0:
        return x.reshape(B, 0)

    if x.ndim == 1:
        # could be (P,) for B==1 or (B,) for P==1; disambiguate:
        if B == 1:
            return x.reshape(1, -1)
        else:
            return x.reshape(B, 1)

    # Ensure first dim is B; if not, just reshape
    if x.shape[0] != B:
        return x.reshape(B, -1)

    return x.reshape(B, -1)


def _delay_stats_from_cir(
    a_cplx: np.ndarray,
    tau_s: np.ndarray,
    B: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (tau_min_s, tau_rms_s) per RX using *path delays* and *path powers*.

    tau_rms is RMS delay spread (sqrt(E[t^2]-E[t]^2)) on *excess delays relative to tau_min*.
    """
    a = _squeeze_to_BxP(a_cplx, B)
    tau = _squeeze_to_BxP(tau_s, B).astype(np.float64)

    p = (np.abs(a) ** 2).astype(np.float64)

    tau_min = np.full((B,), np.nan, dtype=np.float64)
    tau_rms = np.full((B,), np.nan, dtype=np.float64)

    for b in range(B):
        tb = tau[b]
        pb = p[b]

        m = np.isfinite(tb) & (tb >= 0.0) & np.isfinite(pb) & (pb > 0.0)
        if not np.any(m):
            continue

        t = tb[m]
        w = pb[m]

        t0 = float(np.min(t))
        tr = t - t0  # excess-delay axis (stable numerically)

        wsum = float(np.sum(w))
        mu = float(np.sum(w * tr) / wsum)
        mu2 = float(np.sum(w * (tr ** 2)) / wsum)
        var = max(mu2 - mu * mu, 0.0)

        tau_min[b] = t0
        tau_rms[b] = np.sqrt(var)

    return tau_min.astype(np.float32), tau_rms.astype(np.float32)


def _cfr_to_taps(
    h_f: np.ndarray,
    L_taps: int,
    assume_fftshifted: bool = True,
) -> np.ndarray:
    """
    Convert CFR samples -> taps on fixed FFT grid using IFFT.

    If your frequencies are "centered" (negative..positive), your CFR is in FFTSHIFT order,
    so we do ifftshift before ifft.
    """
    h = np.asarray(h_f)
    h = np.squeeze(h)
    if h.ndim == 1:
        h = h[None, :]

    if assume_fftshifted:
        h = np.fft.ifftshift(h, axes=1)

    taps = np.fft.ifft(h, axis=1)
    taps = taps[:, : int(L_taps)]
    return taps.astype(np.complex64)


def _clear_radio_nodes(si_scene) -> None:
    """
    Remove all transmitters/receivers from the Sionna scene so repeated calls don't accumulate nodes.
    """
    for attr in ("transmitters", "receivers"):
        d = getattr(si_scene, attr, None)
        if d is None:
            continue
        try:
            names = list(d.keys())  # common: dict
        except Exception:
            names = [n.name for n in d]  # fallback: list-like
        for name in names:
            try:
                si_scene.remove(name)
            except Exception:
                pass


# ----------------------------
# Drop-in compute_tdl_batch
# ----------------------------

def compute_tdl_batch(
    si_scene,
    tx_xyz: np.ndarray,                 # (3,)
    rx_xyz: np.ndarray,                 # (B,3)
    frequencies_hz: np.ndarray,         # (N,) (typically centered order)
    L_taps: int,
    rt,
    cir_sampling_frequency_hz: float = 1e9,
    assume_centered_frequencies: bool = True,
    return_tau_rms: bool = False,
    return_debug: bool = False,
):
    """
    Returns by default (drop-in):
      wb_loss_db      : (B,) float32
      excess_delay_s  : (B,) float32
      taps_norm       : (B, L_taps) complex64  (unit-power normalized CFR -> IFFT)

    Optional:
      tau_rms_s       : (B,) float32  (computed from CIR path delays & powers)
      debug dict      : contains tau_min_s, tau_rms_ifft_s, etc.
    """
    tx_xyz = np.asarray(tx_xyz, dtype=np.float32).reshape(3)
    rx_xyz = np.asarray(rx_xyz, dtype=np.float32).reshape(-1, 3)
    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    B = rx_xyz.shape[0]

    _clear_radio_nodes(si_scene)

    # Place one TX + B RX
    si_scene.add(Transmitter(name="tx", position=mi.Point3f(tx_xyz)))
    for i in range(B):
        si_scene.add(Receiver(name=f"rx{i:05d}", position=mi.Point3f(rx_xyz[i])))

    # Solve paths
    p_solver = PathSolver()
    paths = p_solver(
        scene=si_scene,
        max_depth=rt.max_depth,
        samples_per_src=rt.samples_per_src,
        los=getattr(rt, "los", True),
        specular_reflection=getattr(rt, "specular_reflection", True),
        diffuse_reflection=getattr(rt, "diffuse_reflection", False),
        refraction=getattr(rt, "refraction", False),
        synthetic_array=getattr(rt, "synthetic_array", True),
        diffraction=getattr(rt, "diffraction", False),
        edge_diffraction=getattr(rt, "edge_diffraction", False),
        diffraction_lit_region=getattr(rt, "diffraction_lit_region", False),
    )

    # CIR for absolute delays (normalize_delays=False) :contentReference[oaicite:2]{index=2}
    a_ri, tau_s = paths.cir(
        sampling_frequency=float(cir_sampling_frequency_hz),
        num_time_steps=1,
        normalize_delays=False,
        out_type="numpy",
    )
    a_cplx = _as_complex(a_ri)

    tau_min_s, tau_rms_s = _delay_stats_from_cir(a_cplx, tau_s, B)

    # CFR for wideband power loss (normalize_delays=True, normalize=False) :contentReference[oaicite:3]{index=3}
    h_ri = paths.cfr(
        frequencies=freqs,
        sampling_frequency=1.0,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy",
    )
    h_raw = _as_complex(h_ri)
    h_raw = np.squeeze(h_raw)
    if h_raw.ndim == 1:
        h_raw = h_raw[None, :]  # (B,N) for 1x1 cases

    # wb_loss from mean(|H|^2)
    power = np.mean(np.abs(h_raw) ** 2, axis=1).astype(np.float32)  # (B,)
    power = np.maximum(power, 1e-12)
    wb_loss_db = (-10.0 * np.log10(power)).astype(np.float32)

    # Unit-power normalize CFR then IFFT -> taps (these taps are *not* a physical CIR)
    h_norm = h_raw / np.sqrt(power)[:, None]
    taps_norm = _cfr_to_taps(h_norm, L_taps=L_taps, assume_fftshifted=assume_centered_frequencies)

    # Excess delay relative to free-space d/c
    d = np.linalg.norm(rx_xyz - tx_xyz[None, :], axis=1).astype(np.float32)
    tau_fs = (d / C0).astype(np.float32)
    excess_delay_s = (tau_min_s - tau_fs).astype(np.float32)

    # Handle no-path: tau_min_s is nan
    bad = ~np.isfinite(excess_delay_s)
    if np.any(bad):
        wb_loss_db[bad] = 200.0
        excess_delay_s[bad] = 0.0
        taps_norm[bad, :] = 0.0
        tau_rms_s[bad] = 0.0

    if return_debug:
        # (optional) show why IFFT-based tau_rms can blow up
        # Compute tau_rms from taps_norm just for comparison (seconds)
        p = (np.abs(taps_norm) ** 2).astype(np.float64)
        ps = np.sum(p, axis=1, keepdims=True)
        p = p / np.maximum(ps, 1e-12)
        # NOTE: time resolution should be Ts = 1/(N*df). We don't infer df here; caller can.
        debug = dict(tau_min_s=tau_min_s, tau_rms_paths_s=tau_rms_s, wb_power_lin=power, bad_mask=bad)
    else:
        debug = None

    if return_tau_rms and return_debug:
        return wb_loss_db, excess_delay_s, taps_norm, tau_rms_s, debug
    if return_tau_rms:
        return wb_loss_db, excess_delay_s, taps_norm, tau_rms_s
    if return_debug:
        return wb_loss_db, excess_delay_s, taps_norm, debug
    return wb_loss_db, excess_delay_s, taps_norm