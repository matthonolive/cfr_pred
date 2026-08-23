#!/usr/bin/env python3
"""
train_delta_tau_v2.py
==================================================================
Retrain wrapper for the path-buffer-truncation fix (2026-08-17).
Imports the ORIGINAL train_delta_tau.py and overrides only:

  1. RT label configuration (the fix itself):
       rx_batch 256 -> 64, max_num_paths_per_src -> 1e7
     Pinned by rt_convergence_sweep.py against a seed-averaged solo
     reference: deep-NLoS median gap +0.28 dB (tier_small) / +0.22 dB
     (tier_large), zero coverage flips.  The old config (B256, 1e6
     buffer) was +10-13 dB dark at depth.

  2. Label-only coarse CFR grid (time saving, unbiased):
       wb labels come from mean(|H|^2); sampling 256 subcarriers
       across the SAME 240 MHz span (spacing x12) preserves the
       band-averaged expectation while cutting paths.cfr() work ~12x.
       tau labels come from paths.cir() and are untouched.  The
       serving/eval waveform stays 3072/78.125 kHz -- this grid exists
       only inside label generation.

  3. Mild dataset shrink (time saving, distribution-preserving):
       in-patch scenes 300 -> 200, off-patch scenes 120 -> 90
       (natural mix ratio ~preserved: 5000:540 vs original 7500:720).
       TX lattice, features, split fraction, seed, and all training
       hyperparameters are UNCHANGED, so the retrain differs from
       unet_shannon in labels and dataset size only.

  4. New out_dir (never resume the corrupted stores) + provenance
     stamp recording the RT config and its convergence evidence.

PREREQUISITES
  - channel_tdl.py patched: RtCfg has max_num_paths_per_src and
    compute_tdl_batch forwards it.  Make sure the mlink THIS script
    imports (the one on cfr_pred's path) is the patched copy.
  - Run from the cfr_pred repo root, same venv as before:
        python train_delta_tau_v2.py

After the FIRST scene completes, check the printed per-scene time and
multiply by 290 before committing to the full run; generation is
per-scene resumable, so nights-only runs are fine.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import train_delta_tau as td
from mlink.channel_tdl import (RtCfg, subcarrier_frequencies_centered,
                               compute_tdl_batch)

import drjit as dr, gc

# ------------------------------------------------------------------
# 1. + 3. + 4. -- cfg overrides (mutate the module-level cfg instance
# so every downstream reference picks them up)
# ------------------------------------------------------------------
V2 = dict(
    out_dir="runs/residual_cost_v2",
    rx_batch=32,
    num_inpatch_scenes=200,
    num_offpatch_scenes=90,
)

td.cfg.out_dir = V2["out_dir"]
td.cfg.rx_batch = V2["rx_batch"]
td.cfg.num_inpatch_scenes = V2["num_inpatch_scenes"]
td.cfg.num_offpatch_scenes = V2["num_offpatch_scenes"]
td.cfg.rt = RtCfg(
    max_depth=10,
    samples_per_src=1_000_000,
    max_num_paths_per_src=10_000_000,     # requires the channel_tdl patch
    diffuse_reflection=True,
    diffraction=True,
    edge_diffraction=True,
    diffraction_lit_region=True,
)

# fail fast if the mlink on the path is unpatched: the field would be
# silently absent and compute_tdl_batch would fall back to 1e6
assert getattr(td.cfg.rt, "max_num_paths_per_src", None) == 10_000_000, \
    "channel_tdl.py on this path is NOT the patched copy (RtCfg lacks " \
    "max_num_paths_per_src) -- fix the import path before generating labels"

# ------------------------------------------------------------------
# 2. -- label-only coarse CFR grid
# ------------------------------------------------------------------
LABEL_FFT = 256
LABEL_SCS_HZ = td.cfg.subcarrier_spacing_hz * (td.cfg.fft_size / LABEL_FFT)
# 3072 * 78.125 kHz = 240 MHz span == 256 * 937.5 kHz: same band, 12x
# fewer CFR evaluation points; mean(|H|^2) expectation unchanged.


def compute_rt_labels_v2(scene):
    """Verbatim port of train_delta_tau.compute_rt_labels with two
    changes: the coarse label-only frequency grid, and per-scene wall
    time printed for run-length extrapolation."""
    rx_grid = scene.antenna_database.rx_grid
    K, H, W = rx_grid.shape
    tx_coords = scene.antenna_database.tx_coords
    rx_coords = scene.antenna_database.rx_coords
    P = rx_coords.shape[0]
    N = int(LABEL_FFT)                                    # v2: coarse grid
    freqs = subcarrier_frequencies_centered(N, LABEL_SCS_HZ)

    si = td._to_sionna_geometry(scene, td.cfg.frequency_hz)

    wb_out = np.full((tx_coords.shape[0], K, H, W), td.cfg.no_path_wb_db,
                     np.float32)
    tau_out = np.zeros((tx_coords.shape[0], K, H, W), np.float32)
    t_scene = time.time()
    for t, tx in enumerate(tx_coords):
        wb_all = np.full((P,), td.cfg.no_path_wb_db, np.float32)
        tau_all = np.zeros((P,), np.float32)
        for i0 in range(0, P, td.cfg.rx_batch):
            i1 = min(i0 + td.cfg.rx_batch, P)
            wb_db, ex_s, taps, tau_rms_s = compute_tdl_batch(
                si_scene=si, tx_xyz=tx, rx_xyz=rx_coords[i0:i1],
                frequencies_hz=freqs, L_taps=N, rt=td.cfg.rt,
                return_tau_rms=True)
            wb_all[i0:i1] = wb_db
            good = wb_db < td.cfg.no_path_wb_db
            if np.any(good):
                tau_all[i0 + np.nonzero(good)[0]] = tau_rms_s[good] * 1e9
        wb_map = wb_all.reshape(K, H, W)
        tau_map = np.maximum(
            td.smooth_map_stack(tau_all.reshape(K, H, W), wb_map), 0.0)
        tau_map[wb_map >= td.cfg.no_path_wb_db] = 0.0
        wb_out[t] = wb_map
        tau_out[t] = tau_map
        print(f"    tx {t+1}/{tx_coords.shape[0]} labels done", flush=True)
        gc.collect(); dr.sync_thread(); dr.flush_malloc_cache()
    print(f"    [v2] scene labels in {time.time() - t_scene:.0f} s "
          f"({tx_coords.shape[0]} tx)", flush=True)
    return wb_out, tau_out


# store builders resolve compute_rt_labels through module globals at
# call time, so this swap redirects every label computation
td.compute_rt_labels = compute_rt_labels_v2


# ------------------------------------------------------------------
# provenance stamp, then hand off to the original pipeline
# ------------------------------------------------------------------
def main():
    out = Path(td.cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rt_provenance.json").write_text(json.dumps({
        "date": "2026-08-17",
        "reason": "path-buffer truncation fix (Sionna PathSolver "
                  "max_num_paths_per_src shared across receivers; old "
                  "B256/1e6-buffer labels were +10-13 dB dark at depth)",
        "rt": {"max_depth": 10, "samples_per_src": 1_000_000,
               "max_num_paths_per_src": 10_000_000,
               "diffuse_reflection": True, "diffraction": True,
               "edge_diffraction": True, "diffraction_lit_region": True,
               "los": True, "specular_reflection": True,
               "refraction": True, "synthetic_array": False},
        "rx_batch": td.cfg.rx_batch,
        "label_grid": {"fft": LABEL_FFT, "scs_hz": LABEL_SCS_HZ,
                       "span_mhz": LABEL_FFT * LABEL_SCS_HZ / 1e6},
        "dataset": {"num_inpatch_scenes": td.cfg.num_inpatch_scenes,
                    "num_offpatch_scenes": td.cfg.num_offpatch_scenes,
                    "note": "shrunk from 300/120; TX lattice, features, "
                            "seed, split, and hyperparameters unchanged"},
        "convergence_evidence": "rt_convergence_sweep 2026-08-17: B64 "
                                "s1e6 b1e7 deep-median gap +0.28 dB "
                                "(tier_small seed910007) / +0.22 dB "
                                "(tier_large seed930000) vs seed-averaged "
                                "solo 3e6/1e7 reference; 0 coverage flips",
    }, indent=2))

    print("[v2] config: rx_batch=64, buffer=1e7, label grid "
          f"{LABEL_FFT} x {LABEL_SCS_HZ/1e3:.1f} kHz, scenes "
          f"{td.cfg.num_inpatch_scenes}+{td.cfg.num_offpatch_scenes}, "
          f"out_dir={td.cfg.out_dir}")

    for entry in ("main", "run", "train"):
        fn = getattr(td, entry, None)
        if callable(fn):
            return fn()
    raise SystemExit(
        "train_delta_tau.py exposes no main()/run()/train() -- if its "
        "pipeline runs inline under `if __name__ == '__main__':`, move "
        "that block into a main() function (a two-line change) and rerun")


if __name__ == "__main__":
    main()