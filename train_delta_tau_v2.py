#!/usr/bin/env python3
"""
train_delta_tau_v2.py
==================================================================
Retrain wrapper for the path-buffer-truncation fix (2026-08-17).
Imports the ORIGINAL train_delta_tau.py and overrides only:

  1. RT label configuration (the fix itself):
       rx_batch 256 -> 32, max_num_paths_per_src -> 1e7
     Pinned by rt_convergence_sweep.py against a seed-averaged solo
     reference: deep-NLoS median gap +0.28 dB (tier_small) / +0.22 dB
     (tier_large), zero coverage flips.  The old config (B256, 1e6
     buffer) was +10-13 dB dark at depth.  rx_batch was lowered from
     64 to 32 after a GPU OOM stall at paths_buffer.shrink() on the
     8 GB RTX 4070; the label value is batch-size independent once the
     buffer is large enough (that is the whole point of the 1e7 raise).

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

  5. Per-batch Dr.Jit allocator flush (2026-08-22, performance only).
     The 1e7-slot PathsBuffer at max_depth=10 with diffraction reserves
     ~6 GB of the 8 GB card before any ray is traced.  Each
     compute_tdl_batch call then leaves ~6 MB of cached-but-free blocks
     in Dr.Jit's allocator; over the 512 batches of a P=16384 scene that
     creep crosses the remaining headroom (observed at batch ~102 on
     offpatch scene 5), WDDM starts paging device memory through host
     RAM, and every subsequent batch runs ~35x slower (0.13 s -> 4.5 s;
     5181 s instead of ~65 s per scene).  A per-scene flush fires too
     late to help.  Measured on the probe: flushing every batch holds
     VRAM flat (~6.3-6.5 GB) at 0.27 s/batch; unflushed is 0.13 s/batch
     until the cliff.  FLUSH_EVERY amortises the flush cost.  Reusing a
     single PathSolver and flushing the kernel cache were tested and
     neither is necessary -- the malloc cache alone explains the creep.
     Labels are unaffected: the flush releases free blocks only.

PREREQUISITES
  - channel_tdl.py patched: RtCfg has max_num_paths_per_src and
    compute_tdl_batch forwards it.  Make sure the mlink THIS script
    imports (the one on cfr_pred's path) is the patched copy.
  - Run from the cfr_pred repo root, same venv as before:
        python train_delta_tau_v2.py

Generation is per-scene resumable via progress.json; a scene that was
interrupted is regenerated from its first batch.  Scenes completed
during the paging slowdown are valid (paging slows compute, it does
not change it) and are kept.
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import drjit as dr

import train_delta_tau as td
from mlink.channel_tdl import (RtCfg, subcarrier_frequencies_centered,
                               compute_tdl_batch)

# ------------------------------------------------------------------
# 1. + 3. + 4. + 5. -- cfg overrides (mutate the module-level cfg
# instance so every downstream reference picks them up)
# ------------------------------------------------------------------
V2 = dict(
    out_dir="runs/residual_cost_v2",
    rx_batch=32,
    num_inpatch_scenes=200,
    num_offpatch_scenes=90,
    flush_every=8,        # dr.flush_malloc_cache() every N batches (1 = every batch)
    batch_log_every=64,   # progress line every N batches
    slow_batch_factor=5,  # also log any batch slower than this x running median
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
LABEL_SCS_HZ = 937.5e3          # 256 x 937.5 kHz = 240 MHz, same span as 3072 x 78.125 kHz
# Rationale: wb_loss = -10 log10(mean_k |H_k|^2) is a band average;
# fewer CFR evaluation points; mean(|H|^2) expectation unchanged.


def _vram_mib() -> int:
    """Device memory in use per nvidia-smi; -1 if unavailable. Called only
    at log points so its ~20 ms cost is negligible."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def compute_rt_labels_v2(scene):
    """Verbatim port of train_delta_tau.compute_rt_labels with three
    changes: the coarse label-only frequency grid, per-batch wall time
    and allocator flush (see docstring item 5), and per-scene wall time
    printed for run-length extrapolation."""
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

    flush_every = int(V2["flush_every"])
    log_every = int(V2["batch_log_every"])
    slow_factor = float(V2["slow_batch_factor"])
    n_batches = (P + td.cfg.rx_batch - 1) // td.cfg.rx_batch

    t_scene = time.time()
    for t, tx in enumerate(tx_coords):
        wb_all = np.full((P,), td.cfg.no_path_wb_db, np.float32)
        tau_all = np.zeros((P,), np.float32)
        batch_times = []
        for b, i0 in enumerate(range(0, P, td.cfg.rx_batch)):
            i1 = min(i0 + td.cfg.rx_batch, P)
            t0 = time.time()
            wb_db, ex_s, taps, tau_rms_s = compute_tdl_batch(
                si_scene=si, tx_xyz=tx, rx_xyz=rx_coords[i0:i1],
                frequencies_hz=freqs, L_taps=N, rt=td.cfg.rt,
                return_tau_rms=True)
            dt = time.time() - t0
            batch_times.append(dt)
            wb_all[i0:i1] = wb_db
            good = wb_db < td.cfg.no_path_wb_db
            if np.any(good):
                tau_all[i0 + np.nonzero(good)[0]] = tau_rms_s[good] * 1e9

            # v2 item 5: release cached-but-free device blocks before the
            # creep reaches the VRAM ceiling
            if flush_every and (b + 1) % flush_every == 0:
                dr.sync_thread()
                dr.flush_malloc_cache()

            median = float(np.median(batch_times)) if len(batch_times) > 8 else None
            is_slow = median is not None and dt > slow_factor * median
            if b % log_every == 0 or is_slow:
                tag = "  SLOW" if is_slow else ""
                print(f"      batch {b:4d}/{n_batches} rx {i0:6d} "
                      f"{dt:6.2f} s  vram {_vram_mib():5d} MiB{tag}",
                      flush=True)

        wb_map = wb_all.reshape(K, H, W)
        tau_map = np.maximum(
            td.smooth_map_stack(tau_all.reshape(K, H, W), wb_map), 0.0)
        tau_map[wb_map >= td.cfg.no_path_wb_db] = 0.0
        wb_out[t] = wb_map
        tau_out[t] = tau_map
        bt = np.asarray(batch_times)
        print(f"    tx {t+1}/{tx_coords.shape[0]} labels done "
              f"(batch median {np.median(bt):.2f} s, max {bt.max():.1f} s)",
              flush=True)
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
        "allocator": {"flush_malloc_cache_every_batches": V2["flush_every"],
                      "note": "2026-08-22: performance only; prevents the "
                              "Dr.Jit malloc-cache creep (~6 MB/batch) from "
                              "reaching the 8 GB VRAM ceiling mid-scene and "
                              "triggering WDDM paging (~35x slowdown). "
                              "Labels unaffected."},
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

    print(f"[v2] config: rx_batch={td.cfg.rx_batch}, buffer=1e7, "
          f"flush_every={V2['flush_every']}, label grid "
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