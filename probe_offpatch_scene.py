#!/usr/bin/env python
"""probe_offpatch_scene.py -- why does one offpatch scene cost 80x its neighbours?

Rebuilds offpatch scenes deterministically (same seeding as
build_offpatch_store), prints geometry stats, then sweeps the RT config on a
small RX batch with a reduced sample budget so every solve is seconds, not
hours.  Compares a "bad" scene against a "good" baseline.

Usage (from ~/cfr_pred, inside sionna-venv):
    python probe_offpatch_scene.py                # bad=4, good=3 (0-based)
    python probe_offpatch_scene.py --bad 4 --good 3 --rx 8 --samples 100000

Nothing here writes to the store; progress.json is untouched.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import replace

import numpy as np
import drjit as dr
import mitsuba as mi
from sionna.rt import Transmitter, Receiver, PathSolver

import train_delta_tau as td
from mlink.channel_tdl import RtCfg, _clear_radio_nodes

# mirror the v2 overrides that affect scene generation / RT
td.cfg.num_offpatch_scenes = 90
BASE_RT = RtCfg(
    max_depth=10,
    samples_per_src=1_000_000,
    max_num_paths_per_src=10_000_000,
    diffuse_reflection=True,
    diffraction=True,
    edge_diffraction=True,
    diffraction_lit_region=True,
)


# ------------------------------------------------------------------
# scene reconstruction (verbatim logic from build_offpatch_store)
# ------------------------------------------------------------------
def rebuild_offpatch(s: int, patch: int = 0):
    rng = td.scene_rng("offpatch", s)
    cands = None
    for _scene_try in range(16):
        base, walls_2d, fm = td.make_full_scene(rng)
        Hf, Wf = fm["H_full"], fm["W_full"]
        for _tx_try in range(64):
            tx_xyz = td.sample_free_tx_xyz(walls_2d, fm["ceiling_h_m"], rng)
            tx_i = int(np.floor(tx_xyz[0] / td.cfg.scale))
            tx_j = int(np.floor(tx_xyz[1] / td.cfg.scale))
            c = td.enumerate_offpatch_crops(tx_i, tx_j, Hf, Wf)
            if c.shape[0] > 0:
                cands = c
                break
        if cands is not None:
            break
    if cands is None:
        raise RuntimeError(f"scene {s}: no feasible (scene, TX, crop)")
    # draw crops in the same order as build_offpatch_store so patch index
    # p reproduces the p-th sample of this scene exactly
    for _p in range(patch + 1):
        i0, j0 = (int(v) for v in cands[int(rng.integers(0, cands.shape[0]))])
    scene = td.make_patch_scene_from_full(base, tx_xyz, fm, i0, j0)
    return scene, walls_2d, fm, tx_xyz, (i0, j0)


def geometry_report(tag, scene, walls_2d, fm, tx_xyz, crop):
    w = np.asarray(walls_2d)
    # count distinct wall cells and horizontal/vertical transitions as a
    # cheap proxy for wedge-edge count
    trans = int(np.sum(w[:, 1:] != w[:, :-1]) + np.sum(w[1:, :] != w[:-1, :]))
    mesh = scene.mesh
    nf = int(mesh.faces.shape[0])
    nv = int(mesh.vertices.shape[0])
    print(f"[{tag}] full {fm['H_full']}x{fm['W_full']} cells, "
          f"ceiling {fm['ceiling_h_m']:.2f} m, z0 {fm['z0_m']:.2f} m, "
          f"z_step {fm['z_step_m']:.2f} m")
    print(f"[{tag}] wall cells {int(w.sum())}/{w.size} "
          f"({100*w.mean():.1f}%), cell transitions {trans}")
    print(f"[{tag}] mesh: {nv} verts, {nf} faces")
    print(f"[{tag}] tx {tx_xyz}  crop origin {crop}  "
          f"P={scene.antenna_database.rx_coords.shape[0]}")
    # nearest wall/ceiling distances for the TX
    tz_to_ceil = fm["ceiling_h_m"] - float(tx_xyz[2])
    tz_to_floor = float(tx_xyz[2]) - td.cfg.scale * td.cfg.floor_h
    print(f"[{tag}] tx clearance: ceiling {tz_to_ceil:.2f} m, "
          f"floor {tz_to_floor:.2f} m")


def vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


# ------------------------------------------------------------------
# one timed solve, returning path count
# ------------------------------------------------------------------
def timed_solve(si, tx_xyz, rx_xyz, rt, solver):
    _clear_radio_nodes(si)
    si.add(Transmitter(name="tx", position=mi.Point3f(tx_xyz)))
    for i in range(rx_xyz.shape[0]):
        si.add(Receiver(name=f"rx{i:05d}", position=mi.Point3f(rx_xyz[i])))
    dr.sync_thread()
    t0 = time.time()
    paths = solver(
        scene=si,
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
        max_num_paths_per_src=getattr(rt, "max_num_paths_per_src", 1_000_000),
    )
    dr.sync_thread()
    dt = time.time() - t0
    # path count: last axis of the amplitude tensor in Sionna 1.x
    try:
        a = paths.a
        a = a[0] if isinstance(a, (tuple, list)) else a
        n_paths = int(a.shape[-1])
    except Exception:
        n_paths = -1
    del paths
    dr.flush_malloc_cache()
    return dt, n_paths


def sweep(tag, scene, rx_n, samples, solver, only=None):
    si = td._to_sionna_geometry(scene, td.cfg.frequency_hz)
    tx = np.asarray(scene.antenna_database.tx_coords[0], np.float32)
    rx = np.asarray(scene.antenna_database.rx_coords[:rx_n], np.float32)
    configs = [
        ("los+spec d3",      dict(max_depth=3,  diffuse_reflection=False, diffraction=False, edge_diffraction=False, diffraction_lit_region=False)),
        ("los+spec d10",     dict(max_depth=10, diffuse_reflection=False, diffraction=False, edge_diffraction=False, diffraction_lit_region=False)),
        ("+diffuse d10",     dict(max_depth=10, diffraction=False, edge_diffraction=False, diffraction_lit_region=False)),
        ("+diffr d10",       dict(max_depth=10, edge_diffraction=False, diffraction_lit_region=False)),
        ("+edge d10",        dict(max_depth=10, diffraction_lit_region=False)),
        ("full (v2) d10",    dict(max_depth=10)),
        ("full (v2) d5",     dict(max_depth=5)),
    ]
    print(f"\n=== {tag}: B={rx_n} rx, samples_per_src={samples:.0e} ===")
    print(f"{'config':<16} {'time s':>8} {'paths':>10} {'VRAM MiB':>9}")
    for name, kw in configs:
        if only and not any(o in name for o in only):
            continue
        rt = replace(BASE_RT, samples_per_src=samples, **kw)
        dt, npth = timed_solve(si, tx, rx, rt, solver)
        print(f"{name:<16} {dt:8.1f} {npth:10d} {vram_mib():9d}", flush=True)
    _clear_radio_nodes(si)
    del si
    dr.flush_malloc_cache()


def production_loop(tag, scene, rx_batch, reuse_solver, every,
                    flush_every=0, kflush_every=0, max_batches=0):
    """Replay compute_rt_labels_v2's inner loop verbatim (fresh PathSolver per
    batch unless --reuse-solver), with per-batch wall time and VRAM so a
    spike vs. a creep is distinguishable."""
    import mlink.channel_tdl as ct
    from mlink.channel_tdl import subcarrier_frequencies_centered, compute_tdl_batch
    if reuse_solver:
        _single = PathSolver()
        ct.PathSolver = lambda: _single          # monkeypatch the name it looks up
    td.cfg.rt = BASE_RT
    td.cfg.rx_batch = rx_batch
    rx_coords = scene.antenna_database.rx_coords
    tx = scene.antenna_database.tx_coords[0]
    P = rx_coords.shape[0]
    freqs = subcarrier_frequencies_centered(256, 937.5e3)
    si = td._to_sionna_geometry(scene, td.cfg.frequency_hz)
    nb = (P + rx_batch - 1) // rx_batch
    print(f"\n=== production replay: {tag}, P={P}, rx_batch={rx_batch}, "
          f"{nb} batches, reuse_solver={reuse_solver}, "
          f"malloc_flush_every={flush_every}, kernel_flush_every={kflush_every} ===")
    print(f"{'batch':>6} {'rx0':>6} {'time s':>8} {'VRAM MiB':>9} {'valid':>6}")
    t_all = time.time()
    times = []
    for b, i0 in enumerate(range(0, P, rx_batch)):
        i1 = min(i0 + rx_batch, P)
        print(f"{b:6d} {i0:6d} {'...':>8}", end="\r", flush=True)
        t0 = time.time()
        wb_db, ex_s, taps, tau_rms_s = compute_tdl_batch(
            si_scene=si, tx_xyz=tx, rx_xyz=rx_coords[i0:i1],
            frequencies_hz=freqs, L_taps=256, rt=td.cfg.rt,
            return_tau_rms=True)
        dt = time.time() - t0
        times.append(dt)
        nvalid = int(np.sum(wb_db < td.cfg.no_path_wb_db))
        if flush_every and (b + 1) % flush_every == 0:
            dr.sync_thread(); dr.flush_malloc_cache()
        if kflush_every and (b + 1) % kflush_every == 0:
            dr.sync_thread(); dr.flush_kernel_cache()
        if b % every == 0 or dt > 5.0 * (np.median(times) if len(times) > 5 else 1.0):
            print(f"{b:6d} {i0:6d} {dt:8.2f} {vram_mib():9d} {nvalid:6d}", flush=True)
        if max_batches and b + 1 >= max_batches:
            break
    times = np.asarray(times)
    print(f"total {time.time()-t_all:.0f} s | batch median {np.median(times):.2f} s "
          f"max {times.max():.1f} s at batch {int(times.argmax())} | "
          f"sum of top-5 {np.sort(times)[-5:].sum():.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bad", type=int, default=4, help="0-based offpatch index")
    ap.add_argument("--good", type=int, default=3, help="0-based baseline index")
    ap.add_argument("--rx", type=int, default=8)
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter on config names, e.g. --only 'full (v2) d10'")
    ap.add_argument("--order", default="good,bad")
    ap.add_argument("--patch", type=int, default=0, help="which of the 6 crops")
    ap.add_argument("--production", action="store_true",
                    help="replay the full per-scene label loop on --bad with per-batch timing")
    ap.add_argument("--reuse-solver", action="store_true")
    ap.add_argument("--every", type=int, default=32, help="print every N batches")
    ap.add_argument("--flush-every", type=int, default=0, help="dr.flush_malloc_cache every N batches")
    ap.add_argument("--kflush-every", type=int, default=0, help="dr.flush_kernel_cache every N batches")
    ap.add_argument("--max-batches", type=int, default=0, help="stop after N batches (0=all)")
    args = ap.parse_args()

    scenes = {}
    for tag, s in (("good", args.good), ("bad", args.bad)):
        scene, walls, fm, tx, crop = rebuild_offpatch(s, args.patch)
        scenes[tag] = scene
        print(f"\n--- offpatch s={s} ({tag}) ---")
        geometry_report(tag, scene, walls, fm, tx, crop)
    if args.geometry_only:
        return
    if args.production:
        production_loop("bad", scenes["bad"], args.rx, args.reuse_solver, args.every,
                        args.flush_every, args.kflush_every, args.max_batches)
        return

    solver = PathSolver()   # single instance, reused
    print(f"\nVRAM at start: {vram_mib()} MiB")
    for tag in args.order.split(","):
        sweep(tag, scenes[tag], args.rx, args.samples, solver, args.only)


if __name__ == "__main__":
    main()