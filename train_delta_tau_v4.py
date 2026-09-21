#!/usr/bin/env python3
"""
train_delta_tau_v4.py
==================================================================
Heavy-wall retrain (2026-09-21): regenerate BOTH stores with an ITU-R
P.2040 material and train with the v3 objective.

Why: the light-wall suite (eps_r=4, sigma=0.01, d=0.1 m -> ~0.8 dB of
absorption per wall at 5.21 GHz) is close to free space, so the
materials-matched multi-wall baseline is within ~2 dB of RT everywhere
and the learned residual has little headroom. A concrete probe on 8
held-out scenes (same geometry, material swapped) moved COST-231's
error to 4.0 / 8.4 / 13.6 dB at 1 / 2 / 3+ walls with an all-positive
bias (direct-ray sum over-predicts loss; energy arrives around walls
through doorways), and log-distance beat COST at depth. That is the
regime a learned model can exploit and the one the paper should
report alongside the light suite.

Inherits, via imports:
  train_delta_tau_v2 : RT label config (max_depth 10, 1e6 samples,
                       1e7 path buffer, rx_batch 32), coarse 256-pt
                       label grid, per-batch Dr.Jit flush, 200+90 scenes
  train_delta_tau_v3 : LOS residual gate OFF, dB smooth-L1 primary
                       (w_db=1.0, w_power=0.2), checkpoint on wb_mae_db,
                       per-depth validation report
Overrides here:
  * default_material_db -> ITU material (eps_r=a, sigma=c*f_GHz^d),
    thickness from --thickness (default 0.10 m to match the probe)
  * out_dir -> runs/residual_cost_v4_<material>  (NEW stores; the v3
    symlink trick is deliberately not used)
  * r_clip_db raised to 80 dB: with 12 dB/wall the residual target
    (RT - COST) reaches -40 dB at 3+ walls in the probe; thicker walls
    could exceed the old +-60 clip
Everything else (TX lattice, features, split, seed, epochs, LR) is
unchanged, so this run differs from v3 in material and stores only.

Run from the cfr_pred repo root, same venv, on the label-generation
machine:
    python train_delta_tau_v4.py [--material concrete] [--thickness 0.10]
After the FIRST scene prints its label time, multiply by 290 before
committing to the full run. Generation is per-scene resumable.

Held-out suite: generate the matching suite from the light-wall scenes
with make_probe_scenes.py (same geometry & placements, material
swapped) using the SAME --material/--thickness:
    python make_probe_scenes.py --material concrete --thickness 0.10 \\
        --out worldbuilding/heldout_concrete \\
        --scenes 'worldbuilding/heldout/tier_*/seed*'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

import train_delta_tau_v3 as v3  # noqa: F401  (v3 objective + report; imports v2 underneath)
import train_delta_tau as td

# ITU-R P.2040 Table 3:  eps_r = a,  sigma = c * f_GHz**d
ITU = {
    "concrete": (5.24, 0.0462, 0.7976),
    "brick": (3.91, 0.0238, 0.16),
    "plasterboard": (2.73, 0.0085, 0.9395),
    "wood": (1.99, 0.0047, 1.0718),
    "glass": (6.31, 0.0036, 1.3394),
}


def itu_params(material: str, freq_hz: float):
    a, c, d = ITU[material]
    f_ghz = float(freq_hz) / 1e9
    return float(a), float(c * f_ghz ** d)


def absorption_db_per_wall(eps_r: float, sigma: float, thickness_m: float) -> float:
    """Low-loss-dielectric absorption through one slab, normal incidence
    (reflection loss at the faces comes on top): alpha ~ sigma*eta0/(2 sqrt(eps_r))."""
    eta0 = 376.73
    alpha_np_per_m = sigma * eta0 / (2.0 * np.sqrt(eps_r))
    return 8.686 * alpha_np_per_m * thickness_m


def make_material_db(eps_r: float, sigma: float, thickness_m: float):
    """Same schema as train_delta_tau.default_material_db; only the EM
    parameters change. The *_loss columns are kept at the original
    values -- the cost feature and Sionna both use eps_r/sigma/thickness
    (the held-out COST bias at 3+ walls, +0.25 dB on light walls, rules
    out a fixed 10 dB/wall term)."""
    def db(freq):
        return pl.DataFrame(data={
            "id": [0], "frequency": [freq], "permittivity": [eps_r],
            "permeability": [1.0], "conductivity": [sigma],
            "transmission_loss_vertical": [10.0],
            "transmission_loss_horizontal": [20.0], "reflection_loss": [9.0],
            "diffraction_loss_min": [8.0], "diffraction_loss_max": [15.0],
            "diffraction_loss": [5.0], "name": ["0"], "thickness": [thickness_m]})
    return db


def self_check(eps_r, sigma, thickness_m):
    """Build one scene through the real generator and verify the material
    that reaches the scene (and therefore RT labels + cost feature)."""
    scene = td.make_scene(td.scene_rng("inpatch", 0))
    mdb = scene.material_database
    got = (float(mdb["permittivity"][0]), float(mdb["conductivity"][0]),
           float(mdb["thickness"][0]))
    want = (eps_r, sigma, thickness_m)
    ok = all(abs(g - w) < 1e-6 * max(1.0, abs(w)) for g, w in zip(got, want))
    print(f"[v4] self-check scene material: eps_r={got[0]} sigma={got[1]:.4f} "
          f"d={got[2]}  -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("material override did not reach make_scene(); "
                         "default_material_db is not being resolved through "
                         "train_delta_tau globals -- do not generate labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", choices=sorted(ITU), default="concrete")
    ap.add_argument("--thickness", type=float, default=0.10, help="wall thickness (m)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="default runs/residual_cost_v4_<material>")
    ap.add_argument("--r_clip_db", type=float, default=80.0)
    args = ap.parse_args()

    eps_r, sigma = itu_params(args.material, td.cfg.frequency_hz)
    out_dir = args.out_dir or f"runs/residual_cost_v4_{args.material}"

    # --- overrides (module globals, resolved at call time by make_scene /
    #     make_full_scene, exactly like v2's compute_rt_labels swap) ---
    td.default_material_db = make_material_db(eps_r, sigma, args.thickness)
    td.cfg.out_dir = out_dir
    td.cfg.r_clip_db = float(args.r_clip_db)
    # v3 objective is already applied by the v3 import; restate for the log
    assert td.cfg.los_gate_residual is False and td.cfg.w_db == 1.0 \
        and td.cfg.ckpt_metric == "wb_mae_db", "v3 objective not applied"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if any(Path(out_dir).glob("store_*")):
        print(f"[v4] existing store_* under {out_dir}: generation will RESUME "
              f"per progress.json (delete the dir to start clean)")

    self_check(eps_r, sigma, args.thickness)
    absorb = absorption_db_per_wall(eps_r, sigma, args.thickness)
    print(f"[v4] material={args.material} eps_r={eps_r} sigma={sigma:.4f} S/m "
          f"d={args.thickness} m  (~{absorb:.1f} dB absorption/wall + face reflection)")
    print(f"[v4] gate={td.cfg.los_gate_residual} w_db={td.cfg.w_db} "
          f"w_power={td.cfg.w_power} ckpt={td.cfg.ckpt_metric} "
          f"r_clip_db={td.cfg.r_clip_db} scenes={td.cfg.num_inpatch_scenes}"
          f"+{td.cfg.num_offpatch_scenes} out_dir={out_dir}")

    prov = {}
    src = Path("runs/residual_cost_v3/rt_provenance.json")
    if src.exists():
        prov = json.loads(src.read_text())
    prov.update({
        "v4_date": "2026-09-21",
        "v4_reason": "heavy-wall regime: regenerate stores with an ITU-R "
                     "P.2040 material; v3 objective unchanged",
        "v4_material": {"name": args.material, "eps_r": eps_r,
                        "sigma_S_per_m": sigma, "thickness_m": args.thickness,
                        "f_ghz": td.cfg.frequency_hz / 1e9,
                        "absorption_db_per_wall_est": round(absorb, 2)},
        "v4_evidence": "8-scene concrete probe on light-wall geometry: "
                       "COST-231 MAE 2.8/4.0/8.4/13.6 dB at LOS/1/2/3+ walls, "
                       "all-positive bias at depth (max +40 dB); log-distance "
                       "beat COST at 2 and 3+ walls",
        "v4_cfg": {"out_dir": out_dir, "r_clip_db": td.cfg.r_clip_db},
        "stores": "regenerated (not reused)",
    })
    (Path(out_dir) / "rt_provenance.json").write_text(json.dumps(prov, indent=2))

    return td.main()


if __name__ == "__main__":
    main()