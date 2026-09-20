#!/usr/bin/env python3
"""
train_delta_tau_v3.py
==================================================================
Objective-only retrain on the v2 stores (2026-09-20).  Imports
train_delta_tau_v2 so every v2 override (RT label config, coarse
label grid, dataset sizes, compute_rt_labels swap) is applied, then
changes ONLY how the network is trained:

  1. LOS residual gate OFF (cfg.los_gate_residual = False).
     Diagnosis: on the held-out suite, COST-231 (= FSPL at LOS) sits
     at 1.55 dB MAE with a +1.19 dB constant offset vs RT, over an RT
     seed-repeat floor of ~0.16 dB.  The gate zeroed the residual
     gradient on every LOS cell, so the network could never learn
     even that constant.  The gate's premise ("cost is essentially
     exact at LOS") is false on v3 RT labels.

  2. dB smooth-L1 becomes the PRIMARY loss (w_db 0.2 -> 1.0), linear
     power NMSE becomes secondary (w_power 1.0 -> 0.2).
     Diagnosis: squared error in linear power scales with power^2, so
     the strongest ungated cells (1-wall, near TX) took ~all of the
     gradient and 2/3+ wall cells were trained only by the 0.2-weight
     dB term.  Held-out result: genuine win at 1 wall, noise at depth.
     The paper scores per-link dB MAE by stratum; train on that.

  3. Checkpoint on wb_mae_db instead of wb_nmse_db (same reason).

  4. Per-depth validation report (LOS / 1 / 2 / 3+ walls, U-Net vs
     COST, MAE and signed bias) appended to every epoch so the deep
     strata are visible during training.  Extra keys land in
     history.csv.

Stores are NOT regenerated: runs/residual_cost_v2/store_* are
symlinked into the new out_dir and reused as-is.  norm_stats.npz is
recomputed (deterministic: same seed, same stores).

SERVING NOTE: ns3unet_spectrum.py reads los_gate_residual from
meta.json, so the residual gate switches off automatically for a v3
checkpoint -- but line ~497 (the coverage-head LOS guarantee) is
under the same flag and must be split onto its own flag first.  See
the note accompanying this file.

Run from the cfr_pred repo root, same venv:
    python train_delta_tau_v3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import train_delta_tau_v2 as v2  # noqa: F401  (applies all v2 overrides on import)
import train_delta_tau as td

# ------------------------------------------------------------------
# config
# ------------------------------------------------------------------
V2_DIR = Path(td.cfg.out_dir)                 # runs/residual_cost_v2
V3 = dict(
    out_dir="runs/residual_cost_v3",
    los_gate_residual=False,
    w_power=0.2,
    w_db=1.0,
    ckpt_metric="wb_mae_db",
)
for k, v in V3.items():
    setattr(td.cfg, k, v)
V3_DIR = Path(td.cfg.out_dir)


def link_v2_stores():
    """Symlink every store_* dir from v2 so the builders see complete
    stores and skip generation.  Refuses to run if v2 has none."""
    V3_DIR.mkdir(parents=True, exist_ok=True)
    stores = sorted(p for p in V2_DIR.glob("store_*") if p.is_dir())
    if not stores:
        raise SystemExit(f"no store_* dirs under {V2_DIR} -- nothing to reuse")
    for src in stores:
        dst = V3_DIR / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve(), target_is_directory=True)
        print(f"[v3] linked {dst} -> {src.resolve()}")
    for src in stores:
        for f in ("x.dat", "y.dat", "meta.json"):
            if not (V3_DIR / src.name / f).exists():
                raise SystemExit(f"{src.name}/{f} missing -- v2 store incomplete")


# ------------------------------------------------------------------
# per-depth validation report (wraps the original, keeps its keys)
# ------------------------------------------------------------------
_orig_report = td.report_metrics
BINS = (("LOS", 0, 0), ("1wall", 1, 1), ("2wall", 2, 2), ("3+wall", 3, 10 ** 6))


@torch.no_grad()
def report_metrics_v3(model, dl, stats, tag="val"):
    m = _orig_report(model, dl, stats, tag)
    K = td.cfg.K_slices
    acc = {b[0]: dict(n=0.0, u_abs=0.0, c_abs=0.0, u_err=0.0, c_err=0.0)
           for b in BINS}
    for batch in dl:
        batch = {k: v.to(td.cfg.device) for k, v in batch.items()}
        pred = model(batch["x"])
        B = pred.shape[0]
        p3 = pred.view(B, K, td.PRED_CH, batch["x"].shape[-2], batch["x"].shape[-1])
        wb_hat = td.reconstruct_wb(p3, batch, stats)
        e_u = wb_hat - batch["wb_rt"]
        e_c = batch["wb_cost"] - batch["wb_rt"]
        nb = torch.round(batch["nobs"])
        mp = batch["m_path"]
        for name, lo, hi in BINS:
            sel = ((nb >= lo) & (nb <= hi)).float() * mp
            a = acc[name]
            a["n"] += sel.sum().item()
            a["u_abs"] += (e_u.abs() * sel).sum().item()
            a["c_abs"] += (e_c.abs() * sel).sum().item()
            a["u_err"] += (e_u * sel).sum().item()
            a["c_err"] += (e_c * sel).sum().item()

    print(f"  [{tag}] by depth   n   unetMAE  costMAE  unetBias  costBias")
    for name, _, _ in BINS:
        a = acc[name]
        n = max(a["n"], 1.0)
        um, cm = a["u_abs"] / n, a["c_abs"] / n
        ub, cb = a["u_err"] / n, a["c_err"] / n
        print(f"    {name:>7} {int(a['n']):>7d}  {um:7.3f}  {cm:7.3f}  "
              f"{ub:+8.3f}  {cb:+8.3f}")
        m[f"mae_{name}_unet"] = um
        m[f"mae_{name}_cost"] = cm
        m[f"bias_{name}_unet"] = ub
        m[f"bias_{name}_cost"] = cb
    return m


td.report_metrics = report_metrics_v3


# ------------------------------------------------------------------
def main():
    link_v2_stores()
    prov = {}
    src = V2_DIR / "rt_provenance.json"
    if src.exists():
        prov = json.loads(src.read_text())
    prov.update({
        "v3_date": "2026-09-20",
        "v3_reason": "objective-only retrain on unchanged v2 stores: LOS "
                     "residual gate removed, dB smooth-L1 primary "
                     "(w_db=1.0, w_power=0.2), checkpoint on wb_mae_db",
        "v3_evidence": "held-out paired per-link diag: exact U-Net==COST "
                       "at LOS (gate), +1.19 dB shared LOS bias, RT "
                       "seed-repeat floor 0.1-0.3 dB vs 1.5-2.2 dB "
                       "model MAE; win at 1 wall only",
        "v3_cfg": V3,
        "stores": f"symlinked from {V2_DIR}",
    })
    (V3_DIR / "rt_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[v3] gate={td.cfg.los_gate_residual} w_db={td.cfg.w_db} "
          f"w_power={td.cfg.w_power} ckpt={td.cfg.ckpt_metric} "
          f"out_dir={td.cfg.out_dir}")
    return td.main()


if __name__ == "__main__":
    main()