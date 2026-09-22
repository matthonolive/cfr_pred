#!/usr/bin/env python3
"""
train_delta_tau_v4.py
==================================================================
Heavy-wall retrain (2026-09-21): regenerate BOTH stores with ITU-R
P.2040 concrete, an RT label configuration converged for that
regime, and sparse per-TX labelling to fit the budget.

Why concrete: the light-wall suite (eps_r=4, sigma=0.01 -> ~0.8 dB of
absorption per wall) is close to free space, so the materials-matched
multi-wall baseline is within ~2 dB of RT everywhere. An 8-scene
concrete probe moved COST-231 to 4.0 / 8.4 / 13.6 dB at 1 / 2 / 3+
walls with an all-positive bias (direct-ray sum over-predicts loss;
energy arrives around walls through doorways).

Why this RT config: rt_convergence_sweep.py on a concrete probe scene
showed PATH-BUFFER TRUNCATION dominates at depth -- at 1e6 rays the
3+-wall bias vs a solo seed-averaged reference was +5.4 / +2.4 /
+1.3 / +0.6 dB for 64 / 32 / 16 / 8 receivers per call (1e7 buffer),
while 1e5 rays undersample the few doorway paths (+1.2 dB). The only
config meeting the +-0.3 dB criterion was 3e5 rays, 16 rx/call, 1e7
buffer (3+ walls +0.29 dB median, seed spread ~0.7 dB, 0 flips).

Why sparse labels + 3x3 TX + 150/60 scenes: per-call time is
overhead-dominated (~0.2 s), so label cost is ~proportional to the
number of (TX, receiver) pairs. 25 TX x 200/90 scenes x 16384 rx at
this config is ~13 days; 9 TX x 150/60 scenes x 50% of cells is ~2.
Diversity lives in scenes, not TX density; the U-Net is trained
pixel-wise with masks, so labelling a random half of the cells per TX
halves RT time at no change to the objective. Unlabelled cells are
stored as NaN in wb_RT and excluded from EVERY loss term (including
coverage, which previously ran over all pixels) and every metric.

Inherits via imports:
  train_delta_tau_v2 : coarse 256-pt label grid, per-batch Dr.Jit
                       flush, RtCfg fields (max_depth 10, diffraction)
  train_delta_tau_v3 : LOS residual gate OFF, dB smooth-L1 primary,
                       checkpoint on wb_mae_db, per-depth report
Overrides here:
  default_material_db  -> ITU material at cfg.frequency_hz
  compute_rt_labels    -> sparse subset per TX, NaN = unlabelled
  ResidualDataset.__getitem__, loss_fn, report_metrics (base) -> label mask
  cfg: rt (samples/buffer), rx_batch, tx_shape/tx_spacing, scene counts,
       r_clip_db (80: residual reaches -40 dB at depth with 12 dB walls),
       out_dir (NEW stores)

Run from the cfr_pred repo root:
    python train_delta_tau_v4.py                       # defaults = the 2-day config
    python train_delta_tau_v4.py --label_fraction 1.0  # dense labels (slower)
Generation is per-scene resumable. A label_config.json guard refuses
to resume a run dir generated under a different configuration.

Held-out suite: make_probe_scenes.py with the SAME --material and
--thickness over worldbuilding/heldout/tier_*/seed*. NOTE: the
evaluation RT in validate_cfr.py solves all STAs of a scene in one
call (up to 14 rx sharing 1e7 slots), which this sweep shows is ~+1 dB
dark at 3+ walls in concrete -- batch <=4 receivers there before
scoring anything on the heavy suite.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import drjit as dr

import train_delta_tau_v3 as v3  # noqa: F401  (v3 objective + report; imports v2 underneath)
import train_delta_tau_v2 as v2
import train_delta_tau as td
from mlink.channel_tdl import subcarrier_frequencies_centered, compute_tdl_batch

ITU = {  # ITU-R P.2040 Table 3:  eps_r = a,  sigma = c * f_GHz**d
    "concrete": (5.24, 0.0462, 0.7976),
    "brick": (3.91, 0.0238, 0.16),
    "plasterboard": (2.73, 0.0085, 0.9395),
    "wood": (1.99, 0.0047, 1.0718),
    "glass": (6.31, 0.0036, 1.3394),
}
ARGS = None   # filled in main(); read by the label function


# ------------------------------------------------------------------
# material
# ------------------------------------------------------------------
def itu_params(material, freq_hz):
    a, c, d = ITU[material]
    return float(a), float(c * (freq_hz / 1e9) ** d)


def absorption_db_per_wall(eps_r, sigma, thickness_m):
    return 8.686 * (sigma * 376.73 / (2.0 * np.sqrt(eps_r))) * thickness_m


def make_material_db(eps_r, sigma, thickness_m):
    """Same schema as train_delta_tau.default_material_db; only the EM
    parameters change (the cost feature and Sionna use eps_r/sigma/d)."""
    def db(freq):
        return pl.DataFrame(data={
            "id": [0], "frequency": [freq], "permittivity": [eps_r],
            "permeability": [1.0], "conductivity": [sigma],
            "transmission_loss_vertical": [10.0],
            "transmission_loss_horizontal": [20.0], "reflection_loss": [9.0],
            "diffraction_loss_min": [8.0], "diffraction_loss_max": [15.0],
            "diffraction_loss": [5.0], "name": ["0"], "thickness": [thickness_m]})
    return db


# ------------------------------------------------------------------
# sparse RT labels (port of compute_rt_labels_v2 with a per-TX subset)
# ------------------------------------------------------------------
def compute_rt_labels_v4(scene):
    """Returns (wb, tau) of shape (T, K, H, W). wb is NaN where unlabelled,
    >= cfg.no_path_wb_db where RT found no path, else the wideband loss."""
    rx_grid = scene.antenna_database.rx_grid
    K, H, W = rx_grid.shape
    tx_coords = scene.antenna_database.tx_coords
    rx_coords = scene.antenna_database.rx_coords
    P = rx_coords.shape[0]
    N = int(v2.LABEL_FFT)
    freqs = subcarrier_frequencies_centered(N, v2.LABEL_SCS_HZ)
    si = td._to_sionna_geometry(scene, td.cfg.frequency_hz)
    sentinel = td.cfg.no_path_wb_db
    rb = int(td.cfg.rx_batch)
    frac = float(ARGS.label_fraction)
    n_lab = P if frac >= 1.0 else max(rb, int(round(frac * P)))
    flush_every = int(v2.V2["flush_every"])

    wb_out = np.full((tx_coords.shape[0], K, H, W), np.nan, np.float32)
    tau_out = np.zeros((tx_coords.shape[0], K, H, W), np.float32)
    t_scene = time.time()
    for t, tx in enumerate(tx_coords):
        # reproducible subset per (seed, TX position) -> resumable
        rng = np.random.default_rng([int(td.cfg.seed), 7_000_003]
                                    + [int(round(float(v) * 1000)) for v in tx])
        sel = np.arange(P) if n_lab >= P else np.sort(rng.choice(P, size=n_lab, replace=False))
        wb_all = np.full((P,), np.nan, np.float32)
        tau_all = np.zeros((P,), np.float32)
        bt = []
        for b, i0 in enumerate(range(0, len(sel), rb)):
            ids = sel[i0:i0 + rb]
            t0 = time.time()
            wb_db, _ex, _taps, tau_rms_s = compute_tdl_batch(
                si_scene=si, tx_xyz=tx, rx_xyz=rx_coords[ids],
                frequencies_hz=freqs, L_taps=N, rt=td.cfg.rt, return_tau_rms=True)
            bt.append(time.time() - t0)
            wb_all[ids] = wb_db
            good = wb_db < sentinel
            if np.any(good):
                tau_all[ids[good]] = tau_rms_s[good] * 1e9
            if flush_every and (b + 1) % flush_every == 0:
                dr.sync_thread(); dr.flush_malloc_cache()
        wb_map = wb_all.reshape(K, H, W)
        # masked median: NaN (unlabelled) and sentinel (dead) both fail
        # `wb < sentinel`, so neither contributes to a neighbour's median
        tau_map = np.maximum(td.smooth_map_stack(tau_all.reshape(K, H, W), wb_map), 0.0)
        tau_map[~(wb_map < sentinel)] = 0.0          # dead OR unlabelled
        wb_out[t] = wb_map
        tau_out[t] = tau_map
        bt = np.asarray(bt)
        print(f"    tx {t+1}/{tx_coords.shape[0]}: {len(sel)}/{P} rx labelled, "
              f"{len(bt)} calls, median {np.median(bt):.2f} s/call", flush=True)
        gc.collect(); dr.sync_thread(); dr.flush_malloc_cache()
    print(f"    [v4] scene labels in {time.time() - t_scene:.0f} s "
          f"({tx_coords.shape[0]} tx)", flush=True)
    return wb_out, tau_out


# ------------------------------------------------------------------
# dataset / loss / metrics with the label mask
# ------------------------------------------------------------------
def getitem_v4(self, i):
    j = int(self.indices[i])
    x = np.array(self.x_mm[j], np.float32)[self.keep_idx, :, :]
    y = np.array(self.y_mm[j], np.float32).reshape(self.K, td.Y_STORE, self.H, self.W)
    wb_raw = y[:, td.S_WBRT]
    lab = np.isfinite(wb_raw)                                   # labelled cells
    wb_rt = torch.from_numpy(np.where(lab, wb_raw, td.cfg.no_path_wb_db).astype(np.float32))
    m_label = torch.from_numpy(lab.astype(np.float32))
    tau = torch.from_numpy(y[:, td.S_TAU])
    wb_cost = torch.from_numpy(y[:, td.S_WBCOST])
    nobs = torch.from_numpy(y[:, td.S_NOBS])

    m_path = (wb_rt < td.cfg.no_path_wb_db).float() * m_label   # labelled AND alive
    m_tau = (m_path.bool() & (tau >= td.cfg.tau_loss_thresh_ns)).float()
    cov = m_path.clone()

    r = torch.clamp(wb_rt - wb_cost, -td.cfg.r_clip_db, td.cfg.r_clip_db)
    r_norm = (r - self.r_mean) / self.r_std * m_path
    tau_t = torch.from_numpy(td.tau_to_target(tau.numpy()))
    tau_norm = (tau_t - self.tau_mean) / self.tau_std * m_tau
    x = (torch.from_numpy(x) - self.x_mean) / self.x_std

    if self.augment:
        k = int(torch.randint(0, 4, (1,)).item())
        flip = torch.rand(1).item() < 0.5
        x, r_norm, tau_norm, cov, wb_cost, wb_rt, nobs, m_path, m_tau, m_label = td._aug(
            [x, r_norm, tau_norm, cov, wb_cost, wb_rt, nobs, m_path, m_tau, m_label], k, flip)

    return {"x": x, "r": r_norm, "tau": tau_norm, "cov": cov,
            "wb_cost": wb_cost, "wb_rt": wb_rt, "nobs": nobs,
            "m_path": m_path, "m_tau": m_tau, "m_label": m_label}


def loss_fn_v4(pred, batch, stats):
    """train_delta_tau.loss_fn with the coverage BCE restricted to
    labelled cells (unlabelled != dead). The other terms already use
    m_path, which getitem_v4 restricts to labelled cells."""
    B, _, H, W = pred.shape
    K = td.cfg.K_slices
    p3 = pred.view(B, K, td.PRED_CH, H, W)
    mp, mt, ml = batch["m_path"], batch["m_tau"], batch["m_label"]

    wb_hat = td.reconstruct_wb(p3, batch, stats)
    wb_rt = batch["wb_rt"]
    g_hat = torch.pow(10.0, torch.clamp(-wb_hat / 10.0, -20.0, 20.0))
    g_tgt = torch.pow(10.0, torch.clamp(-wb_rt / 10.0, -20.0, 20.0))
    l_power = (((g_hat - g_tgt) ** 2) * mp).sum() / ((g_tgt ** 2) * mp).sum().clamp_min(1e-12)

    w = mp
    if td.cfg.db_nobs_gain > 0:
        w = mp * (1.0 + td.cfg.db_nobs_gain * (batch["nobs"] >= td.cfg.nobs_los_thresh).float())
    l_db = (F.smooth_l1_loss(wb_hat, wb_rt, reduction="none") * w).sum() / w.sum().clamp_min(1.0)

    tau_hat = p3[:, :, td.P_TAU]
    l_tau = F.smooth_l1_loss(tau_hat * mt, batch["tau"] * mt, reduction="sum") / mt.sum().clamp_min(1.0)

    l_cov = (F.binary_cross_entropy_with_logits(p3[:, :, td.P_COV], batch["cov"], reduction="none")
             * ml).sum() / ml.sum().clamp_min(1.0)

    return td.cfg.w_power * l_power + td.cfg.w_db * l_db + td.cfg.w_tau * l_tau + td.cfg.w_cov * l_cov


@torch.no_grad()
def report_metrics_v4_base(model, dl, stats, tag="val"):
    """train_delta_tau.report_metrics with coverage accuracy over labelled
    cells only; everything else already conditions on m_path."""
    model.eval()
    K = td.cfg.K_slices
    acc = dict(num=0.0, den=0.0, mae=0.0, n=0.0, mae_los=0.0, n_los=0.0, mae_nlos=0.0,
               n_nlos=0.0, c_num=0.0, c_den=0.0, c_mae=0.0, c_mae_los=0.0, c_mae_nlos=0.0,
               tau=0.0, ntau=0.0, cov_ok=0.0, cov_n=0.0)
    for batch in dl:
        batch = {k: v.to(td.cfg.device) for k, v in batch.items()}
        pred = model(batch["x"])
        B = pred.shape[0]
        p3 = pred.view(B, K, td.PRED_CH, batch["x"].shape[-2], batch["x"].shape[-1])
        wb_hat = td.reconstruct_wb(p3, batch, stats)
        wb_rt, wb_cost, mp = batch["wb_rt"], batch["wb_cost"], batch["m_path"]
        los = (batch["nobs"] < td.cfg.nobs_los_thresh).float() * mp
        nlos = (batch["nobs"] >= td.cfg.nobs_los_thresh).float() * mp
        g_hat = torch.pow(10.0, torch.clamp(-wb_hat / 10.0, -20.0, 20.0))
        g_cost = torch.pow(10.0, torch.clamp(-wb_cost / 10.0, -20.0, 20.0))
        g_tgt = torch.pow(10.0, torch.clamp(-wb_rt / 10.0, -20.0, 20.0))
        acc["num"] += (((g_hat - g_tgt) ** 2) * mp).sum().item()
        acc["c_num"] += (((g_cost - g_tgt) ** 2) * mp).sum().item()
        acc["den"] += ((g_tgt ** 2) * mp).sum().item()
        e = (wb_hat - wb_rt).abs(); ec = (wb_cost - wb_rt).abs()
        acc["mae"] += (e * mp).sum().item(); acc["n"] += mp.sum().item()
        acc["c_mae"] += (ec * mp).sum().item()
        acc["mae_los"] += (e * los).sum().item(); acc["n_los"] += los.sum().item()
        acc["mae_nlos"] += (e * nlos).sum().item(); acc["n_nlos"] += nlos.sum().item()
        acc["c_mae_los"] += (ec * los).sum().item(); acc["c_mae_nlos"] += (ec * nlos).sum().item()
        tau_hat = td.tau_from_target_t(p3[:, :, td.P_TAU] * stats["tau_std"] + stats["tau_mean"])
        tau_tgt = td.tau_from_target_t(batch["tau"] * stats["tau_std"] + stats["tau_mean"])
        mt = batch["m_tau"]
        acc["tau"] += ((tau_hat - tau_tgt).abs() * mt).sum().item(); acc["ntau"] += mt.sum().item()
        ml = batch["m_label"]
        cov_pred = (torch.sigmoid(p3[:, :, td.P_COV]) >= 0.5).float()
        acc["cov_ok"] += ((cov_pred == batch["cov"]).float() * ml).sum().item()
        acc["cov_n"] += ml.sum().item()
    sd = lambda a, b: a / b if b > 0 else 0.0
    nmse = sd(acc["num"], acc["den"]); cnmse = sd(acc["c_num"], acc["den"])
    m = {"wb_nmse_db": 10 * np.log10(max(nmse, 1e-12)),
         "cost_nmse_db": 10 * np.log10(max(cnmse, 1e-12)),
         "wb_mae_db": sd(acc["mae"], acc["n"]), "cost_mae_db": sd(acc["c_mae"], acc["n"]),
         "wb_mae_los": sd(acc["mae_los"], acc["n_los"]), "cost_mae_los": sd(acc["c_mae_los"], acc["n_los"]),
         "wb_mae_nlos": sd(acc["mae_nlos"], acc["n_nlos"]), "cost_mae_nlos": sd(acc["c_mae_nlos"], acc["n_nlos"]),
         "tau_mae_ns": sd(acc["tau"], acc["ntau"]), "cov_acc": sd(acc["cov_ok"], acc["cov_n"])}
    print(f"  [{tag}] NMSE  unet={m['wb_nmse_db']:.2f}dB  cost={m['cost_nmse_db']:.2f}dB"
          f"  | MAE all unet={m['wb_mae_db']:.2f} cost={m['cost_mae_db']:.2f}"
          f"  | LOS unet={m['wb_mae_los']:.2f} cost={m['cost_mae_los']:.2f}"
          f"  | NLOS unet={m['wb_mae_nlos']:.2f} cost={m['cost_mae_nlos']:.2f}"
          f"  | tauMAE={m['tau_mae_ns']:.2f}ns covAcc={m['cov_acc']:.3f}")
    return m


# ------------------------------------------------------------------
def self_check(eps_r, sigma, thickness_m):
    scene = td.make_scene(td.scene_rng("inpatch", 0))
    mdb = scene.material_database
    got = (float(mdb["permittivity"][0]), float(mdb["conductivity"][0]), float(mdb["thickness"][0]))
    want = (eps_r, sigma, thickness_m)
    ok = all(abs(g - w) < 1e-6 * max(1.0, abs(w)) for g, w in zip(got, want))
    n_tx = int(scene.antenna_database.tx_coords.shape[0])
    print(f"[v4] self-check: material eps_r={got[0]} sigma={got[1]:.4f} d={got[2]} -> "
          f"{'OK' if ok else 'MISMATCH'};  TX per scene = {n_tx}")
    if not ok:
        raise SystemExit("material override did not reach make_scene() -- do not generate labels")
    return n_tx


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", choices=sorted(ITU), default="concrete")
    ap.add_argument("--thickness", type=float, default=0.10)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--r_clip_db", type=float, default=80.0)
    # RT label config (from rt_convergence_sweep on concrete)
    ap.add_argument("--samples", type=float, default=3e5)
    ap.add_argument("--rx_batch", type=int, default=16)
    ap.add_argument("--max_paths", type=float, default=1e7)
    ap.add_argument("--flush_every", type=int, default=8)
    # budget
    ap.add_argument("--label_fraction", type=float, default=0.5,
                    help="fraction of receiver cells labelled per TX (1.0 = dense)")
    ap.add_argument("--tx_grid", type=int, default=3, help="TX lattice per patch (n x n)")
    ap.add_argument("--scenes_inpatch", type=int, default=150)
    ap.add_argument("--scenes_offpatch", type=int, default=60)
    ap.add_argument("--s_per_call", type=float, default=0.20,
                    help="measured s/call for the projection (sweep: 0.20 at 3e5:16:1e7)")
    ARGS = args = ap.parse_args()

    eps_r, sigma = itu_params(args.material, td.cfg.frequency_hz)
    out_dir = Path(args.out_dir or f"runs/residual_cost_v4_{args.material}")

    # --- cfg overrides ---
    td.default_material_db = make_material_db(eps_r, sigma, args.thickness)
    td.cfg.out_dir = str(out_dir)
    td.cfg.r_clip_db = float(args.r_clip_db)
    td.cfg.rt = replace(td.cfg.rt, samples_per_src=int(args.samples),
                        max_num_paths_per_src=int(args.max_paths))
    td.cfg.rx_batch = int(args.rx_batch)
    v2.V2["flush_every"] = int(args.flush_every)
    n = int(args.tx_grid)
    td.cfg.tx_shape = (1, n, n)
    td.cfg.tx_spacing_xy = 48.0 / (n - 1) if n > 1 else 0.0   # 5x5 -> 12 (original), 3x3 -> 24
    td.cfg.num_inpatch_scenes = int(args.scenes_inpatch)
    td.cfg.num_offpatch_scenes = int(args.scenes_offpatch)
    assert td.cfg.los_gate_residual is False and td.cfg.w_db == 1.0 \
        and td.cfg.ckpt_metric == "wb_mae_db", "v3 objective not applied"

    # --- hooks ---
    td.compute_rt_labels = compute_rt_labels_v4
    td.ResidualDataset.__getitem__ = getitem_v4
    td.loss_fn = loss_fn_v4
    v3._orig_report = report_metrics_v4_base      # v3's per-depth report wraps this

    # --- config guard (never resume a dir generated under another config) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    label_cfg = {"material": args.material, "eps_r": eps_r, "sigma": round(sigma, 6),
                 "thickness": args.thickness, "samples": int(args.samples),
                 "rx_batch": int(args.rx_batch), "max_paths": int(args.max_paths),
                 "label_fraction": args.label_fraction, "tx_grid": n,
                 "scenes": [args.scenes_inpatch, args.scenes_offpatch],
                 "label_fft": v2.LABEL_FFT, "label_scs_hz": v2.LABEL_SCS_HZ}
    guard = out_dir / "label_config.json"
    if guard.exists():
        prev = json.loads(guard.read_text())
        if prev != label_cfg:
            raise SystemExit(f"{guard} was written under a different configuration:\n"
                             f"  stored : {prev}\n  current: {label_cfg}\n"
                             f"delete {out_dir} or pass a different --out_dir")
    else:
        guard.write_text(json.dumps(label_cfg, indent=2))

    n_tx = self_check(eps_r, sigma, args.thickness)
    P = td.cfg.K_slices * td.cfg.img_hw[0] * td.cfg.img_hw[1]
    n_lab = P if args.label_fraction >= 1.0 else max(args.rx_batch, int(round(args.label_fraction * P)))
    calls_per_tx = int(np.ceil(n_lab / args.rx_batch))
    total_tx = args.scenes_inpatch * n_tx + args.scenes_offpatch * td.cfg.patches_per_scene
    hours = total_tx * calls_per_tx * args.s_per_call / 3600
    absorb = absorption_db_per_wall(eps_r, sigma, args.thickness)
    print(f"[v4] material={args.material} eps_r={eps_r} sigma={sigma:.4f} d={args.thickness} m "
          f"(~{absorb:.1f} dB absorption/wall + face reflection)")
    print(f"[v4] RT labels: {args.samples:.0e} rays, {args.rx_batch} rx/call, "
          f"{args.max_paths:.0e} buffer ({args.max_paths/args.rx_batch:.2e} slots/rx), "
          f"flush every {args.flush_every}")
    print(f"[v4] budget: {n_tx} TX/scene x {args.scenes_inpatch} in-patch + "
          f"{td.cfg.patches_per_scene} x {args.scenes_offpatch} off-patch = {total_tx} TX; "
          f"{n_lab}/{P} cells labelled ({calls_per_tx} calls/TX) -> "
          f"~{hours:.0f} h at {args.s_per_call} s/call")
    print(f"[v4] objective: gate={td.cfg.los_gate_residual} w_db={td.cfg.w_db} "
          f"w_power={td.cfg.w_power} ckpt={td.cfg.ckpt_metric} r_clip_db={td.cfg.r_clip_db} "
          f"out_dir={out_dir}")

    prov = {}
    src = Path("runs/residual_cost_v3/rt_provenance.json")
    if src.exists():
        prov = json.loads(src.read_text())
    prov.update({
        "v4_date": "2026-09-21",
        "v4_reason": "heavy-wall regime: stores regenerated with an ITU-R P.2040 material "
                     "under an RT label config converged for it; sparse per-TX labels",
        "v4_material": {"name": args.material, "eps_r": eps_r, "sigma_S_per_m": sigma,
                        "thickness_m": args.thickness, "f_ghz": td.cfg.frequency_hz / 1e9,
                        "absorption_db_per_wall_est": round(float(absorb), 2)},
        "v4_rt_evidence": "rt_convergence_sweep (concrete, tier_small seed910000, solo 3e6 "
                          "seed-averaged reference): 3+-wall bias +5.4/+2.4/+1.3/+0.6 dB at "
                          "64/32/16/8 rx per call (1e6 rays, 1e7 buffer) = path-buffer "
                          "truncation; 1e5 rays undersamples (+1.2 dB); 3e5:16:1e7 passes "
                          "(+0.29 dB median, seed sd ~0.7 dB, 0 flips)",
        "v4_probe_evidence": "8-scene concrete probe on light-wall geometry: COST-231 MAE "
                             "2.8/4.0/8.4/13.6 dB at LOS/1/2/3+ walls, all-positive bias at "
                             "depth (max +40 dB); log-distance beat COST at 2 and 3+ walls",
        "v4_cfg": label_cfg | {"r_clip_db": td.cfg.r_clip_db, "flush_every": args.flush_every},
        "stores": "regenerated (not reused)",
    })
    (out_dir / "rt_provenance.json").write_text(json.dumps(prov, indent=2))

    rc = td.main()

    # record the label scheme beside the checkpoint (extra keys; the server ignores them)
    mp = out_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
        meta.update({"v4_material": label_cfg["material"], "v4_thickness_m": args.thickness,
                     "v4_label_fraction": args.label_fraction, "v4_tx_grid": n,
                     "v4_rt_label_config": f"{args.samples:.0e}:{args.rx_batch}:{args.max_paths:.0e}"})
        mp.write_text(json.dumps(meta, indent=2))
    return rc


if __name__ == "__main__":
    main()