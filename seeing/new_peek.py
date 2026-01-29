"""
Peek at a memmap run (x.dat/y.dat) and optionally a TorchScript model.

Adds: reconstruct WB loss from delta_pl_db + Friis/FSPL, using electrical_distance feature.

Assumes:
  delta_pl_db = wb_loss_db - fspl_db
  fspl_db = 20*log10(4*pi*electrical_distance)   where electrical_distance = d/lambda

Example:
  python new_peek.py --run runs/delta_3072_Jan7_merged --k 0 --elec-idx 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _read_meta(run: Path) -> dict:
    meta_path = run / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _infer_n_from_file(
    path: Path,
    dtype: np.dtype,
    H: int,
    W: int,
    candidates_C: list[int],
    preferred_N: Optional[int] = None,
    prefer_C: Optional[int] = None,
) -> Tuple[int, int]:
    dt = np.dtype(dtype)
    n_elem = path.stat().st_size // dt.itemsize
    best = None  # (score, N, C)

    for C in candidates_C:
        if C <= 0:
            continue
        denom = int(C) * int(H) * int(W)
        if denom <= 0:
            continue
        if n_elem % denom != 0:
            continue
        N = n_elem // denom
        score = 0
        if preferred_N is not None:
            score += abs(int(N) - int(preferred_N)) * 1000
        if prefer_C is not None:
            score += abs(int(C) - int(prefer_C))
        cand = (score, -int(N), int(C), int(N))
        if best is None or cand < best:
            best = cand

    if best is None:
        raise ValueError(
            f"Could not infer shape for {path}. "
            f"Tried C in {candidates_C} with H={H}, W={W}, dtype={dt}."
        )
    _, _, C, N = best
    return int(N), int(C)


def _memmap_4d(path: Path, dtype: np.dtype, shape: Tuple[int, int, int, int]) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _pick_first_written_sample(y_mm: np.memmap, max_scan: int = 500) -> int:
    n = int(min(max_scan, y_mm.shape[0]))
    for i in range(n):
        a = np.asarray(y_mm[i, 0], dtype=np.float32)
        if np.any(np.isfinite(a)) and (np.nanstd(a) > 1e-6):
            return i
    return 0


def _save_img(arr: np.ndarray, out: Path, title: str, vmin=None, vmax=None, cmap="viridis"):
    plt.figure(figsize=(5, 4), dpi=160)
    plt.imshow(arr, origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()

def _ensure_c11(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None, None]
    return a


def _robust_vlims(arr: np.ndarray, lo=1.0, hi=99.0):
    x = np.asarray(arr, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    vmin = float(np.percentile(x, lo))
    vmax = float(np.percentile(x, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return None, None
    return vmin, vmax


def _make_per_slice_channel_names(
    feat_names, x_per_slice: int
) -> list[str]:
    """
    Best-effort naming:
      - If feat_names is a list and matches x_per_slice -> use it.
      - Otherwise fall back to x_ch00, x_ch01, ...
    """
    if isinstance(feat_names, list) and len(feat_names) == x_per_slice:
        return [str(n) for n in feat_names]
    return [f"x_ch{ci:02d}" for ci in range(x_per_slice)]


def _write_x_channel_map(
    out_path: Path,
    *,
    sample: int,
    H: int,
    W: int,
    K: int,
    C_x: int,
    is_x_stacked: bool,
    x_per_slice: int,
    names_per_slice: list[str],
    keep_idx: Optional[np.ndarray],
):
    lines = []
    lines.append(f"sample={sample}  H={H} W={W}  K={K}  C_x={C_x}")
    lines.append(f"is_x_stacked={is_x_stacked}  x_per_slice={x_per_slice}")
    lines.append("")

    # global channel naming
    def global_name(ci: int) -> str:
        if is_x_stacked and K > 1 and x_per_slice > 0 and (C_x == K * x_per_slice):
            k = ci // x_per_slice
            c = ci % x_per_slice
            return f"k{k}_{names_per_slice[c]}"
        # unknown stacking/shape → just index
        return f"ch{ci:03d}"

    lines.append("GLOBAL x channel index → name")
    for ci in range(C_x):
        lines.append(f"{ci:4d}: {global_name(ci)}")

    if keep_idx is not None and keep_idx.size > 0:
        lines.append("")
        lines.append(f"keep_idx (len={keep_idx.size}) mapping:")
        for j, ci in enumerate(keep_idx.tolist()):
            if 0 <= ci < C_x:
                lines.append(f"  keep[{j:4d}] = {ci:4d} ({global_name(ci)})")
            else:
                lines.append(f"  keep[{j:4d}] = {ci:4d} (OUT OF RANGE!)")

    out_path.write_text("\n".join(lines))


def _dump_x(
    *,
    out: Path,
    meta: dict,
    stats: Optional[dict],
    x_samp: np.ndarray,          # (C_x,H,W) float32
    sample: int,
    k: int,
    H: int,
    W: int,
    K: int,
    is_x_stacked: bool,
    x_off: int,
    x_per_slice: int,
    dump_x: bool,
    dump_x_images: bool,
    dump_x_norm: bool,
):
    C_x = int(x_samp.shape[0])

    # channel naming (best-effort)
    feat_names = meta.get("requested_features") or meta.get("x_channels") or meta.get("dataset_features")
    names_per_slice = _make_per_slice_channel_names(feat_names, x_per_slice)

    # write a mapping text file (very useful for debugging keep_idx / stacking)
    _write_x_channel_map(
        out / f"s{sample:05d}_x_channel_map.txt",
        sample=sample, H=H, W=W, K=K, C_x=C_x,
        is_x_stacked=is_x_stacked, x_per_slice=x_per_slice,
        names_per_slice=names_per_slice,
        keep_idx=(stats.get("keep_idx") if (stats is not None) else None),
    )

    # slice tensor (per-slice channels)
    x_slice = x_samp[x_off:x_off + x_per_slice] if (x_per_slice > 0 and (x_off + x_per_slice) <= C_x) else x_samp

    if dump_x:
        np.savez_compressed(
            out / f"s{sample:05d}_k{k}_x_raw.npz",
            x_full_chw=x_samp.astype(np.float32, copy=False),
            x_slice_chw=x_slice.astype(np.float32, copy=False),
            sample=np.int64(sample),
            k=np.int64(k),
            is_x_stacked=np.int8(1 if is_x_stacked else 0),
        )

    if dump_x_images:
        # dump per-channel PNGs for the selected slice
        for ci in range(int(x_slice.shape[0])):
            name = names_per_slice[ci] if ci < len(names_per_slice) else f"x_ch{ci:02d}"
            arr = x_slice[ci]
            vmin, vmax = _robust_vlims(arr)
            _save_img(
                arr,
                out / f"s{sample:05d}_k{k}_x_{ci:02d}_{name}.png",
                f"x[{ci}] {name} (raw) (s={sample}, k={k})",
                vmin=vmin, vmax=vmax,
            )

    if dump_x_norm:
        if stats is None:
            print("[warn] --dump-x-norm requested but norm_stats.npz not found; skipping.")
            return

        keep_idx = stats.get("keep_idx", None)
        x_mean = _ensure_c11(stats["x_mean"].astype(np.float32))
        x_std  = _ensure_c11(stats["x_std"].astype(np.float32))
        x_std  = np.maximum(x_std, 1e-6)

        x_kept = x_samp
        if keep_idx is not None and np.size(keep_idx) > 0:
            keep_idx = np.asarray(keep_idx, dtype=np.int64)
            x_kept = x_samp[keep_idx, :, :]

        # NOTE: x_mean/x_std are expected to match x_kept channels (your training does this)
        if x_kept.shape[0] != x_mean.shape[0]:
            print(f"[warn] keep/mean mismatch: x_kept has {x_kept.shape[0]} ch but x_mean has {x_mean.shape[0]} ch. Saving anyway.")
        Cn = min(int(x_kept.shape[0]), int(x_mean.shape[0]))
        x_norm = (x_kept[:Cn] - x_mean[:Cn]) / x_std[:Cn]

        np.savez_compressed(
            out / f"s{sample:05d}_k{k}_x_model_input.npz",
            x_kept_chw=x_kept.astype(np.float32, copy=False),
            x_norm_chw=x_norm.astype(np.float32, copy=False),
            x_mean_c11=x_mean.astype(np.float32, copy=False),
            x_std_c11=x_std.astype(np.float32, copy=False),
            keep_idx=(keep_idx if keep_idx is not None else np.array([], dtype=np.int64)),
            sample=np.int64(sample),
            k=np.int64(k),
        )


def _maybe_load_stats(run: Path):
    stats_path = run / "norm_stats.npz"
    if not stats_path.exists():
        return None
    d = np.load(stats_path)
    return {
        "x_mean": d["x_mean"].astype(np.float32),
        "x_std": d["x_std"].astype(np.float32),
        "y_mean": d["y_mean"].astype(np.float32),
        "y_std": d["y_std"].astype(np.float32),
        "keep_idx": d["keep_idx"].astype(np.int64),
    }


def _maybe_load_torchscript(model_path: Path, device: str = "cpu"):
    if not model_path.exists():
        return None
    import torch
    m = torch.jit.load(str(model_path), map_location=device)
    m.eval()
    return m


def _fspl_from_electrical_distance(ed: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # FSPL(dB) = 20 log10( 4*pi * d/lambda ) = 20 log10( 4*pi * electrical_distance )
    ed = np.maximum(ed.astype(np.float32), eps)
    return (20.0 * np.log10(4.0 * np.pi * ed)).astype(np.float32)


def main(
    run_dir: str,
    sample: Optional[int] = None,
    k: int = 0,
    walls_idx: Optional[int] = None,
    elec_idx: Optional[int] = None,      # <-- NEW
    out_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    device: str = "cpu",
    assume_stacked: str = "auto",  # auto|true|false
    dump_x: bool = False,
    dump_x_images: bool = False,
    dump_x_norm: bool = False,
):
    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(run)

    x_path = run / "x.dat"
    y_path = run / "y.dat"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing x.dat/y.dat in {run}")

    meta = _read_meta(run)
    H = int(meta.get("H", 64))
    W = int(meta.get("W", 64))
    K = int(meta.get("K", 1))
    c_in = meta.get("c_in", None)
    y_ch = int(meta.get("y_ch", 3))
    total_meta = meta.get("total_samples", None)
    total_meta = int(total_meta) if total_meta is not None else None

    x_dtype = np.dtype(meta.get("x_dtype", "float32"))
    y_dtype = np.dtype(meta.get("y_dtype", "float16"))

    if assume_stacked not in {"auto", "true", "false"}:
        raise ValueError("assume_stacked must be auto|true|false")

    # Candidate channel counts
    if c_in is None:
        c_in_cands = [4, 3, 5, 6, 8, 1]
    else:
        c_in_cands = [int(c_in)]

    yC_cands = [y_ch]
    xC_cands = c_in_cands.copy()
    if K > 1:
        yC_cands += [y_ch * K]
        xC_cands += [c * K for c in c_in_cands]

    prefer_yC = None
    prefer_xC = None
    if assume_stacked == "true" and K > 1:
        prefer_yC = y_ch * K
        if c_in is not None:
            prefer_xC = int(c_in) * K
    elif assume_stacked == "false":
        prefer_yC = y_ch
        if c_in is not None:
            prefer_xC = int(c_in)

    N_y, C_y = _infer_n_from_file(y_path, y_dtype, H, W, yC_cands, preferred_N=total_meta, prefer_C=prefer_yC)
    N_x, C_x = _infer_n_from_file(x_path, x_dtype, H, W, xC_cands, preferred_N=N_y, prefer_C=prefer_xC)

    N = min(N_x, N_y)
    if N != N_x or N != N_y:
        print(f"[warn] N mismatch: from y={N_y}, from x={N_x}. Using N={N}.")

    y_mm = _memmap_4d(y_path, y_dtype, (N, C_y, H, W))
    x_mm = _memmap_4d(x_path, x_dtype, (N, C_x, H, W))

    is_y_stacked = (C_y == y_ch * K) if K > 1 else (C_y != y_ch)
    if c_in is None:
        c_in_eff = (C_x // K) if (K > 1 and C_x % K == 0) else C_x
    else:
        c_in_eff = int(c_in)

    is_x_stacked = (C_x == c_in_eff * K) if K > 1 else (C_x != c_in_eff)

    if sample is None:
        sample = _pick_first_written_sample(y_mm)

    sample = int(np.clip(sample, 0, N - 1))
    k = int(np.clip(k, 0, max(K - 1, 0)))

    out = Path(out_dir) if out_dir is not None else (run / "peek")
    out.mkdir(parents=True, exist_ok=True)

    # slice offsets
    y_per_slice = y_ch
    y_off = (k * y_per_slice) if (is_y_stacked and K > 1) else 0

    x_per_slice = c_in_eff
    x_off = (k * x_per_slice) if (is_x_stacked and K > 1) else 0

    # -------------------------
    # Figure out channel indices by meta y_channels
    # -------------------------
    y_names = meta.get("y_channels", None)
    y_idx = {}
    if isinstance(y_names, list):
        y_idx = {str(name): i for i, name in enumerate(y_names)}

    # Defaults consistent with YOUR training script:
    # 0=delta_pl_db, 1=ex, 2=tau, 3=wb_loss_db
    i_delta = y_idx.get("delta_pl_db", 0)
    i_ex    = y_idx.get("excess_delay_ns_sm", 1)
    i_tau   = y_idx.get("tau_rms_ns_sm", 2)
    i_wb    = y_idx.get("wb_loss_db", 3) if y_ch >= 4 else None

    # -------------------------
    # Determine per-slice input channel indices
    # -------------------------
    feat_names = meta.get("requested_features") or meta.get("x_channels")
    if isinstance(feat_names, list) and "binary_walls" in feat_names:
        walls_idx_eff = int(feat_names.index("binary_walls"))
    else:
        walls_idx_eff = int(walls_idx) if walls_idx is not None else 0
        if walls_idx is None:
            print("[warn] meta lacks x_channels; assuming binary_walls is per-slice channel 0 (override --walls-idx).")

    # electrical_distance: usually next after binary_walls in your requested list
    if isinstance(feat_names, list) and "electrical_distance" in feat_names:
        elec_idx_eff = int(feat_names.index("electrical_distance"))
    else:
        elec_idx_eff = int(elec_idx) if elec_idx is not None else 1
        if elec_idx is None:
            print("[warn] meta lacks x_channels; assuming electrical_distance is per-slice channel 1 (override --elec-idx).")

    # -------------------------
    # Export LABEL maps
    # -------------------------
    y_samp = np.asarray(y_mm[sample], dtype=np.float32)  # (C_y,H,W)

    delta_lab = y_samp[y_off + i_delta]
    ex_lab    = y_samp[y_off + i_ex]  if (y_ch >= 2) else None
    tau_lab   = y_samp[y_off + i_tau] if (y_ch >= 3) else None
    wb_lab    = y_samp[y_off + i_wb]  if (i_wb is not None and (y_off + i_wb) < y_samp.shape[0]) else None

    _save_img(delta_lab, out / f"s{sample:05d}_k{k}_deltaPL_label.png", f"delta_pl_db label (s={sample}, k={k})")
    if ex_lab is not None:
        _save_img(ex_lab, out / f"s{sample:05d}_k{k}_ex_label.png", f"Excess delay label (ns) (s={sample}, k={k})")
    if tau_lab is not None:
        _save_img(tau_lab, out / f"s{sample:05d}_k{k}_tauRMS_label.png", f"Tau RMS label (ns) (s={sample}, k={k})")
    if wb_lab is not None:
        _save_img(wb_lab, out / f"s{sample:05d}_k{k}_wbTRUE_label.png", f"WB true label (dB) (s={sample}, k={k})")

    # -------------------------
    # Export INPUT maps (walls + electrical_distance + FSPL)
    # -------------------------
    
    stats = _maybe_load_stats(run)
    x_samp = np.asarray(x_mm[sample], dtype=np.float32)  # (C_x,H,W)

    _dump_x(
        out=out,
        meta=meta,
        stats=stats,
        x_samp=x_samp,
        sample=sample,
        k=k,
        H=H, W=W, K=K,
        is_x_stacked=is_x_stacked,
        x_off=x_off,
        x_per_slice=x_per_slice,
        dump_x=dump_x,
        dump_x_images=dump_x_images,
        dump_x_norm=dump_x_norm,
    )

    walls = x_samp[x_off + walls_idx_eff]
    walls_bin = (walls > 0.5).astype(np.float32)

    elec = x_samp[x_off + elec_idx_eff]
    fspl = _fspl_from_electrical_distance(elec)

    _save_img(walls, out / f"s{sample:05d}_k{k}_walls_raw.png", f"binary_walls raw (s={sample}, k={k})", vmin=0.0, vmax=1.0, cmap="gray")
    _save_img(walls_bin, out / f"s{sample:05d}_k{k}_walls_bin.png", f"binary_walls bin (s={sample}, k={k})", vmin=0.0, vmax=1.0, cmap="gray")
    _save_img(elec, out / f"s{sample:05d}_k{k}_elecDist.png", f"electrical_distance=d/λ (s={sample}, k={k})")
    _save_img(fspl, out / f"s{sample:05d}_k{k}_fspl.png", f"FSPL from electrical_distance (dB) (s={sample}, k={k})")

    # Reconstruct WB from delta label
    wb_from_delta_lab = fspl + delta_lab
    _save_img(wb_from_delta_lab, out / f"s{sample:05d}_k{k}_wbFromDelta_label.png",
              f"WB = FSPL + delta_pl (label) (dB) (s={sample}, k={k})")

    if wb_lab is not None:
        _save_img(wb_from_delta_lab - wb_lab, out / f"s{sample:05d}_k{k}_wbFromDelta_minus_TRUE_label.png",
                  f"(FSPL+delta) - WB_true (label) (dB) (s={sample}, k={k})")

    # -------------------------
    # Optional: model prediction
    # -------------------------
    
    if model_path is None:
        cand = run / "model.pt"
        model_path = str(cand) if cand.exists() else None

    model = _maybe_load_torchscript(Path(model_path), device=device) if model_path is not None else None

    if model is not None and stats is not None:
        import torch

        x_t = torch.from_numpy(x_samp.astype(np.float32)).unsqueeze(0)  # (1,C,H,W)
        x_mean = torch.from_numpy(stats["x_mean"]).to(x_t.dtype)
        x_std  = torch.from_numpy(stats["x_std"]).to(x_t.dtype)
        y_mean = torch.from_numpy(stats["y_mean"]).to(x_t.dtype)
        y_std  = torch.from_numpy(stats["y_std"]).to(x_t.dtype)
        keep_idx = stats.get("keep_idx", None)

        # Apply keep_idx BEFORE normalization, same as training
        if keep_idx is not None:
            keep_idx_t = torch.from_numpy(keep_idx).long()
            x_t = x_t[:, keep_idx_t]

        x_n = (x_t - x_mean) / torch.clamp(x_std, min=1e-6)
        x_n = x_n.to(device)

        with torch.no_grad():
            pred_n = model(x_n)  # (1,Cy,H,W)

        pred_p = (pred_n.cpu() * y_std + y_mean).squeeze(0).numpy()  # (C_y,H,W)

        # predicted channels
        delta_pr = pred_p[y_off + i_delta]
        _save_img(delta_pr, out / f"s{sample:05d}_k{k}_deltaPL_pred.png", f"delta_pl_db pred (s={sample}, k={k})")
        _save_img(delta_pr - delta_lab, out / f"s{sample:05d}_k{k}_deltaPL_err.png", f"delta pred - label (dB) (s={sample}, k={k})")

        wb_from_delta_pr = fspl + delta_pr
        _save_img(wb_from_delta_pr, out / f"s{sample:05d}_k{k}_wbFromDelta_pred.png",
                  f"WB = FSPL + delta_pl (pred) (dB) (s={sample}, k={k})")
        _save_img(wb_from_delta_pr - wb_from_delta_lab, out / f"s{sample:05d}_k{k}_wbFromDelta_err.png",
                  f"(FSPL+delta) pred - label (dB) (s={sample}, k={k})")

        if wb_lab is not None:
            _save_img(wb_from_delta_pr - wb_lab, out / f"s{sample:05d}_k{k}_wbFromDelta_minus_TRUE_pred.png",
                      f"(FSPL+delta_pred) - WB_true(label) (dB) (s={sample}, k={k})")

        if ex_lab is not None:
            ex_pr = pred_p[y_off + i_ex]
            _save_img(ex_pr, out / f"s{sample:05d}_k{k}_ex_pred.png", f"Excess delay pred (ns) (s={sample}, k={k})")
            _save_img(ex_pr - ex_lab, out / f"s{sample:05d}_k{k}_ex_err.png", f"Excess delay pred - label (ns) (s={sample}, k={k})")

        if tau_lab is not None:
            tau_pr = pred_p[y_off + i_tau]
            _save_img(tau_pr, out / f"s{sample:05d}_k{k}_tauRMS_pred.png", f"Tau RMS pred (ns) (s={sample}, k={k})")
            _save_img(tau_pr - tau_lab, out / f"s{sample:05d}_k{k}_tauRMS_err.png", f"Tau RMS pred - label (ns) (s={sample}, k={k})")

        # Metrics (use WB_true mask if available)
        no_path = float(meta.get("no_path_wb_db", meta.get("no_path_wb", 199.5)))
        if wb_lab is not None:
            m = (wb_lab < no_path).astype(np.float32)
            if m.sum() > 0:
                mae = (np.abs(wb_from_delta_pr - wb_lab) * m).sum() / m.sum()
                g_pr = np.power(10.0, -wb_from_delta_pr / 10.0)
                g_lb = np.power(10.0, -wb_lab / 10.0)
                nmse = ((g_pr - g_lb) ** 2 * m).sum() / np.maximum(((g_lb ** 2) * m).sum(), 1e-12)
                nmse_db = 10.0 * np.log10(np.maximum(nmse, 1e-12))
                print(f"WB_from_delta vs WB_true (s={sample}, k={k}): MAE={mae:.3f} dB, NMSE={nmse:.4f} ({nmse_db:.2f} dB)")
        else:
            print("[warn] No wb_loss_db channel found; skipping WB_true-based metrics.")

    print("\nWrote outputs to:", out)
    print(f"Inferred shapes: N={N}, x=(N,{C_x},{H},{W}), y=(N,{C_y},{H},{W})")
    print(f"Assumptions: K={K}, y_ch={y_ch}, c_in={c_in_eff}, y_stacked={is_y_stacked}, x_stacked={is_x_stacked}")
    print(f"Using y indices: delta={i_delta}, ex={i_ex}, tau={i_tau}, wb_true={i_wb}")
    print(f"Using x per-slice indices: walls={walls_idx_eff}, electrical_distance={elec_idx_eff}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory containing x.dat/y.dat/meta.json")
    ap.add_argument("--sample", type=int, default=None, help="Sample index (default: auto-pick)")
    ap.add_argument("--k", type=int, default=0, help="Slice index k (default: 0)")
    ap.add_argument("--walls-idx", type=int, default=None, help="Per-slice channel index of binary_walls (default: 0)")
    ap.add_argument("--elec-idx", type=int, default=None, help="Per-slice channel index of electrical_distance (default: 1)")
    ap.add_argument("--out", default=None, help="Output directory (default: <run>/peek)")
    ap.add_argument("--model", default=None, help="TorchScript model path (default: <run>/model.pt if present)")
    ap.add_argument("--device", default="cpu", help="Torch device for inference (cpu, cuda:0, ...)")
    ap.add_argument("--assume-stacked", default="auto", choices=["auto", "true", "false"], help="Force stacked/unstacked interpretation")
    ap.add_argument("--dump-x", action="store_true", help="Dump raw x tensors to .npz (full + selected slice)")
    ap.add_argument("--dump-x-images", action="store_true", help="Dump one PNG per input channel for selected slice")
    ap.add_argument("--dump-x-norm", action="store_true", help="Dump kept + normalized x as fed to the model (needs norm_stats.npz)")
    args = ap.parse_args()

    main(
        run_dir=args.run,
        sample=args.sample,
        k=args.k,
        walls_idx=args.walls_idx,
        elec_idx=args.elec_idx,
        out_dir=args.out,
        model_path=args.model,
        device=args.device,
        assume_stacked=args.assume_stacked,
    )
