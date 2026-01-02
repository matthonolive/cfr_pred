import json
from pathlib import Path
import numpy as np

def load_meta(run: Path):
    return json.loads((run / "meta.json").read_text())

def assert_same(meta0, meta, keys):
    for k in keys:
        if meta0.get(k) != meta.get(k):
            raise ValueError(f"meta mismatch on '{k}': {meta0.get(k)} != {meta.get(k)}")

def file_n_samples(path: Path, dtype: np.dtype, ch: int, H: int, W: int) -> int:
    nbytes = path.stat().st_size
    bps = int(dtype.itemsize) * int(ch) * int(H) * int(W)
    if bps <= 0:
        raise ValueError("bad bytes-per-sample")
    n = nbytes // bps
    rem = nbytes - n * bps
    if rem != 0:
        print(f"WARNING: {path} has {rem} trailing bytes (not a whole sample)")
    return int(n)

def main(out_run="runs/tau_rms_3072_200k_merged",
         runs=("runs/temp_mer_1", "runs/3072_200k_taupaths4"),
         chunk=128):

    runs = [Path(r) for r in runs]
    metas = [load_meta(r) for r in runs]
    m0 = metas[0]

    # These must match across shards
    keys = ["H", "W", "K", "num_tx", "c_in", "y_ch", "x_dtype", "y_dtype"]
    for mi in metas[1:]:
        assert_same(m0, mi, keys)

    H, W = int(m0["H"]), int(m0["W"])
    K = int(m0["K"])
    c_in = int(m0["c_in"])     # per-slice feature channels
    y_ch = int(m0["y_ch"])     # per-slice label channels (3)
    C = K * c_in               # stacked input channels
    Y = K * y_ch               # stacked output channels

    x_dtype = np.dtype(m0["x_dtype"])
    y_dtype = np.dtype(m0["y_dtype"])

    # Derive actual sample counts from FILE SIZES (more trustworthy than meta)
    ns_list = []
    for r in runs:
        nx = file_n_samples(r / "x.dat", x_dtype, C, H, W)
        ny = file_n_samples(r / "y.dat", y_dtype, Y, H, W)
        if nx != ny:
            raise ValueError(f"{r}: x/y sample mismatch: nx={nx}, ny={ny}")
        ns_list.append(nx)

    total_samples = int(sum(ns_list))
    print("Merging (file-derived) samples:", ns_list, "=> total", total_samples)

    out_run = Path(out_run)
    out_run.mkdir(parents=True, exist_ok=True)

    x_out = np.memmap(out_run / "x.dat", dtype=x_dtype, mode="w+", shape=(total_samples, C, H, W))
    y_out = np.memmap(out_run / "y.dat", dtype=y_dtype, mode="w+", shape=(total_samples, Y, H, W))

    # Write merged meta (make it explicit this is stacked)
    meta_out = dict(m0)
    meta_out["total_samples"] = total_samples
    meta_out["merged_from"] = [str(r) for r in runs]
    meta_out["x_ch_total"] = C
    meta_out["y_ch_total"] = Y
    meta_out["format"] = "2p5d_stacked"
    (out_run / "meta.json").write_text(json.dumps(meta_out, indent=2))

    # Copy shard-by-shard
    offset = 0
    for r, n in zip(runs, ns_list):
        x_in = np.memmap(r / "x.dat", dtype=x_dtype, mode="r", shape=(n, C, H, W))
        y_in = np.memmap(r / "y.dat", dtype=y_dtype, mode="r", shape=(n, Y, H, W))

        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            x_out[offset + i0: offset + i1] = x_in[i0:i1]
            y_out[offset + i0: offset + i1] = y_in[i0:i1]

        offset += n
        x_out.flush(); y_out.flush()
        print("Copied", r, "offset now", offset)

    print("Done. Wrote:", out_run)

if __name__ == "__main__":
    main()