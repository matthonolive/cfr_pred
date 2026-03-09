import json
import numpy as np
from scipy.ndimage import median_filter
from pathlib import Path

out_dir = Path("runs/delta_3072_10int")
meta = json.loads((out_dir / "meta.json").read_text())

total_samples = meta["total_samples"]
H, W, K, y_ch = meta["H"], meta["W"], meta["K"], meta["y_ch"]

y_mm = np.memmap(out_dir / "y.dat", dtype="float16", mode="r",
                 shape=(total_samples, y_ch * K, H, W))

rng = np.random.default_rng(42)
idx = rng.choice(total_samples, size=min(200, total_samples), replace=False)

no_path = 199.5

for size in [1, 3, 5, 7, 9]:
    residuals = []
    for i in idx:
        y_s = np.array(y_mm[i], dtype=np.float32)
        for k in range(K):
            delta = y_s[k * y_ch + 0]          # delta channel
            wb    = y_s[k * y_ch + 3]          # wb channel for mask
            mask  = wb < no_path

            if mask.sum() < 10:
                continue

            smoothed = median_filter(delta, size=size)
            res = (delta - smoothed)[mask]
            residuals.append(res)

    residuals = np.concatenate(residuals)
    print(f"median {size}x{size}: residual std = {residuals.std():.2f} dB, "
          f"mean abs = {np.mean(np.abs(residuals)):.2f} dB, "
          f"p90 = {np.percentile(np.abs(residuals), 90):.2f} dB")