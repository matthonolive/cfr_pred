import json
import numpy as np
from pathlib import Path

out_dir = Path("runs/delta_3072_10int")
meta = json.loads((out_dir / "meta.json").read_text())

total_samples = meta["total_samples"]
H, W, K, y_ch = meta["H"], meta["W"], meta["K"], meta["y_ch"]
c_in = meta["c_in"]

x_mm = np.memmap(out_dir / "x.dat", dtype="float32", mode="r",
                 shape=(total_samples, c_in * K, H, W))
y_mm = np.memmap(out_dir / "y.dat", dtype="float16", mode="r",
                 shape=(total_samples, y_ch * K, H, W))

# You need to know which channel index the cost feature lands on.
# From your dataset_features: ["binary_walls", "electrical_distance", "cost", "height_cond"]
# binary_walls = 1ch, electrical_distance = 1ch, cost = 1ch (channel index 2 per slice)
# Adjust these if your channel counts differ:
cost_ch_in_slice = 2  # 0-indexed within each slice's features

no_path = 199.5
rng = np.random.default_rng(42)
idx = rng.choice(total_samples, size=min(300, total_samples), replace=False)

errs_cost = []
errs_friis = []  # delta=0 baseline (pure Friis)

for i in idx:
    x_s = np.array(x_mm[i], dtype=np.float32)
    y_s = np.array(y_mm[i], dtype=np.float32)

    for k in range(K):
        delta_tgt = y_s[k * y_ch + 0]
        wb_tgt    = y_s[k * y_ch + 3]
        cost_feat = x_s[k * c_in + cost_ch_in_slice]  # -FSPL + 10*log10(wall_trans)
        ed        = x_s[k * c_in + 1]                 # electrical_distance

        mask = wb_tgt < no_path
        if mask.sum() < 10:
            continue

        # reconstruct FSPL from electrical distance
        ed_safe = np.maximum(ed, 1e-6)
        fspl = 20 * np.log10(4 * np.pi * ed_safe)

        # COST-231 predicted delta = -(cost_feat + fspl) = wall attenuation in dB
        cost_predicted_delta = -cost_feat - fspl

        errs_cost.append(np.abs(cost_predicted_delta[mask] - delta_tgt[mask]))
        errs_friis.append(np.abs(delta_tgt[mask]))

errs_cost = np.concatenate(errs_cost)
errs_friis = np.concatenate(errs_friis)

print(f"Pure Friis (delta=0):  MAE = {errs_friis.mean():.2f} dB, "
      f"p50 = {np.median(errs_friis):.2f} dB, p90 = {np.percentile(errs_friis, 90):.2f} dB")
print(f"COST-231 feature:      MAE = {errs_cost.mean():.2f} dB, "
      f"p50 = {np.median(errs_cost):.2f} dB, p90 = {np.percentile(errs_cost, 90):.2f} dB")
print(f"Your UNet:             MAE ~ 4.28 dB (from training logs)")
print(f"Irreducible floor:     MAE ~ 3.00 dB (from diagnostic)")