import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import torch

# ── Config ──────────────────────────────────────────────────────────────
out_dir    = Path("runs/delta_3072_10int_merged")
gauss_sigma = 2.0                    # Gaussian blur σ (pixels)
sample_idx  = 0                      # which sample to visualise
k_slice     = 0                      # which height slice (0..K-1)
no_path     = 199.5                  # sentinel for "no propagation path"

# ── Load metadata & memmaps ─────────────────────────────────────────────
meta = json.loads((out_dir / "meta.json").read_text())
total_samples = meta["total_samples"]
H, W     = meta["H"], meta["W"]
K        = meta["K"]
y_ch     = meta["y_ch"]
c_in     = meta["c_in"]
keep_idx = np.array(meta["keep_idx"], dtype=np.int64)

x_mm = np.memmap(out_dir / "x.dat", dtype="float32", mode="r",
                 shape=(total_samples, c_in * K, H, W))
y_mm = np.memmap(out_dir / "y.dat", dtype="float16", mode="r",
                 shape=(total_samples, y_ch * K, H, W))

# ── Load normalisation stats ────────────────────────────────────────────
stats = np.load(out_dir / "norm_stats.npz")
x_mean = stats["x_mean"]
x_std  = stats["x_std"]
y_mean = stats["y_mean"]
y_std  = stats["y_std"]

if x_mean.ndim == 1: x_mean = x_mean[:, None, None]
if x_std.ndim  == 1: x_std  = x_std[:, None, None]
if y_mean.ndim == 1: y_mean = y_mean[:, None, None]
if y_std.ndim  == 1: y_std  = y_std[:, None, None]

# ── Load the TorchScript U-Net ──────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load(str(out_dir / "model.pt"), map_location=device)
model.eval()

# ── Prepare the sample ──────────────────────────────────────────────────
i = sample_idx

# --- label (physical space) ---
y_phys = np.array(y_mm[i], dtype=np.float32)          # (y_ch*K, H, W)
delta_ch = k_slice * y_ch + 0
wb_ch    = k_slice * y_ch + 3

delta_label = y_phys[delta_ch]                          # (H, W)
wb_label    = y_phys[wb_ch]
mask = wb_label < no_path                               # True where path exists

# --- Gaussian-blurred label ---
delta_blurred = gaussian_filter(delta_label, sigma=gauss_sigma).astype(np.float32)

# --- U-Net prediction (forward pass) ---
x_full = np.array(x_mm[i], dtype=np.float32)
x_kept = x_full[keep_idx]
x_norm = (x_kept - x_mean) / x_std

x_t = torch.from_numpy(x_norm[None]).to(device)
with torch.no_grad():
    pred_norm = model(x_t).cpu().numpy()[0]              # (y_ch*K, H, W)

# unnormalise prediction → physical units
ys = y_std.reshape(-1, 1, 1) if y_std.ndim != 3 else y_std
ym = y_mean.reshape(-1, 1, 1) if y_mean.ndim != 3 else y_mean
pred_phys = pred_norm * ys + ym

delta_pred = pred_phys[delta_ch]

# --- error map: prediction minus blurred label ---
error_map = delta_pred - delta_blurred

# mask out no-path pixels for display
delta_blurred_disp = np.where(mask, delta_blurred, np.nan)
delta_pred_disp    = np.where(mask, delta_pred,    np.nan)
error_disp         = np.where(mask, error_map,     np.nan)

# ── Print summary stats on the error (on-path only) ────────────────────
err_valid = error_map[mask]
print(f"Sample {i}, k={k_slice}, Gaussian σ={gauss_sigma}")
print(f"  on-path pixels : {mask.sum()}")
print(f"  error  mean    : {err_valid.mean():.3f} dB")
print(f"  error  std     : {err_valid.std():.3f} dB")
print(f"  error  MAE     : {np.mean(np.abs(err_valid)):.3f} dB")
print(f"  error  p90 abs : {np.percentile(np.abs(err_valid), 90):.3f} dB")

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].imshow(delta_blurred_disp, cmap="inferno", origin="upper")
axes[0].set_title(f"Gaussian-blurred Δ (dB)\nσ={gauss_sigma}")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(delta_pred_disp, cmap="inferno", origin="upper")
axes[1].set_title(f"U-Net predicted Δ (dB)")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

vabs = max(abs(np.nanmin(error_disp)), abs(np.nanmax(error_disp)))
im2 = axes[2].imshow(error_disp, cmap="RdBu_r", origin="upper",
                       vmin=-vabs, vmax=vabs)
axes[2].set_title(f"Error: pred − blurred (dB)\nMAE={np.mean(np.abs(err_valid)):.2f} dB")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.tight_layout()
fig.savefig(out_dir / "gaussian_vs_unet.png", dpi=200)
plt.show()
print(f"Saved → {out_dir / 'gaussian_vs_unet.png'}")