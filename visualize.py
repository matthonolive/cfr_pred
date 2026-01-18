#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def infer_io_base_from_torchscript(m: torch.jit.RecursiveScriptModule):
    """
    Infer:
      - in_ch  from the first conv weight's in_channels
      - base   from the first conv weight's out_channels
      - out_ch from the final 1x1 conv weight's out_channels (heuristic: kernel=1)
    """
    conv_weights = []
    for name, p in m.named_parameters():
        if p.ndim == 4:  # conv weight [out_c, in_c, kH, kW]
            conv_weights.append((name, p))

    if not conv_weights:
        raise RuntimeError("Could not find any Conv2d weights in the TorchScript model.")

    # First conv: smallest layer by name/order tends to be early; sort by name for stability
    conv_weights.sort(key=lambda t: t[0])
    first_name, first_w = conv_weights[0]
    base = int(first_w.shape[0])
    in_ch = int(first_w.shape[1])

    # Find a 1x1 conv that maps to output channels; pick the one with kernel=1 and smallest in_c ~ base
    one_by_one = [(n, w) for (n, w) in conv_weights if int(w.shape[2]) == 1 and int(w.shape[3]) == 1]
    if not one_by_one:
        # fallback: last conv in sorted list
        out_ch = int(conv_weights[-1][1].shape[0])
    else:
        # choose the 1x1 with smallest in_channels (often base) and last-ish name
        one_by_one.sort(key=lambda t: (int(t[1].shape[1]), t[0]))
        out_ch = int(one_by_one[0][1].shape[0])

    return in_ch, base, out_ch, first_name


def add_box(ax, x, y, w, h, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return box


def arrow(ax, x0, y0, x1, y1, rad=0.0, lw=1.2, style="-|>"):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color="black",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to TorchScript model.pt")
    ap.add_argument("--out_prefix", type=str, default="unet_block", help="Output file prefix (no extension)")
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--title", type=str, default="U-Net surrogate (UNet3)")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    # Load on CPU for portability
    m = torch.jit.load(str(model_path), map_location="cpu").eval()
    in_ch, base, out_ch, first_conv_name = infer_io_base_from_torchscript(m)

    # UNet3 channel pattern (as in your training_tdl.py UNet3): base,2b,4b,8b
    b1, b2, b3, b4 = base, 2*base, 4*base, 8*base

    # Spatial resolutions for 3 pools starting from HxW
    H, W = args.H, args.W
    H2, W2 = H//2, W//2
    H4, W4 = H//4, W//4
    H8, W8 = H//8, W//8

    # Figure layout (manual coordinates in [0,1] space)
    fig = plt.figure(figsize=(11, 6.2))
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.96, args.title, fontsize=14, weight="bold", va="top")
    ax.text(0.02, 0.92,
            f"Loaded: {model_path}\n"
            f"Inferred: in_ch={in_ch}, base={base}, out_ch={out_ch}  (first conv: {first_conv_name})",
            fontsize=9, va="top")

    # Box sizes
    bw, bh = 0.18, 0.12

    # Encoder x positions
    x_enc = 0.08
    x_mid = 0.41
    x_dec = 0.70

    # Y positions by scale
    y1 = 0.70
    y2 = 0.52
    y3 = 0.34
    y4 = 0.16

    # Input + output
    inp = add_box(ax, 0.02, y1, 0.05, bh,
                  f"Input\n{H}×{W}\nC={in_ch}", fontsize=9)

    out = add_box(ax, 0.93, y1, 0.05, bh,
                  f"Output\n{H}×{W}\nC={out_ch}", fontsize=9)

    # Encoder blocks
    e1 = add_box(ax, x_enc, y1, bw, bh, f"E1: ResBlock×2\n{H}×{W}, C={b1}")
    e2 = add_box(ax, x_enc, y2, bw, bh, f"E2: ResBlock×2\n{H2}×{W2}, C={b2}")
    e3 = add_box(ax, x_enc, y3, bw, bh, f"E3: ResBlock×2\n{H4}×{W4}, C={b3}")

    # Pool labels (small)
    p1 = add_box(ax, x_enc + bw + 0.02, y1 - 0.06, 0.08, 0.06, "MaxPool\n/2", fontsize=9)
    p2 = add_box(ax, x_enc + bw + 0.02, y2 - 0.06, 0.08, 0.06, "MaxPool\n/2", fontsize=9)
    p3 = add_box(ax, x_enc + bw + 0.02, y3 - 0.06, 0.08, 0.06, "MaxPool\n/2", fontsize=9)

    # Bottleneck
    mid = add_box(ax, x_mid, y4, bw, bh, f"MID: ResBlock×2\n{H8}×{W8}, C={b4}")

    # Decoder blocks + upconvs
    u3 = add_box(ax, x_mid + bw + 0.03, y3, 0.11, 0.06, "UpConv\n×2", fontsize=9)
    d3 = add_box(ax, x_dec, y3, bw, bh, f"D3: concat + ResBlock×2\n{H4}×{W4}, C={b3}")

    u2 = add_box(ax, x_mid + bw + 0.03, y2, 0.11, 0.06, "UpConv\n×2", fontsize=9)
    d2 = add_box(ax, x_dec, y2, bw, bh, f"D2: concat + ResBlock×2\n{H2}×{W2}, C={b2}")

    u1 = add_box(ax, x_mid + bw + 0.03, y1, 0.11, 0.06, "UpConv\n×2", fontsize=9)
    d1 = add_box(ax, x_dec, y1, bw, bh, f"D1: concat + ResBlock×2\n{H}×{W}, C={b1}")

    head = add_box(ax, x_dec + bw + 0.03, y1, 0.10, 0.06, "1×1 Conv", fontsize=9)

    # Main forward arrows
    arrow(ax, 0.07, y1 + bh/2, x_enc, y1 + bh/2)
    arrow(ax, x_enc + bw, y1 + bh/2, x_enc + bw + 0.02, y1 + bh/2 - 0.03)  # to pool1 box
    arrow(ax, x_enc + bw + 0.10, y1 + bh/2 - 0.03, x_enc + bw, y2 + bh/2)  # to e2

    arrow(ax, x_enc + bw, y2 + bh/2, x_enc + bw + 0.02, y2 + bh/2 - 0.03)  # to pool2
    arrow(ax, x_enc + bw + 0.10, y2 + bh/2 - 0.03, x_enc + bw, y3 + bh/2)  # to e3

    arrow(ax, x_enc + bw, y3 + bh/2, x_enc + bw + 0.02, y3 + bh/2 - 0.03)  # to pool3
    arrow(ax, x_enc + bw + 0.10, y3 + bh/2 - 0.03, x_mid, y4 + bh/2)       # to mid

    # Decoder arrows: mid -> u3 -> d3 -> u2 -> d2 -> u1 -> d1 -> head -> out
    arrow(ax, x_mid + bw, y4 + bh/2, x_mid + bw + 0.03, y3 + 0.03)
    arrow(ax, x_mid + bw + 0.14, y3 + 0.03, x_dec, y3 + bh/2)

    arrow(ax, x_dec + bw, y3 + bh/2, x_mid + bw + 0.03, y2 + 0.03, rad=0.0)
    arrow(ax, x_mid + bw + 0.14, y2 + 0.03, x_dec, y2 + bh/2)

    arrow(ax, x_dec + bw, y2 + bh/2, x_mid + bw + 0.03, y1 + 0.03, rad=0.0)
    arrow(ax, x_mid + bw + 0.14, y1 + 0.03, x_dec, y1 + bh/2)

    arrow(ax, x_dec + bw, y1 + bh/2, x_dec + bw + 0.03, y1 + 0.03)
    arrow(ax, x_dec + bw + 0.13, y1 + 0.03, 0.93, y1 + bh/2)

    # Skip connections (encoder -> decoder) as curved arrows
    arrow(ax, x_enc + bw, y1 + bh/2, x_dec, y1 + bh/2, rad=0.20, lw=1.0)
    arrow(ax, x_enc + bw, y2 + bh/2, x_dec, y2 + bh/2, rad=0.20, lw=1.0)
    arrow(ax, x_enc + bw, y3 + bh/2, x_dec, y3 + bh/2, rad=0.20, lw=1.0)

    ax.text(0.52, 0.06, "Skip connections: concatenation (encoder feature maps → decoder)", fontsize=9, ha="center")

    out_prefix = Path(args.out_prefix)
    fig.tight_layout()

    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".svg", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", dpi=200, bbox_inches="tight")
    print(f"[ok] wrote: {out_prefix}.pdf / .svg / .png")


if __name__ == "__main__":
    main()
