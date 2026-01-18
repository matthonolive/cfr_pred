#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import shutil
import torch


def infer_io_base(m):
    convs = [(n, p) for (n, p) in m.named_parameters() if p.ndim == 4]  # [out_c,in_c,kH,kW]
    if not convs:
        raise RuntimeError("No Conv2d weights found in model parameters.")
    convs.sort(key=lambda t: t[0])

    first_name, first_w = convs[0]
    in_ch = int(first_w.shape[1])
    base = int(first_w.shape[0])

    one_by_one = [(n, w) for (n, w) in convs if int(w.shape[2]) == 1 and int(w.shape[3]) == 1]
    if one_by_one:
        one_by_one.sort(key=lambda t: (int(t[1].shape[1]), t[0]))
        out_ch = int(one_by_one[0][1].shape[0])
    else:
        out_ch = int(convs[-1][1].shape[0])

    return in_ch, base, out_ch, first_name


def make_dot(in_ch, base, out_ch, H, W, title):
    b1, b2, b3, b4 = base, 2 * base, 4 * base, 8 * base
    H2, W2 = H // 2, W // 2
    H4, W4 = H // 4, W // 4
    H8, W8 = H // 8, W // 8

    def note(name, lines):
        # keep labels compact for thesis
        return f"{name}\\n" + "\\n".join(lines)

    # colors (tweak if you like)
    c_inout = "#FEF3C7"   # warm
    c_enc   = "#DBEAFE"   # blue-ish
    c_mid   = "#F1F5F9"   # gray
    c_dec   = "#DCFCE7"   # green-ish
    c_head  = "#FFE4E6"   # light pink

    dot = f"""digraph UNet3 {{
  graph [
    rankdir=TB,
    splines=ortho,
    nodesep=0.35,
    ranksep=0.30,
    pad=0.18,
    fontsize=16,
    labelloc="t",
    label="{title}",
    size="6,4!",
    ratio=fill
  ];

  node [
    shape=box,
    style="rounded,filled",
    color="#111827",
    fontname="Helvetica",
    fontsize=11,
    margin="0.10,0.06"
  ];

  edge [
    color="#111827",
    arrowsize=0.8,
    fontname="Helvetica",
    fontsize=10
  ];

  // --- nodes ---
  In     [fillcolor="{c_inout}", label="{note('Input', [f'{H}×{W}', f'C={in_ch}'])}"];
  E1     [fillcolor="{c_enc}",   label="{note('E1',  [f'ResBlock×2', f'{H}×{W}',  f'C={b1}'])}"];
  E2     [fillcolor="{c_enc}",   label="{note('E2',  [f'ResBlock×2', f'{H2}×{W2}', f'C={b2}'])}"];
  E3     [fillcolor="{c_enc}",   label="{note('E3',  [f'ResBlock×2', f'{H4}×{W4}', f'C={b3}'])}"];
  MID    [fillcolor="{c_mid}",   label="{note('MID', [f'ResBlock×2', f'{H8}×{W8}', f'C={b4}'])}"];
  D3     [fillcolor="{c_dec}",   label="{note('D3',  [f'Up + concat', 'ResBlock×2', f'{H4}×{W4}', f'C={b3}'])}"];
  D2     [fillcolor="{c_dec}",   label="{note('D2',  [f'Up + concat', 'ResBlock×2', f'{H2}×{W2}', f'C={b2}'])}"];
  D1     [fillcolor="{c_dec}",   label="{note('D1',  [f'Up + concat', 'ResBlock×2', f'{H}×{W}',  f'C={b1}'])}"];
  Head   [fillcolor="{c_head}",  label="{note('1×1 Conv', [f'{H}×{W}', f'C={out_ch}'])}"];
  Output [fillcolor="{c_inout}", label="{note('Output', [f'{H}×{W}', f'C={out_ch}'])}"];

  // --- U-shape layout constraints (ranks = rows) ---
  {{ rank=same; In; }}
  {{ rank=same; E1; D1; Head; Output; }}
  {{ rank=same; E2; D2; }}
  {{ rank=same; E3; D3; }}
  {{ rank=same; MID; }}

  // enforce left-to-right ordering within each rank
  E1 -> D1     [style=invis, weight=10];
  D1 -> Head   [style=invis, weight=10];
  Head -> Output [style=invis, weight=10];

  E2 -> D2     [style=invis, weight=10];
  E3 -> D3     [style=invis, weight=10];

  // --- main forward path (down left, then up right) ---
  In  -> E1;
  E1  -> E2 [label="↓ /2"];
  E2  -> E3 [label="↓ /2"];
  E3  -> MID [label="↓ /2"];

  // these edges go upward (Graphviz will route them cleanly with splines=ortho)
  MID -> D3 [label="↑ ×2"];
  D3  -> D2 [label="↑ ×2"];
  D2  -> D1 [label="↑ ×2"];

  D1  -> Head -> Output;

  // --- skip connections (concat) ---
  E1 -> D1 [style=dashed, constraint=false, label="skip"];
  E2 -> D2 [style=dashed, constraint=false, label="skip"];
  E3 -> D3 [style=dashed, constraint=false, label="skip"];
}}
"""
    # tiny helper: keep labels short but let us visually emphasize one line if you want
    # (Graphviz doesn't do bold in plain labels reliably, so we just keep it normal text)
    return dot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to TorchScript model.pt")
    ap.add_argument("--out_prefix", default="unet_block_u", help="Output prefix (no extension)")
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--title", default="U-Net surrogate (UNet3)")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    m = torch.jit.load(str(model_path), map_location="cpu").eval()
    in_ch, base, out_ch, first_conv = infer_io_base(m)
    print(f"[info] inferred in_ch={in_ch}, base={base}, out_ch={out_ch} (first conv: {first_conv})")

    dot_text = make_dot(in_ch, base, out_ch, args.H, args.W, args.title)

    out_prefix = Path(args.out_prefix)
    dot_path = out_prefix.with_suffix(".dot")
    dot_path.write_text(dot_text)
    print(f"[ok] wrote {dot_path}")

    dot_bin = shutil.which("dot")
    if not dot_bin:
        print("[err] Graphviz 'dot' not found. Install: sudo apt-get install -y graphviz")
        return

    subprocess.run([dot_bin, "-Tsvg", str(dot_path), "-o", str(out_prefix.with_suffix(".svg"))], check=True)
    subprocess.run([dot_bin, "-Tpdf", str(dot_path), "-o", str(out_prefix.with_suffix(".pdf"))], check=True)
    subprocess.run([dot_bin, "-Tpng", str(dot_path), "-o", str(out_prefix.with_suffix(".png"))], check=True)
    print(f"[ok] rendered {out_prefix}.svg / .pdf / .png")


if __name__ == "__main__":
    main()
