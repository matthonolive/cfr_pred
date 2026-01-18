#!/usr/bin/env python3
"""
Export a TorchScript model.pt to ONNX for visualization (e.g., in Netron).

Usage:
  python export_unet_onnx.py --model model.pt --out unet.onnx --in_ch 4 --K 4 --H 64 --W 64
"""

import argparse
import sys
from pathlib import Path

import torch


def try_forward(model, device, c, h, w):
    x = torch.randn(1, c, h, w, device=device)
    with torch.no_grad():
        y = model(x)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to TorchScript model.pt")
    ap.add_argument("--out", type=str, default="unet.onnx", help="Output ONNX path")
    ap.add_argument("--in_ch", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--dynamic", action="store_true", help="Enable dynamic batch/H/W axes in ONNX")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[err] model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    print(f"[info] loading TorchScript: {model_path}")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    # Try channel conventions:
    #  - some setups expect C=in_ch
    #  - others expect C=in_ch*K (height-slices stacked as channels)
    candidates = [args.in_ch, args.in_ch * args.K]
    tried = []

    x = y = None
    used_c = None
    for c in candidates:
        try:
            tried.append(c)
            x, y = try_forward(model, device, c, args.H, args.W)
            used_c = c
            break
        except Exception as e:
            print(f"[warn] forward failed for input C={c}: {type(e).__name__}: {e}")

    if used_c is None:
        print(f"[err] forward failed for all candidates {tried}. "
              f"Your model might expect different input channels or shapes.", file=sys.stderr)
        sys.exit(2)

    print(f"[ok] forward worked with input shape: {tuple(x.shape)}")
    # y might be a Tensor or a tuple/list of Tensors depending on scripting
    if isinstance(y, torch.Tensor):
        print(f"[ok] output shape: {tuple(y.shape)}")
    elif isinstance(y, (tuple, list)):
        print("[ok] output is a tuple/list:")
        for i, yi in enumerate(y):
            if isinstance(yi, torch.Tensor):
                print(f"  y[{i}] shape: {tuple(yi.shape)}")
            else:
                print(f"  y[{i}] type: {type(yi)}")
    else:
        print(f"[warn] output type: {type(y)} (still exportable sometimes)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic:
        # Works best if output is a single tensor
        dynamic_axes = {"x": {0: "batch", 2: "H", 3: "W"}}
        if isinstance(y, torch.Tensor):
            dynamic_axes["y"] = {0: "batch", 2: "H", 3: "W"}

    print(f"[info] exporting to ONNX: {out_path} (opset={args.opset})")
    torch.onnx.export(
        model,
        x,
        str(out_path),
        opset_version=18,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes=dynamic_axes,  # note: dynamic_axes is for dynamo=False
        dynamo=False,               # <-- IMPORTANT
    )
    print("[ok] exported ONNX.")

    # Optional: validate ONNX if package is available
    try:
        import onnx  # type: ignore
        m = onnx.load(str(out_path))
        onnx.checker.check_model(m)
        print("[ok] onnx.checker: model is valid.")
    except ImportError:
        print("[info] onnx not installed; skipping onnx.checker. (pip install onnx)")
    except Exception as e:
        print(f"[warn] onnx.checker failed: {type(e).__name__}: {e}")

    print("\nNext:")
    print(f"  - Open {out_path} in Netron to view the graph.")
    print("  - If you want a layer-by-layer table with shapes, use torchinfo.summary(model, input_data=x).")


if __name__ == "__main__":
    main()
