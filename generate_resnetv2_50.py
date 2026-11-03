"""
generate_resnetv2_50.py -- export the PRE-ACTIVATION ResNet-50 v2 (timm `resnetv2_50`) to linalg-on-tensors MLIR,
plus input and PyTorch reference logits, for the rvm pipeline.

  pip install timm            # (torch + torch_mlir are already installed for generate_resnet50.py)
  python3 generate_resnetv2_50.py [--weights resnetv2_50.a1h_in1k] [--input gray|random] [--prefix preact]

Writes (in the current directory, never touching your torchvision files):
  resnetv2_50.mlir   input_<prefix>.bin   ref_logits_<prefix>.bin
Architecture: BN -> ReLU -> conv ordering (He et al., "Identity Mappings"), BatchNorm (not GroupNorm), plain convs.
"""
import argparse
import collections
import sys

import numpy as np
import torch
import timm
from torch_mlir import fx

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="resnetv2_50.a1h_in1k", help="timm model name (with pretrained tag)")
ap.add_argument("--input", default="gray", choices=["gray", "random"])
ap.add_argument("--prefix", default="preact")
args = ap.parse_args()

print(f"Loading timm {args.weights} ...")
model = timm.create_model(args.weights, pretrained=True).eval()

# ---- sanity: this must be the BatchNorm / plain-conv flavour, otherwise the pipeline would need new ops
names = collections.Counter(type(m).__name__ for m in model.modules())
print("module census:", dict(names))
bad = [n for n in names if n in ("GroupNorm", "GroupNormAct", "StdConv2d", "ScaledStdConv2d", "FilterResponseNorm2d")]
if bad:
    sys.exit(f"ERROR: {bad} present -- that is the BiT (GroupNorm + weight-standardisation) variant, not the "
             f"BatchNorm pre-activation ResNet-50 v2. Pick a different --weights (e.g. resnetv2_50.a1h_in1k).")
print("stem :", model.stem)
b0 = model.stages[0].blocks[0]
print("first block submodule order:", [n for n, _ in b0.named_children()])
print("head :", type(model.head).__name__, "| fc =", model.head.fc)

# ---- deterministic input
if args.input == "gray":
    x = torch.ones(1, 3, 224, 224) * 0.5
else:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, 224, 224, generator=g)

with torch.no_grad():
    y = model(x)
assert y.shape == (1, 1000), y.shape
x.numpy().astype(np.float32).tofile(f"input_{args.prefix}.bin")
y.numpy().astype(np.float32).tofile(f"ref_logits_{args.prefix}.bin")
top5 = torch.topk(y[0], 5).indices.tolist()
print(f"-> PYTORCH EXPECTED ARGMAX CLASS: {int(torch.argmax(y))}   top-5: {top5}")

print("Exporting to MLIR linalg-on-tensors ...")
with torch.no_grad():
    module = fx.export_and_import(model, x, output_type="linalg-on-tensors")
with open("resnetv2_50.mlir", "w") as f:
    f.write(str(module))
print("Saved resnetv2_50.mlir, input_%s.bin, ref_logits_%s.bin" % (args.prefix, args.prefix))
