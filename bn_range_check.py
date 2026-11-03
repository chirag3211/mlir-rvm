#!/usr/bin/env python3
"""Reports the range of the folded BatchNorm constants (alpha = gamma/sqrt(var+eps), beta' = beta - mean*alpha) of a
model, i.e. what the fp16 conv epilogue has to represent.   usage: python3 bn_range_check.py resnetv2_50.mlir"""
import sys
import numpy as np
sys.path.insert(0, ".")
import lower_to_c as L
from torch_mlir import ir

text = open(sys.argv[1]).read()
blobs = L.parse_resources(text)
with ir.Context() as ctx, ir.Location.unknown():
    ctx.allow_unregistered_dialects = True
    mod = ir.Module.parse(text)
    print("folded BN chains:", L.fuse_batchnorm(mod, blobs, "main"))
    fn = [f for f in mod.body.operations if f.operation.name == "func.func"][0].operation
    rows = []
    for op in fn.regions[0].blocks[0].operations:
        op = op.operation
        if L._is_bn_relu_generic(op):
            a = np.asarray(L._const_eval(op.operands[1], blobs), np.float64).reshape(-1)
            b = np.asarray(L._const_eval(op.operands[2], blobs), np.float64).reshape(-1)
            rows.append((len(rows), a.size, np.abs(a).max(), np.abs(b).max(), int((np.abs(a) > 1e3).sum()),
                         int((np.abs(b) > 6e4).sum()), int(np.isinf(a.astype(np.float16)).sum() + np.isinf(b.astype(np.float16)).sum()),
                         float(np.abs(np.float16(a).astype(np.float64) - a).max() / max(np.abs(a).max(), 1e-30))))
    print("layer   C   max|alpha|   max|beta'|  #|alpha|>1e3  #|beta'|>6e4  #fp16-inf")
    for r in rows:
        print("%5d %4d %12.4g %12.4g %12d %12d %10d" % r[:7])
    print("TOTAL fp16 overflows:", sum(r[6] for r in rows), " worst max|alpha|: %.4g  worst max|beta'|: %.4g" % (max(r[2] for r in rows), max(r[3] for r in rows)))
