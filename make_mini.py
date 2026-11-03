#!/usr/bin/env python3
"""Builds mini.mlir (+ input.bin, ref_logits.bin) mimicking torch-mlir's linalg-on-tensors ResNet ops."""
import sys
import numpy as np

rng = np.random.default_rng(0)
f32 = np.float32
res = {}          # resource name -> array
lines = []
cnt = [0]


def v(p="v"):
    cnt[0] += 1
    return f"%{p}{cnt[0]}"


def T(shape):
    return "tensor<" + "x".join(str(s) for s in shape) + "xf32>"


def const(name, arr):
    res[name] = arr.astype(f32)
    r = v("w")
    lines.append(f"  {r} = arith.constant dense_resource<{name}> : {T(arr.shape)}")
    return r


def conv(x, xs, w, ws, stride, tag):
    n, c, h, wd = xs
    f, _, kh, kw = ws
    oh, ow = (h - kh) // stride + 1, (wd - kw) // stride + 1
    os_ = (n, f, oh, ow)
    z = v("z"); e = v("e"); fl = v("f"); o = v("o")
    lines.append(f"  {z} = arith.constant 0.0 : f32")
    lines.append(f"  {e} = tensor.empty() : {T(os_)}")
    lines.append(f"  {fl} = linalg.fill ins({z} : f32) outs({e} : {T(os_)}) -> {T(os_)}")
    lines.append(f"  {o} = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} "
                 f"ins({x}, {w} : {T(xs)}, {T(ws)}) outs({fl} : {T(os_)}) -> {T(os_)}")
    return o, os_


def pad(x, xs, p, val, tag):
    n, c, h, w = xs
    os_ = (n, c, h + 2 * p, w + 2 * p)
    cv = v("c"); o = v("p")
    lines.append(f"  {cv} = arith.constant {val} : f32")
    lines.append(f"  {o} = tensor.pad {x} low[0, 0, {p}, {p}] high[0, 0, {p}, {p}] {{")
    lines.append("  ^bb0(%a: index, %b: index, %c: index, %d: index):")
    lines.append(f"    tensor.yield {cv} : f32")
    lines.append(f"  }} : {T(xs)} to {T(os_)}")
    return o, os_


def bn(x, xs, name):
    c = xs[1]
    ps = []
    for k, nm in enumerate(["g", "b", "m", "v"]):
        arr = rng.uniform(0.5, 1.5, c) if nm in ("g", "v") else rng.uniform(-0.5, 0.5, c)
        res_name = f"{name}_{nm}"
        r = const(res_name, arr)
        e = v("x")
        lines.append(f"  {e} = tensor.expand_shape {r} [[0, 1, 2, 3]] output_shape [1, {c}, 1, 1] : {T((c,))} into {T((1, c, 1, 1))}")
        ps.append((e, arr.astype(f32)))
    (g, ga), (b, ba), (m, ma), (vv, va) = ps
    em = v("e"); o = v("o")
    lines.append(f"  {em} = tensor.empty() : {T(xs)}")
    lines.append(f"  {o} = linalg.generic {{indexing_maps = [#id4, #bc4, #bc4, #bc4, #bc4, #id4], iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]}} "
                 f"ins({x}, {m}, {vv}, {g}, {b} : {T(xs)}, {T((1,c,1,1))}, {T((1,c,1,1))}, {T((1,c,1,1))}, {T((1,c,1,1))}) outs({em} : {T(xs)}) {{")
    lines.append("  ^bb0(%in: f32, %m: f32, %v: f32, %g: f32, %b: f32, %out: f32):")
    lines.append("    %eps64 = arith.constant 1.000000e-05 : f64")
    lines.append("    %eps = arith.truncf %eps64 : f64 to f32")
    lines.append("    %ve = arith.addf %v, %eps : f32")
    lines.append("    %rs = math.rsqrt %ve : f32")
    lines.append("    %d = arith.subf %in, %m : f32")
    lines.append("    %n = arith.mulf %d, %rs : f32")
    lines.append("    %s = arith.mulf %n, %g : f32")
    lines.append("    %r = arith.addf %s, %b : f32")
    lines.append("    linalg.yield %r : f32")
    lines.append(f"  }} -> {T(xs)}")
    return o, (ga, ba, ma, va)


def relu(x, xs):
    em = v("e"); o = v("o")
    lines.append(f"  {em} = tensor.empty() : {T(xs)}")
    lines.append(f"  {o} = linalg.generic {{indexing_maps = [#id4, #id4], iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]}} "
                 f"ins({x} : {T(xs)}) outs({em} : {T(xs)}) {{")
    lines.append("  ^bb0(%in: f32, %out: f32):")
    lines.append("    %z = arith.constant 0.000000e+00 : f32")
    lines.append("    %c = arith.cmpf ugt, %in, %z : f32")
    lines.append("    %s = arith.select %c, %in, %z : f32")
    lines.append("    linalg.yield %s : f32")
    lines.append(f"  }} -> {T(xs)}")
    return o


def add(x, y, xs):
    em = v("e"); o = v("o")
    lines.append(f"  {em} = tensor.empty() : {T(xs)}")
    lines.append(f"  {o} = linalg.generic {{indexing_maps = [#id4, #id4, #id4], iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]}} "
                 f"ins({x}, {y} : {T(xs)}, {T(xs)}) outs({em} : {T(xs)}) {{")
    lines.append("  ^bb0(%a: f32, %b: f32, %out: f32):")
    lines.append("    %s = arith.addf %a, %b : f32")
    lines.append("    linalg.yield %s : f32")
    lines.append(f"  }} -> {T(xs)}")
    return o


# numpy references -------------------------------------------------------------
def np_conv(x, w, stride):
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    oh, ow = (h - kh) // stride + 1, (wd - kw) // stride + 1
    out = np.zeros((n, f, oh, ow), f32)
    for i in range(oh):
        for j in range(ow):
            patch = x[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return out


def np_bn(x, p):
    g, b, m, vv = (q.reshape(1, -1, 1, 1) for q in p)
    return ((x - m) / np.sqrt(vv + f32(1e-5)) * g + b).astype(f32)


def np_pad(x, p, val):
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), constant_values=val)


def np_maxpool(x, k, s):
    n, c, h, w = x.shape
    oh, ow = (h - k) // s + 1, (w - k) // s + 1
    out = np.zeros((n, c, oh, ow), f32)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i * s:i * s + k, j * s:j * s + k].max(axis=(2, 3))
    return out


def main(outdir):
    x_in = rng.standard_normal((1, 3, 8, 8)).astype(f32)
    lines.append("func.func @main(%x: tensor<1x3x8x8xf32>) -> tensor<1x5xf32> {")
    # stem: pad, conv3x3, bn, relu
    p1, s = pad("%x", (1, 3, 8, 8), 1, "0.0", "stem")
    w1 = rng.standard_normal((4, 3, 3, 3)).astype(f32) * 0.3
    w1r = const("w_conv1", w1)
    c1, s1 = conv(p1, s, w1r, w1.shape, 1, "c1")
    b1, bp1 = bn(c1, s1, "bn1")
    r1 = relu(b1, s1)
    # maxpool 3x3/2 with -inf pad
    pp, ps = pad(r1, s1, 1, "0xFF800000", "mp")
    win = v("win"); mpe = v("e"); mpf = v("f"); ninf = v("c"); mp = v("o")
    mps = (1, 4, 4, 4)
    lines.append(f"  {ninf} = arith.constant 0xFF800000 : f32")
    lines.append(f"  {win} = tensor.empty() : tensor<3x3xf32>")
    lines.append(f"  {mpe} = tensor.empty() : {T(mps)}")
    lines.append(f"  {mpf} = linalg.fill ins({ninf} : f32) outs({mpe} : {T(mps)}) -> {T(mps)}")
    lines.append(f"  {mp} = linalg.pooling_nchw_max {{dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}} "
                 f"ins({pp}, {win} : {T(ps)}, tensor<3x3xf32>) outs({mpf} : {T(mps)}) -> {T(mps)}")
    # residual-ish block: 1x1 conv, bn, add skip, relu
    w2 = rng.standard_normal((4, 4, 1, 1)).astype(f32) * 0.3
    w2r = const("w_conv2", w2)
    c2, s2 = conv(mp, mps, w2r, w2.shape, 1, "c2")
    b2, bp2 = bn(c2, s2, "bn2")
    a = add(b2, mp, s2)
    r2 = relu(a, s2)
    # downsample 1x1 stride 2 conv + relu
    w3 = rng.standard_normal((6, 4, 1, 1)).astype(f32) * 0.3
    w3r = const("w_conv3", w3)
    c3, s3 = conv(r2, s2, w3r, w3.shape, 2, "c3")
    r3 = relu(c3, s3)
    # global avg pool as generic reduction, then divf, collapse
    z = v("z"); ge = v("e"); gf = v("f"); gs = v("o")
    gshape = (1, 6, 1, 1)
    lines.append(f"  {z} = arith.constant 0.0 : f32")
    lines.append(f"  {ge} = tensor.empty() : {T(gshape)}")
    lines.append(f"  {gf} = linalg.fill ins({z} : f32) outs({ge} : {T(gshape)}) -> {T(gshape)}")
    lines.append(f"  {gs} = linalg.generic {{indexing_maps = [#id4, #red4], iterator_types = [\"parallel\", \"parallel\", \"reduction\", \"reduction\"]}} "
                 f"ins({r3} : {T(s3)}) outs({gf} : {T(gshape)}) {{")
    lines.append("  ^bb0(%in: f32, %out: f32):")
    lines.append("    %s = arith.addf %in, %out : f32")
    lines.append("    linalg.yield %s : f32")
    lines.append(f"  }} -> {T(gshape)}")
    gde = v("e"); gd = v("o")
    lines.append(f"  {gde} = tensor.empty() : {T(gshape)}")
    lines.append(f"  {gd} = linalg.generic {{indexing_maps = [#id4, #id4], iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]}} "
                 f"ins({gs} : {T(gshape)}) outs({gde} : {T(gshape)}) {{")
    lines.append("  ^bb0(%in: f32, %out: f32):")
    lines.append("    %c4 = arith.constant 4.000000e+00 : f32")
    lines.append("    %q = arith.divf %in, %c4 : f32")
    lines.append("    linalg.yield %q : f32")
    lines.append(f"  }} -> {T(gshape)}")
    fl = v("o")
    lines.append(f"  {fl} = tensor.collapse_shape {gd} [[0], [1, 2, 3]] : {T(gshape)} into {T((1, 6))}")
    # fc: transpose weight, matmul, bias (one splat-constant bias to test dense<> globals)
    wfc = rng.standard_normal((5, 6)).astype(f32) * 0.5
    wfr = const("w_fc", wfc)
    te = v("e"); tr = v("o")
    lines.append(f"  {te} = tensor.empty() : {T((6, 5))}")
    lines.append(f"  {tr} = linalg.transpose ins({wfr} : {T((5, 6))}) outs({te} : {T((6, 5))}) permutation = [1, 0]")
    mz = v("z"); me = v("e"); mf = v("f"); mm = v("o")
    lines.append(f"  {mz} = arith.constant 0.0 : f32")
    lines.append(f"  {me} = tensor.empty() : {T((1, 5))}")
    lines.append(f"  {mf} = linalg.fill ins({mz} : f32) outs({me} : {T((1, 5))}) -> {T((1, 5))}")
    lines.append(f"  {mm} = linalg.matmul ins({fl}, {tr} : {T((1, 6))}, {T((6, 5))}) outs({mf} : {T((1, 5))}) -> {T((1, 5))}")
    bias = v("w"); be = v("e"); bo = v("o")
    lines.append(f"  {bias} = arith.constant dense<2.000000e+00> : {T((5,))}")
    lines.append(f"  {be} = tensor.empty() : {T((1, 5))}")
    lines.append(f"  {bo} = linalg.generic {{indexing_maps = [#id2, #bc2, #id2], iterator_types = [\"parallel\", \"parallel\"]}} "
                 f"ins({mm}, {bias} : {T((1, 5))}, {T((5,))}) outs({be} : {T((1, 5))}) {{")
    lines.append("  ^bb0(%a: f32, %b: f32, %out: f32):")
    lines.append("    %s = arith.addf %a, %b : f32")
    lines.append("    linalg.yield %s : f32")
    lines.append(f"  }} -> {T((1, 5))}")
    lines.append(f"  return {bo} : {T((1, 5))}")
    lines.append("}")

    # ---- numpy reference
    x = np_pad(x_in, 1, 0.0)
    x = np_conv(x, w1, 1); x = np_bn(x, bp1); x = np.maximum(x, 0)
    x = np_pad(x, 1, -np.inf); x = np_maxpool(x, 3, 2); skip = x
    x = np_conv(x, w2, 1); x = np_bn(x, bp2); x = np.maximum(x + skip, 0)
    x = np.maximum(np_conv(x, w3, 2), 0)
    x = x.sum(axis=(2, 3)) / f32(4)
    ref = (x @ wfc.T + f32(2.0)).astype(f32)

    header = ("#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>\n"
              "#bc4 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>\n"
              "#red4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>\n"
              "#id2 = affine_map<(d0, d1) -> (d0, d1)>\n"
              "#bc2 = affine_map<(d0, d1) -> (d1)>\n")
    blobs = ",\n".join(f'      {k}: "0x04000000{a.tobytes().hex()}"' for k, a in res.items())
    text = header + "\n".join(lines) + "\n\n{-#\n  dialect_resources: {\n    builtin: {\n" + blobs + "\n    }\n  }\n#-}\n"
    open(f"{outdir}/mini.mlir", "w").write(text)
    x_in.tofile(f"{outdir}/input.bin")
    ref.tofile(f"{outdir}/ref_logits.bin")
    print("reference logits:", ref.ravel())


if __name__ == "__main__":
    main(sys.argv[1])
