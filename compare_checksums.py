#!/usr/bin/env python3
"""
compare_checksums.py HOST.log TARGET.log [--model model.c] [--all] [--tol 0.02]

Compares the per-GEMM checksums (abs-sums) printed by a host run and a target run.
  --model model.c : annotate every checkpoint with its GEMM shape (M,N,K) and row stride in bytes
  --all           : print every checkpoint (default: only the first few, the first bad one, and the last)
The 'weights' row is informational only: summing 25M floats sequentially in fp32 saturates on the target.
"""
import argparse, math, re, struct

ap = argparse.ArgumentParser()
ap.add_argument("host"); ap.add_argument("target")
ap.add_argument("--model"); ap.add_argument("--all", action="store_true"); ap.add_argument("--tol", type=float, default=0.02)
a = ap.parse_args()

def f(h): return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]
def load(p):
    d = {}
    for m in re.finditer(r"CK (-?\d+) ([0-9a-fA-F]+) ([0-9a-fA-F]+)", open(p, errors="replace").read()):
        d[int(m.group(1))] = (f(m.group(2)), f(m.group(3)))
    return d
host, tgt = load(a.host), load(a.target)
if not tgt: raise SystemExit("no CK lines in target log (did it finish? look for DONE)")

shapes = {}
if a.model:
    n = 0
    for line in open(a.model):
        m = re.search(r"rvm_conv2d_nchw\((.*)\);", line)
        if m:
            g = [x.strip() for x in m.group(1).split(",")]
            C, H, W, F, KH, KW, OH, OW = (int(x) for x in g[3:11])
            n += 1; shapes[n] = (F, OH * OW, C * KH * KW, f"conv{KH}x{KW}")
            continue
        m = re.search(r"rvm_matmul\((.*)\);", line)
        if m:
            g = [x.strip() for x in m.group(1).split(",")]
            M, N, K = (int(x) for x in g[3:6])
            n += 1; shapes[n] = (M, N, K, "matmul")

def status(h, t):
    if not (math.isfinite(h) and math.isfinite(t)): return "NAN/INF", float("inf")
    if t == 0 and h != 0: return "ZERO", 1.0
    rel = abs(h - t) / max(abs(h), 1e-12)
    return ("ok" if rel <= a.tol else "BAD"), rel

rows = []
for k in sorted(x for x in host if x >= 1):
    h = host[k][1]; t = tgt[k][1] if k in tgt else float("nan")
    st, rel = status(h, t)
    rows.append((k, h, t, rel, st))
first = next((r[0] for r in rows if r[4] != "ok"), None)

def line(r):
    k, h, t, rel, st = r
    sh = shapes.get(k)
    shp = f"M={sh[0]:<5} N={sh[1]:<6} K={sh[2]:<5} {sh[3]:<8} rowstride(B)={2*sh[1]:<6}" if sh else ""
    return f"gemm {k:<3} {shp} host={h:<12.6g} target={t:<12.6g} rel={rel:<8.3g} {st}" + ("   <-- first divergence" if k == first else "")

print("input   :", "host=%.6g target=%.6g" % (host[-1][1], tgt[-1][1]) if -1 in host and -1 in tgt else "n/a")
if -2 in host and -2 in tgt:
    print("weights : host=%.6g target=%.6g   (informational: fp32 sequential sum saturates on the target)" % (host[-2][1], tgt[-2][1]))
print()
for r in rows:
    if a.all or r[0] <= 3 or r[0] == first or (first and abs(r[0] - first) <= 1) or r[0] == rows[-1][0]:
        print(line(r))
bad = [r for r in rows if r[4] != "ok"]
print(f"\n{len(rows)} GEMM checkpoints, {len(bad)} not ok (rel tol {a.tol}); zero outputs on target: {sum(1 for r in rows if r[4]=='ZERO')}")
print("FIRST DIVERGENCE: gemm %d" % first if first else "No divergence.")
if shapes and bad:
    zs = [shapes[r[0]][1] for r in rows if r[4] == "ZERO" and r[0] in shapes]
    if zs: print("N (=> rowstride 2N bytes) of the all-zero GEMMs: min=%d max=%d" % (min(zs), max(zs)))
    print("NOTE: everything after the first bad GEMM is contaminated, so only the FIRST bad row is conclusive.")
