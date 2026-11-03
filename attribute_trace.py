#!/usr/bin/env python3
"""
attribute_trace.py -- attribute executed instructions (spike -l trace) to source statements / layers.

Two subcommands, driven by attribute_trace.sh:
  prepare : static. Maps EVERY code address of the ELF to the statement of --main-src (model.c, or riscv-dnn's test.c)
            that it belongs to (DWARF line + inline info, so inlined kernels are attributed to their call site), and
            writes  blocks.json  and  pcmap.txt  (pc -> block id) for the streaming awk pass.
  report  : after the run. Reads blk_mn.txt (block mnemonic count), pc_counts.txt (count pc mnemonic) and prints
            totals by category, per block, per block kind, per innermost function and hot source lines.
Needs the ELF built with -g.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, OrderedDict

STD_M = re.compile(r"^(mul[a-z]*|mv|mret|min[a-z]*|max[a-z]*)$")


def category(mn):
    m = mn[2:] if mn.startswith("c.") else mn
    if m.startswith("m") and not STD_M.match(m):
        if m.startswith("mset"):                         return "matrix config"
        if re.match(r"ml[a-z0-9]*[.]", m):               return "matrix load"
        if re.match(r"ms[a-z0-9]*[.]", m):               return "matrix store"
        if re.match(r"mf[a-z]*cvt", m):                  return "matrix cvt"
        if re.match(r"(mf[a-z]*?(wma|ma|macc)|mma[a-z]*)[.]", m): return "matrix mac"
        return "matrix other"
    if m.startswith("v"):                                return "vector"
    if re.fullmatch(r"(l[bhwd]u?|lr[.].*|fl[hwdq])", m): return "scalar load"
    if re.fullmatch(r"(s[bhwd]|sc[.].*|fs[hwdq])", m):   return "scalar store"
    if m.startswith("f"):                                return "scalar fp"
    if re.fullmatch(r"(b(eq|ne|lt|ge|ltu|geu|eqz|nez|lez|gez|ltz|gtz)|j|jal|jalr|jr|call|ret|tail)", m):
        return "branch/jump"
    return "scalar int/other"


CATS = ["matrix config", "matrix load", "matrix store", "matrix mac", "matrix cvt", "matrix other", "vector",
        "scalar load", "scalar store", "scalar fp", "branch/jump", "scalar int/other"]


def find_tool(cands):
    for c in cands:
        p = shutil.which(c) or (c if os.path.isfile(c) else None)
        if p:
            return p
    return None


def exec_sections(readelf, elf):
    out = subprocess.run([readelf, "-S", "-W", elf], capture_output=True, text=True).stdout
    secs = []
    for ln in out.splitlines():
        if "]" not in ln:
            continue
        t = ln.split("]", 1)[1].split()
        if len(t) >= 7 and t[1] == "PROGBITS" and "X" in t[6]:
            secs.append((t[0], int(t[2], 16), int(t[4], 16)))
    return secs


def resolve(addr2line, elf, addrs):
    """addrs: list[int] -> dict addr -> [(func, file, line), ...] innermost first"""
    inp = "".join(f"0x{a:x}\n" for a in addrs)
    out = subprocess.run([addr2line, "-e", elf, "-a", "-f", "-i", "-C"], input=inp, capture_output=True,
                         text=True).stdout.splitlines()
    res, cur, buf = {}, None, []

    def flush():
        if cur is None:
            return
        frames = []
        for k in range(0, len(buf) - 1, 2):
            fn, loc = buf[k], buf[k + 1]
            loc = re.sub(r" \(discriminator \d+\)$", "", loc)
            f, _, l = loc.rpartition(":")
            frames.append((fn, f, int(l) if l.isdigit() else 0))
        res[cur] = frames

    for ln in out:
        if re.fullmatch(r"0x[0-9a-fA-F]+", ln) and len(buf) % 2 == 0:
            flush()
            cur, buf = int(ln, 16), []
        else:
            buf.append(ln)
    flush()
    return res


# ------------------------------------------------------------------ model.c block parsing
def parse_generated(path):
    """Group the statements of the *_forward function of a lower_to_c.py model.c into blocks.
    Returns (blocks, line->block id). A block = one top-level (4-space indent) statement, with the preceding
    /* #i label */ marker as its label."""
    lines = open(path).read().split("\n")
    in_fwd, blocks, line2b = False, [], {}
    pending = None
    cur = None
    for n, ln in enumerate(lines, 1):
        if not in_fwd:
            if re.match(r"static .*_forward\(", ln):
                in_fwd = True
            continue
        if ln.startswith("}"):
            in_fwd = False
            continue
        ind = len(ln) - len(ln.lstrip(" "))
        s = ln.strip()
        if not s:
            continue
        mk = re.match(r"/\* #(\d+) (.*) \*/$", s)
        if ind == 4 and mk:
            pending = mk.group(2)
            continue
        if ind == 4 and not s.startswith("}"):
            kind = (pending or "").split(" ->")[0].split(" ")[0] if pending else None
            if pending is None:
                if s.startswith("rvm_"):
                    kind, label = s.split("(")[0], s[:70]
                elif s.startswith("for") or s.startswith("{"):
                    kind, label = "loop", s[:70]
                else:
                    kind, label = "setup", s[:70]
            else:
                label = pending
            cur = {"id": len(blocks), "label": label, "kind": kind or "other", "first": n, "last": n}
            blocks.append(cur)
            pending = None
        if cur is not None:
            cur["last"] = n
            line2b[n] = cur["id"]
    # merge consecutive 'setup' singletons into one
    return blocks, line2b


def group_kind(b):
    k = b["kind"]
    lab = b["label"]
    if k.startswith("loops["):
        return lab.split(" ->")[0]
    if k.startswith("rvm.conv"):
        return "rvm.conv2d (software im2col + GEMM)"
    return k.split(" ")[0]


# ------------------------------------------------------------------ prepare
def cmd_prepare(a):
    a2l = a.addr2line or find_tool(["riscv64-unknown-elf-addr2line", os.path.join(os.path.dirname(a.clang or ""), "llvm-addr2line"),
                                    "llvm-addr2line", "addr2line"])
    rde = a.readelf or find_tool(["riscv64-unknown-elf-readelf", os.path.join(os.path.dirname(a.clang or ""), "llvm-readelf"),
                                  "llvm-readelf", "readelf"])
    if not a2l or not rde:
        sys.exit("need addr2line and readelf (riscv64-unknown-elf-* or llvm-*)")
    os.makedirs(a.out, exist_ok=True)
    main_base = os.path.basename(a.main_src)
    generated = "generated by lower_to_c.py" in open(a.main_src).read(400) if os.path.exists(a.main_src) else False
    if generated:
        blocks, line2b = parse_generated(a.main_src)
        print(f"{a.main_src}: {len(blocks)} statement blocks in the forward function")
    else:
        blocks, line2b = [], {}
    secs = exec_sections(rde, a.elf)
    addrs = []
    for name, base, size in secs:
        addrs += list(range(base, base + size, 2))
    print(f"resolving {len(addrs)} code addresses in {[s[0] for s in secs]} with {os.path.basename(a2l)} ...")
    fr = resolve(a2l, a.elf, addrs)
    by_line = {}
    pcmap = []
    for ad in addrs:
        frames = fr.get(ad) or []
        outer = None
        for f in frames:                      # innermost -> outermost: keep the OUTERMOST frame in main_src
            if os.path.basename(f[1]) == main_base and f[2] > 0:
                outer = f
        if outer is None:
            continue
        if generated:
            bid = line2b.get(outer[2], -1)
            if bid < 0:
                bid = -1                       # model.c line outside the forward function (main/printf)
        else:
            key = outer[2]
            if key not in by_line:
                by_line[key] = len(blocks)
                txt = ""
                if os.path.exists(a.main_src):
                    L = open(a.main_src).read().split("\n")
                    txt = L[key - 1].strip() if 0 < key <= len(L) else ""
                blocks.append({"id": len(blocks), "label": f"{main_base}:{key}  {txt[:70]}",
                               "kind": (re.split(r"[ (]", txt)[0] if txt else "stmt"), "first": key, "last": key})
            bid = by_line[key]
        pcmap.append((ad, bid))
    if generated:
        blocks.append({"id": -1, "label": f"{main_base} outside forward (main, printf of logits)", "kind": "main", "first": 0, "last": 0})
    with open(os.path.join(a.out, "pcmap.txt"), "w") as f:
        for ad, bid in pcmap:
            f.write(f"0x{ad:016x} {bid}\n")
    json.dump({"elf": a.elf, "main_src": a.main_src, "blocks": blocks, "addr2line": a2l, "generated": generated},
              open(os.path.join(a.out, "blocks.json"), "w"))
    print(f"wrote {a.out}/pcmap.txt ({len(pcmap)} addresses) and blocks.json ({len(blocks)} blocks)")


# ------------------------------------------------------------------ report
def M(x):
    return f"{x / 1e6:8.2f}M"


def cmd_report(a):
    meta = json.load(open(os.path.join(a.out, "blocks.json")))
    blocks = {b["id"]: b for b in meta["blocks"]}
    blk = defaultdict(Counter)      # bid -> category -> count
    blk_mn = defaultdict(Counter)
    tot = Counter()
    for ln in open(os.path.join(a.out, "blk_mn.txt")):
        p = ln.split()
        if len(p) != 3:
            continue
        bid, mn, c = p[0], p[1], int(p[2])
        bid = -1 if bid in ("startup", "-1") else int(bid)
        cat = category(mn)
        blk[bid][cat] += c
        blk_mn[bid][mn] += c
        tot[cat] += c
    total = sum(tot.values())
    print(f"\nTOTAL {total:,} instructions\n")
    print("BY CATEGORY")
    for c in CATS:
        print(f"  {c:18}{tot[c]:>16,}  {100 * tot[c] / total:6.2f}%")

    rows = []
    for bid, cc in blk.items():
        b = blocks.get(bid, {"id": bid, "label": "(before first statement / startup / unattributed)", "kind": "startup"})
        rows.append((bid, b, sum(cc.values()), cc))
    rows.sort(key=lambda r: r[0] if r[0] >= 0 else 10 ** 9)

    print("\nBY BLOCK (program order; blocks below 0.05% are summed at the end)")
    hdr = f"{'blk':>4} {'total':>9} {'%':>6} {'matrix':>8} {'ld':>8} {'st':>8} {'fp':>8} {'int/br':>8}  label"
    print(hdr)
    small = Counter(); small_n = 0
    for bid, b, t, cc in rows:
        if t / total < 0.0005:
            small.update(cc); small_n += 1
            continue
        mat = sum(v for k, v in cc.items() if k.startswith("matrix"))
        ib = cc["scalar int/other"] + cc["branch/jump"] + cc["vector"]
        print(f"{bid:>4} {M(t)} {100 * t / total:5.2f}% {M(mat)} {M(cc['scalar load'])} {M(cc['scalar store'])} "
              f"{M(cc['scalar fp'])} {M(ib)}  {b['label'][:70]}")
    if small_n:
        print(f"  ... {small_n} small blocks: {sum(small.values()):,} instructions")

    print("\nBY BLOCK KIND (what kind of pass costs how much)")
    kinds = defaultdict(Counter); kn = Counter()
    for bid, b, t, cc in rows:
        k = group_kind(b) if "kind" in b else "?"
        kinds[k].update(cc); kn[k] += 1
    print(f"{'kind':58}{'#':>4}{'total':>10}{'%':>7}{'matrix':>9}{'ld+st':>9}{'fp':>9}{'int+br':>9}")
    for k, cc in sorted(kinds.items(), key=lambda kv: -sum(kv[1].values())):
        t = sum(cc.values())
        mat = sum(v for kk, v in cc.items() if kk.startswith("matrix"))
        print(f"{k[:57]:58}{kn[k]:>4}{M(t)}{100 * t / total:6.1f}%{M(mat)}{M(cc['scalar load'] + cc['scalar store'])}"
              f"{M(cc['scalar fp'])}{M(cc['scalar int/other'] + cc['branch/jump'] + cc['vector'])}")

    # --- innermost function / hot lines from per-pc counts
    pcs = {}
    for ln in open(os.path.join(a.out, "pc_counts.txt")):
        p = ln.split()
        if len(p) >= 3:
            pcs[int(p[1], 16)] = (int(p[0]), p[2])
    fr = resolve(meta["addr2line"], meta["elf"], sorted(pcs))
    fn_c, line_c, line_mn = Counter(), Counter(), defaultdict(Counter)
    for pc, (c, mn) in pcs.items():
        frames = fr.get(pc) or [("??", "??", 0)]
        inner = frames[0]
        fn_c[inner[0]] += c
        key = (os.path.basename(inner[1]), inner[2], inner[1])
        line_c[key] += c
        line_mn[key][mn] += c
    print("\nBY INNERMOST FUNCTION (inlined helpers count under their own name)")
    for fn, c in fn_c.most_common(a.top):
        print(f"  {c:>14,} {100 * c / total:6.2f}%  {fn[:70]}")
    print(f"\nHOT SOURCE LINES (top {a.top})")
    for (base, line, full), c in line_c.most_common(a.top):
        txt = ""
        for cand in [full] + [os.path.join(r, base) for r in a.src_roots]:
            if os.path.exists(cand):
                L = open(cand, errors="replace").read().split("\n")
                txt = L[line - 1].strip() if 0 < line <= len(L) else ""
                break
        top_mn = ",".join(m for m, _ in line_mn[(base, line, full)].most_common(3))
        print(f"  {c:>14,} {100 * c / total:6.2f}%  {base}:{line:<5} {txt[:58]:58} [{top_mn}]")

    with open(os.path.join(a.out, "attribution_blocks.csv"), "w") as f:
        f.write("block,label,kind,total," + ",".join(CATS) + "\n")
        for bid, b, t, cc in rows:
            f.write(f"{bid},\"{b['label']}\",{b['kind']},{t}," + ",".join(str(cc[c]) for c in CATS) + "\n")
    with open(os.path.join(a.out, "attribution_lines.csv"), "w") as f:
        f.write("file,line,count\n")
        for (base, line, full), c in line_c.most_common():
            f.write(f"{base},{line},{c}\n")
    print(f"\nwrote {a.out}/attribution_blocks.csv, attribution_lines.csv")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--elf", required=True); p.add_argument("--main-src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--addr2line"); p.add_argument("--readelf"); p.add_argument("--clang")
    r = sp.add_parser("report")
    r.add_argument("--out", required=True); r.add_argument("--top", type=int, default=25)
    r.add_argument("--src-roots", nargs="*", default=[])
    a = ap.parse_args()
    (cmd_prepare if a.cmd == "prepare" else cmd_report)(a)


if __name__ == "__main__":
    main()
