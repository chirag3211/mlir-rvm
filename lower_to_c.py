#!/usr/bin/env python3
"""
lower_to_c.py  --  linalg-on-tensors MLIR  ->  C for the riscv-matrix toolchain

    python3 lower_to_c.py model.mlir OUTDIR [--input input.bin] [--func main]

Writes OUTDIR/model.c and OUTDIR/weights.bin (fp32, 64-byte aligned).

Pipeline
  1. one-shot-bufferize (identity layouts)          tensors -> memrefs
  2. linalg.conv_2d_nchw_fchw / linalg.matmul        -> rvm.* marker ops (matrix GEMM in C runtime)
  3. convert-linalg-to-loops, lower-affine           everything else -> scf/arith/memref loops
  4. C emitter walks the result. Any op it does not know is a hard error (never silently dropped).

Needs only the torch-mlir wheel (python bindings); no torch import.
"""
import argparse
import math
import os
import re
import sys

import numpy as np
from torch_mlir import ir
from torch_mlir.passmanager import PassManager


class Unsupported(Exception):
    pass


def isa(x, cls):
    """Works with both old (Cls.isinstance) and new (auto-downcast) MLIR python bindings."""
    if hasattr(cls, "isinstance"):
        return cls.isinstance(x)
    return isinstance(x, cls)


# ----------------------------------------------------------------------------- helpers
def parse_resources(text):
    """name -> raw bytes for every dialect_resources blob (strips the 4-byte alignment word)."""
    blobs = {}
    m = re.search(r"\{-#(.*?)#-\}", text, re.S)
    if m:
        for name, hx in re.findall(r'([A-Za-z0-9_.$\-]+)\s*:\s*"0x([0-9a-fA-F]+)"', m.group(1)):
            blobs[name] = bytes.fromhex(hx)[4:]
    return blobs


def int_list(attr):
    s = str(attr)
    m = re.search(r"array<i64:?\s*([^>]*)>", s)
    if m:
        return [int(x) for x in m.group(1).split(",") if x.strip()]
    m = re.search(r"dense<\[?([^\]>]*)\]?>\s*:\s*(?:tensor|vector)<([0-9x]+)x", s)
    if m:
        vals = [int(x) for x in m.group(1).split(",") if x.strip()]
        n = int(np.prod([int(d) for d in m.group(2).split("x")]))
        return vals if len(vals) == n else vals * n
    raise Unsupported(f"cannot read integer list from attribute: {s}")


def ctype(t):
    if isa(t, ir.IndexType):
        return "long"
    if isa(t, ir.IntegerType):
        return {1: "int", 8: "signed char", 16: "short", 32: "int", 64: "long"}[ir.IntegerType(t).width]
    if isa(t, ir.F32Type):
        return "float"
    if isa(t, ir.F64Type):
        return "double"
    raise Unsupported(f"unsupported scalar type {t}")


def is_float(t):
    return isa(t, ir.F32Type) or isa(t, ir.F64Type)


def row_major(shape):
    st, acc = [], 1
    for d in reversed(shape):
        st.append(acc)
        acc *= d
    return list(reversed(st))


class MR:
    """A memref value: element (base + off + sum idx*stride) of a flat float array."""

    def __init__(self, base, mtype, h16=False, nhwc=False):
        mt = ir.MemRefType(mtype)
        if not isa(mt.element_type, ir.F32Type):
            raise Unsupported(f"only f32 memrefs are supported, got {mtype}")
        self.shape = list(mt.shape)
        if any(d < 0 for d in self.shape):
            raise Unsupported(f"dynamic shape not supported: {mtype}")
        strides, off = mt.get_strides_and_offset()
        if off < 0 or any(s < 0 for s in strides):
            raise Unsupported(f"dynamic stride/offset not supported: {mtype}")
        self.strides, self.off, self.base = list(strides), int(off), base
        self.h16, self.nhwc = h16, nhwc
        if nhwc:                                  # logical NCHW indexing, physical NHWC fp16 storage
            if len(self.shape) != 4 or self.shape[0] != 1:
                raise Unsupported(f"NHWC activation must be 4-D with N=1, got {self.shape}")
            _, C_, H_, W_ = self.shape
            self.strides, self.off = [H_ * W_ * C_, 1, W_ * C_, C_], 0

    @property
    def numel(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def contiguous(self):
        if self.nhwc:
            return True
        rm = row_major(self.shape)
        return all(s == r for s, r, d in zip(self.strides, rm, self.shape) if d > 1)

    def ptr(self):
        return f"({self.base} + {self.off})" if self.off else self.base


CMPF = {1: "==", 2: ">", 3: ">=", 4: "<", 5: "<=", 6: "!=", 8: "==", 9: ">", 10: ">=", 11: "<", 12: "<=", 13: "!="}
CMPI = {0: "==", 1: "!=", 2: "<", 3: "<=", 4: ">", 5: ">=", 6: "<", 7: "<=", 8: ">", 9: ">="}
BIN = {"arith.addf": "+", "arith.subf": "-", "arith.mulf": "*", "arith.divf": "/", "arith.addi": "+",
       "arith.subi": "-", "arith.muli": "*", "arith.divsi": "/", "arith.remsi": "%", "arith.andi": "&",
       "arith.ori": "|", "arith.xori": "^", "arith.shli": "<<", "arith.shrsi": ">>"}
CASTS = {"arith.truncf", "arith.extf", "arith.sitofp", "arith.fptosi", "arith.index_cast", "arith.index_castui",
         "arith.extsi", "arith.trunci", "arith.extui", "arith.uitofp"}
VIEWS = {"memref.subview", "memref.expand_shape", "memref.collapse_shape", "memref.cast"}


def walk(op):
    yield op
    for r in op.operation.regions:
        for b in r.blocks:
            for o in b.operations:
                yield from walk(o)


# ----------------------------------------------------------------------------- passes
def bufferize(mod):
    PassManager.parse(
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries "
        "function-boundary-type-conversion=identity-layout-map})").run(mod.operation)


def mark_matrix_ops(mod):
    """Replace conv/matmul by opaque rvm.* ops so they survive loop lowering."""
    i64 = ir.IntegerType.get_signless(64)
    n = 0
    for f in mod.body.operations:
        if f.operation.name != "func.func":
            continue
        for op in list(f.regions[0].blocks[0].operations):
            nm = op.operation.name
            if nm == "linalg.conv_2d_nchw_fchw":
                st = int_list(op.operation.attributes["strides"])
                dl = int_list(op.operation.attributes["dilations"])
                attrs = {"sh": ir.IntegerAttr.get(i64, st[0]), "sw": ir.IntegerAttr.get(i64, st[1]),
                         "dh": ir.IntegerAttr.get(i64, dl[0]), "dw": ir.IntegerAttr.get(i64, dl[1])}
                ir.Operation.create("rvm.conv2d_nchw", operands=list(op.operation.operands),
                                    attributes=attrs, ip=ir.InsertionPoint(op.operation))
                op.operation.erase()
                n += 1
            elif nm == "linalg.matmul":
                ir.Operation.create("rvm.matmul", operands=list(op.operation.operands),
                                    ip=ir.InsertionPoint(op.operation))
                op.operation.erase()
                n += 1
    return n


def attr_names(op):
    return [getattr(a, "name", a) for a in _op(op).attributes]


def _op(x):
    return x.operation if hasattr(x, "operation") else x


def _owner(v):
    try:
        o = v.owner
        if isinstance(o, ir.Block):      # block argument (e.g. the function's input)
            return None
        return _op(o)
    except Exception:
        return None


def _is_zero_const(v):
    o = _owner(v)
    if o is None or o.name != "arith.constant":
        return False
    try:
        return float(ir.FloatAttr(o.attributes["value"]).value) == 0.0
    except Exception:
        return False


def _fill_value_is_zero(op):
    """linalg.fill(0.0) or a linalg.map whose body just yields a constant 0.0 (torch-mlir's zero pad)."""
    op = _op(op)
    if op.name == "linalg.fill":
        return _is_zero_const(op.operands[0])
    if op.name == "linalg.map":
        ops = list(op.regions[0].blocks[0].operations)
        return bool(ops) and _op(ops[-1]).name == "linalg.yield" and _is_zero_const(_op(ops[-1]).operands[0])
    return False


def fold_pad_and_fill(mod):
    """Two peephole rewrites on bufferized IR, applied to each rvm.conv2d_nchw:
       (1) pad folding : alloc P; fill/map(P)=0; subview(P)<-copy X; conv(P,..)  ==>  conv(X,..) with pt/pl attrs
       (2) overwrite   : alloc O; fill(O)=0; conv(..,O)                          ==>  conv(..,O) with ow=1
       The runtime then handles borders while building im2col and writes (not accumulates) the output.
       A pattern is only applied when it matches exactly; anything else is left untouched."""
    n_pad = n_ow = 0
    for f in mod.body.operations:
        if f.operation.name != "func.func":
            continue
        blk = f.regions[0].blocks[0]
        n_conv = sum(1 for o in blk.operations if o.operation.name == "rvm.conv2d_nchw")
        for ck in range(n_conv):
            ops = [o for o in blk.operations]
            convs = [(i, o) for i, o in enumerate(ops) if o.operation.name == "rvm.conv2d_nchw"]
            ci, c = convs[ck]
            uses = {}
            for i, top in enumerate(ops):
                for o in walk(top):
                    for k, v in enumerate(o.operation.operands):
                        uses.setdefault(v, []).append((i, o.operation, k))

            def recreate(extra, new_in=None):
                cop = c.operation
                operands = list(cop.operands)
                if new_in is not None:
                    operands[0] = new_in
                attrs = {n: cop.attributes[n] for n in attr_names(cop)}
                attrs.update(extra)
                new = ir.Operation.create("rvm.conv2d_nchw", operands=operands, attributes=attrs,
                                          ip=ir.InsertionPoint(cop))
                cop.erase()
                return new

            i64 = ir.IntegerType.get_signless(64)
            cur = c.operation
            # ---- (1) pad folding
            P = cur.operands[0]
            po = _owner(P)
            if po is not None and po.name == "memref.alloc":
                others = [(i, o, k) for (i, o, k) in uses.get(P, []) if o != cur]
                fills = [u for u in others if u[1].name in ("linalg.map", "linalg.fill")]
                subs = [u for u in others if u[1].name == "memref.subview"]
                deallocs = [u for u in others if u[1].name == "memref.dealloc"]
                if (len(fills) == 1 and len(subs) == 1 and len(fills) + len(subs) + len(deallocs) == len(others)
                        and _fill_value_is_zero(fills[0][1])):
                    sv = subs[0][1]
                    su = uses.get(sv.results[0], [])
                    cps = [u for u in su if u[1].name == "memref.copy" and u[2] == 1]
                    if len(su) == 1 and len(cps) == 1:
                        offs, sizes, strs = (int_list(sv.attributes[k]) for k in
                                             ("static_offsets", "static_sizes", "static_strides"))
                        pshape = list(ir.MemRefType(P.type).shape)
                        cp = cps[0][1]
                        X = cp.operands[0]
                        xshape = list(ir.MemRefType(X.type).shape)
                        if (len(pshape) == 4 and len(offs) == 4 and all(x == 1 for x in strs)
                                and offs[0] == 0 and offs[1] == 0 and sizes == xshape
                                and sizes[:2] == pshape[:2] and all(o >= 0 for o in offs)
                                and ir.MemRefType(X.type).element_type == ir.MemRefType(P.type).element_type):
                            new = recreate({"pt": ir.IntegerAttr.get(i64, offs[2]),
                                            "pl": ir.IntegerAttr.get(i64, offs[3]),
                                            "pb": ir.IntegerAttr.get(i64, pshape[2] - offs[2] - sizes[2]),
                                            "pr": ir.IntegerAttr.get(i64, pshape[3] - offs[3] - sizes[3])}, X)
                            for dead in [cp, sv, fills[0][1]] + [u[1] for u in deallocs]:
                                dead.erase()
                            po.erase()
                            cur = new
                            n_pad += 1
            # ---- (2) overwrite instead of accumulate
            ops = [o for o in blk.operations]
            ci = next(i for i, o in enumerate(ops) if o.operation == cur)
            uses = {}
            for i, top in enumerate(ops):
                for o in walk(top):
                    for k, v in enumerate(o.operation.operands):
                        uses.setdefault(v, []).append((i, o.operation, k))
            O = cur.operands[2]
            oo = _owner(O)
            if oo is not None and oo.name == "memref.alloc" and "ow" not in attr_names(cur):
                before = [u for u in uses.get(O, []) if u[0] < ci]
                aliased = any(u[1].name in VIEWS for u in uses.get(O, []))
                fill = None
                if len(before) == 1 and not aliased:
                    u0 = before[0][1]
                    if u0.name in ("linalg.fill", "linalg.map") and _fill_value_is_zero(u0):
                        fill = u0                     # (2a) O = alloc; fill(O, 0); conv
                    elif u0.name == "memref.copy" and before[0][2] == 1:
                        # (2b) O = alloc; copy(Z -> O); conv     with Z provably all-zero at the copy
                        Z = u0.operands[0]
                        zo = _owner(Z)
                        ic = before[0][0]
                        if zo is not None and zo.name == "memref.alloc" and Z != O:
                            pre = [u for u in uses.get(Z, []) if u[0] < ic]
                            zf = [u for u in pre if u[1].name in ("linalg.fill", "linalg.map")]
                            rest = [u for u in pre if u not in zf]
                            if (len(zf) == 1 and _fill_value_is_zero(zf[0][1])
                                    and all((u[1].name == "memref.copy" and u[2] == 0) or u[1].name == "memref.dealloc"
                                            for u in rest)
                                    and not any(u[1].name in VIEWS for u in uses.get(Z, []))):
                                fill = u0             # the copy plays the role of the fill
                if fill is not None:
                    attrs = {n: cur.attributes[n] for n in attr_names(cur)}
                    attrs["ow"] = ir.IntegerAttr.get(i64, 1)
                    ir.Operation.create("rvm.conv2d_nchw", operands=list(cur.operands), attributes=attrs,
                                        ip=ir.InsertionPoint(cur))
                    cur.erase()
                    fill.erase()
                    n_ow += 1
        # drop zero-fills whose buffer is no longer read by anything
        changed = True
        while changed:
            changed = False
            ops = [o for o in blk.operations]
            uses = {}
            for i, top in enumerate(ops):
                for o in walk(top):
                    for k, v in enumerate(o.operation.operands):
                        uses.setdefault(v, []).append((i, o.operation, k))
            for o in ops:
                if o.operation.name != "memref.alloc":
                    continue
                us = uses.get(o.operation.results[0], [])
                if us and all((u[1].name in ("linalg.fill", "linalg.map") and _fill_value_is_zero(u[1]))
                              or u[1].name == "memref.dealloc" for u in us):
                    for u in us:
                        u[1].erase()
                    o.operation.erase()
                    changed = True
                    break
    return n_pad, n_ow


def _float_const(v):
    o = _owner(v)
    if o is None or o.name != "arith.constant":
        return None
    try:
        return float(ir.FloatAttr(o.attributes["value"]).value)
    except Exception:
        return None


def _fill_value(op):
    """constant stored by linalg.fill / a linalg.map that just yields a constant; None if anything else."""
    op = _op(op)
    if op.name == "linalg.fill":
        return _float_const(op.operands[0])
    if op.name == "linalg.map":
        ops = list(op.regions[0].blocks[0].operations)
        if ops and _op(ops[-1]).name == "linalg.yield":
            return _float_const(_op(ops[-1]).operands[0])
    return None


def _is_neginf_fill(op):
    v = _fill_value(op)
    return v is not None and math.isinf(v) and v < 0


def fold_maxpool(mod):
    """Bufferized IR:  alloc P; fill(P,-inf); subview(P) <- copy X; [alloc O; fill(O,-inf);] pooling_nchw_max(P,W,O)
       ==>  rvm.maxpool_nchw(X, O)  with pt/pl attrs: windows are clamped at the border (max-identity padding),
       so the padded copy, its fill and the output pre-fill all disappear.  Only applied when the pattern matches."""
    i64 = ir.IntegerType.get_signless(64)
    n_fold = 0
    for f in mod.body.operations:
        if f.operation.name != "func.func":
            continue
        blk = f.regions[0].blocks[0]
        while True:
            ops = [o for o in blk.operations]
            uses = {}
            for i, top in enumerate(ops):
                for o in walk(top):
                    for k, v in enumerate(o.operation.operands):
                        uses.setdefault(v, []).append((i, o.operation, k))
            done = False
            for ci, c in enumerate(ops):
                cur = c.operation
                if cur.name != "linalg.pooling_nchw_max" or "fold_failed" in attr_names(cur):
                    continue
                P, W, O = cur.operands[0], cur.operands[1], cur.operands[2]
                po, oo = _owner(P), _owner(O)
                if po is None or po.name != "memref.alloc":
                    continue
                others = [u for u in uses.get(P, []) if u[1] != cur]
                fills = [u for u in others if u[1].name in ("linalg.map", "linalg.fill")]
                subs = [u for u in others if u[1].name == "memref.subview"]
                deallocs = [u for u in others if u[1].name == "memref.dealloc"]
                if not (len(fills) == 1 and len(subs) == 1 and len(fills) + len(subs) + len(deallocs) == len(others)
                        and _is_neginf_fill(fills[0][1])):
                    continue
                sv = subs[0][1]
                su = uses.get(sv.results[0], [])
                cps = [u for u in su if u[1].name == "memref.copy" and u[2] == 1]
                if not (len(su) == 1 and len(cps) == 1):
                    continue
                offs, sizes, strs = (int_list(sv.attributes[k]) for k in ("static_offsets", "static_sizes", "static_strides"))
                cp = cps[0][1]
                X = cp.operands[0]
                pshape, xshape = list(ir.MemRefType(P.type).shape), list(ir.MemRefType(X.type).shape)
                if not (len(pshape) == 4 and len(offs) == 4 and all(x == 1 for x in strs) and offs[0] == 0 and offs[1] == 0
                        and sizes == xshape and sizes[:2] == pshape[:2] and all(o >= 0 for o in offs)):
                    continue
                if not all(ir.MemRefType(t.type).element_type == ir.MemRefType(P.type).element_type for t in (X, O)):
                    continue
                st, dl = int_list(cur.attributes["strides"]), int_list(cur.attributes["dilations"])
                wshape = list(ir.MemRefType(W.type).shape)
                # output fill (-inf) that can be dropped because the new op overwrites every output element
                ofill = None
                if oo is not None and oo.name == "memref.alloc":
                    before = [u for u in uses.get(O, []) if u[0] < ci]
                    if (len(before) == 1 and before[0][1].name in ("linalg.fill", "linalg.map")
                            and _is_neginf_fill(before[0][1]) and not any(u[1].name in VIEWS for u in uses.get(O, []))):
                        ofill = before[0][1]
                attrs = {"kh": ir.IntegerAttr.get(i64, wshape[0]), "kw": ir.IntegerAttr.get(i64, wshape[1]),
                         "sh": ir.IntegerAttr.get(i64, st[0]), "sw": ir.IntegerAttr.get(i64, st[1]),
                         "dh": ir.IntegerAttr.get(i64, dl[0]), "dw": ir.IntegerAttr.get(i64, dl[1]),
                         "pt": ir.IntegerAttr.get(i64, offs[2]), "pl": ir.IntegerAttr.get(i64, offs[3])}
                ir.Operation.create("rvm.maxpool_nchw", operands=[X, O], attributes=attrs, ip=ir.InsertionPoint(cur))
                cur.erase()
                dead = [cp, sv, fills[0][1]] + [u[1] for u in deallocs]
                if ofill is not None:
                    dead.append(ofill)
                for d in dead:
                    d.erase()
                po.erase()
                n_fold += 1
                done = True
                break
            if not done:
                break
    return n_fold


def _dest_operands(op):
    """operands an op may write (everything else is read-only); conservative for unknown ops."""
    op = _op(op)
    n = op.name
    ops = list(op.operands)
    if n in ("linalg.generic", "linalg.map", "linalg.fill", "linalg.pooling_nchw_max", "linalg.copy"):
        if "operandSegmentSizes" in attr_names(op):
            seg = [int(x) for x in re.findall(r"-?\d+", str(op.attributes["operandSegmentSizes"]).split(":")[-1])]
            return ops[seg[0]:]
        return ops[-1:]
    if n == "memref.copy":
        return [ops[1]]
    if n == "rvm.conv2d_nchw":
        return [ops[2]]
    if n in ("rvm.maxpool_nchw", "rvm.maxpool_bn"):
        return [ops[1]]
    if n == "rvm.conv2d_epi":
        return [ops[2]] + ([ops[6]] if len(ops) > 5 else [])
    if n in ("memref.alloc", "memref.get_global", "memref.dealloc", "arith.constant"):
        return []
    return ops


def _body_names(g):
    return [o.operation.name for o in g.regions[0].blocks[0].operations
            if o.operation.name not in ("linalg.yield", "arith.constant")]


def _is_bn_relu_generic(g):
    g = _op(g)
    return (g.name == "linalg.generic" and len(g.operands) == 4
            and _body_names(g) == ["arith.mulf", "arith.addf", "arith.cmpf", "arith.select"])


def _is_add_generic(g):
    g = _op(g)
    return (g.name == "linalg.generic" and len(g.operands) == 3 and _body_names(g) == ["arith.addf"]
            and "reduction" not in str(g.attributes["iterator_types"]))


def fuse_conv_epilogue(mod, mode="all", report=None):
    """Bufferized IR (after fold-pad / fuse-bn):   conv -> O ; [add(O,S) -> A ;] bn_relu(. , alpha, beta) -> Y
       ==> one rvm.conv2d_epi(in, w, Y, alpha, beta [, S, A])    (riscv-dnn conv_bn_relu_rvm / conv_add_bn_relu_rvm)
       The fused op sits where the BN generic was.  Applied only when no other op touches the buffers involved in
       between; anything else is left untouched.  Returns (n_bn, n_add_bn); `report` (a dict) receives the reasons
       why remaining convs were not fused."""
    i64 = ir.IntegerType.get_signless(64)
    n_bn = n_add = 0
    why = {}
    for f in mod.body.operations:
        if f.operation.name != "func.func":
            continue
        blk = f.regions[0].blocks[0]
        progress = True
        while progress:
            progress = False
            why = {}
            ops = [o.operation for o in blk.operations]
            pos = {id(o): i for i, o in enumerate(ops)}
            uses = {}
            for i, top in enumerate(ops):
                for o in walk(top):
                    for k, v in enumerate(o.operation.operands):
                        uses.setdefault(v, []).append((i, o.operation, k))

            def skip(reason):
                why[reason] = why.get(reason, 0) + 1

            for ci, conv in enumerate(ops):
                if conv.name != "rvm.conv2d_nchw":
                    continue
                if "ow" not in attr_names(conv):
                    skip("conv accumulates into its output (not overwrite-only)")
                    continue
                X, Wt, O = conv.operands[0], conv.operands[1], conv.operands[2]
                us = _dedupe_users(sorted([u for u in uses.get(O, []) if u[1] != conv and u[1].name != "memref.dealloc"], key=lambda u: u[0]))
                if not us:
                    skip("conv output has no reader")
                    continue
                first = us[0][1]
                add_op, S, A, bn = None, None, None, None
                if _is_bn_relu_generic(first) and first.operands[0] == O:
                    inplace = (first.operands[3] == O)      # BN overwrites the conv output: later readers see the BN result
                    others = [u for u in us if not (u[1] == first)]
                    if others and not (inplace and all(u[0] > us[0][0] for u in others)):
                        skip("BN input (conv output) has other readers")
                        continue
                    if mode == "add":
                        continue
                    bn = first
                elif _is_add_generic(first) and O in (first.operands[0], first.operands[1]):
                    if mode == "bn":
                        continue
                    add_op = first
                    S = first.operands[1] if first.operands[0] == O else first.operands[0]
                    A = first.operands[2]
                    if S == O:
                        skip("add of the conv output with itself")
                        continue
                    if A != O and len(us) != 1:
                        skip("add: conv output has other readers")
                        continue
                    if A == O and any(u[1] is not first and u[0] < us[0][0] for u in us):
                        skip("add: odd aliasing")
                        continue
                    ai = pos[id(add_op)]
                    cands = [u for u in uses.get(A, []) if u[0] > ai and _is_bn_relu_generic(u[1]) and u[1].operands[0] == A and u[2] == 0]
                    if not cands:
                        skip("add is not followed by a BN+ReLU generic")
                        continue
                    bn = sorted(cands, key=lambda u: u[0])[0][1]
                else:
                    skip("first reader of the conv output is neither BN+ReLU nor an add (%s)" % first.name)
                    continue
                bi = pos[id(bn)]
                alpha, beta, Y = bn.operands[1], bn.operands[2], bn.operands[3]
                src = A if add_op is not None else O
                if bn.operands[0] != src:
                    skip("BN does not read the add / conv result")
                    continue
                # Y may reuse the conv-output buffer (it is dead afterwards) but never anything the fused op still reads
                bad_alias = [X, Wt, alpha, beta] + ([S, A] if add_op is not None else [])
                names = ["X(conv input)", "W", "alpha", "beta"] + (["S(residual)", "A(add result)"] if add_op is not None else [])
                hit = [n for n, v in zip(names, bad_alias) if v == Y]
                redirect = skip_a = False
                if hit == ["X(conv input)"] and O.type == Y.type:
                    redirect = True          # BN wrote over the conv input: write the fused result into the dead conv output instead
                elif hit == ["A(add result)"]:
                    skip_a = True            # BN overwrote the add result in place: that value is dead, so do not store it
                elif hit:
                    skip("BN output buffer is the same as " + "/".join(hit))
                    continue
                chain = {id(conv), id(bn)} | ({id(add_op)} if add_op is not None else set())
                guarded_w = [X, Wt, alpha, beta] + ([S] if add_op is not None else [])
                guarded_all = [Y] + ([A] if add_op is not None else [])
                ok = True
                for j in range(ci + 1, bi):
                    mid = ops[j]
                    if id(mid) in chain:
                        continue
                    dests = _dest_operands(mid)
                    if any(v in dests for v in guarded_w):
                        skip("a buffer read by the fused op is overwritten in between"); ok = False; break
                    allops = [v for o2 in walk(mid) for v in o2.operation.operands]
                    if any(v in allops for v in guarded_all):
                        skip("an output buffer of the fused op is touched in between"); ok = False; break
                if not ok:
                    continue
                if redirect:
                    # later readers of the old BN result (== buffer X) now read the conv-output buffer O
                    late = [(j2, op2, k2) for (j2, op2, k2) in uses.get(Y, []) if j2 > bi]
                    for _, op2, k2 in late:
                        op2.operands[k2] = O
                    Y = O
                attrs = {n: conv.attributes[n] for n in attr_names(conv)}
                if skip_a:
                    attrs["skip_a"] = ir.IntegerAttr.get(i64, 1)
                attrs["relu"] = ir.IntegerAttr.get(i64, 1)
                attrs["has_add"] = ir.IntegerAttr.get(i64, 1 if add_op is not None else 0)
                operands = [X, Wt, Y, alpha, beta] + ([S, A] if add_op is not None else [])
                ir.Operation.create("rvm.conv2d_epi", operands=operands, attributes=attrs, ip=ir.InsertionPoint(bn))
                for dead in ([add_op] if add_op is not None else []) + [conv, bn]:
                    dead.erase()
                if add_op is not None:
                    n_add += 1
                else:
                    n_bn += 1
                # drop the now-unused conv-output buffer
                oo = _owner(O)
                if oo is not None and oo.name == "memref.alloc":
                    rest = [u for o2 in blk.operations for u in [o2.operation]
                            if any(v == O for v in u.operands) and u.name != "memref.dealloc"]
                    if not rest:
                        for o2 in list(blk.operations):
                            if o2.operation.name == "memref.dealloc" and o2.operation.operands[0] == O:
                                o2.operation.erase()
                        oo.erase()
                progress = True
                break
    if report is not None:
        report.update(why)
    return n_bn, n_add


def _dedupe_users(us):
    """(index, op, operand#) triples -> one entry per distinct op (an op that reads AND writes a buffer counts once)"""
    out = []
    for u in us:
        if not any(v[1] == u[1] for v in out):
            out.append(u)
    return out


def fuse_pool_bn(mod, report=None):
    """rvm.maxpool_nchw(X,O) ; bn_relu(O,alpha,beta)->Y   ==>   rvm.maxpool_bn(X,Y,alpha,beta)   (riscv-dnn maxpool_bn_relu)"""
    i64 = ir.IntegerType.get_signless(64)
    n = 0
    why = {}
    for f in mod.body.operations:
        if f.operation.name != "func.func":
            continue
        blk = f.regions[0].blocks[0]
        progress = True
        while progress:
            progress = False
            why.clear()
            ops = [o.operation for o in blk.operations]
            pos = {id(o): i for i, o in enumerate(ops)}
            uses = {}
            for i, top in enumerate(ops):
                for o in walk(top):
                    for k, v in enumerate(o.operation.operands):
                        uses.setdefault(v, []).append((i, o.operation, k))
            for pi, pool in enumerate(ops):
                if pool.name != "rvm.maxpool_nchw" or len(pool.operands) != 2:
                    continue
                X, O = pool.operands[0], pool.operands[1]
                us = _dedupe_users(sorted([u for u in uses.get(O, []) if u[1] != pool and u[1].name != "memref.dealloc"], key=lambda u: u[0]))
                if not us or not _is_bn_relu_generic(us[0][1]) or us[0][1].operands[0] != O:
                    why["maxpool output is not read first by a BN+ReLU"] = 1
                    continue
                inplace = (us[0][1].operands[3] == O)     # in-place BN: later readers of O read the BN result
                if [u for u in us[1:] if not (inplace and u[0] > us[0][0])]:
                    why["maxpool output has other readers besides the BN"] = 1
                    continue
                bn = us[0][1]; bi = us[0][0]
                alpha, beta, Y = bn.operands[1], bn.operands[2], bn.operands[3]
                if Y in (X, alpha, beta):
                    why["BN output aliases an input of the fused op"] = 1
                    continue
                ok = True
                for j in range(pi + 1, bi):
                    mid = ops[j]
                    if any(v in _dest_operands(mid) for v in (X, alpha, beta)) or \
                       any(v in [w for o2 in walk(mid) for w in o2.operation.operands] for v in (Y,) if Y != O):
                        ok = False
                        break
                if not ok:
                    why["a buffer of the fused op is touched in between"] = 1
                    continue
                attrs = {nm: pool.attributes[nm] for nm in attr_names(pool)}
                attrs["relu"] = ir.IntegerAttr.get(i64, 1)
                ir.Operation.create("rvm.maxpool_bn", operands=[X, Y, alpha, beta], attributes=attrs, ip=ir.InsertionPoint(bn))
                pool.erase(); bn.erase()
                n += 1
                if Y != O:
                    oo = _owner(O)
                    if oo is not None and oo.name == "memref.alloc":
                        rest = [o2.operation for o2 in blk.operations
                                if any(v == O for v in o2.operation.operands) and o2.operation.name != "memref.dealloc"]
                        if not rest:
                            for o2 in list(blk.operations):
                                if o2.operation.name == "memref.dealloc" and o2.operation.operands[0] == O:
                                    o2.operation.erase()
                            oo.erase()
                progress = True
                break
    if report is not None:
        report.update(why)
    return n


def lower_rest(mod):
    PassManager.parse("builtin.module(func.func(convert-linalg-to-loops),lower-affine,canonicalize)").run(mod.operation)


# ----------------------------------------------------------------------------- BatchNorm fusion (tensor level)
def _gen_body_ops(g):
    return [o.operation for o in g.regions[0].blocks[0].operations if o.operation.name != "linalg.yield"]


def _body_binop(g, name):
    """linalg.generic with 2 ins whose body is exactly `r = <name> in0, in1`; returns True if so."""
    ops = _gen_body_ops(g)
    if len(ops) != 1 or ops[0].name != name:
        return False
    a = list(g.regions[0].blocks[0].arguments)
    return list(ops[0].operands) == [a[0], a[1]]


def _is_relu(g):
    """1-input generic: select(cmpf ugt/ogt in, 0.0), in, 0.0)."""
    if len(g.operands) != 2:
        return False
    ops = _gen_body_ops(g)
    if [o.name for o in ops] != ["arith.cmpf", "arith.select"]:
        return False
    a = list(g.regions[0].blocks[0].arguments)
    c, s = ops
    if c.operands[0] != a[0] or not _is_zero_const(c.operands[1]):
        return False
    if not re.search(r"\b(ugt|ogt)\b", str(c)):
        return False
    return s.operands[1] == a[0] and _is_zero_const(s.operands[2]) and s.operands[0] == c.results[0]


def _sole_user(v):
    us = list(v.uses)
    if len(us) != 1:
        return None
    return _op(us[0].owner), us[0].operand_number


def _const_eval(v, blobs, env=None):
    """Evaluate a constant tensor/scalar value in numpy. Handles arith.constant (dense / dense_resource / scalar),
    expand/collapse_shape, 1-D identity-map linalg.generic over such values, and scalar float arith/math ops."""
    if env is not None and v in env:
        return env[v]
    o = _owner(v)
    if o is None:
        raise Unsupported("non-constant value")
    n = o.name
    if n == "arith.constant":
        at = o.attributes["value"]
        s = str(at)
        m = re.search(r"dense_resource<([^>]+)>", s)
        if m:
            raw = blobs[m.group(1)]
            shp = list(ir.RankedTensorType(v.type).shape)
            return np.frombuffer(raw, dtype=np.float32).reshape(shp)
        if isa(at, ir.FloatAttr) if hasattr(ir.FloatAttr, "isinstance") else isinstance(at, ir.FloatAttr):
            return np.float64(ir.FloatAttr(at).value)
        d = ir.DenseElementsAttr(at)
        shp = list(ir.RankedTensorType(v.type).shape)
        return (np.full(shp, d.get_splat_value(), np.float32) if d.is_splat else np.array(d, dtype=np.float32).reshape(shp))
    if n in ("tensor.expand_shape", "tensor.collapse_shape"):
        shp = list(ir.RankedTensorType(v.type).shape)
        return _const_eval(o.operands[0], blobs, env).reshape(shp)
    if n in ("arith.truncf", "arith.extf"):
        x = _const_eval(o.operands[0], blobs, env)
        return np.asarray(x).astype(np.float32 if "f32" in str(v.type) else np.float64)[()] if np.ndim(x) == 0 else \
            np.asarray(x).astype(np.float32 if "f32" in str(v.type) else np.float64)
    bin_ops = {"arith.addf": np.add, "arith.subf": np.subtract, "arith.mulf": np.multiply, "arith.divf": np.divide}
    if n in bin_ops:
        return bin_ops[n](_const_eval(o.operands[0], blobs, env), _const_eval(o.operands[1], blobs, env))
    if n == "math.rsqrt":
        x = _const_eval(o.operands[0], blobs, env)
        return (1.0 / np.sqrt(x)).astype(np.asarray(x).dtype)
    if n == "math.sqrt":
        return np.sqrt(_const_eval(o.operands[0], blobs, env))
    if n == "linalg.generic":
        nin = len(o.operands) - 1
        ins = [_const_eval(o.operands[i], blobs, env) for i in range(nin)]
        if any(np.ndim(a) != 1 or a.shape != ins[0].shape for a in ins):
            raise Unsupported("only equal-shape 1-D constant generics")
        blk = o.regions[0].blocks[0]
        args = list(blk.arguments)
        loc = {args[i]: ins[i] for i in range(nin)}
        res = None
        for bo in blk.operations:
            bo = bo.operation
            if bo.name == "linalg.yield":
                res = _const_eval(bo.operands[0], blobs, loc)
            else:
                loc[bo.results[0]] = _const_eval(bo.results[0], blobs, loc)
        return res
    raise Unsupported(f"cannot evaluate {n}")


def fuse_batchnorm(mod, blobs, func="main"):
    """Fold  y = relu?(((x - mean) * invstd) * gamma + beta)  (torch-mlir's inference BatchNorm decomposition:
    four per-channel 4-D generics, optional ReLU generic) into ONE generic  y = relu?(x * alpha + beta')
    with alpha = gamma*invstd, beta' = beta - mean*alpha, evaluated at lowering time.
    Returns (n_folded, n_with_relu, n_skipped)."""
    fn = None
    for f in mod.body.operations:
        if f.operation.name == "func.func" and ir.StringAttr(f.operation.attributes["sym_name"]).value == func:
            fn = f.operation
    blk = fn.regions[0].blocks[0]
    chains, skipped = [], 0
    for op in list(blk.operations):
        op = op.operation
        if op.name != "linalg.generic" or len(op.operands) != 3 or not _body_binop(op, "arith.subf"):
            continue
        try:
            x, mean_v = op.operands[0], op.operands[1]
            xt = ir.RankedTensorType(x.type)
            xs = list(xt.shape)
            if len(xs) != 4 or xs[0] != 1 or str(xt.element_type) != "f32":
                raise Unsupported("not 4-D N=1 f32")
            C = xs[1]
            chain, params, cur = [op], [mean_v], op
            for kind in ("arith.mulf", "arith.mulf", "arith.addf"):
                su = _sole_user(cur.results[0])
                if su is None or su[1] != 0:
                    raise Unsupported("chain broken")
                nxt = su[0]
                if nxt.name != "linalg.generic" or len(nxt.operands) != 3 or not _body_binop(nxt, kind):
                    raise Unsupported("chain broken")
                chain.append(nxt); params.append(nxt.operands[1]); cur = nxt
            relu = False
            su = _sole_user(cur.results[0])
            if su is not None and su[0].name == "linalg.generic" and _is_relu(su[0]):
                chain.append(su[0]); relu = True
            vals = []
            for p in params:
                if list(ir.RankedTensorType(p.type).shape) != [C, 1, 1]:
                    raise Unsupported("param shape")
                a = np.asarray(_const_eval(p, blobs), np.float64).reshape(-1)
                if a.size != C:
                    raise Unsupported("param size")
                vals.append(a)
            mean, invstd, gamma, beta = vals
            alpha = invstd * gamma
            beta2 = beta - mean * alpha
            chains.append((chain, x, xt, C, alpha, beta2, relu))
        except (Unsupported, KeyError, ValueError):
            skipped += 1
    n_relu = 0
    f32 = ir.F32Type.get()
    for chain, x, xt, C, alpha, beta2, relu in chains:
        last = chain[-1]
        ctt = ir.RankedTensorType.get([C, 1, 1], f32)
        with ir.InsertionPoint(last):
            ca = ir.Operation.create("arith.constant", results=[ctt], attributes={
                "value": ir.DenseElementsAttr.get(alpha.astype(np.float32).reshape(C, 1, 1), type=ctt)})
            cb = ir.Operation.create("arith.constant", results=[ctt], attributes={
                "value": ir.DenseElementsAttr.get(beta2.astype(np.float32).reshape(C, 1, 1), type=ctt)})
        xty = str(xt)
        body_relu = ("%c = arith.cmpf ugt, %t2, %z : f32\n      %r = arith.select %c, %t2, %z : f32\n"
                     if relu else "%r = arith.addf %t, %ib : f32\n")
        if relu:
            body = ("%t = arith.mulf %in, %ia : f32\n      %t2 = arith.addf %t, %ib : f32\n      %z = arith.constant 0.000000e+00 : f32\n"
                    "      %c = arith.cmpf ugt, %t2, %z : f32\n      %r = arith.select %c, %t2, %z : f32\n")
        else:
            body = "%t = arith.mulf %in, %ia : f32\n      %r = arith.addf %t, %ib : f32\n"
        proto = f"""
#m4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#mc = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
func.func @proto(%x: {xty}, %a: tensor<{C}x1x1xf32>, %b: tensor<{C}x1x1xf32>, %o: {xty}) -> {xty} {{
  %r = linalg.generic {{indexing_maps = [#m4, #mc, #mc, #m4], iterator_types = ["parallel","parallel","parallel","parallel"]}}
      ins(%x, %a, %b : {xty}, tensor<{C}x1x1xf32>, tensor<{C}x1x1xf32>) outs(%o : {xty}) {{
    ^bb0(%in: f32, %ia: f32, %ib: f32, %out: f32):
      {body}      linalg.yield %r : f32
  }} -> {xty}
  return %r : {xty}
}}"""
        pm = ir.Module.parse(proto)
        gen = [o.operation for o in list(pm.body.operations)[0].regions[0].blocks[0].operations
               if o.operation.name == "linalg.generic"][0]
        new = gen.clone(ip=ir.InsertionPoint(last))
        outs = last.operands[len(last.operands) - 1]
        new.operands[0] = x
        new.operands[1] = ca.results[0]
        new.operands[2] = cb.results[0]
        new.operands[3] = outs
        last.results[0].replace_all_uses_with(new.results[0])
        for o in reversed(chain):
            o.erase()
        n_relu += int(relu)
    # dead-code cleanup of the now-unused per-channel precomputation
    pure = {"arith.constant", "tensor.expand_shape", "tensor.collapse_shape", "linalg.generic", "tensor.empty", "arith.truncf"}
    changed = True
    while changed:
        changed = False
        for o in reversed(list(blk.operations)):
            o = o.operation
            if o.name in pure and all(len(list(r.uses)) == 0 for r in o.results):
                o.erase(); changed = True
    return len(chains), n_relu, skipped


# ----------------------------------------------------------------------------- emitter
class Slot:
    def __init__(self, idx, size):
        self.idx, self.size, self.busy_until = idx, size, -1


class Emitter:
    def __init__(self, mod, blobs, checksums=False, fp16_weights=False, nhwc=False):
        self.mod, self.blobs = mod, blobs
        self.fp16w, self.h16_vals, self.h16_names = fp16_weights, set(), set()
        self.nhwc, self.hwio_vals, self.hwio_names = nhwc, set(), set()
        if nhwc and checksums:
            raise Unsupported('--checksums is not supported with --nhwc-fp16')
        self.ck, self.ck_idx = checksums, 0
        self.lines, self.slots = [], []
        self.names, self.mr = {}, {}
        self.cnt = 0
        self.weights = bytearray()
        self.w_off = {}
        self.max = {"A": 1, "B": 1, "C": 1}
        self.globals = {}
        self.yield_targets = []
        self.depth = 0
        for op in mod.body.operations:
            if op.operation.name == "memref.global":
                self.globals[ir.StringAttr(op.operation.attributes["sym_name"]).value] = op.operation
        self.rvm_calls = 0
        self.n_ext = 0

    # -- naming / output
    def fresh(self, p="t"):
        self.cnt += 1
        return f"{p}{self.cnt}"

    def name(self, v):
        if v not in self.names:
            raise Unsupported(f"value used before definition: {v}")
        return self.names[v]

    def emit(self, s):
        self.lines.append("    " * (self.depth + 1) + s)

    # -- weights
    def add_weight(self, glob, h16=False, hwio=False):
        sym = ir.StringAttr(glob.attributes["sym_name"]).value
        if (sym, h16, hwio) in self.w_off:
            return self.w_off[(sym, h16, hwio)]
        mt = ir.MemRefType(ir.TypeAttr(glob.attributes["type"]).value)
        shape = list(mt.shape)
        n = int(np.prod(shape)) if shape else 1
        iv = glob.attributes["initial_value"]
        s = str(iv)
        m = re.search(r"dense_resource<([^>]+)>", s)
        if m:
            raw = self.blobs.get(m.group(1))
            if raw is None:
                raise Unsupported(f"resource blob {m.group(1)} not found in file")
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            d = ir.DenseElementsAttr(iv)
            arr = np.full(n, d.get_splat_value(), np.float32) if d.is_splat else np.array(d, dtype=np.float32)
        if arr.size != n:
            raise Unsupported(f"{sym}: blob has {arr.size} floats, type needs {n}")
        if hwio:                                   # conv weights [F,C,KH,KW] -> [KH,KW,C,F]
            if len(shape) != 4:
                raise Unsupported(f"{sym}: HWIO layout needs a 4-D conv weight")
            arr = np.ascontiguousarray(arr.reshape(shape).transpose(2, 3, 1, 0)).reshape(-1)
        pad = (-len(self.weights)) % 64
        self.weights.extend(b"\0" * pad)
        off = len(self.weights)
        # fp16 weights: numpy float32->float16 is round-to-nearest-even, same as the target's fcvt.h.s
        self.weights.extend(arr.astype(np.float16 if h16 else np.float32).tobytes())
        self.w_off[(sym, h16, hwio)] = off
        return off

    # -- liveness for buffer reuse (top-level ops only)
    def prepass(self, top_ops):
        root, self.last_use, self.root_size = {}, {}, {}
        nid = 0
        for i, op in enumerate(top_ops):
            nm = op.operation.name
            for o in walk(op):
                for v in o.operation.operands:
                    r = root.get(v)
                    if r is not None:
                        self.last_use[r] = i
            if nm in ("memref.alloc", "memref.alloca"):
                root[op.operation.results[0]] = nid
                self.root_size[nid] = MR("x", op.operation.results[0].type).numel * 4
                self.last_use[nid] = i
                nid += 1
            elif nm in VIEWS:
                r = root.get(op.operation.operands[0])
                if r is not None:
                    root[op.operation.results[0]] = r
            elif nm == "func.return":
                for v in op.operation.operands:
                    if v in root:
                        self.last_use[root[v]] = 1 << 30
        self.root = root

    def assign_slot(self, rid, idx):
        need = self.root_size[rid]
        best = None
        for s in self.slots:
            if s.busy_until < idx and s.size >= need and (best is None or s.size < best.size):
                best = s
        if best is None:
            best = Slot(len(self.slots), need)
            self.slots.append(best)
        best.busy_until = self.last_use[rid]
        return best

    # -- addressing
    def addr(self, mr, idxs):
        terms = [str(mr.off)] if mr.off else []
        for i, s, d in zip(idxs, mr.strides, mr.shape):
            if d == 1 or s == 0:
                continue
            terms.append(self.name(i) if s == 1 else f"{self.name(i)}*{s}")
        return " + ".join(terms) if terms else "0"

    # -- function
    def emit_function(self, f):
        blk = f.regions[0].blocks[0]
        ft = ir.FunctionType(ir.TypeAttr(f.attributes["function_type"]).value)
        params = []
        for k, a in enumerate(blk.arguments):
            if not isa(a.type, ir.MemRefType):
                raise Unsupported("only memref arguments supported")
            nm = f"arg{k}"
            params.append(f"float* {nm}")
            if self.nhwc and len(ir.MemRefType(a.type).shape) == 4:
                m0 = MR(nm, a.type)
                _, C_, H_, W_ = m0.shape
                self.emit(f"static f16 nhwc_in{k}[{m0.numel}] __attribute__((aligned(64)));")
                self.emit(f"rvm_nchw_to_nhwc({nm}, nhwc_in{k}, {C_}, {H_}, {W_});")
                self.mr[a] = MR(f"nhwc_in{k}", a.type, h16=True, nhwc=True)
            else:
                self.mr[a] = MR(nm, a.type)
        if self.ck:
            self.emit("rvm_ck(-1, arg0, %d);" % self.mr[blk.arguments[0]].numel)
            self.emit("rvm_ck(-2, (const float*)weights_blob, WEIGHT_FLOATS);")
        top = list(blk.operations)
        if self.fp16w:
            self.find_fp16_weights(top)
        self.prepass(top)
        self.top_index = {id(o): i for i, o in enumerate(top)}
        for i, op in enumerate(top):
            self.cur_top = i
            self.annotate(i, op)
            self.emit_op(op)
        self.out_numel = None
        if len(ft.results) == 1 and isa(ft.results[0], ir.MemRefType):
            self.out_numel = MR("x", ft.results[0]).numel
        return ", ".join(params)

    def find_fp16_weights(self, top):
        """get_global results used ONLY as a conv weight (rvm.conv2d_nchw operand 1) or a matmul B
        (rvm.matmul operand 1) are stored as fp16, so the runtime needs no per-call fp32->fp16 conversion."""
        uses = {}
        for op in top:
            for o in walk(op):
                for k, v in enumerate(o.operation.operands):
                    uses.setdefault(v, []).append((o.operation.name, k))
        ok = {("rvm.conv2d_nchw", 1), ("rvm.matmul", 1), ("rvm.conv2d_epi", 1), ("rvm.conv2d_epi", 3), ("rvm.conv2d_epi", 4),
              ("rvm.maxpool_bn", 2), ("rvm.maxpool_bn", 3)}
        for op in top:
            if op.operation.name == "memref.get_global":
                u = uses.get(op.operation.results[0], [])
                if u and all(x in ok for x in u):
                    self.h16_vals.add(op.operation.results[0])
                    if self.nhwc and all(x in (("rvm.conv2d_nchw", 1), ("rvm.conv2d_epi", 1)) for x in u):
                        self.hwio_vals.add(op.operation.results[0])

    def annotate(self, i, op):
        """Emit a one-line /* #i ... */ marker before every top-level statement that generates real work, so
        attribute_trace.py can attribute executed instructions to layers / passes. No effect on code generation."""
        nm = op.operation.name
        if (nm in ("arith.constant", "memref.get_global", "memref.alloc", "memref.alloca", "memref.dealloc",
                   "func.return") or nm in VIEWS):
            return
        def shp(v):
            try:
                return "x".join(str(d) for d in ir.MemRefType(v.type).shape) or "scalar"
            except Exception:
                return "?"
        if nm.startswith("rvm."):
            ops = list(op.operation.operands)
            lab = f"{nm} " + " ".join(shp(v) for v in ops)
        elif nm == "memref.copy":
            lab = f"copy {shp(op.operation.operands[0])}"
        else:
            skip = {"arith.constant", "arith.index_cast", "arith.index_castui", "scf.yield", "scf.for", "scf.if",
                    "memref.load", "memref.store", "linalg.yield"}
            kinds, target = set(), None
            for o in walk(op):
                n2 = o.operation.name
                if n2 in ("memref.store",) and target is None:
                    target = shp(o.operation.operands[1])
                elif (n2.startswith("arith.") or n2.startswith("math.")) and n2 not in skip:
                    kinds.add(n2.split(".")[1])
            lab = f"loops[{','.join(sorted(kinds)) or 'copy/fill'}] -> {target or '?'}"
        self.emit(f"/* #{i} {lab} */")

    def emit_block(self, block):
        self.depth += 1
        for op in block.operations:
            self.emit_op(op)
        self.depth -= 1

    # -- single op
    def emit_op(self, op):
        o = op.operation
        nm = o.name
        res = list(o.results)

        if nm == "arith.constant":
            a = o.attributes["value"]
            t = res[0].type
            if isa(a, ir.IntegerAttr):
                val = ir.IntegerAttr(a).value
                if isa(t, ir.IntegerType) and ir.IntegerType(t).width == 1:
                    val = 1 if val else 0
                expr = str(val)
            elif isa(a, ir.FloatAttr):
                v = ir.FloatAttr(a).value
                f32 = isa(t, ir.F32Type)
                if math.isinf(v):
                    expr = ("-" if v < 0 else "") + ("__builtin_inff()" if f32 else "__builtin_inf()")
                elif math.isnan(v):
                    expr = '__builtin_nanf("")' if f32 else '__builtin_nan("")'
                else:
                    expr = repr(float(v)) + ("f" if f32 else "")
            else:
                raise Unsupported(f"arith.constant with non-scalar value: {a}")
            n = self.fresh()
            self.emit(f"const {ctype(t)} {n} = {expr};")
            self.names[res[0]] = n
            return

        if nm in BIN:
            a, b = (self.name(x) for x in o.operands)
            self.scalar(res[0], f"{a} {BIN[nm]} {b}")
            return
        if nm in ("arith.maximumf", "arith.maxnumf", "arith.minimumf", "arith.minnumf"):
            a, b = (self.name(x) for x in o.operands)
            fn = ("__builtin_fmax" if "max" in nm else "__builtin_fmin") + ("f" if isa(res[0].type, ir.F32Type) else "")
            self.scalar(res[0], f"{fn}({a}, {b})")
            return
        if nm in ("arith.maxsi", "arith.minsi"):
            a, b = (self.name(x) for x in o.operands)
            self.scalar(res[0], f"(({a}) {'>' if 'max' in nm else '<'} ({b}) ? ({a}) : ({b}))")
            return
        if nm == "arith.negf":
            self.scalar(res[0], f"-{self.name(o.operands[0])}")
            return
        if nm == "arith.select":
            c, a, b = (self.name(x) for x in o.operands)
            self.scalar(res[0], f"({c} ? {a} : {b})")
            return
        if nm == "arith.cmpf":
            p = ir.IntegerAttr(o.attributes["predicate"]).value
            a, b = (self.name(x) for x in o.operands)
            expr = {0: "0", 7: "1", 14: "0", 15: "1"}.get(p) or f"{a} {CMPF[p]} {b}"
            self.scalar(res[0], expr)
            return
        if nm == "arith.cmpi":
            p = ir.IntegerAttr(o.attributes["predicate"]).value
            a, b = (self.name(x) for x in o.operands)
            if p >= 6:
                a, b = f"(unsigned long){a}", f"(unsigned long){b}"
            self.scalar(res[0], f"{a} {CMPI[p]} {b}")
            return
        if nm in CASTS:
            self.scalar(res[0], f"({ctype(res[0].type)}){self.name(o.operands[0])}")
            return
        if nm in ("math.rsqrt", "math.sqrt", "math.absf"):
            x = self.name(o.operands[0])
            f32 = isa(res[0].type, ir.F32Type)
            sfx = "f" if f32 else ""
            one = "1.0f" if f32 else "1.0"
            expr = {"math.rsqrt": f"{one} / __builtin_sqrt{sfx}({x})", "math.sqrt": f"__builtin_sqrt{sfx}({x})",
                    "math.absf": f"__builtin_fabs{sfx}({x})"}[nm]
            self.scalar(res[0], expr)
            return

        # ---- memref
        if nm in ("memref.alloc", "memref.alloca"):
            if self.depth != 0:
                raise Unsupported("alloc inside a loop body")
            rid = self.root[res[0]]
            slot = self.assign_slot(rid, self.cur_top)
            n = self.fresh("m")
            if self.nhwc and len(ir.MemRefType(res[0].type).shape) == 4:
                self.emit(f"f16* const {n} = (f16*)pool{slot.idx};")
                self.mr[res[0]] = MR(n, res[0].type, h16=True, nhwc=True)
            else:
                self.emit(f"float* const {n} = (float*)pool{slot.idx};")
                self.mr[res[0]] = MR(n, res[0].type)
            return
        if nm == "memref.dealloc":
            return
        if nm == "memref.get_global":
            sym = ir.FlatSymbolRefAttr(o.attributes["name"]).value
            h16 = res[0] in self.h16_vals
            off = self.add_weight(self.globals[sym], h16, res[0] in self.hwio_vals)
            n = self.fresh("g")
            if res[0] in self.hwio_vals:
                self.hwio_names.add(n)
            if h16:
                self.h16_names.add(n)
                self.emit(f"const rvm_w16* const {n} = (const rvm_w16*)(weights_blob + {off});")
            else:
                self.emit(f"const float* const {n} = (const float*)(weights_blob + {off});")
            self.mr[res[0]] = MR(n, res[0].type)
            return
        if nm in VIEWS:
            if nm == "memref.subview":
                for k in ("static_offsets", "static_sizes", "static_strides"):
                    if any(v == -(1 << 63) for v in int_list(o.attributes[k])):
                        raise Unsupported("memref.subview with dynamic offsets/sizes/strides")
            src = self.mr[o.operands[0]]
            if src.h16:
                rt = ir.MemRefType(res[0].type)
                if nm == "memref.subview" or (len(rt.shape) == 4 and list(rt.shape) != src.shape):
                    raise Unsupported("view of an NHWC fp16 activation that changes its layout (use --fold-pad/--fold-pool)")
                if len(rt.shape) == 4:
                    m = MR(src.base, res[0].type, h16=True, nhwc=True)
                elif src.shape[2:] == [1, 1] and int(np.prod(rt.shape)) == src.numel:
                    m = MR(src.base, res[0].type, h16=True)     # [1,C,1,1] -> flat: same linear order
                else:
                    raise Unsupported(f"unsupported reshape of an NHWC fp16 activation {src.shape} -> {list(rt.shape)}")
            else:
                m = MR(src.base, res[0].type)
            self.mr[res[0]] = m
            return
        if nm == "memref.load":
            mr = self.mr[o.operands[0]]
            self.scalar(res[0], f"{mr.base}[{self.addr(mr, list(o.operands)[1:])}]")
            return
        if nm == "memref.store":
            v = self.name(o.operands[0])
            mr = self.mr[o.operands[1]]
            if mr.h16:
                v = f"RVM_H16({v})"
            self.emit(f"{mr.base}[{self.addr(mr, list(o.operands)[2:])}] = {v};")
            return
        if nm == "memref.copy":
            s, d = self.mr[o.operands[0]], self.mr[o.operands[1]]
            self.emit("{")
            self.depth += 1
            ivs = []
            for k, n in enumerate(s.shape):
                iv = self.fresh("c")
                ivs.append(iv)
                self.emit(f"for (long {iv} = 0; {iv} < {n}; {iv}++)")
            sa = " + ".join([str(s.off)] + [f"{iv}*{st}" for iv, st in zip(ivs, s.strides)])
            da = " + ".join([str(d.off)] + [f"{iv}*{st}" for iv, st in zip(ivs, d.strides)])
            self.emit(f"    {d.base}[{da}] = {s.base}[{sa}];")
            self.depth -= 1
            self.emit("}")
            return

        # ---- control flow
        if nm == "scf.for":
            lb, ub, st = (self.name(x) for x in list(o.operands)[:3])
            inits = list(o.operands)[3:]
            body = o.regions[0].blocks[0]
            iv = self.fresh("i")
            carried = []
            for init, res_v, barg in zip(inits, res, list(body.arguments)[1:]):
                if isa(init.type, ir.MemRefType):
                    raise Unsupported("memref iter_args")
                cn = self.fresh("r")
                self.emit(f"{ctype(init.type)} {cn} = {self.name(init)};")
                self.names[res_v] = cn
                self.names[barg] = cn
                carried.append(cn)
            self.names[body.arguments[0]] = iv
            self.emit(f"for (long {iv} = {lb}; {iv} < {ub}; {iv} += {st}) {{")
            self.yield_targets.append(carried)
            self.emit_block(body)
            self.yield_targets.pop()
            self.emit("}")
            return
        if nm == "scf.if":
            c = self.name(o.operands[0])
            outs = []
            for r in res:
                cn = self.fresh("r")
                self.emit(f"{ctype(r.type)} {cn};")
                self.names[r] = cn
                outs.append(cn)
            self.emit(f"if ({c}) {{")
            self.yield_targets.append(outs)
            self.emit_block(o.regions[0].blocks[0])
            if len(o.regions[1].blocks) > 0:
                self.emit("} else {")
                self.emit_block(o.regions[1].blocks[0])
            self.yield_targets.pop()
            self.emit("}")
            return
        if nm == "scf.yield":
            for tgt, v in zip(self.yield_targets[-1], o.operands):
                self.emit(f"{tgt} = {self.name(v)};")
            return
        if nm == "func.return":
            if len(o.operands) == 1:
                m = self.mr[o.operands[0]]
                if m.h16:
                    self.emit(f"static float retbuf[{m.numel}];")
                    if m.nhwc and m.shape[2:] != [1, 1]:       # NHWC fp16 -> NCHW fp32 (logical order)
                        _, C_, H_, W_ = m.shape
                        self.emit(f"for (int c = 0; c < {C_}; c++) for (int h = 0; h < {H_}; h++) for (int w = 0; w < {W_}; w++)")
                        self.emit(f"    retbuf[(c * {H_} + h) * {W_} + w] = (float){m.base}[(h * {W_} + w) * {C_} + c];")
                    else:
                        self.emit(f"for (long i = 0; i < {m.numel}; i++) retbuf[i] = (float){m.base}[i];")
                    self.emit("return retbuf;")
                else:
                    self.emit(f"return {m.ptr()};")
            else:
                self.emit("return 0;")
            return

        # ---- matrix ops
        if nm == "rvm.conv2d_nchw":
            self.emit_conv(o)
            return
        if nm == "rvm.matmul":
            self.emit_matmul(o)
            return
        if nm == "rvm.maxpool_nchw":
            self.emit_maxpool(o)
            return
        if nm == "rvm.conv2d_epi":
            self.emit_conv_epi(o)
            return
        if nm == "rvm.maxpool_bn":
            self.emit_maxpool_bn(o)
            return

        raise Unsupported(f"unsupported op: {nm}")

    def scalar(self, res, expr):
        n = self.fresh()
        self.emit(f"const {ctype(res.type)} {n} = {expr};")
        self.names[res] = n

    # -- matrix op emission
    def need_contig(self, what, *mrs):
        for m in mrs:
            if not m.contiguous():
                raise Unsupported(f"{what}: operand is not contiguous (strides {m.strides} for shape {m.shape})")

    def emit_conv_nhwc(self, o):
        i, w, out = (self.mr[x] for x in o.operands)
        if not (i.nhwc and out.nhwc):
            raise Unsupported("NHWC conv needs NHWC fp16 input and output buffers")
        if w.base not in self.hwio_names:
            raise Unsupported("NHWC conv weight is not stored as HWIO fp16")
        N, C, H, W = i.shape
        F, C2, KH, KW = w.shape
        N2, F2, OH, OW = out.shape
        if not (C == C2 and F == F2 and N == N2 == 1):
            raise Unsupported(f"conv shape mismatch {i.shape} {w.shape} {out.shape}")
        names = attr_names(o)
        a = {k: ir.IntegerAttr(o.attributes[k]).value for k in ("sh", "sw", "dh", "dw")}
        g = lambda k: ir.IntegerAttr(o.attributes[k]).value if k in names else 0
        pt, pl, pb, pr = g("pt"), g("pl"), g("pb"), g("pr")
        if "pt" in names and "pb" not in names:
            raise Unsupported("fold-pad output lacks pb/pr; regenerate with the current lower_to_c.py")
        if max(pt, pl, pb, pr) > 255 or max(H, W, OH, OW) > 0xFFFF:
            raise Unsupported("conv geometry exceeds the msetinsh/msetoutsh field widths")
        self.max["A"] = max(self.max["A"], F * C * KH * KW)
        overwrite = g("ow") == 1
        fn = "rvm_conv_unfold_wh" if overwrite else "rvm_conv_unfold_wh_acc"
        if not overwrite:
            self.max["C"] = max(self.max["C"], F * OH * OW)
        self.emit(f"{fn}({i.ptr()}, {w.ptr()}, {out.ptr()}, {C}, {H}, {W}, {F}, {KH}, {KW}, {OH}, {OW}, "
                  f"{a['sh']}, {a['sw']}, {a['dh']}, {a['dw']}, {pt}, {pb}, {pl}, {pr});")
        self.rvm_calls += 1

    def emit_conv_epi(self, o):
        if not self.nhwc:
            raise Unsupported("rvm.conv2d_epi needs --nhwc-fp16")
        ops = [self.mr[x] for x in o.operands]
        i, w, out, al, be = ops[:5]
        has_add = len(ops) > 5
        if not (i.nhwc and out.nhwc and (not has_add or (ops[5].nhwc and ops[6].nhwc))):
            raise Unsupported("fused conv epilogue needs NHWC fp16 buffers")
        if w.base not in self.hwio_names or al.base not in self.h16_names or be.base not in self.h16_names:
            raise Unsupported("fused conv epilogue: weights / alpha / beta are not fp16 blobs")
        N, C, H, W = i.shape
        F, C2, KH, KW = w.shape
        names = attr_names(o)
        a = {k: ir.IntegerAttr(o.attributes[k]).value for k in ("sh", "sw", "dh", "dw")}
        g = lambda k: ir.IntegerAttr(o.attributes[k]).value if k in names else 0
        pt, pl, pb, pr = g("pt"), g("pl"), g("pb"), g("pr")
        if "pt" in names and "pb" not in names:
            raise Unsupported("fold-pad output lacks pb/pr; regenerate with the current lower_to_c.py")
        _, F2, OH, OW = out.shape
        if not (C == C2 and F == F2 and N == 1):
            raise Unsupported(f"conv shape mismatch {i.shape} {w.shape} {out.shape}")
        if max(pt, pl, pb, pr) > 255 or max(H, W, OH, OW) > 0xFFFF:
            raise Unsupported("conv geometry exceeds the msetinsh/msetoutsh field widths")
        self.max["A"] = max(self.max["A"], F * C * KH * KW)
        self.max["B"] = max(self.max["B"], 2 * F)
        add_s, add_o = (ops[5].ptr(), ops[6].ptr()) if has_add else ("0", "0")
        if has_add and "skip_a" in names:
            add_o = "0"
        self.emit(f"rvm_conv_unfold_epi_wh({i.ptr()}, {w.ptr()}, {out.ptr()}, {al.ptr()}, {be.ptr()}, {add_s}, {add_o}, "
                  f"{C}, {H}, {W}, {F}, {KH}, {KW}, {OH}, {OW}, {a['sh']}, {a['sw']}, {a['dh']}, {a['dw']}, {pt}, {pb}, {pl}, {pr});")
        self.rvm_calls += 1

    def emit_conv(self, o):
        if self.nhwc:
            return self.emit_conv_nhwc(o)
        i, w, out = (self.mr[x] for x in o.operands)
        self.need_contig("conv", i, w, out)
        N, C, H, W = i.shape
        F, C2, KH, KW = w.shape
        N2, F2, OH, OW = out.shape
        if not (C == C2 and F == F2 and N == N2):
            raise Unsupported(f"conv shape mismatch {i.shape} {w.shape} {out.shape}")
        a = {k: ir.IntegerAttr(o.attributes[k]).value for k in ("sh", "sw", "dh", "dw")}
        self.max["A"] = max(self.max["A"], F * C * KH * KW)
        self.max["B"] = max(self.max["B"], C * KH * KW * OH * OW, C * H * W)
        self.max["C"] = max(self.max["C"], F * OH * OW)
        fn = "rvm_conv2d_nchw_wh" if w.base in self.h16_names else "rvm_conv2d_nchw"
        names = attr_names(o)
        ext = ""
        if any(k in names for k in ("pt", "pl", "ow")):      # produced by fold_pad_and_fill
            g = lambda k: ir.IntegerAttr(o.attributes[k]).value if k in names else 0
            fn += "_x"
            ext = f", {g('pt')}, {g('pl')}, {g('ow')}"
            self.n_ext += 1
        for n in range(N):
            self.emit(f"{fn}({i.ptr()} + {n * C * H * W}, {w.ptr()}, {out.ptr()} + {n * F * OH * OW}, "
                      f"{C}, {H}, {W}, {F}, {KH}, {KW}, {OH}, {OW}, {a['sh']}, {a['sw']}, {a['dh']}, {a['dw']}{ext});")
        self.rvm_calls += 1
        self.checkpoint(out)

    def emit_maxpool_bn(self, o):
        if not self.nhwc:
            raise Unsupported("rvm.maxpool_bn needs --nhwc-fp16")
        i, out, al, be = (self.mr[x] for x in o.operands)
        if not (i.nhwc and out.nhwc) or al.base not in self.h16_names or be.base not in self.h16_names:
            raise Unsupported("fused maxpool+BN needs NHWC fp16 buffers and fp16 alpha/beta blobs")
        _, C, H, W = i.shape
        _, _, OH, OW = out.shape
        a = {k: ir.IntegerAttr(o.attributes[k]).value for k in ("kh", "kw", "sh", "sw", "dh", "dw", "pt", "pl")}
        if a["dh"] != 1 or a["dw"] != 1:
            raise Unsupported("maxpool with dilation")
        self.max["B"] = max(self.max["B"], 2 * C)
        self.emit(f"rvm_maxpool_bn_relu_nhwc({i.ptr()}, {out.ptr()}, {al.ptr()}, {be.ptr()}, {C}, {H}, {W}, {OH}, {OW}, "
                  f"{a['kh']}, {a['kw']}, {a['sh']}, {a['sw']}, {a['pt']}, {a['pl']});")

    def emit_maxpool(self, o):
        i, out = (self.mr[x] for x in o.operands)
        if self.nhwc:
            if not (i.nhwc and out.nhwc):
                raise Unsupported("NHWC maxpool needs NHWC fp16 buffers")
            _, C, H, W = i.shape
            _, _, OH, OW = out.shape
            a = {k: ir.IntegerAttr(o.attributes[k]).value for k in ("kh", "kw", "sh", "sw", "dh", "dw", "pt", "pl")}
            if a["dh"] != 1 or a["dw"] != 1:
                raise Unsupported("maxpool with dilation")
            self.emit(f"rvm_maxpool_nhwc({i.ptr()}, {out.ptr()}, {C}, {H}, {W}, {OH}, {OW}, {a['kh']}, {a['kw']}, "
                      f"{a['sh']}, {a['sw']}, {a['pt']}, {a['pl']});")
            return
        self.need_contig("maxpool", i, out)
        N, C, H, W = i.shape
        N2, C2, OH, OW = out.shape
        if N != N2 or C != C2:
            raise Unsupported(f"maxpool shape mismatch {i.shape} {out.shape}")
        a = {k: ir.IntegerAttr(o.attributes[k]).value for k in ("kh", "kw", "sh", "sw", "dh", "dw", "pt", "pl")}
        if a["dh"] != 1 or a["dw"] != 1:
            raise Unsupported("maxpool with dilation")
        for n in range(N):
            self.emit(f"rvm_maxpool_nchw({i.ptr()} + {n * C * H * W}, {out.ptr()} + {n * C * OH * OW}, "
                      f"{C}, {H}, {W}, {OH}, {OW}, {a['kh']}, {a['kw']}, {a['sh']}, {a['sw']}, {a['pt']}, {a['pl']});")
        self.checkpoint(out)

    def checkpoint(self, mr):
        if self.ck:
            self.ck_idx += 1
            self.emit(f"rvm_ck({self.ck_idx}, {mr.ptr()}, {mr.numel});")

    def emit_matmul(self, o):
        if self.nhwc:
            raise Unsupported("rvm.matmul is not supported with --nhwc-fp16 (the FC layer must be a 1x1 conv)")
        a, b, c = (self.mr[x] for x in o.operands)
        self.need_contig("matmul", a, b, c)
        M, K = a.shape
        K2, N = b.shape
        if K != K2 or c.shape != [M, N]:
            raise Unsupported(f"matmul shape mismatch {a.shape} {b.shape} {c.shape}")
        self.max["A"] = max(self.max["A"], M * K)
        self.max["B"] = max(self.max["B"], K * N)
        self.max["C"] = max(self.max["C"], M * N)
        fn = "rvm_matmul_bh" if b.base in self.h16_names else "rvm_matmul"
        self.emit(f"{fn}({a.ptr()}, {b.ptr()}, {c.ptr()}, {M}, {N}, {K});")
        self.rvm_calls += 1
        self.checkpoint(c)


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("outdir")
    ap.add_argument("--input", help="input.bin (fp32 NCHW) to embed; omit to use zeros")
    ap.add_argument("--func", default="main")
    ap.add_argument("--checksums", action="store_true", help="print a checksum after every GEMM (debug)")
    ap.add_argument("--fp16-weights", action="store_true",
                    help="store conv weights / matmul B as fp16 in weights.bin (no per-inference conversion)")
    ap.add_argument("--fold-pad", action="store_true",
                    help="fold explicit zero-padding into the conv and drop the zero-fill + accumulate of conv outputs")
    ap.add_argument("--nhwc-fp16", action="store_true",
                    help="fp16 NHWC activations + hardware-unfold (mlufa) convolutions, HWIO fp16 weights "
                         "(implies --fp16-weights; use with --fold-pad --fold-pool)")
    ap.add_argument("--fuse-epilogue", action="store_true",
                    help="with --nhwc-fp16: fuse BN+ReLU (and the residual add) into the conv output tile (needs --fuse-bn)")
    ap.add_argument("--fuse-epilogue-mode", choices=["all", "bn", "add"], default="all",
                    help="debug: fuse only conv+BN+ReLU ('bn'), only conv+add+BN+ReLU ('add'), or both")
    ap.add_argument("--fuse-pool-bn", action="store_true",
                    help="with --nhwc-fp16 --fold-pool --fuse-bn: fuse the BN+ReLU after the maxpool into it (RVV kernel, like riscv-dnn)")
    ap.add_argument("--fold-pool", action="store_true",
                    help="fold the -inf pad (+ its fill/copy and the output pre-fill) into a border-clamped maxpool")
    ap.add_argument("--fuse-bn", action="store_true",
                    help="fold inference BatchNorm (+ReLU) into a single y=relu(x*alpha+beta) pass per layer")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"reading {args.mlir} ...")
    text = open(args.mlir).read()
    blobs = parse_resources(text)
    print(f"  {len(blobs)} weight blobs, {sum(len(b) for b in blobs.values()) / 1e6:.1f} MB")

    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        mod = ir.Module.parse(text)
        del text
        if args.fuse_bn:
            n_ok, n_relu, n_skip = fuse_batchnorm(mod, blobs, args.func)
            print(f"fuse-bn: {n_ok} BatchNorm chains folded ({n_relu} with ReLU), {n_skip} skipped")
        print("bufferizing ...")
        bufferize(mod)
        print(f"marking matrix ops: {mark_matrix_ops(mod)} conv/matmul -> rvm.*")
        if args.fold_pad:
            n_pad, n_ow = fold_pad_and_fill(mod)
            print(f"fold-pad: {n_pad} pads folded into conv, {n_ow} conv outputs overwrite-only")
        if args.fold_pool:
            print(f"fold-pool: {fold_maxpool(mod)} maxpool(s) rewritten to rvm.maxpool_nchw")
        if args.fuse_epilogue:
            if not args.nhwc_fp16:
                sys.exit("--fuse-epilogue needs --nhwc-fp16")
            why = {}
            n_bn, n_add = fuse_conv_epilogue(mod, args.fuse_epilogue_mode, why)
            print(f"fuse-epilogue: {n_bn} conv+BN+ReLU and {n_add} conv+add+BN+ReLU fused")
            for k, v in sorted(why.items(), key=lambda kv: -kv[1]):
                print(f"  not fused ({v}x): {k}")
        if args.fuse_pool_bn:
            if not args.nhwc_fp16:
                sys.exit("--fuse-pool-bn needs --nhwc-fp16")
            pw = {}
            print(f"fuse-pool-bn: {fuse_pool_bn(mod, pw)} maxpool+BN+ReLU fused" + "".join(f"  [{k}]" for k in pw))
        print("lowering remaining linalg to loops ...")
        lower_rest(mod)

        fn = None
        for f in mod.body.operations:
            if f.operation.name == "func.func" and ir.StringAttr(f.operation.attributes["sym_name"]).value == args.func:
                fn = f.operation
        if fn is None:
            sys.exit(f"function @{args.func} not found")

        em = Emitter(mod, blobs, args.checksums, args.fp16_weights or args.nhwc_fp16, args.nhwc_fp16)
        params = em.emit_function(fn)
        em.weights.extend(b"\0" * ((-len(em.weights)) % 4))
        in_numel = MR("x", fn.regions[0].blocks[0].arguments[0].type).numel

    # absolute paths: .incbin searches the -I directories in order, so a stale weights.bin in an earlier
    # include dir (e.g. the project root) would silently win over the freshly generated one.
    wpath = os.path.abspath(os.path.join(args.outdir, "weights.bin"))
    ipath = os.path.abspath(os.path.join(args.outdir, "input.bin"))
    pool = "\n".join(f"static char pool{s.idx}[{s.size}] __attribute__((aligned(64)));" for s in em.slots)
    out_n = em.out_numel or 0
    total_pool = sum(s.size for s in em.slots)
    src = f"""/* generated by lower_to_c.py -- do not edit */
{"#define RVM_CHECKSUMS 1" if args.checksums else ""}
{"#define RVM_NHWC 1" if args.nhwc_fp16 else ""}
#define WEIGHT_FLOATS {len(em.weights) // 4}
#define RVM_MAX_A {em.max['A']}
#define RVM_MAX_B {em.max['B']}
#define RVM_MAX_C {em.max['C']}
#include <stdint.h>
#include <stdio.h>
#ifdef HOST
#define EMBED(var, file) extern uint8_t var[]; \\
  __asm__(".section .scdata.params,\\"a\\",@progbits\\n\\t.global " #var "\\n\\t.balign 64\\n" #var ":\\n\\t.incbin \\"" file "\\"\\n\\t.previous\\n")
#else
#include "incbin.h"
#define EMBED(var, file) INCBIN(var, file, ".scdata.params")
#endif
#include "rvm_runtime.h"

EMBED(weights_blob, "{wpath}");
{('EMBED(input_blob, "%s");' % ipath) if args.input else 'static float input_blob[1] ;'}

/* activation buffers (reused by liveness): {len(em.slots)} slots, {total_pool / 1e6:.1f} MB */
{pool}

static float* {args.func}_forward({params or 'void'}) {{
{chr(10).join(em.lines)}
}}

int main(void) {{
"""
    if args.input:
        src += "    float* out = %s_forward((float*)input_blob);\n" % args.func
    else:
        src += "    static float zeros[%d];\n    float* out = %s_forward(zeros);\n" % (
            in_numel, args.func)
    src += f"""    for (int i = 0; i < {out_n}; i++) {{
        union {{ float f; unsigned u; }} x;
        x.f = out[i];
        printf("LOGIT %d %x\\n", i, x.u);
    }}
    printf("DONE\\n");
    return 0;
}}
"""
    open(os.path.join(args.outdir, "model.c"), "w").write(src)
    open(os.path.join(args.outdir, "weights.bin"), "wb").write(bytes(em.weights))
    print(f"wrote {args.outdir}/model.c  ({len(em.lines)} lines, {em.rvm_calls} matrix GEMM calls, "
          f"{len(em.slots)} buffer slots = {total_pool / 1e6:.1f} MB)")
    print(f"wrote {args.outdir}/weights.bin ({len(em.weights) / 1e6:.1f} MB)  fp16 weight tensors: {len(em.h16_names)}")
    print(f"scratch: A={em.max['A']} B={em.max['B']} C={em.max['C']} fp16 elements")


if __name__ == "__main__":
    try:
        main()
    except Unsupported as e:
        sys.exit(f"UNSUPPORTED: {e}")
