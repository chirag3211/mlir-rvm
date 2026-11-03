#!/bin/bash
# Checks whether weights.bin really is in memory on spike (plain and with ~41 MB of .bss like the model).
# Prereq: build_dbg/{weights.bin,input.bin} from run_model_debug.sh (or set OUTW to another dir that has them).
cd "$(dirname "$0")"; HERE=$PWD
R="${R:-$HOME/test/riscv-dnn}"
CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"
SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"
OUTW="${OUTW:-$HERE/build_dbg}"
OUT="$HERE/wcheck_build"; mkdir -p "$OUT"
[ -f "$OUTW/weights.bin" ] || { echo "no $OUTW/weights.bin"; exit 1; }
WW=$(( $(stat -c %s "$OUTW/weights.bin") / 4 ))
BASE="--target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
   -mabi=lp64d -mcmodel=medany -static -O2 -fno-common -fno-builtin-printf -std=gnu99 \
   -I/usr/include/newlib -I$R/include -I$R/include/matrix -I$R/include/common -I$R/src -I$HERE -I$OUTW -DWEIGHT_WORDS=$WW"
$CL $BASE -c "$R/include/common/crt.S"      -o "$OUT/crt.o"
$CL $BASE -c "$R/include/common/syscalls.c" -o "$OUT/sc.o"

# host expectation: nonzero words per chunk
python3 - "$OUTW/weights.bin" > "$OUT/expected.txt" <<'PY'
import sys, numpy as np
w = np.fromfile(sys.argv[1], dtype=np.uint32)
print("W0..5", " ".join("%x" % x for x in w[:6]))
for c in range(0, (len(w) + (1<<20) - 1) >> 20):
    s = w[c<<20:(c+1)<<20]
    print("CHUNK %d nz=%d of %d" % (c, int((s != 0).sum()), len(s)))
PY

for v in nobss bss; do
  X=""; [ $v = bss ] && X="-DMIMIC_BSS"
  echo; echo "================ wcheck: $v ================"
  $CL $BASE $X -c wcheck.c -o "$OUT/wcheck_$v.o" || exit 1
  $CL --target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
      -static -nostdlib -fuse-ld=lld -T "$R/include/common/test.ld" \
      "$OUT/wcheck_$v.o" "$OUT/crt.o" "$OUT/sc.o" -o "$OUT/wcheck_$v.elf" || exit 1
  $SPIKE -p1 --isa=rv64gcv_zfh_zvfh_matrix "$OUT/wcheck_$v.elf" > "$OUT/wcheck_$v.log"
  echo "spike exit code: $?"
  head -4 "$OUT/wcheck_$v.log"
  echo "expected: $(head -1 "$OUT/expected.txt")"
  python3 - "$OUT/expected.txt" "$OUT/wcheck_$v.log" <<'PY'
import re, sys
exp = dict((int(m.group(1)), (int(m.group(2)), int(m.group(3)))) for m in re.finditer(r"CHUNK (\d+) nz=(\d+) of (\d+)", open(sys.argv[1]).read()))
got = dict((int(m.group(1)), (int(m.group(2)), int(m.group(3)))) for m in re.finditer(r"CHUNK (\d+) nz=(\d+) of (\d+)", open(sys.argv[2], errors="replace").read()))
bad = [c for c in sorted(exp) if got.get(c) != exp[c]]
print("chunks expected=%d got=%d MISMATCHING=%d" % (len(exp), len(got), len(bad)))
for c in bad[:30]:
    print("  chunk %d (words %d..): host nz=%s  spike nz=%s" % (c, c << 20, exp[c][0], got.get(c, ("missing",))[0]))
PY
done

echo; echo "================ ELF layout ================"
RE=""
for t in "$(dirname "$CL")/llvm-readelf" riscv64-unknown-elf-readelf readelf llvm-readelf; do command -v "$t" >/dev/null 2>&1 && { RE="$t"; break; }; done
NM=""
for t in "$(dirname "$CL")/llvm-nm" riscv64-unknown-elf-nm nm llvm-nm; do command -v "$t" >/dev/null 2>&1 && { NM="$t"; break; }; done
for ELF in "$OUT/wcheck_bss.elf" "$HERE/build/resnet50.elf"; do
  [ -f "$ELF" ] || continue
  echo "---- $ELF   (readelf=$RE nm=$NM)"
  [ -n "$RE" ] && "$RE" -S -W "$ELF" | head -40 && "$RE" -l -W "$ELF" | head -30
  [ -n "$NM" ] && "$NM" "$ELF" | grep -Ei 'weights_blob|input_blob|_end|bss|edata|stack|heap|tohost|pool[0-9]|rvm_tmp' || true
done
echo; echo "---- test.ld"; cat "$R/include/common/test.ld"
echo; echo "---- crt.S"; cat "$R/include/common/crt.S"
echo; echo "---- incbin.h"; cat "$R/include/incbin.h" 2>/dev/null || find "$R/include" -name incbin.h -exec cat {} \;
