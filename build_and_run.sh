#!/bin/bash
# resnet50_v2.mlir -> model.c -> RISC-V matrix ELF -> spike, then compare logits with PyTorch.
# Prereq: generate_resnet50.py has produced resnet50_v2.mlir, input.bin, ref_logits.bin (see the 3 added lines).
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
R="${R:-$HOME/test/riscv-dnn}"
CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"
SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"
BUILD="${BUILD:-build}"; MLIR="${MLIR:-resnet50_v2.mlir}"; INP="${INP:-input.bin}"; REF="${REF:-ref_logits.bin}"
OUT="$HERE/$BUILD"; mkdir -p "$OUT"

# Env overrides (defaults = the torchvision model):  MLIR=... INP=... REF=... BUILD=build_dir
# 1. MLIR -> C  (replaces bufferize.py + transpile_to_c.py)
cp "$INP" "$OUT/input.bin"
python3 lower_to_c.py "$MLIR" "$OUT" --input input.bin $LOWER_FLAGS

# 2. Compile + link (same flags as before; -I$R/include for incbin.h, -I$OUT for the .incbin files)
F="--target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
   -mabi=lp64d -mcmodel=medany -static -O2 -ffast-math -fno-common -fno-builtin-printf -std=gnu99 -g \
   -I/usr/include/newlib -I$R/include -I$R/include/matrix -I$R/include/common -I$R/src -I$HERE -I$OUT"
$CL $F -c "$OUT/model.c"                  -o "$OUT/model.o"
$CL $F -c "$R/include/common/crt.S"       -o "$OUT/crt.o"
$CL $F -c "$R/include/common/syscalls.c"  -o "$OUT/sc.o"
$CL --target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
    -static -nostdlib -fuse-ld=lld -T "$R/include/common/test.ld" \
    "$OUT/model.o" "$OUT/crt.o" "$OUT/sc.o" -o "$OUT/resnet50.elf"
ls -lh "$OUT/resnet50.elf"

# 3. Run on spike.
#    TRACE=0 : fast correctness run (no instruction trace)
#    TRACE=1 : (default) also stream the instruction trace into a mnemonic histogram -- much slower
TRACE="${TRACE:-1}"
set +e
if [ "$TRACE" = "1" ]; then
  $SPIKE -p1 --isa=rv64gcv_zfh_zvfh_matrix -l "$OUT/resnet50.elf" 2>&1 >"$OUT/run_stdout.log" \
   | awk '{for(i=1;i<=NF;i++) if($i ~ /^\(0x/){m=$(i+1); gsub(/"/,"",m); c[m]++; break}} END{for(k in c) print c[k],k}' \
   | sort -rn > "$OUT/mnemonic_histogram.txt"
  echo "spike exit code: ${PIPESTATUS[0]}"
else
  time $SPIKE -p1 --isa=rv64gcv_zfh_zvfh_matrix "$OUT/resnet50.elf" > "$OUT/run_stdout.log"
  echo "spike exit code: $?"
fi
set -e

# 4. Results
python3 compare_logits.py "$OUT/run_stdout.log" "$REF"
if [ "$TRACE" = "1" ]; then
  echo "total instructions: $(awk '{s+=$1} END{print s}' "$OUT/mnemonic_histogram.txt")"
  echo "== matrix instructions =="
  grep -E "^[0-9]+ (mset|ml[a-z]*e[0-9]+\.|ms[a-z]*e[0-9]+\.|mf)" "$OUT/mnemonic_histogram.txt" || echo "none found"
  echo "== top 15 mnemonics =="; head -15 "$OUT/mnemonic_histogram.txt"
fi
