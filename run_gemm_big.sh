#!/bin/bash
# Model-scale kernel test on spike (plain build, then -DRVM_BARRIERS build), plus ELF layout dump.
# Needs: rvm_runtime.h (the NEW one) and gemm_big.c in this directory.
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD
R="${R:-$HOME/test/riscv-dnn}"
CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"
SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"
BIN="$(dirname "$CL")"
OUT="$HERE/gemm_big_build"; mkdir -p "$OUT"
BASE="--target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
   -mabi=lp64d -mcmodel=medany -static -O2 -ffast-math -fno-common -fno-builtin-printf -std=gnu99 \
   -I/usr/include/newlib -I$R/include -I$R/include/matrix -I$R/include/common -I$R/src -I$HERE"
$CL $BASE -c "$R/include/common/crt.S"      -o "$OUT/crt.o"
$CL $BASE -c "$R/include/common/syscalls.c" -o "$OUT/sc.o"

run_variant() {  # name extra-flags
  local name="$1" extra="$2"
  echo; echo "================ gemm_big: $name ($extra) ================"
  $CL $BASE $extra -c gemm_big.c -o "$OUT/gemm_big_$name.o"
  $CL --target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
      -static -nostdlib -fuse-ld=lld -T "$R/include/common/test.ld" \
      "$OUT/gemm_big_$name.o" "$OUT/crt.o" "$OUT/sc.o" -o "$OUT/gemm_big_$name.elf"
  set +e
  time $SPIKE -p1 --isa=rv64gcv_zfh_zvfh_matrix "$OUT/gemm_big_$name.elf" | tee "$OUT/gemm_big_$name.log"
  echo "spike exit code: ${PIPESTATUS[0]}"
  set -e
}
run_variant plain ""
run_variant barriers "-DRVM_BARRIERS"

echo; echo "================ ELF layout ================"
for ELF in "$OUT/gemm_big_plain.elf" "$HERE/build/resnet50.elf"; do
  [ -f "$ELF" ] || continue
  echo "---- $ELF"
  "$BIN/llvm-readelf" -S -l "$ELF" | grep -Ev "^\s*$" | head -60
  "$BIN/llvm-nm" "$ELF" | grep -Ei ' (rvm_tmp|pool[0-9]|_end|__bss|_edata|_stack|__stack|_heap|heap_|weights_blob|input_blob|tohost|fromhost)' || true
done
echo; echo "---- test.ld"; cat "$R/include/common/test.ld"
