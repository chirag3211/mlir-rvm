#!/bin/bash
# Checks a standalone matrix-kernel test on spike (default: unfold_test = plain hardware-unfold conv).
#   ./run_unfold_test.sh                    -> unfold_test.c  (conv)
#   T=unfold_epi_test ./run_unfold_test.sh  -> unfold_epi_test.c  (conv + fused BN/ReLU [+ add] epilogue)
#   ./run_unfold_test.sh host               -> same, with gcc and the functional model emu_matrix.h (no spike)
# Needs in this directory: rvm_runtime.h (the NEW one), rvm_unfold.h, emu_matrix.h and the test .c file.
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD
if [ "${1:-}" = host ]; then
  gcc -DHOST -O2 -DFULL -I"$HERE" -o ${T:-unfold_test}_host ${T:-unfold_test}.c -lm && ./${T:-unfold_test}_host; exit $?
fi
R="${R:-$HOME/test/riscv-dnn}"
CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"
SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"
OUT="$HERE/unfold_build"; mkdir -p "$OUT"
BASE="--target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
   -mabi=lp64d -mcmodel=medany -static -O2 -ffast-math -fno-common -fno-builtin-printf -std=gnu99 \
   -I/usr/include/newlib -I$R/include -I$R/include/matrix -I$R/include/common -I$R/src -I$HERE"
$CL $BASE -c "$R/include/common/crt.S"      -o "$OUT/crt.o"
$CL $BASE -c "$R/include/common/syscalls.c" -o "$OUT/sc.o"
$CL $BASE -c ${T:-unfold_test}.c -o "$OUT/${T:-unfold_test}.o"
$CL --target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
    -static -nostdlib -fuse-ld=lld -T "$R/include/common/test.ld" \
    "$OUT/${T:-unfold_test}.o" "$OUT/crt.o" "$OUT/sc.o" -o "$OUT/${T:-unfold_test}.elf"
set +e
time $SPIKE -p1 --isa=rv64gcv_zfh_zvfh_matrix "$OUT/${T:-unfold_test}.elf" | tee "$OUT/${T:-unfold_test}.log"
echo "spike exit code: ${PIPESTATUS[0]}"
