#!/bin/bash
# Isolates the matrix kernel on spike. Run after build_and_run.sh has produced build/crt.o and build/sc.o.
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD
R="${R:-$HOME/test/riscv-dnn}"
CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"
SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"
OUT="$HERE/gemm_test_build"; mkdir -p "$OUT"
F="--target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
   -mabi=lp64d -mcmodel=medany -static -O2 -ffast-math -fno-common -fno-builtin-printf -std=gnu99 \
   -I/usr/include/newlib -I$R/include -I$R/include/matrix -I$R/include/common -I$R/src -I$HERE"
$CL $F -c gemm_test.c                    -o "$OUT/gemm_test.o"
$CL $F -c "$R/include/common/crt.S"      -o "$OUT/crt.o"
$CL $F -c "$R/include/common/syscalls.c" -o "$OUT/sc.o"
$CL --target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
    -static -nostdlib -fuse-ld=lld -T "$R/include/common/test.ld" \
    "$OUT/gemm_test.o" "$OUT/crt.o" "$OUT/sc.o" -o "$OUT/gemm_test.elf"
set +e
$SPIKE -p1 --isa=rv64gcv_zfh_zvfh_matrix "$OUT/gemm_test.elf" | tee "$OUT/gemm_test.log"
echo "spike exit code: ${PIPESTATUS[0]}"
