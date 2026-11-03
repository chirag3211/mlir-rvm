#!/bin/bash
# Rebuilds the full model with RVM_DEBUG (+ per-GEMM checksums) and runs two variants on spike:
#   dbg           : RVM_DEBUG only
#   dbg_barriers  : RVM_DEBUG + RVM_BARRIERS (noinline GEMM + compiler memory barriers)
# Prints the DBG lines of the first GEMMs and the checksum comparison against the host run.
# Prereq: resnet50_v2.mlir, input.bin, ref_logits.bin here; the NEW rvm_runtime.h.
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD
R="${R:-$HOME/test/riscv-dnn}"
CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"
SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"
OUT="$HERE/build_dbg"; mkdir -p "$OUT"
cp input.bin "$OUT/input.bin"
python3 lower_to_c.py resnet50_v2.mlir "$OUT" --input input.bin --checksums

echo; echo "== host reference (exact fp16 emulation)"
( cd "$OUT" && gcc -DHOST -DEMU_FP16_IN -DEMU_FP16_OUT -O2 -fno-math-errno -I"$HERE" -o host_ck model.c -lm && ./host_ck > host_ck.log )
echo "host checkpoints: $(grep -c '^CK' "$OUT/host_ck.log")"

BASE="--target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
  -mabi=lp64d -mcmodel=medany -static -fno-common -fno-builtin-printf -std=gnu99 -O2 -ffast-math \
  -I/usr/include/newlib -I$R/include -I$R/include/matrix -I$R/include/common -I$R/src -I$HERE -I$OUT"

run_variant() {  # name extra-flags
  local name="$1" extra="$2"
  echo; echo "================ spike variant: $name ($extra) ================"
  $CL $BASE $extra -c "$OUT/model.c"                 -o "$OUT/model_$name.o"
  $CL $BASE $extra -c "$R/include/common/crt.S"      -o "$OUT/crt_$name.o"
  $CL $BASE $extra -c "$R/include/common/syscalls.c" -o "$OUT/sc_$name.o"
  $CL --target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
      -static -nostdlib -fuse-ld=lld -T "$R/include/common/test.ld" \
      "$OUT/model_$name.o" "$OUT/crt_$name.o" "$OUT/sc_$name.o" -o "$OUT/dbg_$name.elf"
  set +e
  $SPIKE -p1 --isa=rv64gcv_zfh_zvfh_matrix "$OUT/dbg_$name.elf" > "$OUT/spike_$name.log"
  echo "spike exit code: $?"
  set -e
  echo "--- DBG lines (first GEMMs, plus any later all-zero C):"
  grep '^DBG' "$OUT/spike_$name.log" | head -40 || echo "(none -- did it finish? look for DONE)"
  echo "--- checksum comparison:"
  python3 compare_checksums.py "$OUT/host_ck.log" "$OUT/spike_$name.log" --model "$OUT/model.c" | tail -12 || true
  python3 compare_logits.py "$OUT/spike_$name.log" ref_logits.bin | head -3 || true
}
run_variant dbg "-DRVM_DEBUG"
run_variant dbg_barriers "-DRVM_DEBUG -DRVM_BARRIERS"
