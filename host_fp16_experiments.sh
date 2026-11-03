#!/bin/bash
# Reproduces the matrix unit's fp16 numerics on the host and shows WHICH rounding stage hurts accuracy.
# usage: ./host_fp16_experiments.sh [DIR_WITH_model.c] [ref_logits.bin]      (defaults: host_check ref_logits.bin)
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD
DIR="${1:-host_check}"; REF="${2:-ref_logits.bin}"
run() {  # name, defines...
  local name="$1"; shift
  ( cd "$DIR" && gcc -DHOST "$@" -O2 -fno-math-errno -I"$HERE" -o "exp_$name" model.c -lm && ./"exp_$name" > "exp_$name.log" )
  local out; out=$(python3 "$HERE/compare_logits.py" "$DIR/exp_$name.log" "$REF")
  printf "%-28s%s\n" "$name" "$(echo "$out" | head -1)"
}
echo "variant                     error vs PyTorch fp32 reference"
run fp32_baseline
run fp16_inputs_only    -DEMU_FP16_IN
run fp16_output_only    -DEMU_FP16_OUT
run fp16_in_and_out     -DEMU_FP16_IN -DEMU_FP16_OUT
echo
echo "argmax / top-5 of the full emulation:"
python3 "$HERE/compare_logits.py" "$DIR/exp_fp16_in_and_out.log" "$REF" | tail -2 || true
