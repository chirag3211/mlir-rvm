#!/bin/bash
# 10-second check that lower_to_c.py + rvm_runtime.h work on YOUR machine (host gcc, no RISC-V needed).
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p selftest
python3 make_mini.py selftest
python3 lower_to_c.py selftest/mini.mlir selftest --input selftest/input.bin
# .incbin resolves weights.bin/input.bin relative to the build dir (same reason build_and_run.sh passes -I$OUT)
(cd selftest && gcc -DHOST -O2 -ffast-math -I.. -o mini_host model.c && ./mini_host > run.log)
python3 compare_logits.py selftest/run.log selftest/ref_logits.bin
