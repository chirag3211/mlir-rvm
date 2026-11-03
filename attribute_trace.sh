#!/bin/bash
# Per-layer / per-pass instruction attribution of a spike run.
#   ./attribute_trace.sh ELF MAIN_SRC [OUTDIR] [SRC_ROOT ...]
#   ./attribute_trace.sh build_preact_A3/resnet50.elf build_preact_A3/model.c
#   riscv-dnn (build the test with -g, e.g. DEFS='... -g'):
#   ./attribute_trace.sh ~/test/riscv-dnn/test/models/resnet50/build/0/test.elf \
#       ~/test/riscv-dnn/test/models/resnet50/test.c dnn_attr  ~/test/riscv-dnn/src
# Needs the ELF built with -g. Runs spike with -l once (same cost as TRACE=1).
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD
ELF="$1"; SRC="$2"; OUT="${3:-$(dirname "$ELF")/attr}"; shift 3 2>/dev/null || shift $# ; ROOTS="$*"
SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"; CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"
ISA="${ISA:-rv64gcv_zfh_zvfh_matrix}"
mkdir -p "$OUT"
python3 attribute_trace.py prepare --elf "$ELF" --main-src "$SRC" --out "$OUT" --clang "$CL"
echo "running spike with -l (streamed; this takes as long as a TRACE=1 run) ..."
${SPIKE_CMD_PREFIX:-} $SPIKE -p1 --isa=$ISA -l "$ELF" 2>&1 >"$OUT/run_stdout.log" \
 | awk -v mapf="$OUT/pcmap.txt" -v blkf="$OUT/blk_mn.txt" -v pcf="$OUT/pc_counts.txt" '
   BEGIN { while ((getline l < mapf) > 0) { split(l, a, " "); blk[a[1]] = a[2] } cur = "startup" }
   { for (i = 2; i <= NF; i++) if ($i ~ /^\(0x/) {
       pc = $(i-1); m = $(i+1); gsub(/"/, "", m)
       if (pc in blk) cur = blk[pc]
       bm[cur SUBSEP m]++; p[pc]++; pm[pc] = m; break } }
   END { for (k in bm) { split(k, a, SUBSEP); print a[1], a[2], bm[k] > blkf }
         for (k in p) print p[k], k, pm[k] > pcf }'
python3 attribute_trace.py report --out "$OUT" --top 25 --src-roots "$HERE" $ROOTS | tee "$OUT/attribution_report.txt"
