#!/bin/bash
# Exact instruction sequences for the RISC-V matrix build.
#   ./trace_instructions.sh kernel              one 8x8x8 GEMM: FULL executed trace (scalar + matrix) + its assembly
#   ./trace_instructions.sh model-asm [BUILD]   assembly (.s) of the whole network's generated C   (default BUILD=build_preact)
#   ./trace_instructions.sh model-trace N [BUILD]   first N executed MATRIX instructions of the real network (PC, encoding, text)
#   ./trace_instructions.sh model-full [BUILD]  EVERY executed instruction of the network, gzip-compressed (HUGE, hours)
set -uo pipefail
cd "$(dirname "$0")"; HERE=$PWD
R="${R:-$HOME/test/riscv-dnn}"
CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"
SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"
ISA=rv64gcv_zfh_zvfh_matrix
MODE="${1:-}"; shift || true
FLAGS="--target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions \
  -mabi=lp64d -mcmodel=medany -static -O2 -ffast-math -fno-common -fno-builtin-printf -std=gnu99 \
  -I/usr/include/newlib -I$R/include -I$R/include/matrix -I$R/include/common -I$R/src -I$HERE"
# matrix mnemonics (same pattern build_and_run.sh uses for its histogram)
MATRIX_RE='(mset|ml[a-z]*e[0-9]+\.|ms[a-z]*e[0-9]+\.|mf|mla|mlb|msc|mz)'

case "$MODE" in
kernel)
  D="$HERE/trace_kernel"; rm -rf "$D"; mkdir -p "$D"; cp gemm_diag.c rvm_runtime.h "$D/"
  F="$FLAGS -I$D -DRVM_BARRIERS"            # RVM_BARRIERS keeps rvm_gemm_f16 a separate (non-inlined) function
  $CL $F -S "$D/gemm_diag.c" -o "$D/gemm_diag.s" || exit 1
  $CL $F -c "$D/gemm_diag.c" -o "$D/t.o" && $CL $F -c "$R/include/common/crt.S" -o "$D/crt.o" && \
  $CL $F -c "$R/include/common/syscalls.c" -o "$D/sc.o" && \
  $CL --target=riscv64-unknown-elf -march=rv64gv0p10zfh0p1 -menable-experimental-extensions -static -nostdlib \
      -fuse-ld=lld -T "$R/include/common/test.ld" "$D/t.o" "$D/crt.o" "$D/sc.o" -o "$D/diag.elf" || exit 1
  $SPIKE -p1 --isa=$ISA -l "$D/diag.elf" 2> "$D/trace_full.log" > "$D/out.log"
  cat "$D/out.log"
  # trace of just the GEMM: from the first executed instruction inside rvm_gemm_f16 to its return
  START=$(grep -n -m1 -E " $MATRIX_RE" "$D/trace_full.log" | cut -d: -f1)
  echo; echo "total executed instructions: $(wc -l < "$D/trace_full.log")   first matrix instruction at trace line $START"
  grep -E " $MATRIX_RE" "$D/trace_full.log" | sed 's/^core *0: //' > "$D/trace_matrix_only.log"
  echo "files in $D/:"
  echo "  gemm_diag.s           compiler assembly (look for the label rvm_gemm_f16:)"
  echo "  trace_full.log        every executed instruction of the whole program"
  echo "  trace_matrix_only.log only the matrix instructions, in execution order"
  echo; echo "--- first 40 executed matrix instructions:"; head -40 "$D/trace_matrix_only.log"
  ;;
model-asm)
  B="${1:-build_preact}"; [ -f "$B/model.c" ] || { echo "no $B/model.c"; exit 1; }
  $CL $FLAGS -I"$HERE/$B" -DRVM_BARRIERS -S "$B/model.c" -o "$B/model.s" && \
  echo "wrote $B/model.s  ($(wc -l < "$B/model.s") lines).  GEMM kernel: grep -n 'rvm_gemm_f16:' $B/model.s"
  ;;
model-trace)
  N="${1:-200}"; B="${2:-build_preact}"; ELF="$B/resnet50.elf"; [ -f "$ELF" ] || { echo "no $ELF"; exit 1; }
  OUT="$B/matrix_trace_first_${N}.log"
  # stop as soon as N lines are collected (head closes the pipe -> spike gets SIGPIPE and exits)
  $SPIKE -p1 --isa=$ISA -l "$ELF" 2>&1 >/dev/null | grep --line-buffered -E " $MATRIX_RE" | head -n "$N" > "$OUT"
  echo "wrote $OUT"; head -40 "$OUT"
  ;;
model-full)
  B="${1:-build_preact}"; ELF="$B/resnet50.elf"; [ -f "$ELF" ] || { echo "no $ELF"; exit 1; }
  echo "WARNING: this is the entire instruction stream (billions of lines). Streaming to $B/full_trace.log.gz ..."
  $SPIKE -p1 --isa=$ISA -l "$ELF" 2>&1 >/dev/null | gzip -1 > "$B/full_trace.log.gz"
  ls -lh "$B/full_trace.log.gz"
  ;;
*) sed -n 2,7p "$0"; exit 1 ;;
esac
