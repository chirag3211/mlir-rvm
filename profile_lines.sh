#!/bin/bash
# Per-source-line instruction profile of an ELF run on spike.
# usage: ./profile_lines.sh build_preact_h16/resnet50.elf [TOPN]
# needs the ELF built with -g (see below). -g does not change generated code.
set -euo pipefail
ELF=$1; TOP=${2:-40}
SPIKE=${SPIKE:-$HOME/test/spike-install/bin/spike}
A2L=${A2L:-riscv64-unknown-elf-addr2line}
W=$(mktemp -d)
if [ -n "${LINES:-}" ]; then cp "$LINES" "$W/lines.txt"; SKIP=1; else SKIP=0; fi
if [ "$SKIP" = 0 ]; then
# 1. executed-instruction count per PC (PC = the field right before "(0x<opcode>)")
$SPIKE -p1 --isa=rv64gcv_zfh_zvfh_matrix -l "$ELF" 2>&1 >/dev/null |
  awk '{for(i=2;i<=NF;i++) if($i ~ /^\(0x/){c[$(i-1)]++; break}} END{for(p in c) print p, c[p]}' > "$W/pc.txt"
# 2. PC -> file:line, then sum per line
cut -d' ' -f1 "$W/pc.txt" | "$A2L" -e "$ELF" > "$W/loc.txt"
paste -d' ' "$W/loc.txt" <(cut -d' ' -f2 "$W/pc.txt") |
  sed 's/ (discriminator [0-9]*)//' |
  awk '{c[$1]+=$2; t+=$2} END{for(k in c) printf "%d %.2f %s\n", c[k], 100*c[k]/t, k}' | sort -rn > "$W/lines.txt"
fi
# 3. print top lines with the source text
printf "%-12s %-6s %s\n" COUNT PCT LOCATION
head -n "$TOP" "$W/lines.txt" | while read -r n pct loc; do
  f=${loc%:*}; l=${loc##*:}; src=""
  if [ -f "$f" ] && [[ "$l" =~ ^[0-9]+$ ]]; then src=$(sed -n "${l}p" "$f" | sed 's/^ *//' | cut -c1-110); fi
  printf "%-12s %-6s %s   %s\n" "$n" "$pct" "$(basename "$f"):$l" "$src"
done
[ "$SKIP" = 0 ] && cp "$W/lines.txt" "${ELF%.elf}_line_profile.txt"; echo "(raw data: $W/lines.txt)"
