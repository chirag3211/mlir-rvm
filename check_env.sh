#!/bin/bash
# Verifies every external dependency the pipeline needs. Exit code != 0 if something is missing.
R="${R:-$HOME/test/riscv-dnn}"; CL="${CL:-$HOME/test/llvm-project/build/bin/clang}"; SPIKE="${SPIKE:-$HOME/test/spike-install/bin/spike}"
bad=0
chk() { if [ -e "$2" ]; then printf "  ok   %-22s %s\n" "$1" "$2"; else printf "  MISSING %-19s %s\n" "$1" "$2"; bad=1; fi; }
chk clang "$CL"; chk ld.lld "$(dirname "$CL")/ld.lld"; chk spike "$SPIKE"
chk "riscv-dnn include" "$R/include/matrix/matrix_intrinsic.h"; chk "crt.S" "$R/include/common/crt.S"
chk "test.ld" "$R/include/common/test.ld"; chk "syscalls.c" "$R/include/common/syscalls.c"; chk "incbin.h" "$R/include/incbin.h"
chk "newlib headers" /usr/include/newlib/stdio.h
python3 - <<'PY' || bad=1
import importlib
for m in ("torch", "torchvision", "torch_mlir", "timm", "numpy", "PIL"):
    try:
        mod = importlib.import_module(m); print("  ok   python %-14s %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:
        print("  MISSING python %-11s %s" % (m, e)); raise SystemExit(1)
PY
command -v gcc >/dev/null && echo "  ok   gcc (host checks)" || { echo "  MISSING gcc"; bad=1; }
[ -x "$CL" ] && "$CL" --version | head -1
exit $bad
