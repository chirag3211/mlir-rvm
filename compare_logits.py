#!/usr/bin/env python3
"""compare_logits.py RUN_LOG REF_LOGITS.bin  -- parse 'LOGIT i hex' lines and compare with the PyTorch reference."""
import re, struct, sys
import numpy as np
log, ref_path = sys.argv[1], sys.argv[2]
got = {}
for m in re.finditer(r"LOGIT (\d+) ([0-9a-fA-F]+)", open(log, errors="replace").read()):
    got[int(m.group(1))] = struct.unpack("<f", struct.pack("<I", int(m.group(2), 16)))[0]
ref = np.fromfile(ref_path, dtype=np.float32).ravel()
if len(got) != len(ref):
    sys.exit(f"FAIL: got {len(got)} logits, expected {len(ref)} (did the program finish? look for DONE in the log)")
out = np.array([got[i] for i in range(len(ref))], np.float32)
err = np.abs(out - ref)
print(f"max abs err {err.max():.3e}   mean abs err {err.mean():.3e}   ref range [{ref.min():.3f}, {ref.max():.3f}]")
print("top-5 ref:", list(np.argsort(-ref)[:5]), " top-5 got:", list(np.argsort(-out)[:5]))
print("argmax match:", int(np.argmax(out)) == int(np.argmax(ref)))
