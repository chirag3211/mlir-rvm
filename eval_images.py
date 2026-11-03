#!/usr/bin/env python3
"""
eval_images.py -- run REAL images through the spike-simulated RISC-V matrix build and compare with PyTorch fp32.

No re-lowering / recompiling: the input image is embedded in the ELF (build_*/resnet50.elf) as 602112 raw fp32 bytes,
so for every image this script patches those bytes in a copy of the ELF and runs spike on the copy.

  python3 eval_images.py --elf build_preact/resnet50.elf --orig-input input_preact_rand.bin --fetch-samples
  python3 eval_images.py --elf build_preact/resnet50.elf --orig-input input_preact_rand.bin --images photos/*.jpg --jobs 3

--orig-input : the input.bin that was embedded when that ELF was built (used to locate the bytes; must be unique in
               the ELF, so use a RANDOM input, not the constant-gray one).
--arch timm --weights resnetv2_50.a1h_in1k   (pre-activation v2, default)
--arch torchvision                            (resnet50 IMAGENET1K_V2)
Each spike run takes ~1.5 min; --jobs N runs N in parallel (each spike process needs ~2 GB RAM + a 290 MB ELF copy).
"""
import argparse
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SAMPLES = {
    "dog.jpg": "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "cats.jpg": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "bus.jpg": "http://images.cocodataset.org/val2017/000000000139.jpg",
}


def fetch_samples(d):
    os.makedirs(d, exist_ok=True)
    out = []
    for name, url in SAMPLES.items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            print(f"downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
                shutil.copyfileobj(r, f)
        out.append(p)
    return out


def find_unique(elf, needle):
    with open(elf, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        a = m.find(needle)
        if a < 0:
            sys.exit(f"ERROR: the bytes of --orig-input were not found in {elf}. Was the ELF built with that exact input?")
        b = m.find(needle, a + 1)
        if b >= 0:
            sys.exit("ERROR: --orig-input occurs more than once in the ELF (constant image?). "
                     "Rebuild the ELF with a random input (generate_resnetv2_50.py --input random).")
    return a


def patch_elf(src, dst, offset, data):
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(offset)
        f.write(data)


def parse_logits(text, n=1000):
    got = {int(m.group(1)): struct.unpack("<f", struct.pack("<I", int(m.group(2), 16)))[0]
           for m in re.finditer(r"LOGIT (\d+) ([0-9a-fA-F]+)", text)}
    if len(got) != n:
        return None
    return np.array([got[i] for i in range(n)], np.float32)


def run_spike(spike, elf, isa):
    r = subprocess.run([spike, "-p1", f"--isa={isa}", elf], capture_output=True, text=True, errors="replace")
    return r.stdout


def load_model(arch, weights):
    import torch
    if arch == "timm":
        import timm
        m = timm.create_model(weights, pretrained=True).eval()
        cfg = timm.data.resolve_data_config({}, model=m)
        assert tuple(cfg["input_size"]) == (3, 224, 224), cfg
        tf = timm.data.create_transform(**cfg)
        try:
            names = timm.data.ImageNetInfo().index_to_description
        except Exception:
            names = lambda i: str(i)
    else:
        import torchvision.models as tvm
        W = tvm.ResNet50_Weights.IMAGENET1K_V2
        m = tvm.resnet50(weights=W).eval()
        tf = W.transforms()
        cats = W.meta["categories"]
        names = lambda i: cats[i]
    return m, tf, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--orig-input", required=True)
    ap.add_argument("--arch", default="timm", choices=["timm", "torchvision"])
    ap.add_argument("--weights", default="resnetv2_50.a1h_in1k")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--fetch-samples", action="store_true", help="download 3 sample images into ./sample_images")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--spike", default=os.environ.get("SPIKE", os.path.expanduser("~/test/spike-install/bin/spike")))
    ap.add_argument("--isa", default="rv64gcv_zfh_zvfh_matrix")
    ap.add_argument("--keep", action="store_true", help="keep the per-image logits (.bin) next to the images")
    a = ap.parse_args()

    imgs = list(a.images)
    if a.fetch_samples:
        imgs += fetch_samples("sample_images")
    if not imgs:
        sys.exit("no images: pass --images ... and/or --fetch-samples")

    orig = open(a.orig_input, "rb").read()
    assert len(orig) == 3 * 224 * 224 * 4, "orig input must be 1x3x224x224 fp32"
    off = find_unique(a.elf, orig)
    print(f"input located in ELF at file offset {off:#x}")

    import torch
    from PIL import Image
    model, tf, names = load_model(a.arch, a.weights)

    jobs = []
    for p in imgs:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        assert tuple(x.shape) == (1, 3, 224, 224), x.shape
        with torch.no_grad():
            ref = model(x)[0].numpy().astype(np.float32)
        jobs.append((p, x.numpy().astype(np.float32).tobytes(), ref))

    tmp = tempfile.mkdtemp(prefix="evalimg_")

    def work(k):
        p, data, ref = jobs[k]
        elf = os.path.join(tmp, f"img{k}.elf")
        patch_elf(a.elf, elf, off, data)
        out = run_spike(a.spike, elf, a.isa)
        os.remove(elf)
        return parse_logits(out, len(ref))

    print(f"running {len(jobs)} image(s) on spike, {a.jobs} at a time ...")
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(work, range(len(jobs))))
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'image':<24}{'pytorch top1':<28}{'spike top1':<28}{'top1':<6}{'top5∩':<7}{'maxerr':<10}{'cos':<9}")
    n_ok = n_top5 = 0
    errs = []
    for (p, _, ref), got in zip(jobs, results):
        base = os.path.basename(p)
        if got is None:
            print(f"{base:<24}FAILED (spike did not print all logits)")
            continue
        r1, g1 = int(np.argmax(ref)), int(np.argmax(got))
        r5, g5 = set(np.argsort(-ref)[:5].tolist()), set(np.argsort(-got)[:5].tolist())
        err = float(np.abs(ref - got).max())
        cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
        n_ok += r1 == g1
        n_top5 += len(r5 & g5) == 5
        errs.append(err)
        print(f"{base:<24}{(str(r1) + ' ' + str(names(r1)))[:26]:<28}{(str(g1) + ' ' + str(names(g1)))[:26]:<28}"
              f"{'ok' if r1 == g1 else 'DIFF':<6}{len(r5 & g5)}/5    {err:<10.3e}{cos:<9.6f}")
        if a.keep:
            got.tofile(p + ".spike_logits.bin"); ref.tofile(p + ".ref_logits.bin")
    n = len(errs)
    print(f"\ntop-1 agreement: {n_ok}/{len(jobs)}   identical top-5 set: {n_top5}/{len(jobs)}   "
          f"mean max-abs-err: {np.mean(errs):.3e}" if n else "\nno successful runs")


if __name__ == "__main__":
    main()
