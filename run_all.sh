#!/bin/bash
# From-scratch end-to-end run of everything (≈ 10 min of spike time + model downloads).
#   1. torchvision ResNet-50 (post-activation, IMAGENET1K_V2): generate -> lower -> spike -> compare
#   2. timm ResNetV2-50 (pre-activation):                       generate -> lower -> spike -> compare
#   3. real images through the pre-activation ELF
# Skips nothing; re-run ./clean.sh first if you want a pristine tree.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p logs
echo "== 0. toolchain check";  ./check_env.sh
echo "== 1. torchvision ResNet-50"
python3 generate_resnet50.py 2>&1 | tee logs/gen_torchvision.log
TRACE=0 ./build_and_run.sh 2>&1 | tee logs/run_torchvision.log
echo "== 2. pre-activation ResNetV2-50 (random input)"
python3 generate_resnetv2_50.py --input random --prefix preact_rand 2>&1 | tee logs/gen_preact.log
MLIR=resnetv2_50.mlir INP=input_preact_rand.bin REF=ref_logits_preact_rand.bin BUILD=build_preact \
  TRACE=0 ./build_and_run.sh 2>&1 | tee logs/run_preact.log
echo "== 3. real images"
python3 eval_images.py --elf build_preact/resnet50.elf --orig-input input_preact_rand.bin \
  --fetch-samples --jobs 2 2>&1 | tee logs/eval_images.log
echo; echo "== SUMMARY"; grep -h "argmax match\|top-1 agreement" logs/run_*.log logs/eval_images.log
