rvm_pipeline -- PyTorch -> MLIR (linalg-on-tensors) -> C -> RISC-V matrix-extension ELF -> spike
================================================================================================
Expected layout (paths are the script defaults; override with env R / CL / SPIKE):
  ~/test/riscv-dnn          matrix toolchain headers, crt.S, test.ld, incbin.h
  ~/test/llvm-project/build/bin/clang (+ ld.lld)     clang with the matrix extension
  ~/test/spike-install/bin/spike                     spike with --isa=rv64gcv_zfh_zvfh_matrix
  ~/rvm_pipeline            THIS directory

Quick start        ./check_env.sh      then      ./run_all.sh
Clean up           ./clean.sh          (removes builds, logs, traces; keeps sources + generated models)

FILES
 Core:  lower_to_c.py          MLIR -> model.c + weights.bin. Embeds weights/input by ABSOLUTE path (a relative
                               .incbin silently picked up a stale weights.bin from an earlier -I dir: keep this!)
        rvm_runtime.h          matrix GEMM kernel + im2col conv wrapper. -DRVM_DEBUG / -DRVM_BARRIERS are optional
        build_and_run.sh       build + run on spike. Env: MLIR INP REF BUILD TRACE
        compare_logits.py      compare "LOGIT i hex" lines with ref_logits.bin
 Models: generate_resnet50.py       torchvision ResNet-50 (post-activation, IMAGENET1K_V2) -> resnet50_v2.mlir, input.bin, ref_logits.bin
         generate_resnetv2_50.py    timm resnetv2_50 (pre-activation) -> resnetv2_50.mlir, input_<prefix>.bin, ref_logits_<prefix>.bin
 Real images: eval_images.py        patches an image into a copy of the ELF and runs spike (no rebuild)
 Tracing:     trace_instructions.sh static asm / executed-instruction traces
 Diagnostics: gemm_test.c run_gemm_test.sh   small-shape kernel tests
              gemm_big.c run_gemm_big.sh     model-scale kernel tests
              wcheck.c run_wcheck.sh         is weights.bin really in memory? (found the stale-file bug)
              run_model_debug.sh compare_checksums.py   per-GEMM checksums host vs spike
              make_mini.py selftest_host.sh host_fp16_experiments.sh     host-only checks
 Env:   check_env.sh clean.sh run_all.sh
