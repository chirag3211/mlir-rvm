# MLIR → RISC-V Matrix Extension (RVM): ResNet-50 v2

Lowers `timm` `resnetv2_50` through MLIR to C that uses the RISC-V matrix extension intrinsics, runs it on
`spike`, and compares **executed instructions** with the hand-written ResNet-50 in
[riscv-dnn](https://github.com/riscv-stc) (kernels are modeled on riscv-dnn's `conv_bn_relu_rvm`,
`conv_add_bn_relu_rvm` and `maxpool_bn_relu`).

## Result

| | instructions | 
|---|---|
| MLIR-RVM (this repo) | 214,680,060 |
| riscv-dnn | 230,150,057 |
| ratio | 0.93x |

Matrix instruction counts match riscv-dnn within about 0.1%. Output is correct: argmax and top-5 match the
PyTorch reference, and 3/3 real images give the same top-1 (logit cosine 1.000000).
Side-by-side histograms: [`docs/mlir_vs_dnn_histograms.html`](docs/mlir_vs_dnn_histograms.html)
(data in `results/*.json`).

## Pipeline

```
timm resnetv2_50 ──torch-mlir──▶ linalg-on-tensors MLIR ──lower_to_c.py──▶ model.c + fp16 weights
      ──riscv-stc clang 14──▶ ELF ──spike (rv64gcv_zfh_zvfh_matrix)──▶ logits + instruction trace
```

`lower_to_c.py` is a custom script because there is no MLIR backend/dialect for the RISC-V matrix extension
(neither upstream nor, as far as I checked, in the riscv-stc LLVM tree). It does bufferization, BN folding,
pad/pool folding, fused conv+BN+ReLU(+add) and maxpool+BN+ReLU, and emits C calling the kernels in
`rvm_runtime.h` / `rvm_unfold.h` (hardware im2col via `mlufa`/`msetsk`, `mfwma`).

Flags (all on by default in `build_and_run.sh`):
`--nhwc-fp16 --fold-pad --fuse-bn --fold-pool --fuse-epilogue --fuse-pool-bn`

## Requirements

- `riscv-dnn` checkout (headers, `crt.S`, `test.ld`, `incbin.h`) – env `R`, default `~/test/riscv-dnn`
- riscv-stc clang with the matrix extension (+ `ld.lld`) – env `CL`
- spike with `--isa=rv64gcv_zfh_zvfh_matrix` – env `SPIKE`
- Python 3 with `torch`, `torch_mlir`, `timm`, `numpy` (model export only)

## Expected layout

Paths are the script defaults; override them with the env vars `R`, `CL` and `SPIKE`.

```
~/test/riscv-dnn                                   matrix headers, crt.S, test.ld, incbin.h
~/test/llvm-project/build/bin/clang (+ ld.lld)     clang with the matrix extension
~/test/spike-install/bin/spike                     spike with --isa=rv64gcv_zfh_zvfh_matrix
~/rvm_pipeline                                     this repository
```

`check_env.sh` checks that these exist.

## Notes

- `lower_to_c.py` embeds the weights and input by **absolute** path in `.incbin`. A relative path once silently
  picked up a stale `weights.bin` from an earlier `-I` directory, so keep it absolute.
- `-DRVM_DEBUG` and `-DRVM_BARRIERS` are optional compile flags for `rvm_runtime.h`.
- `eval_images.py` patches a real image into a copy of the ELF and runs spike, so no rebuild is needed.

## Run

```bash
python3 generate_resnetv2_50.py                       # -> resnetv2_50.mlir, input_preact.bin, ref_logits_preact.bin
MLIR=resnetv2_50.mlir INP=input_preact.bin REF=ref_logits_preact.bin BUILD=build_preact ./build_and_run.sh
# TRACE=0 for a fast correctness-only run; TRACE=1 (default) also writes build_preact/mnemonic_histogram.txt

python3 compare_histograms.py mlir=build_preact/mnemonic_histogram.txt dnn=<riscv-dnn histogram>.txt
python3 eval_images.py ...                            # real-image check (see script header)
```

## Files

| file | purpose |
|---|---|
| `lower_to_c.py` | MLIR → C lowering and all fusions |
| `rvm_runtime.h`, `rvm_unfold.h` | RVM kernels (GEMM, hardware-unfold conv, fused epilogues, maxpool+BN+ReLU in RVV) |
| `emu_matrix.h` | functional host model of the matrix ops, for tests without spike |
| `build_and_run.sh` | C generation, compile, spike run, logit comparison |
| `generate_resnetv2_50.py`, `generate_resnet50.py` | model export (timm pre-activation / torchvision) |
| `compare_logits.py`, `compare_histograms.py`, `eval_images.py` | correctness and instruction-mix comparison |
| `trace_instructions.sh`, `attribute_trace.*` | static/dynamic instruction attribution |
| `unfold_test.c`, `unfold_epi_test.c`, `run_unfold_test.sh`, `gemm_big.c`, `run_gemm_big.sh`, `pooltest.c`, `wcheck.c`, `run_wcheck.sh` | kernel-level tests |
| `make_mini.py`, `selftest_host.sh`, `bn_range_check.py`, `compare_checksums.py`, `run_model_debug.sh`, `check_env.sh` | host-side checks and debugging |

## Known gaps vs riscv-dnn

- Scalar integer instructions: 121.9M vs 139.7M. dnn uses 32-bit ops, this uses 64-bit address arithmetic.
- The FC layer is lowered as a 1x1 conv, not dnn's matmul; the zero tile uses `mlc` (+145K `mlce16`).
- Instruction counts include `main`/printf (about 5.0M) and the input conversion (about 0.76M).
