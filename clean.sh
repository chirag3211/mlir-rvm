#!/bin/bash
# Removes everything that is regenerated (builds, logs, traces, downloads). Keeps sources, generated MLIR and inputs.
cd "$(dirname "$0")"
rm -rf build build_* gemm_big_build gemm_test_build wcheck_build trace_kernel diag_* host_check selftest_* logs sample_images \
       nohup.out *.log
echo "cleaned. kept:"; ls
