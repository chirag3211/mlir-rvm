/* gemm_test.c -- isolates the matrix kernel from the rest of the network.
 * Tests rvm_gemm_f16 on many (M,N,K) and rvm_conv2d_nchw on small convs against float references.
 * Inputs are multiples of 1/128 so they are exact in fp16; the only expected error is fp16 rounding of the output.
 * Output uses only %d / %s (the repo's printf has no %f). Values are printed in thousandths.
 */
#define RVM_MAX_A 70000
#define RVM_MAX_B 140000
#define RVM_MAX_C 140000
#include "rvm_runtime.h"

static unsigned rng = 12345u;
static float rnd(void) {
  rng = rng * 1664525u + 1013904223u;
  return ((int)((rng >> 16) & 0xff) - 128) / 128.0f;
}
static int milli(float x) { return (int)(x * 1000.0f + (x >= 0 ? 0.5f : -0.5f)); }
static float fabs_(float x) { return x < 0 ? -x : x; }

static f16 A[70000], B[140000], C[140000];
static float refC[140000];

static int test_gemm(int M, int N, int K) {
  for (long i = 0; i < (long)M * K; i++) A[i] = (f16)rnd();
  for (long i = 0; i < (long)K * N; i++) B[i] = (f16)rnd();
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float s = 0.f;
      for (int k = 0; k < K; k++) s += (float)A[(long)i * K + k] * (float)B[(long)k * N + j];
      refC[(long)i * N + j] = s;
    }
  for (long i = 0; i < (long)M * N; i++) C[i] = (f16)12345.0f;   /* poison: untouched outputs show up */
  rvm_gemm_f16(A, B, C, M, N, K);
  int bad = 0, fi = -1, fj = -1;
  float maxerr = 0, fgot = 0, fref = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float got = (float)C[(long)i * N + j], ref = refC[(long)i * N + j];
      float err = fabs_(got - ref), tol = 2e-3f * fabs_(ref) + 2e-3f;
      if (err > maxerr) maxerr = err;
      if (err > tol) {
        if (!bad) { fi = i; fj = j; fgot = got; fref = ref; }
        bad++;
      }
    }
  printf("GEMM M=%d N=%d K=%d : %s  bad=%d/%d maxerr(milli)=%d", M, N, K, bad ? "FAIL" : "ok  ", bad, M * N, milli(maxerr));
  if (bad) printf("  first bad (i=%d,j=%d) got(milli)=%d ref(milli)=%d", fi, fj, milli(fgot), milli(fref));
  printf("\n");
  return bad != 0;
}

static float cin[20000], cw[8 * 16 * 9], cout_[40000], cref[40000];

static int test_conv(int C_, int H, int W, int F, int KH, int KW, int SH, int SW) {
  int OH = (H - KH) / SH + 1, OW = (W - KW) / SW + 1;
  for (int i = 0; i < C_ * H * W; i++) cin[i] = rnd();
  for (int i = 0; i < F * C_ * KH * KW; i++) cw[i] = rnd();
  for (int i = 0; i < F * OH * OW; i++) { cout_[i] = 0.f; cref[i] = 0.f; }
  for (int f = 0; f < F; f++)
    for (int oh = 0; oh < OH; oh++)
      for (int ow = 0; ow < OW; ow++) {
        float s = 0.f;
        for (int c = 0; c < C_; c++)
          for (int r = 0; r < KH; r++)
            for (int t = 0; t < KW; t++)
              s += cin[(c * H + oh * SH + r) * W + ow * SW + t] * cw[((f * C_ + c) * KH + r) * KW + t];
        cref[(f * OH + oh) * OW + ow] = s;
      }
  rvm_conv2d_nchw(cin, cw, cout_, C_, H, W, F, KH, KW, OH, OW, SH, SW, 1, 1);
  int bad = 0;
  float maxerr = 0;
  for (int i = 0; i < F * OH * OW; i++) {
    float err = fabs_(cout_[i] - cref[i]), tol = 2e-3f * fabs_(cref[i]) + 2e-3f;
    if (err > maxerr) maxerr = err;
    if (err > tol) bad++;
  }
  printf("CONV C=%d H=%d W=%d F=%d K=%dx%d s=%d : %s  bad=%d/%d maxerr(milli)=%d\n", C_, H, W, F, KH, KW, SH,
         bad ? "FAIL" : "ok  ", bad, F * OH * OW, milli(maxerr));
  return bad != 0;
}

int main(void) {
#ifndef HOST
  SET_MBA0_FP16();
  int tm = msettilem(1000), tn = msettilen(1000), tk = msettilek(1000);
  printf("granted tile sizes for fp16 (m,n,k) = %d %d %d\n", tm, tn, tk);
#endif
  int fails = 0;
  /* aligned / powers of two */
  fails += test_gemm(8, 8, 8);
  fails += test_gemm(16, 16, 16);
  fails += test_gemm(32, 32, 32);
  fails += test_gemm(64, 64, 64);
  /* odd / non-multiples */
  fails += test_gemm(13, 17, 19);
  fails += test_gemm(1, 64, 64);
  fails += test_gemm(64, 1, 64);
  fails += test_gemm(64, 64, 1);
  /* ResNet-like: N=49 / 196 / 784 (odd and even row strides), K=147 / 576 / 64 */
  fails += test_gemm(64, 49, 64);
  fails += test_gemm(16, 49, 147);
  fails += test_gemm(64, 196, 64);
  fails += test_gemm(64, 196, 147);
  fails += test_gemm(16, 784, 64);
  fails += test_gemm(8, 784, 147);
  fails += test_gemm(1, 1000, 64);
  fails += test_gemm(1, 1000, 100);
  /* LARGE ROW STRIDES. Bytes per row = 2*N (B and C) or 2*K (A). The stem conv has N=12544 -> 25088 B,
     which none of the shapes above reach (max was N=1000 -> 2000 B). */
  fails += test_gemm(8, 2048, 8);    /*  4096 B */
  fails += test_gemm(8, 4096, 8);    /*  8192 B */
  fails += test_gemm(8, 8192, 8);    /* 16384 B */
  fails += test_gemm(8, 12544, 8);   /* 25088 B  <- stem conv N */
  fails += test_gemm(8, 16384, 8);   /* 32768 B */
  fails += test_gemm(4, 32768, 4);   /* 65536 B */
  fails += test_gemm(8, 8, 2048);    /* A stride  4096 B */
  fails += test_gemm(8, 8, 4608);    /* A stride  9216 B  (512*3*3) */
  fails += test_gemm(8, 8, 8192);    /* A stride 16384 B */
  /* whole conv wrapper (im2col + fp16 converts + GEMM + accumulate) */
  fails += test_conv(3, 10, 10, 4, 3, 3, 1, 1);
  fails += test_conv(4, 4, 4, 6, 1, 1, 2, 2);
  fails += test_conv(16, 16, 16, 8, 3, 3, 2, 2);
  fails += test_conv(8, 9, 9, 8, 1, 1, 1, 1);
  /* stem-like conv: N = 112*112 = 12544 (25088-byte rows) */
  fails += test_conv(1, 114, 114, 2, 3, 3, 1, 1);
  fails += test_conv(1, 130, 130, 2, 1, 1, 1, 1);   /* N = 16900 */
  printf(fails ? "SUMMARY: %d test(s) FAILED\n" : "SUMMARY: ALL PASS (%d)\n", fails);
  printf("DONE\n");
  return 0;
}
