/* gemm_big.c -- MODEL-SCALE test of the matrix kernel.
 * Uses the SAME scratch buffers (rvm_tmpA/B/C, sized like the real model) and the exact ResNet-50 GEMM shapes.
 * Only sampled rows/outputs are checked against a float reference (a full reference would take too long on spike),
 * but the ENTIRE C is scanned for "untouched" (poison) and exactly-zero elements.
 * printf here has no %f: values are printed in thousandths.
 */
#define RVM_MAX_A 2359296
#define RVM_MAX_B 2048000
#define RVM_MAX_C 802816
#include "rvm_runtime.h"

static unsigned rng = 12345u;
static float rnd(void) {
  rng = rng * 1664525u + 1013904223u;
  return ((int)((rng >> 16) & 0xff) - 128) / 128.0f;      /* multiples of 1/128: exact in fp16 */
}
static int milli(float x) { return (int)(x * 1000.0f + (x >= 0 ? 0.5f : -0.5f)); }
static float fabs_(float x) { return x < 0 ? -x : x; }

#define POISON 12345.0f
static int test_gemm(int M, int N, int K) {
  f16 *A = rvm_tmpA, *B = rvm_tmpB, *C = rvm_tmpC;
  for (long i = 0; i < (long)M * K; i++) A[i] = (f16)rnd();
  for (long i = 0; i < (long)K * N; i++) B[i] = (f16)rnd();
  for (long i = 0; i < (long)M * N; i++) C[i] = (f16)POISON;
  rvm_gemm_f16(A, B, C, M, N, K);

  long untouched = 0, zeros = 0;
  for (long i = 0; i < (long)M * N; i++) {
    float v = (float)C[i];
    if (v == POISON) untouched++;
    else if (v == 0.0f) zeros++;
  }
  /* sampled rows, all columns */
  int rows[6] = {0, 1, M / 3, M / 2, M - 2, M - 1};
  int bad = 0, checked = 0;
  float maxerr = 0;
  for (int r = 0; r < 6; r++) {
    int i = rows[r];
    if (i < 0 || i >= M || (r > 0 && i == rows[r - 1])) continue;
    for (int j = 0; j < N; j++) {
      float s = 0.f;
      for (int k = 0; k < K; k++) s += (float)A[(long)i * K + k] * (float)B[(long)k * N + j];
      float got = (float)C[(long)i * N + j];
      float err = fabs_(got - s), tol = 2e-3f * fabs_(s) + 5e-3f;
      if (err > maxerr) maxerr = err;
      if (err > tol) bad++;
      checked++;
    }
  }
  int fail = bad || untouched;
  printf("GEMM M=%d N=%d K=%d : %s  sampled bad=%d/%d untouched=%d zeros=%d/%d maxerr(milli)=%d\n",
         M, N, K, fail ? "FAIL" : "ok  ", bad, checked, (int)untouched, (int)zeros, M * N, milli(maxerr));
  return fail;
}

static float cin[3 * 230 * 230], cw[64 * 3 * 7 * 7], cout_[64 * 112 * 112];

/* the real stem: padded input 3x230x230, 64 filters 7x7 stride 2 -> 64x112x112 (GEMM M=64 N=12544 K=147) */
static int test_stem(void) {
  const int C_ = 3, H = 230, W = 230, F = 64, KH = 7, KW = 7, S = 2, OH = 112, OW = 112;
  for (int i = 0; i < C_ * H * W; i++) cin[i] = rnd();
  for (int i = 0; i < F * C_ * KH * KW; i++) cw[i] = rnd();
  for (int i = 0; i < F * OH * OW; i++) cout_[i] = 0.f;
  rvm_conv2d_nchw(cin, cw, cout_, C_, H, W, F, KH, KW, OH, OW, S, S, 1, 1);
  long zeros = 0;
  for (long i = 0; i < (long)F * OH * OW; i++) if (cout_[i] == 0.f) zeros++;
  int pos[5][2] = {{0, 0}, {0, 111}, {111, 0}, {55, 60}, {111, 111}};
  int bad = 0, checked = 0;
  float maxerr = 0;
  for (int p = 0; p < 5; p++)
    for (int f = 0; f < F; f++) {
      int oh = pos[p][0], ow = pos[p][1];
      float s = 0.f;
      for (int c = 0; c < C_; c++)
        for (int r = 0; r < KH; r++)
          for (int t = 0; t < KW; t++)
            s += cin[(c * H + oh * S + r) * W + ow * S + t] * cw[((f * C_ + c) * KH + r) * KW + t];
      float got = cout_[(f * OH + oh) * OW + ow];
      float err = fabs_(got - s), tol = 2e-3f * fabs_(s) + 5e-3f;
      if (err > maxerr) maxerr = err;
      if (err > tol) bad++;
      checked++;
    }
  printf("STEM conv 3x230x230 -> 64x112x112 : %s  sampled bad=%d/%d zeros=%d/%d maxerr(milli)=%d\n",
         bad ? "FAIL" : "ok  ", bad, checked, (int)zeros, F * OH * OW, milli(maxerr));
  return bad != 0;
}

int main(void) {
#ifndef HOST
  SET_MBA0_FP16();
  int tm = msettilem(1000), tn = msettilen(1000), tk = msettilek(1000);
  printf("granted tile sizes for fp16 (m,n,k) = %d %d %d\n", tm, tn, tk);
#endif
  printf("scratch addrs (low32): A=%x B=%x C=%x\n", (unsigned)(unsigned long)rvm_tmpA,
         (unsigned)(unsigned long)rvm_tmpB, (unsigned)(unsigned long)rvm_tmpC);
  int fails = 0;
  /* exact ResNet-50 GEMM shapes, in the order the network first uses them */
  fails += test_gemm(64, 12544, 147);   /* gemm 1  stem            */
  fails += test_gemm(64, 3136, 64);     /* gemm 2                  */
  fails += test_gemm(64, 3136, 576);    /* gemm 3                  */
  fails += test_gemm(256, 3136, 64);    /* gemm 4                  */
  fails += test_gemm(64, 3136, 256);    /* gemm 6                  */
  fails += test_gemm(128, 3136, 256);   /* gemm 12                 */
  fails += test_gemm(128, 784, 1152);   /* gemm 13                 */
  fails += test_gemm(512, 784, 128);    /* gemm 14                 */
  fails += test_gemm(256, 196, 2304);   /* gemm 26                 */
  fails += test_gemm(1024, 196, 256);   /* gemm 27                 */
  fails += test_gemm(512, 49, 4608);    /* gemm 45                 */
  fails += test_gemm(2048, 49, 512);    /* gemm 46                 */
  fails += test_gemm(512, 49, 2048);    /* gemm 48                 */
  fails += test_gemm(1, 1000, 2048);    /* gemm 54  fc             */
  /* the stem through the whole conv wrapper with the real shapes */
  fails += test_stem();
  printf(fails ? "SUMMARY: %d test(s) FAILED\n" : "SUMMARY: ALL PASS (%d)\n", fails);
  printf("DONE\n");
  return 0;
}
