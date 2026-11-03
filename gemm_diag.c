/* gemm_diag.c -- ONE 8x8x8 GEMM, every element dumped, nothing else touches the matrix unit.
 * Output: addresses, then per row "got(milli) | ref(milli)". Zero columns / wrong rows are visible at a glance. */
#define RVM_MAX_A 4096
#define RVM_MAX_B 4096
#define RVM_MAX_C 4096
#include "rvm_runtime.h"

static f16 A[64] __attribute__((aligned(64))), B[64] __attribute__((aligned(64))), C[64] __attribute__((aligned(64)));
static unsigned rng = 12345u;
static float rnd(void) { rng = rng * 1664525u + 1013904223u; return ((int)((rng >> 16) & 0xff) - 128) / 128.0f; }
static int milli(float x) { return (int)(x * 1000.0f + (x >= 0 ? 0.5f : -0.5f)); }

int main(void) {
  printf("addr A=%p B=%p C=%p\n", (void *)A, (void *)B, (void *)C);
  for (int i = 0; i < 64; i++) { A[i] = (f16)rnd(); B[i] = (f16)rnd(); C[i] = (f16)12345.0f; }
  rvm_gemm_f16(A, B, C, 8, 8, 8);
  int bad = 0;
  for (int i = 0; i < 8; i++) {
    printf("row %d got:", i);
    for (int j = 0; j < 8; j++) printf(" %6d", milli((float)C[i * 8 + j]));
    printf("  | ref:");
    for (int j = 0; j < 8; j++) {
      float s = 0.f;
      for (int k = 0; k < 8; k++) s += (float)A[i * 8 + k] * (float)B[k * 8 + j];
      printf(" %6d", milli(s));
      float d = (float)C[i * 8 + j] - s; if (d < 0) d = -d;
      if (d > 2e-3f * (s < 0 ? -s : s) + 2e-3f) bad++;
    }
    printf("\n");
  }
  printf("RESULT: %s (%d of 64 wrong)\n", bad ? "FAIL" : "PASS", bad);
  printf("DONE\n");
  return 0;
}
