/* unfold_test.c -- checks rvm_conv_unfold (hardware-unfold convolution) against a plain reference.
 * HOST build (-DHOST): uses the functional model of the matrix instructions in emu_matrix.h.
 * Target build: the real instructions (run on spike).  -DFULL checks every output element (host only is practical).
 * printf has no %f: errors are printed in thousandths. */
#define RVM_MAX_A 1
#define RVM_MAX_B 1
#define RVM_MAX_C 1
#include "rvm_runtime.h"
#include "rvm_unfold.h"

static unsigned rng = 12345u;
static float rnd(void) { rng = rng * 1664525u + 1013904223u; return ((int)((rng >> 16) & 0xff) - 128) / 128.0f; }
static int milli(float x) { return (int)(x * 1000.0f + (x >= 0 ? 0.5f : -0.5f)); }
static float fabs_(float x) { return x < 0 ? -x : x; }

#define MAX_IN 802816
#define MAX_W 2359296
#define MAX_OUT 802816
static f16 g_in[MAX_IN], g_w[MAX_W], g_out[MAX_OUT];

static float ref_at(int C, int H, int W, int F, int KH, int KW, int S, int P, int oh, int ow, int f) {
  float s = 0.f;
  for (int r = 0; r < KH; r++) {
    int ih = oh * S - P + r;
    if (ih < 0 || ih >= H) continue;
    for (int t = 0; t < KW; t++) {
      int iw = ow * S - P + t;
      if (iw < 0 || iw >= W) continue;
      for (int c = 0; c < C; c++) s += (float)g_in[((long)ih * W + iw) * C + c] * (float)g_w[((long)(r * KW + t) * C + c) * F + f];
    }
  }
  return s;
}

static int test_conv(const char *name, int C, int H, int W, int F, int KH, int KW, int S, int P) {
  const int OH = (H + 2 * P - KH) / S + 1, OW = (W + 2 * P - KW) / S + 1;
  for (long i = 0; i < (long)H * W * C; i++) g_in[i] = (f16)rnd();
  for (long i = 0; i < (long)KH * KW * C * F; i++) g_w[i] = (f16)rnd();
  for (long i = 0; i < (long)OH * OW * F; i++) g_out[i] = (f16)12288.0f;
  rvm_conv_unfold(g_in, g_w, g_out, C, H, W, F, KH, KW, OH, OW, S, S, 1, 1, P, P, P, P);
  long untouched = 0;
  for (long i = 0; i < (long)OH * OW * F; i++) if ((float)g_out[i] == 12288.0f) untouched++;
  int bad = 0, checked = 0;
  float maxerr = 0;
#ifdef FULL
  for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) for (int f = 0; f < F; f++) {
#else
  int pos[9][2] = {{0, 0}, {0, OW - 1}, {OH - 1, 0}, {OH - 1, OW - 1}, {OH / 2, OW / 2}, {0, OW / 2}, {OH / 2, 0}, {OH - 1, OW / 2}, {OH / 3, OW / 3}};
  for (int p = 0; p < 9; p++) for (int f = 0; f < F; f++) { int oh = pos[p][0], ow = pos[p][1];
#endif
    float s = ref_at(C, H, W, F, KH, KW, S, P, oh, ow, f);
    float got = (float)g_out[((long)oh * OW + ow) * F + f];
    float err = fabs_(got - s), tol = 3e-3f * fabs_(s) + 5e-3f;
    if (err > maxerr) maxerr = err;
    if (err > tol) bad++;
    checked++;
  }
  int fail = bad || untouched;
  printf("CONV %-9s C=%d %dx%d F=%d K=%dx%d S=%d P=%d -> %dx%d : %s bad=%d/%d untouched=%d maxerr(milli)=%d\n", name, C, H, W, F, KH, KW, S, P,
         OH, OW, fail ? "FAIL" : "ok  ", bad, checked, (int)untouched, milli(maxerr));
  return fail;
}

int main(void) {
#ifndef HOST
  SET_MBA0_FP16();
  printf("granted tile sizes for fp16 (m,n,k) = %d %d %d\n", msettilem(1000), msettilen(1000), msettilek(1000));
#endif
  int fails = 0;
  fails += test_conv("tiny", 2, 4, 4, 3, 3, 3, 1, 1);
  fails += test_conv("oddch", 5, 9, 11, 7, 3, 3, 2, 1);
  fails += test_conv("nopad", 17, 8, 8, 19, 3, 3, 1, 0);
  fails += test_conv("k5s2", 3, 10, 7, 4, 5, 5, 2, 2);
  fails += test_conv("stem", 3, 224, 224, 64, 7, 7, 2, 3);
  fails += test_conv("3x3s1", 64, 56, 56, 64, 3, 3, 1, 1);
  fails += test_conv("1x1s1", 64, 56, 56, 256, 1, 1, 1, 0);
  fails += test_conv("1x1s2", 256, 56, 56, 512, 1, 1, 2, 0);
  fails += test_conv("3x3s2", 128, 56, 56, 128, 3, 3, 2, 1);
  fails += test_conv("deep", 512, 7, 7, 512, 3, 3, 1, 1);
  printf(fails ? "SUMMARY: %d test(s) FAILED\n" : "SUMMARY: ALL PASS (%d)\n", fails);
  printf("DONE\n");
  return fails != 0;
}
