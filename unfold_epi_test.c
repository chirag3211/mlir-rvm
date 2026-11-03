/* unfold_epi_test.c -- conv + fused BN/ReLU [+ residual add] epilogue (rvm_conv_unfold_epi) vs a plain reference.
 * Host: functional model (emu_matrix.h).  Target: real instructions on spike.  -DFULL = check every element. */
#define RVM_MAX_A 1
#define RVM_MAX_B 1
#define RVM_MAX_C 1
#include "rvm_runtime.h"
#include "rvm_unfold.h"

static unsigned rng = 777u;
static float rnd(void) { rng = rng * 1664525u + 1013904223u; return ((int)((rng >> 16) & 0xff) - 128) / 128.0f; }
static int milli(float x) { return (int)(x * 1000.0f + (x >= 0 ? 0.5f : -0.5f)); }
static float fabs_(float x) { return x < 0 ? -x : x; }
#define MAX_IN 401408
#define MAX_W 589824
#define MAX_OUT 401408
static f16 g_in[MAX_IN], g_w[MAX_W], g_out[MAX_OUT], g_add[MAX_OUT], g_aout[MAX_OUT], g_alpha[1024], g_beta[1024];

static float conv_at(int C, int H, int W, int F, int KH, int KW, int S, int P, int oh, int ow, int f) {
  float s = 0.f;
  for (int r = 0; r < KH; r++) { int ih = oh * S - P + r; if (ih < 0 || ih >= H) continue;
    for (int t = 0; t < KW; t++) { int iw = ow * S - P + t; if (iw < 0 || iw >= W) continue;
      for (int c = 0; c < C; c++) s += (float)g_in[((long)ih * W + iw) * C + c] * (float)g_w[((long)(r * KW + t) * C + c) * F + f]; } }
  return s;
}
static int test_epi(const char *name, int C, int H, int W, int F, int KH, int KW, int S, int P, int with_add) {
  const int OH = (H + 2 * P - KH) / S + 1, OW = (W + 2 * P - KW) / S + 1;
  for (long i = 0; i < (long)H * W * C; i++) g_in[i] = (f16)rnd();
  for (long i = 0; i < (long)KH * KW * C * F; i++) g_w[i] = (f16)(rnd() * 0.25f);
  for (int f = 0; f < F; f++) { g_alpha[f] = (f16)(rnd() + 1.25f); g_beta[f] = (f16)(rnd() * 0.5f); }
  for (long i = 0; i < (long)OH * OW * F; i++) { g_add[i] = (f16)(rnd() * 2.f); g_out[i] = (f16)12288.0f; g_aout[i] = (f16)12288.0f; }
  rvm_conv_unfold_epi(g_in, g_w, g_out, g_alpha, g_beta, with_add ? g_add : 0, with_add ? g_aout : 0,
                      C, H, W, F, KH, KW, OH, OW, S, S, 1, 1, P, P, P, P);
  long untouched = 0;
  for (long i = 0; i < (long)OH * OW * F; i++) {
    if ((float)g_out[i] == 12288.0f) untouched++;
    if (with_add && (float)g_aout[i] == 12288.0f) untouched++;
    if (!with_add && (float)g_aout[i] != 12288.0f) untouched++;           /* addout must stay untouched */
  }
  int bad = 0, checked = 0; float maxerr = 0;
#ifdef FULL
  for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) for (int f = 0; f < F; f++) {
#else
  int pos[7][2] = {{0, 0}, {0, OW - 1}, {OH - 1, 0}, {OH - 1, OW - 1}, {OH / 2, OW / 2}, {0, OW / 2}, {OH / 3, OW / 3}};
  for (int p = 0; p < 7; p++) for (int f = 0; f < F; f++) { int oh = pos[p][0], ow = pos[p][1];
#endif
    float s = conv_at(C, H, W, F, KH, KW, S, P, oh, ow, f);
    long o = ((long)oh * OW + ow) * F + f;
    float a = with_add ? s + (float)g_add[o] : s;
    float y = a * (float)g_alpha[f] + (float)g_beta[f]; if (y < 0) y = 0;
    float e1 = fabs_((float)g_out[o] - y), t1 = 8e-3f * fabs_(y) + 1e-2f; if (e1 > maxerr) maxerr = e1; if (e1 > t1) bad++;
    if (with_add) { float e2 = fabs_((float)g_aout[o] - a), t2 = 5e-3f * fabs_(a) + 1e-2f; if (e2 > maxerr) maxerr = e2; if (e2 > t2) bad++; }
    checked++;
  }
  int fail = bad || untouched;
  printf("EPI  %-9s C=%d %dx%d F=%d K=%dx%d S=%d P=%d add=%d: %s bad=%d/%d untouched=%d maxerr(milli)=%d\n", name, C, H, W, F, KH, KW, S, P,
         with_add, fail ? "FAIL" : "ok  ", bad, checked, (int)untouched, milli(maxerr));
  return fail;
}
int main(void) {
#ifndef HOST
  SET_MBA0_FP16();
#endif
  int fails = 0;
  fails += test_epi("tiny", 2, 4, 4, 3, 3, 3, 1, 1, 0);
  fails += test_epi("tiny+add", 2, 4, 4, 3, 3, 3, 1, 1, 1);
  fails += test_epi("odd", 5, 9, 11, 7, 3, 3, 2, 1, 1);
  fails += test_epi("1x1", 64, 14, 14, 64, 1, 1, 1, 0, 0);
  fails += test_epi("1x1+add", 64, 14, 14, 256, 1, 1, 1, 0, 1);
  fails += test_epi("3x3", 32, 14, 14, 32, 3, 3, 1, 1, 0);
  fails += test_epi("3x3s2", 32, 14, 14, 48, 3, 3, 2, 1, 0);
  fails += test_epi("stem", 3, 64, 64, 16, 7, 7, 2, 3, 0);
  fails += test_epi("7x7+add", 128, 7, 7, 256, 1, 1, 1, 0, 1);
  printf(fails ? "SUMMARY: %d test(s) FAILED\n" : "SUMMARY: ALL PASS (%d)\n", fails);
  printf("DONE\n");
  return fails != 0;
}
