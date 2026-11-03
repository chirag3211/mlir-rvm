#ifndef RVM_UNFOLD_H
#define RVM_UNFOLD_H
/* Hardware-unfold convolution (no software im2col), same instruction sequence as riscv-dnn's conv_bn_relu_rvm
 * without the BN/ReLU epilogue.  Needs f16 (rvm_runtime.h) and, on the target, matrix_intrinsic.h.
 *   in  : NHWC, [H][W][C]            (pixel stride C elements; the hardware pads out-of-image taps itself)
 *   w   : HWIO, [KH][KW][C][F]
 *   out : NHWC, [OH][OW][F]          (written, not accumulated)
 * PT/PB/PL/PR are the zero padding (top/bottom/left/right), DH/DW the dilation (the simulator handles dilation
 * through the msetsk kernel-position field only). */
#ifdef HOST
#include "emu_matrix.h"
#endif

static void rvm_conv_unfold(const f16 *in, const f16 *w, f16 *out, int C, int H, int W, int F, int KH, int KW,
                            int OH, int OW, int SH, int SW, int DH, int DW, int PT, int PB, int PL, int PR) {
  const long ds = sizeof(f16);
  const long sin = (long)C * ds, sw_ = (long)F * ds, sout = (long)F * ds;
  const int M = OH * OW;
  msetinsh(((long)H << 16) | W, ((long)PT << 24) | (PB << 16) | (PL << 8) | PR);
  msetoutsh(((long)OH << 16) | OW, ((long)DH << 24) | (DW << 16) | (SH << 8) | SW);
  msetpadval(0);
  int tm = 0, tn = 0, tk = 0;
  for (int i = 0; i < M; i += tm) {
    SET_MBA0_FP16();
    tm = msettilem(M - i);
    const int ohp = i / OW, owp = i - ohp * OW;
    for (int j = 0; j < F; j += tn) {
      SET_MBA0_FP16();
      tn = msettilen(F - j);
      SET_MBA0_FP32();
      mfloat32_t acc;
      acc = mfsub_f_mm(acc, acc);
      for (int skh = 0; skh < KH; skh++) {
        const int hin = ohp * SH - PT + skh * DH;
        for (int skw = 0; skw < KW; skw++) {
          const int win = owp * SW - PL + skw * DW;
          msetsk((long)(((unsigned)hin << 16) | ((unsigned)win & 0xFFFF)), (long)(((unsigned)(skw * DW) << 16) | (unsigned)owp));
          const f16 *pa = in + ((long)hin * W + win) * C;
          const f16 *pb = w + ((long)(skh * KW + skw) * C) * F + j;
          for (int skc = 0; skc < C; skc += tk) {
            SET_MBA0_FP16();
            tk = msettilek(C - skc);
            mfloat16_t a = mlufa_m(pa + skc, sin);
            mfloat16_t b = mlb_m(pb + (long)skc * F, sw_);
            SET_MBA0_FP16_FP32();
            acc = mfwma_mm(acc, a, b);
          }
        }
      }
      SET_MBA0_FP32_FP16();
      mfloat16_t d = mfncvt_f_fw_m(acc);
      SET_MBA0_FP16();
      msc_m(d, out + (long)i * F + j, sout);
    }
  }
}

/* Same as rvm_conv_unfold with the riscv-dnn epilogue fused into every output tile:
 *     [acc1 += add ; addout = acc1]  (only if add != NULL: residual stream)
 *     out = max(acc1 * alpha + beta, 0)                      (folded BatchNorm + ReLU)
 * alpha/beta are per output channel (fp16, F elements); add/addout are NHWC fp16 like out.
 * (conv_bn_relu_rvm / conv_add_bn_relu_rvm in riscv-dnn.)  Loads of alpha/beta use row stride 0, so the
 * first row is replicated by the load itself; mbccr is kept to match the reference instruction sequence. */
static const f16 rvm_zero_row[64] = {0};

static void rvm_conv_unfold_epi(const f16 *in, const f16 *w, f16 *out, const f16 *alpha, const f16 *beta,
                                const f16 *add, f16 *addout, int C, int H, int W, int F, int KH, int KW,
                                int OH, int OW, int SH, int SW, int DH, int DW, int PT, int PB, int PL, int PR) {
  const long ds = sizeof(f16);
  const long sin = (long)C * ds, sw_ = (long)F * ds, sout = (long)F * ds;
  const int M = OH * OW;
  msetinsh(((long)H << 16) | W, ((long)PT << 24) | (PB << 16) | (PL << 8) | PR);
  msetoutsh(((long)OH << 16) | OW, ((long)DH << 24) | (DW << 16) | (SH << 8) | SW);
  msetpadval(0);
  int tm = 0, tn = 0, tk = 0;
  for (int i = 0; i < M; i += tm) {
    SET_MBA0_FP16();
    tm = msettilem(M - i);
    const int ohp = i / OW, owp = i - ohp * OW;
    for (int j = 0; j < F; j += tn) {
      SET_MBA0_FP16();
      tn = msettilen(F - j);
      mfloat16_t madd;
      if (add) madd = mlc_m(add + (long)i * F + j, sout);
      mfloat16_t mbeta = mlc_m(beta + j, 0);
      mfloat16_t malpha = mlc_m(alpha + j, 0);
      /* zero tile for the ReLU, loaded from memory: riscv-dnn builds it as x - x on an uninitialised register,
       * which is NaN whenever that register happens to hold a stale inf/NaN bit pattern. */
      mfloat16_t zero = mlc_m(rvm_zero_row, 0);
      SET_MBA0_FP32();
      mfloat32_t acc;
      acc = mfsub_f_mm(acc, acc);
      for (int skh = 0; skh < KH; skh++) {
        const int hin = ohp * SH - PT + skh * DH;
        for (int skw = 0; skw < KW; skw++) {
          const int win = owp * SW - PL + skw * DW;
          msetsk((long)(((unsigned)hin << 16) | ((unsigned)win & 0xFFFF)), (long)(((unsigned)(skw * DW) << 16) | (unsigned)owp));
          const f16 *pa = in + ((long)hin * W + win) * C;
          const f16 *pb = w + ((long)(skh * KW + skw) * C) * F + j;
          for (int skc = 0; skc < C; skc += tk) {
            SET_MBA0_FP16();
            tk = msettilek(C - skc);
            mfloat16_t a = mlufa_m(pa + skc, sin);
            mfloat16_t b = mlb_m(pb + (long)skc * F, sw_);
            SET_MBA0_FP16_FP32();
            acc = mfwma_mm(acc, a, b);
          }
        }
      }
      SET_MBA0_FP32_FP16();
      mfloat16_t d = mfncvt_f_fw_m(acc);
      SET_MBA0_FP16();
      if (add) {
        d = mfadd_mm(d, madd);
        if (addout) msc_m(d, addout + (long)i * F + j, sout);
      }
      mbeta = mbccr_m(mbeta);
      malpha = mbccr_m(malpha);
      d = mfmul_mm(d, malpha);
      d = mfadd_mm(d, mbeta);
      d = mfmax_mm(d, zero);
      msc_m(d, out + (long)i * F + j, sout);
    }
  }
}
#endif
