#ifndef EMU_MATRIX_H
#define EMU_MATRIX_H
/* HOST-ONLY functional model of the matrix-extension instructions used by the unfold convolution.
 * Semantics are taken from riscv-isa-sim (branch matrix): riscv/matrix_unit.cc (set_insh/set_moutsh/set_msk/set_pad,
 * msettile*) and riscv/m_ext_macros.h (MTU_UF_TR_ACC_LD = mlufa*, MTU_LS_LEN).  Tile geometry is a compile-time
 * parameter so the kernel can be checked against several (EMU_MROWS, EMU_MCOLS) combinations.
 * f16 is the element type of memory (float on the host; see rvm_runtime.h). */
#include <string.h>
#ifndef EMU_MROWS
#define EMU_MROWS 8           /* mrows = MLEN/RLEN */
#endif
#ifndef EMU_MCOLS
#define EMU_MCOLS 128         /* mcols = RLEN, in bits */
#endif
#define EMU_NMAX16 (EMU_MCOLS / 16)

typedef struct { float v[EMU_MROWS][EMU_NMAX16]; } emu_tile;
typedef emu_tile mfloat16_t;
typedef emu_tile mfloat32_t;

static struct {
  int sew;
  int tm, tn, tk;
  long inh, inw; int pt, pb, pl, pr;
  long outh, outw; int sh, sw, dh, dw;
  long skin_h, skin_w, skout_kw, skout_ow;
  unsigned padv;
} emu = {16, 0, 0, 0};

static inline int emu_min(int a, int b) { return a < b ? a : b; }
static inline void msettypei(int li) { emu.sew = (li == 2) ? 32 : 16; }
static inline void msettypehi(int hi) { (void)hi; }
static inline int msettilem(int avl) { emu.tm = emu_min(avl, EMU_MROWS); return emu.tm; }
static inline int msettilen(int avl) { emu.tn = emu_min(avl, EMU_MCOLS / emu.sew); return emu.tn; }
static inline int msettilek(int avl) {
  int kmax = emu_min(EMU_MROWS, EMU_MCOLS / emu.sew);
  emu.tk = emu_min(avl, kmax); return emu.tk;
}
#define SET_MBA0_FP16()      do { msettypei(1); msettypehi(1); } while (0)
#define SET_MBA0_FP32()      do { msettypei(2); msettypehi(4); } while (0)
#define SET_MBA0_FP16_FP32() do { msettypei(1); msettypehi(5); } while (0)
#define SET_MBA0_FP32_FP16() do { msettypei(2); msettypehi(5); } while (0)

/* msetinsh rs1 = hin<<16 | win ; rs2 = pt<<24 | pb<<16 | pl<<8 | pr */
static inline void msetinsh(long rs1, long rs2) {
  emu.inh = (rs1 >> 16) & 0xFFFF; emu.inw = rs1 & 0xFFFF;
  emu.pr = rs2 & 0xFF; emu.pl = (rs2 >> 8) & 0xFF; emu.pb = (rs2 >> 16) & 0xFF; emu.pt = (rs2 >> 24) & 0xFF;
}
/* msetoutsh rs1 = hout<<16 | wout ; rs2 = dh<<24 | dw<<16 | sh<<8 | sw */
static inline void msetoutsh(long rs1, long rs2) {
  emu.outh = (rs1 >> 16) & 0xFFFF; emu.outw = rs1 & 0xFFFF;
  emu.sw = rs2 & 0xFF; emu.sh = (rs2 >> 8) & 0xFF; emu.dw = (rs2 >> 16) & 0xFF; emu.dh = (rs2 >> 24) & 0xFF;
}
static inline void msetpadval(long rs1) { emu.padv = (unsigned)(rs1 & 0xFFFFFFFF); }
/* msetsk rs1 = inposh<<16 | inposw (both int16) ; rs2 = krposw<<16 | outposw */
static inline void msetsk(long rs1, long rs2) {
  emu.skin_w = (short)(rs1 & 0xFFFF); emu.skin_h = (short)((rs1 >> 16) & 0xFFFF);
  emu.skout_ow = rs2 & 0xFFFF; emu.skout_kw = (rs2 >> 16) & 0xFFFF;
}

/* mlufa: MTU_UF_TR_ACC_LD(0, elt, 'a'): tile m x k, whole tile pre-set to the pad value, then one input pixel
 * (k contiguous channels) per tile row, advancing over the output positions exactly like the simulator. */
static mfloat16_t mlufa_m(const f16 *p, long stride2) {
  mfloat16_t t;
  float padf = (float)emu.padv;           /* pad value 0 -> 0.0 (raw fp16 bits 0) */
  if (emu.padv != 0) padf = 0.f / 0.f;     /* only raw 0 is modelled; anything else is a bug here */
  for (int i = 0; i < EMU_MROWS; i++) for (int j = 0; j < EMU_NMAX16; j++) t.v[i][j] = padf;
  long baseAddr = (long)(const char *)p;
  long sh = emu.sh, sw = emu.sw, inh = emu.inh, inw = emu.inw, pl = emu.pl, outw = emu.outw;
  long inposh = emu.skin_h, inposw = emu.skin_w, krposw = emu.skout_kw, outposw = emu.skout_ow;
  for (int i = 0; i < emu.tm; ++i) {
    if (inposh >= 0 && inposh < inh && inposw >= 0 && inposw < inw)
      for (int j = 0; j < emu.tk; ++j)
        t.v[i][j] = (float)*(const f16 *)(baseAddr + j * (long)sizeof(f16));
    outposw++;
    if (outposw > outw - 1) {
      outposw = 0;
      baseAddr += (inw - inposw - pl) * stride2 + (sh - 1) * inw * stride2 + krposw * stride2;
      inposw = -pl + krposw;
      inposh += sh;
    } else {
      inposw += sw;
      baseAddr += sw * stride2;
    }
  }
  return t;
}
static mfloat16_t mlb_m(const f16 *p, long stride) {
  mfloat16_t t; memset(&t, 0, sizeof t);
  for (int k = 0; k < emu.tk; k++)
    for (int n = 0; n < emu.tn; n++) t.v[k][n] = (float)*(const f16 *)((const char *)p + k * stride + n * sizeof(f16));
  return t;
}
static mfloat16_t mla_m(const f16 *p, long stride) {
  mfloat16_t t; memset(&t, 0, sizeof t);
  for (int m = 0; m < emu.tm; m++)
    for (int k = 0; k < emu.tk; k++) t.v[m][k] = (float)*(const f16 *)((const char *)p + m * stride + k * sizeof(f16));
  return t;
}
static mfloat32_t mfsub_f_mm(mfloat32_t a, mfloat32_t b) { (void)a; (void)b; mfloat32_t t; memset(&t, 0, sizeof t); return t; }
static mfloat32_t mfsub_mm(mfloat32_t a, mfloat32_t b) { return mfsub_f_mm(a, b); }
static mfloat32_t mfwma_mm(mfloat32_t acc, mfloat16_t a, mfloat16_t b) {
  for (int m = 0; m < emu.tm; m++)
    for (int n = 0; n < emu.tn; n++) {
      float s = acc.v[m][n];
      for (int k = 0; k < emu.tk; k++) s += a.v[m][k] * b.v[k][n];
      acc.v[m][n] = s;
    }
  return acc;
}
static mfloat16_t mfncvt_f_fw_m(mfloat32_t a) {
  mfloat16_t t = a;
  for (int m = 0; m < emu.tm; m++) for (int n = 0; n < emu.tn; n++) { volatile _Float16 h = (_Float16)a.v[m][n]; t.v[m][n] = (float)h; }
  return t;
}
static void msc_m(mfloat16_t t, f16 *p, long stride) {
  for (int m = 0; m < emu.tm; m++)
    for (int n = 0; n < emu.tn; n++) *(f16 *)((char *)p + m * stride + n * sizeof(f16)) = (f16)t.v[m][n];
}

/* ---- epilogue instructions (conv + BN + ReLU [+ residual add]); the register-bank distinction between tile and
 * accumulator registers is not modelled: all tiles are emu_tile.  fp16 arithmetic = float op rounded to fp16. */
static inline float emu_r16(float x) { volatile _Float16 h = (_Float16)x; return (float)h; }
static mfloat16_t mlc_m(const f16 *p, long stride) {            /* mlce16.m: tile_m x tile_n, row stride in bytes */
  mfloat16_t t; memset(&t, 0, sizeof t);
  for (int m = 0; m < emu.tm; m++)
    for (int n = 0; n < emu.tn; n++) t.v[m][n] = (float)*(const f16 *)((const char *)p + m * stride + n * sizeof(f16));
  return t;
}
static mfloat16_t mbccr_m(mfloat16_t a) {                        /* mbccr.m: broadcast row 0 to every row */
  for (int m = 1; m < emu.tm; m++) for (int n = 0; n < emu.tn; n++) a.v[m][n] = a.v[0][n];
  return a;
}
static mfloat16_t mfadd_mm(mfloat16_t a, mfloat16_t b) {
  for (int m = 0; m < emu.tm; m++) for (int n = 0; n < emu.tn; n++) a.v[m][n] = emu_r16(a.v[m][n] + b.v[m][n]);
  return a;
}
static mfloat16_t mfmul_mm(mfloat16_t a, mfloat16_t b) {
  for (int m = 0; m < emu.tm; m++) for (int n = 0; n < emu.tn; n++) a.v[m][n] = emu_r16(a.v[m][n] * b.v[m][n]);
  return a;
}
static mfloat16_t mfmax_mm(mfloat16_t a, mfloat16_t b) {
  for (int m = 0; m < emu.tm; m++) for (int n = 0; n < emu.tn; n++) a.v[m][n] = a.v[m][n] > b.v[m][n] ? a.v[m][n] : b.v[m][n];
  return a;
}
#endif
