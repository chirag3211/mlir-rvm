#ifndef RVM_RUNTIME_H
#define RVM_RUNTIME_H
/*
 * Runtime used by the C that lower_to_c.py generates.
 *
 *  - rvm_gemm_f16 : C[MxN] = A[MxK] * B[KxN], fp16 in / fp16 out (fp32 accumulate)
 *                   On the target this is the matrix-extension tile loop (copied from
 *                   riscv_hw_matmul, which already compiled with the matrix clang).
 *                   With -DHOST it is a plain C reference so the generated code can be
 *                   verified with gcc on the host.
 *  - rvm_conv2d_nchw / rvm_matmul : fp32 in memory, converted to fp16 around the GEMM.
 *                   Both ACCUMULATE into the output (linalg semantics: out += a*b).
 *
 * The converts and im2col are ordinary scalar code, so they show up in the
 * instruction trace next to the matrix instructions. Account for that when comparing.
 */
#include <stdint.h>
#include <stdio.h>

#ifndef RVM_MAX_A
#define RVM_MAX_A 1
#endif
#ifndef RVM_MAX_B
#define RVM_MAX_B 1
#endif
#ifndef RVM_MAX_C
#define RVM_MAX_C 1
#endif

/* -DRVM_BARRIERS : make the GEMM a non-inlined function and put compiler memory barriers around the
 *                  scratch fills / the GEMM / the accumulate, so clang cannot reorder or fold the
 *                  scalar accesses to rvm_tmpA/B/C around the (opaque) matrix instructions. */
#ifdef RVM_BARRIERS
#define RVM_GEMM_ATTR __attribute__((noinline))
#define RVM_BARRIER() __asm__ volatile("" ::: "memory")
#else
#define RVM_GEMM_ATTR
#define RVM_BARRIER() ((void)0)
#endif

#ifdef HOST
typedef float f16;
/* Host-only numerics emulation of the matrix unit (build with -DEMU_FP16_IN and/or -DEMU_FP16_OUT):
 *   EMU_FP16_IN  : weights / im2col activations are rounded to fp16 before the GEMM
 *   EMU_FP16_OUT : the GEMM result is rounded to fp16 (accumulation itself stays fp32, as on the target)
 * Both together reproduce the target's fp16-in / fp32-accumulate / fp16-out arithmetic. */
static inline float rnd16(float x) { volatile _Float16 h = (_Float16)x; return (float)h; }
#ifdef EMU_FP16_IN
#define RND_IN(x) rnd16(x)
#else
#define RND_IN(x) (x)
#endif
#ifdef EMU_FP16_OUT
#define RND_OUT(x) rnd16(x)
#else
#define RND_OUT(x) (x)
#endif
static RVM_GEMM_ATTR void rvm_gemm_f16(f16 *A, f16 *B, f16 *C, int M, int N, int K) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float acc = 0.f;
      for (int k = 0; k < K; k++) acc += A[(long)i * K + k] * B[(long)k * N + j];
      C[(long)i * N + j] = RND_OUT(acc);
    }
}
#else
#define RND_IN(x) (x)
#include "matrix_intrinsic.h"
typedef _Float16 f16;

static RVM_GEMM_ATTR void rvm_gemm_f16(f16 *A, f16 *B, f16 *C, int M, int N, int K) {
  const int ds = sizeof(f16);
  int tm = 0, tn = 0, tk = 0;
  RVM_BARRIER();
  for (int i = 0; i < M; i += tm) {
    SET_MBA0_FP16();
    tm = msettilem(M - i);
    for (int j = 0; j < N; j += tn) {
      tn = msettilen(N - j);
      SET_MBA0_FP32();
      mfloat32_t acc;
      acc = mfsub_mm(acc, acc);
      for (int k = 0; k < K; k += tk) {
        SET_MBA0_FP16();
        tk = msettilek(K - k);
        mfloat16_t a = mla_m(A + (long)i * K + k, K * ds);
        mfloat16_t b = mlb_m(B + (long)k * N + j, N * ds);
        SET_MBA0_FP16_FP32();
        acc = mfwma_mm(acc, a, b);
      }
      SET_MBA0_FP32_FP16();
      mfloat16_t d = mfncvt_f_fw_m(acc);
      SET_MBA0_FP16();
      msc_m(d, C + (long)i * N + j, N * ds);
      __asm__ volatile("" ::: "memory");
    }
  }
  RVM_BARRIER();
}
#endif

#ifdef RVM_CHECKSUMS
/* debug: sum and abs-sum of a buffer, printed as raw float bits (%x) so host and target logs compare exactly */
static void rvm_ck(int idx, const float *p, long n) {
  float s = 0.f, a = 0.f;
  for (long i = 0; i < n; i++) { float v = p[i]; s += v; a += v < 0 ? -v : v; }
  union { float f; unsigned u; } x, y;
  x.f = s; y.f = a;
  printf("CK %d %x %x\n", idx, x.u, y.u);
}
#endif

static f16 rvm_tmpA[RVM_MAX_A] __attribute__((aligned(64)));
static f16 rvm_tmpB[RVM_MAX_B] __attribute__((aligned(64)));
static f16 rvm_tmpC[RVM_MAX_C] __attribute__((aligned(64)));

#ifdef RVM_DEBUG
/* -DRVM_DEBUG : for the first RVM_DBG_CALLS GEMMs print abs-sum / zero-count of A, B (before) and C (after),
 * plus the scratch addresses >>0 (low 32 bits). Any later GEMM whose C is all zero is also reported. */
#ifndef RVM_DBG_CALLS
#define RVM_DBG_CALLS 4
#endif
static int rvm_dbg_n;
static unsigned rvm_fbits(float f) { union { float f; unsigned u; } x; x.f = f; return x.u; }
static float rvm_abs16(const f16 *p, long n, long *nz) {
  float s = 0.f; long z = 0;
  for (long i = 0; i < n; i++) { float v = (float)p[i]; if (v == 0.f) z++; s += v < 0 ? -v : v; }
  *nz = z; return s;
}
static void rvm_dbg_pre(const char *t, int M, int N, int K) {
  if (rvm_dbg_n >= RVM_DBG_CALLS) return;
  long za, zb;
  float sa = rvm_abs16(rvm_tmpA, (long)M * K, &za), sb = rvm_abs16(rvm_tmpB, (long)K * N, &zb);
  printf("DBG %d %s M=%d N=%d K=%d A.abs=%x A.zeros=%d/%d B.abs=%x B.zeros=%d/%d pA=%x pB=%x pC=%x\n",
         rvm_dbg_n, t, M, N, K, rvm_fbits(sa), (int)za, M * K, rvm_fbits(sb), (int)zb, K * N,
         (unsigned)(unsigned long)rvm_tmpA, (unsigned)(unsigned long)rvm_tmpB, (unsigned)(unsigned long)rvm_tmpC);
}
static void rvm_dbg_post(int M, int N) {
  long zc; float sc = rvm_abs16(rvm_tmpC, (long)M * N, &zc);
  if (rvm_dbg_n < RVM_DBG_CALLS || sc == 0.f)
    printf("DBG %d post  C.abs=%x C.zeros=%d/%d\n", rvm_dbg_n, rvm_fbits(sc), (int)zc, M * N);
  rvm_dbg_n++;
}
#define RVM_DBG_PRE(t, M, N, K) rvm_dbg_pre(t, M, N, K)
#define RVM_DBG_POST(M, N) rvm_dbg_post(M, N)
#else
#define RVM_DBG_PRE(t, M, N, K) ((void)0)
#define RVM_DBG_POST(M, N) ((void)0)
#endif

/* fp16 weights stored directly in weights.bin (lower_to_c.py --fp16-weights): no per-call conversion.
 * On the target f16 == _Float16, so the blob is handed to the GEMM as-is. On the host f16 is float
 * (numerics emulation), so the *host-only* path widens the weights into the scratch buffer. */
typedef _Float16 rvm_w16;

#define RVM_INLINE static inline __attribute__((always_inline))

/* Prevents clang from turning a loop into a memset/memcpy call (the libc stub copies byte-wise when unaligned). */
#define RVM_NOIDIOM() __asm__ volatile("")

RVM_INLINE void rvm_zero16(f16 *d, long n) {
  for (long i = 0; i < n; i++) { d[i] = (f16)0; RVM_NOIDIOM(); }
}

/* out[F,OH,OW] (+)= conv(in[C,H,W], A[F,C*KH*KW]); A is row-major fp16.
 * PT/PL : implicit zero padding (top/left) -- `in` is the UNPADDED input and out-of-range taps read as 0;
 *         bottom/right padding is implied by OH/OW.
 * OWR   : 1 = the output was known to be zero, so write instead of accumulate (no readback of out). */
RVM_INLINE void rvm_conv_core(const float *in, f16 *A, float *out,
                              int C, int H, int W, int F, int KH, int KW,
                              int OH, int OW, int SH, int SW, int DH, int DW,
                              int PT, int PL, int OWR) {
  const int M = F, K = C * KH * KW, N = OH * OW;
  /* any tap outside [0,H)x[0,W) (incl. bottom/right padding implied by OH/OW) needs the bounds-aware path */
  const int nopad = PT == 0 && PL == 0 && (OH - 1) * SH + (KH - 1) * DH < H && (OW - 1) * SW + (KW - 1) * DW < W;
  if (nopad && KH == 1 && KW == 1 && SH == 1 && SW == 1) {
    for (long i = 0; i < (long)K * N; i++) rvm_tmpB[i] = (f16)RND_IN(in[i]);
  } else if (nopad) {
    for (int c = 0; c < C; c++)
      for (int r = 0; r < KH; r++)
        for (int s = 0; s < KW; s++) {
          const long row = ((long)c * KH + r) * KW + s;
          for (int oh = 0; oh < OH; oh++) {
            const float *src = in + ((long)c * H + (long)oh * SH + (long)r * DH) * W + (long)s * DW;
            f16 *dst = rvm_tmpB + row * N + (long)oh * OW;
            for (int ow = 0; ow < OW; ow++) dst[ow] = (f16)RND_IN(src[(long)ow * SW]);
          }
        }
  } else {
    for (int c = 0; c < C; c++)
      for (int r = 0; r < KH; r++)
        for (int s = 0; s < KW; s++) {
          const long row = ((long)c * KH + r) * KW + s;
          /* valid output columns: 0 <= ow*SW + off < W */
          const int off = s * DW - PL;
          int lo = off >= 0 ? 0 : (-off + SW - 1) / SW;
          int hi = (W - 1 - off) < 0 ? 0 : (W - 1 - off) / SW + 1;
          if (hi > OW) hi = OW;
          if (lo > hi) lo = hi;
          for (int oh = 0; oh < OH; oh++) {
            const int ih = oh * SH + r * DH - PT;
            f16 *dst = rvm_tmpB + row * N + (long)oh * OW;
            if (ih < 0 || ih >= H) { rvm_zero16(dst, OW); continue; }
            const float *src = in + ((long)c * H + ih) * W + off;
            rvm_zero16(dst, lo);
            for (int ow = lo; ow < hi; ow++) dst[ow] = (f16)RND_IN(src[(long)ow * SW]);
            rvm_zero16(dst + hi, OW - hi);
          }
        }
  }
  RVM_BARRIER();
  RVM_DBG_PRE("conv", M, N, K);   /* with fp16 weights the A stats printed here are of the scratch, not of A */
  rvm_gemm_f16(A, rvm_tmpB, rvm_tmpC, M, N, K);
  RVM_BARRIER();
  RVM_DBG_POST(M, N);
  if (OWR) for (long i = 0; i < (long)M * N; i++) out[i] = (float)rvm_tmpC[i];
  else     for (long i = 0; i < (long)M * N; i++) out[i] += (float)rvm_tmpC[i];
}

/* fp32 weights, converted on every call. _x = with implicit padding / overwrite (see rvm_conv_core) */
static void rvm_conv2d_nchw_x(const float *in, const float *w, float *out,
                              int C, int H, int W, int F, int KH, int KW,
                              int OH, int OW, int SH, int SW, int DH, int DW, int PT, int PL, int OWR) {
  for (long i = 0; i < (long)F * C * KH * KW; i++) rvm_tmpA[i] = (f16)RND_IN(w[i]);
  rvm_conv_core(in, rvm_tmpA, out, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, PT, PL, OWR);
}
static void rvm_conv2d_nchw(const float *in, const float *w, float *out,
                            int C, int H, int W, int F, int KH, int KW,
                            int OH, int OW, int SH, int SW, int DH, int DW) {
  rvm_conv2d_nchw_x(in, w, out, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, 0, 0, 0);
}

/* weights already fp16 in the blob */
static void rvm_conv2d_nchw_wh_x(const float *in, const rvm_w16 *w, float *out,
                                 int C, int H, int W, int F, int KH, int KW,
                                 int OH, int OW, int SH, int SW, int DH, int DW, int PT, int PL, int OWR) {
#ifdef HOST
  for (long i = 0; i < (long)F * C * KH * KW; i++) rvm_tmpA[i] = (f16)(float)w[i];
  rvm_conv_core(in, rvm_tmpA, out, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, PT, PL, OWR);
#else
  rvm_conv_core(in, (f16 *)w, out, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, PT, PL, OWR);
#endif
}
static void rvm_conv2d_nchw_wh(const float *in, const rvm_w16 *w, float *out,
                               int C, int H, int W, int F, int KH, int KW,
                               int OH, int OW, int SH, int SW, int DH, int DW) {
  rvm_conv2d_nchw_wh_x(in, w, out, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, 0, 0, 0);
}

RVM_INLINE void rvm_matmul_core(f16 *B, float *C, int M, int N, int K) {
  RVM_BARRIER();
  RVM_DBG_PRE("matmul", M, N, K);
  rvm_gemm_f16(rvm_tmpA, B, rvm_tmpC, M, N, K);
  RVM_BARRIER();
  RVM_DBG_POST(M, N);
  for (long i = 0; i < (long)M * N; i++) C[i] += (float)rvm_tmpC[i];
}

/* C[M,N] += A[M,K] * B[K,N] */
static void rvm_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  for (long i = 0; i < (long)M * K; i++) rvm_tmpA[i] = (f16)RND_IN(A[i]);
  for (long i = 0; i < (long)K * N; i++) rvm_tmpB[i] = (f16)RND_IN(B[i]);
  rvm_matmul_core(rvm_tmpB, C, M, N, K);
}

/* same, B (weights) already fp16 in the blob */
static void rvm_matmul_bh(const float *A, const rvm_w16 *B, float *C, int M, int N, int K) {
  for (long i = 0; i < (long)M * K; i++) rvm_tmpA[i] = (f16)RND_IN(A[i]);
#ifdef HOST
  for (long i = 0; i < (long)K * N; i++) rvm_tmpB[i] = (f16)(float)B[i];
  rvm_matmul_core(rvm_tmpB, C, M, N, K);
#else
  rvm_matmul_core((f16 *)B, C, M, N, K);
#endif
}

/* out[C,OH,OW] = maxpool(in[C,H,W]) with the implicit -inf padding (PT,PL) handled by clamping the window at the
 * border: no padded copy, no pre-fill of the output.  Every output element is written. */
static void rvm_maxpool_nchw(const float *in, float *out, int C, int H, int W, int OH, int OW,
                             int KH, int KW, int SH, int SW, int PT, int PL) {
  /* output columns whose full KWxKW window lies inside the image: [owlo, owhi) */
  int owlo = (PL + SW - 1) / SW;
  int owhi = (W + PL - KW) >= 0 ? (W + PL - KW) / SW + 1 : 0;
  if (owhi > OW) owhi = OW;
  if (owlo > owhi) owlo = owhi;
  for (int c = 0; c < C; c++) {
    const float *ic = in + (long)c * H * W;
    float *oc = out + (long)c * OH * OW;
    for (int oh = 0; oh < OH; oh++) {
      const int h0 = oh * SH - PT;
      const int r0 = h0 < 0 ? -h0 : 0, r1 = (H - h0 < KH) ? H - h0 : KH;
      float *o = oc + (long)oh * OW;
      int ow = 0;
      while (ow < OW) {
        if (KH == 3 && KW == 3 && r0 == 0 && r1 == 3 && ow >= owlo && ow < owhi) {
          /* interior fast path: fully unrolled 3x3 window */
          const float *p0 = ic + (long)h0 * W + (ow * SW - PL);
          const float *p1 = p0 + W, *p2 = p1 + W;
          for (; ow < owhi; ow++, p0 += SW, p1 += SW, p2 += SW) {
            float m0 = __builtin_fmaxf(p0[0], p0[1]); m0 = __builtin_fmaxf(m0, p0[2]);
            float m1 = __builtin_fmaxf(p1[0], p1[1]); m1 = __builtin_fmaxf(m1, p1[2]);
            float m2 = __builtin_fmaxf(p2[0], p2[1]); m2 = __builtin_fmaxf(m2, p2[2]);
            m0 = __builtin_fmaxf(m0, m1);
            o[ow] = __builtin_fmaxf(m0, m2);
          }
          continue;
        }
        const int w0 = ow * SW - PL;
        const int s0 = w0 < 0 ? -w0 : 0, s1 = (W - w0 < KW) ? W - w0 : KW;
        float m = -__builtin_inff();
        for (int r = r0; r < r1; r++) {
          const float *row = ic + (long)(h0 + r) * W + w0;
          for (int s = s0; s < s1; s++) m = __builtin_fmaxf(m, row[s]);
        }
        o[ow] = m;
        ow++;
      }
    }
  }
}

#ifdef RVM_NHWC
/* ---- fp16 NHWC activations + hardware-unfold convolutions (lower_to_c.py --nhwc-fp16) ----
 * Activations live as fp16 [H][W][C]; conv weights are HWIO fp16 in the blob; the matrix unit does the
 * im2col itself (mlufa) and applies the zero padding.  See rvm_unfold.h. */
#include "rvm_unfold.h"
#ifndef RND_OUT
#define RND_OUT(x) (x)
#endif
#define RVM_H16(x) ((f16)RND_OUT(x))

/* one-time input conversion: fp32 NCHW -> fp16 NHWC */
static void rvm_nchw_to_nhwc(const float *in, f16 *out, int C, int H, int W) {
  for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++) out[((long)h * W + w) * C + c] = (f16)RND_IN(in[((long)c * H + h) * W + w]);
}

static void rvm_conv_unfold_wh(const f16 *in, const rvm_w16 *w, f16 *out, int C, int H, int W, int F, int KH, int KW,
                               int OH, int OW, int SH, int SW, int DH, int DW, int PT, int PB, int PL, int PR) {
#ifdef HOST
  for (long i = 0; i < (long)F * C * KH * KW; i++) rvm_tmpA[i] = (f16)(float)w[i];
  rvm_conv_unfold(in, rvm_tmpA, out, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, PT, PB, PL, PR);
#else
  rvm_conv_unfold(in, (const f16 *)w, out, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, PT, PB, PL, PR);
#endif
}

/* conv + folded BN + ReLU [+ residual add]; alpha/beta (fp16, F each) and w are fp16 blobs */
static void rvm_conv_unfold_epi_wh(const f16 *in, const rvm_w16 *w, f16 *out, const rvm_w16 *alpha, const rvm_w16 *beta,
                                   const f16 *add, f16 *addout, int C, int H, int W, int F, int KH, int KW,
                                   int OH, int OW, int SH, int SW, int DH, int DW, int PT, int PB, int PL, int PR) {
#ifdef HOST
  for (long i = 0; i < (long)F * C * KH * KW; i++) rvm_tmpA[i] = (f16)(float)w[i];
  for (int i = 0; i < F; i++) { rvm_tmpB[i] = (f16)(float)alpha[i]; rvm_tmpB[F + i] = (f16)(float)beta[i]; }
  rvm_conv_unfold_epi(in, rvm_tmpA, out, rvm_tmpB, rvm_tmpB + F, add, addout, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, PT, PB, PL, PR);
#else
  rvm_conv_unfold_epi(in, (const f16 *)w, out, (const f16 *)alpha, (const f16 *)beta, add, addout,
                      C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, PT, PB, PL, PR);
#endif
}

/* out += conv(...)  (only used where the output was pre-initialised, e.g. the FC bias) */
static void rvm_conv_unfold_wh_acc(const f16 *in, const rvm_w16 *w, f16 *out, int C, int H, int W, int F, int KH, int KW,
                                   int OH, int OW, int SH, int SW, int DH, int DW, int PT, int PB, int PL, int PR) {
  rvm_conv_unfold_wh(in, w, rvm_tmpC, C, H, W, F, KH, KW, OH, OW, SH, SW, DH, DW, PT, PB, PL, PR);
  for (long i = 0; i < (long)OH * OW * F; i++) out[i] = RVM_H16((float)out[i] + (float)rvm_tmpC[i]);
}

/* NHWC fp16 max pooling with implicit -inf padding (window clamped at the border) */
static void rvm_maxpool_nhwc(const f16 *in, f16 *out, int C, int H, int W, int OH, int OW,
                             int KH, int KW, int SH, int SW, int PT, int PL) {
  for (int oh = 0; oh < OH; oh++) {
    const int h0 = oh * SH - PT;
    const int r0 = h0 < 0 ? -h0 : 0, r1 = (H - h0 < KH) ? H - h0 : KH;
    for (int ow = 0; ow < OW; ow++) {
      const int w0 = ow * SW - PL;
      const int s0 = w0 < 0 ? -w0 : 0, s1 = (W - w0 < KW) ? W - w0 : KW;
      f16 *o = out + ((long)oh * OW + ow) * C;
      for (int c = 0; c < C; c++) {
        float m = -__builtin_inff();
        for (int r = r0; r < r1; r++) {
          const f16 *row = in + ((long)(h0 + r) * W + w0) * C + c;
          for (int s = s0; s < s1; s++) m = __builtin_fmaxf(m, (float)row[(long)s * C]);
        }
        o[c] = (f16)m;
      }
    }
  }
}
/* maxpool (window clamped at the border) + folded BN + ReLU in one pass over fp16 NHWC data.
 * Target: RVV fp16, vectorised over the channel axis exactly like riscv-dnn's maxpool_bn_relu
 * (one vle16 per tap and per output pixel; VLEN=1024 covers 64 channels in one register).
 * BN/ReLU are applied in fp16 (mul, add, max), as in dnn.  Host: scalar model with the same rounding. */
#ifndef HOST
#include <riscv_vector.h>
#endif
static void rvm_maxpool_bn_relu_nhwc(const f16 *in, f16 *out, const rvm_w16 *alpha, const rvm_w16 *beta,
                                     int C, int H, int W, int OH, int OW, int KH, int KW, int SH, int SW, int PT, int PL) {
#ifdef HOST
  for (int c = 0; c < C; c++) { rvm_tmpB[c] = (f16)(float)alpha[c]; rvm_tmpB[C + c] = (f16)(float)beta[c]; }
  for (int oh = 0; oh < OH; oh++) {
    const int h0 = oh * SH - PT;
    const int r0 = h0 < 0 ? -h0 : 0, r1 = (H - h0 < KH) ? H - h0 : KH;
    for (int ow = 0; ow < OW; ow++) {
      const int w0 = ow * SW - PL;
      const int s0 = w0 < 0 ? -w0 : 0, s1 = (W - w0 < KW) ? W - w0 : KW;
      f16 *o = out + ((long)oh * OW + ow) * C;
      for (int c = 0; c < C; c++) {
        float m = -65500.0f;
        for (int r = r0; r < r1; r++)
          for (int s = s0; s < s1; s++) m = __builtin_fmaxf(m, (float)in[((long)(h0 + r) * W + (w0 + s)) * C + c]);
        float y = RVM_H16(RVM_H16(m * (float)rvm_tmpB[c]) + (float)rvm_tmpB[C + c]);
        o[c] = (f16)(y > 0.0f ? y : 0.0f);
      }
    }
  }
#else
  for (int c0 = 0; c0 < C;) {
    const size_t vl = vsetvl_e16m1(C - c0);
    vfloat16m1_t va = vle16_v_f16m1((const _Float16 *)alpha + c0, vl);
    vfloat16m1_t vb = vle16_v_f16m1((const _Float16 *)beta + c0, vl);
    vfloat16m1_t vz = vfmv_v_f_f16m1((_Float16)0.0, vl);
    for (int oh = 0; oh < OH; oh++) {
      const int h0 = oh * SH - PT;
      const int r0 = h0 < 0 ? -h0 : 0, r1 = (H - h0 < KH) ? H - h0 : KH;
      for (int ow = 0; ow < OW; ow++) {
        const int w0 = ow * SW - PL;
        const int s0 = w0 < 0 ? -w0 : 0, s1 = (W - w0 < KW) ? W - w0 : KW;
        vfloat16m1_t vm = vfmv_v_f_f16m1((_Float16)-65500.0, vl);
        for (int r = r0; r < r1; r++) {
          const f16 *row = in + ((long)(h0 + r) * W + (w0 + s0)) * C + c0;
          for (int s = s0; s < s1; s++, row += C) vm = vfmax_vv_f16m1(vm, vle16_v_f16m1((const _Float16 *)row, vl), vl);
        }
        vm = vfmul_vv_f16m1(vm, va, vl);
        vm = vfadd_vv_f16m1(vm, vb, vl);
        vm = vfmax_vv_f16m1(vm, vz, vl);
        vse16_v_f16m1((_Float16 *)(out + ((long)oh * OW + ow) * C + c0), vm, vl);
      }
    }
    c0 += (int)vl;
  }
#endif
}
#endif /* RVM_NHWC */

#endif
