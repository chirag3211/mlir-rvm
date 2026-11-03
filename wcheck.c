/* wcheck.c -- is weights_blob actually present in memory on spike?
 * Embeds weights.bin / input.bin exactly like model.c does, then prints addresses, the first words,
 * and the number of non-zero 32-bit words per 1M-word (4 MB) chunk. Compare with the host numbers.
 * Build with -DMIMIC_BSS to also add ~41 MB of .bss (the model's activation pools + scratch). */
#include <stdint.h>
#include <stdio.h>
#ifdef HOST
#define EMBED(var, file) extern uint8_t var[]; \
  __asm__(".section .scdata.params,\"a\",@progbits\n\t.global " #var "\n\t.balign 64\n" #var ":\n\t.incbin \"" file "\"\n\t.previous\n")
#else
#include "incbin.h"
#define EMBED(var, file) INCBIN(var, file, ".scdata.params")
#endif
#ifndef WEIGHT_WORDS
#error define WEIGHT_WORDS
#endif
EMBED(weights_blob, "weights.bin");
EMBED(input_blob, "input.bin");

#ifdef MIMIC_BSS
static char bigbss[41000000] __attribute__((aligned(64)));
#endif

int main(void) {
  const unsigned *w = (const unsigned *)weights_blob;
  const unsigned *in = (const unsigned *)input_blob;
#ifdef MIMIC_BSS
  ((volatile char *)bigbss)[0] = 1;
  printf("bigbss addr=%x\n", (unsigned)(unsigned long)bigbss);
#endif
  printf("weights_blob addr=%x  input_blob addr=%x\n", (unsigned)(unsigned long)weights_blob, (unsigned)(unsigned long)input_blob);
  printf("W0..5 %x %x %x %x %x %x\n", w[0], w[1], w[2], w[3], w[4], w[5]);
  printf("I0..3 %x %x %x %x\n", in[0], in[1], in[2], in[3]);
  const long CH = 1L << 20;
  for (long c = 0; c * CH < WEIGHT_WORDS; c++) {
    long lo = c * CH, hi = lo + CH; if (hi > WEIGHT_WORDS) hi = WEIGHT_WORDS;
    long nz = 0;
    for (long i = lo; i < hi; i++) nz += (w[i] != 0);
    printf("CHUNK %d nz=%d of %d\n", (int)c, (int)nz, (int)(hi - lo));
  }
  printf("DONE\n");
  return 0;
}
