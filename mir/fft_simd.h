#ifndef FFT_SIMD_MIR_H_2
#define FFT_SIMD_MIR_H_2 

#include <complex.h>
#include <stdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>

// Twiddle
typedef struct{
  float* tw;
  uint32_t* idx;
  uint32_t* in;
  uint32_t len;
} fft_tw_t;

fft_tw_t init_fft_simd(uint32_t len);

void free_fft_simd(fft_tw_t* tw);

void fft_simd(fft_tw_t const* tw, uint32_t len, const float _Complex* in, float _Complex* out);

void ifft_simd(fft_tw_t const* tw, uint32_t len, const float _Complex* in, float _Complex* out);

#endif
