#include <complex.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "fft_simd.h"

#define LENGTH 4*1024

int64_t time_now_ns(void)
{
  struct timespec tms;
  if (clock_gettime(CLOCK_MONOTONIC_RAW,&tms)) {
    return -1;
  }
  int64_t nanos = tms.tv_sec * 1000000000;
  nanos += tms.tv_nsec;
  return nanos;
}

int main() 
{
  fft_tw_t tw = init_fft_simd(LENGTH);
  _Complex float* in = calloc(LENGTH, sizeof(_Complex float));

    for(size_t i = 0; i < LENGTH; ++i){
      *(float*)&in[i] = i; // Real
//      *((float*)&in[i]+1) = 0; // Img
    }

   _Complex float* out = calloc(LENGTH, sizeof(_Complex float));
  
  int64_t const t1 = time_now_ns();
  fft_simd(&tw, LENGTH, in, out);
  int64_t const t2 = time_now_ns();
  ifft_simd(&tw, LENGTH, out, in);
  int64_t const t3 = time_now_ns();

//  for(size_t i = 0; i < LENGTH; ++i){
//    printf("%lu Re: %f Im: %f \n", i, *((float*)&in[0][i]),*(((float*)&in[0][i])+1));
//  }

  printf("FFT %ld Inverse %ld \n", t2-t1, t3-t2);

//  free_fft_simd(&tw);
  return EXIT_SUCCESS;
}
