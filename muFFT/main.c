#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "muFFT/fft.h"
#include "muFFT/fft_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static
int64_t time_now_ns(void)
{
  struct timespec tms;
  if (clock_gettime(CLOCK_REALTIME,&tms)) {
    return -1;
  }
  int64_t nanos = tms.tv_sec * 1000000000;
  nanos += tms.tv_nsec;
  return nanos; 
}

int main()
{
  unsigned N = 4096;
  int direction = 1;
  unsigned flags = 0;

  cfloat *input = mufft_alloc(N * sizeof(cfloat));
  cfloat *output = mufft_alloc(N * sizeof(cfloat));

  for (unsigned i = 0; i < N; i++)
  {
    float real = i; //(float)rand() / RAND_MAX - 0.5f;
    float imag = 0.0;// (float)rand() / RAND_MAX - 0.5f;
    input[i] = cfloat_create(real, imag);
  }

  mufft_plan_1d *muplan = mufft_create_plan_1d_c2c(N, direction, flags);
  mufft_assert(muplan != NULL);

  int64_t now = time_now_ns();
  mufft_execute_plan_1d(muplan, output, input);
  int64_t stop = time_now_ns();

  printf("Elapsed time %ld \n", stop - now);


//  for(int i =0; i < N; ++i){
//    printf("Real %lf Im %lf \n", output[i].real, output[i].imag  );
//  }

  mufft_free(input);
  mufft_free(output);
  mufft_free_plan_1d(muplan);
  return 0;
}

