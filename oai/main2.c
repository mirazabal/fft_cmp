

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

#include "oai_dfts.h"

#include <stdint.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>

#define Re(x)   (_Generic((x), \
    complex float       : ((float *)       &(x)), \
    complex double      : ((double *)      &(x)), \
    complex long double : ((long double *) &(x)))[0])

#define Im(x)   (_Generic((x), \
    complex float       : ((float *)       &(x)), \
    complex double      : ((double *)      &(x)), \
    complex long double : ((long double *) &(x)))[1])

#define LENGTH 8*1024 

static
int64_t time_now_ns(void)
{
  struct timespec tms;

  if (clock_gettime(CLOCK_REALTIME,&tms)) {
    return -1;
  }
  int64_t nanos = tms.tv_sec * 1000000000 + tms.tv_nsec;
  return nanos; 
}

int main()
{
  int16_t* in = NULL; 
  int rc = posix_memalign( (void**)&in, 64, sizeof(int16_t)*LENGTH );
  assert(rc == 0 && in != NULL);

  int16_t* out = NULL; 
  rc = posix_memalign( (void**)&out, 64, sizeof(int16_t)*LENGTH );
  assert(rc == 0 && out != NULL);

  for(int i = 0; i < LENGTH/2; ++i){
    in[i*2] = i;
  }

  int64_t const t0 = time_now_ns();
  idft4096(in,out, 0);
  int64_t const t1 = time_now_ns();
  dft4096(out,in, 0);
  int64_t const t2 = time_now_ns();

  printf("Elapsed time ns 4096 = %ld invers %ld \n", t1-t0, t2-t1);

  free(in);
  free(out);
  return EXIT_SUCCESS;
}

