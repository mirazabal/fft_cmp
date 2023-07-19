#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>

#define N 4096

int64_t time_now_ns(void)
{
  struct timespec tms;

  /* The C11 way */
  /* if (! timespec_get(&tms, TIME_UTC))  */

  /* POSIX.1-2008 way */
  if (clock_gettime(CLOCK_REALTIME,&tms)) {
    return -1;
  }
  /* seconds, multiplied with 1 million */
  int64_t nanos = tms.tv_sec * 1000000000;
  /* Add full microseconds */
  nanos += tms.tv_nsec;
  /* round up if necessary */
  return nanos;
}

int main(void) {
  fftwf_complex in[N], out[N], in2[N]; /* double [2] */
  fftwf_plan p, q;
  int i;

  /* forward Fourier transform, save the result in 'out' */
  //p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
  p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_PATIENT);

  /* prepare a cosine wave */
  for (i = 0; i < N; i++) {
    in[i][0] = i; //cos(3 * 2*M_PI*i/N);
    in[i][1] = 0;
  }

  int64_t now = time_now_ns();
  fftwf_execute(p);
  int64_t stop = time_now_ns();


  for (i = 0; i < N; i++)
    printf("freq: %3d %+9.5f %+9.5f I\n", i, out[i][0], out[i][1]);
  fftwf_destroy_plan(p);

  printf("Elapsed time ns %ld \n", stop - now);

  /* backward Fourier transform, save the result in 'in2' */
  printf("\nInverse transform:\n");
  //q = fftwf_plan_dft_1d(N, out, in2, FFTW_BACKWARD, FFTW_MEASURE);
  q = fftwf_plan_dft_1d(N, out, in2, FFTW_BACKWARD, FFTW_PATIENT);

  now = time_now_ns();
  fftwf_execute(q);
  stop = time_now_ns();

  /* normalize */
  for (i = 0; i < N; i++) {
    in2[i][0] *= 1./N;
    in2[i][1] *= 1./N;
  }
  for (i = 0; i < N; i++){
//    printf("recover: %3d %+9.5f %+9.5f I vs. %+9.5f %+9.5f I\n",
//        i, in[i][0], in[i][1], in2[i][0], in2[i][1]);
  }
  printf("Elapsed time %ld \n", stop - now);

  fftwf_destroy_plan(q);

  fftwf_cleanup();
  return 0;
}


