#include "fft_simd.h"
#include "vec_simd.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define sin_pi_4 0.7071067812

static
void gen(uint32_t* in, int in_val, int* idx, int s, int N)
{
  assert(N > 7);

  if(N == 8){
    in[*idx] = in_val;
    *idx += 1;
    return;
  }

  gen(in, in_val, idx, s*2, N/2);
  gen(in, in_val+s, idx, s*2, N/2);
}

fft_tw_t init_fft_simd(uint32_t len)
{
  assert(len != 0);
  assert((len & (len -1)) == 0 && "Only power of 2 supported");
  assert(len > 8 && "Minimum FFT supported is 8");
  
  fft_tw_t dst = {.len = len};
 
  // tz*(N + N/2 + N/4 + N/8 ... ) = tz*(2N-1)
  uint32_t const tz = __builtin_ctz(len); 
  dst.tw = (float*)calloc(2*tz*(2*len + 1), sizeof(float));
  assert(dst.tw != NULL && "Memory exhausted");

  dst.idx = (uint32_t*)calloc(tz, sizeof(uint32_t)); 
  assert(dst.idx != NULL && "Memory exhausted");

  float const two_pi_by_n = 2*M_PI / len;

  float tmp[2*len];
  for(size_t i = 0, j = 0 ; i < len; i++, j+=2){
    tmp[j] = cosf(two_pi_by_n * i);    // real
    tmp[j+1] = sinf(two_pi_by_n * i);  // imag
  }

  for(size_t i = 0; i < tz; ++i){
    uint32_t idx = 0; 
    for(size_t a = i ; a > 0; --a){
      idx += len >> (a-1); 
    }
    dst.idx[i] = idx; // i = 0, val 0; i = 1, val = 4096; i = 2, val = 4096+2048 

    for(size_t j = 0; j < len >> i; j++){
     dst.tw[2*(dst.idx[i] + j)] = tmp[j* (1 << (1+i))];
     dst.tw[2*(dst.idx[i] + j) + 1] = tmp[(j*(1 << (1+i))) + 1]; 
    }
  }

  dst.in = (uint32_t*)calloc(len >> 3, sizeof(uint32_t));
  assert(dst.in != NULL && "Memory exhausted");
  int idx = 0;
  gen(dst.in, 0, &idx, 2, len);

  return dst;
}

void free_fft_simd(fft_tw_t* tw)
{
  assert(tw != NULL);

  assert(tw->idx != NULL);
  free(tw->idx);

  assert(tw->tw != NULL);
  free(tw->tw);
}

static
void fft8_simd(float* __restrict__    in, float* __restrict__    out, int stride)
{
  float a0r = in[0];
  float a0i = in[1];

  float a1r = in[stride];
  float a1i = in[stride+1];

  float a2r = in[2*stride];
  float a2i = in[2*stride+1];

  float a3r = in[3*stride];
  float a3i = in[3*stride+1];

  float a4r = in[4*stride];
  float a4i = in[4*stride+1];

  float a5r = in[5*stride];
  float a5i = in[5*stride+1];

  float a6r = in[6*stride];
  float a6i = in[6*stride+1];

  float a7r = in[7*stride];
  float a7i = in[7*stride+1];

  // Stage 1
  vec4f_simd_t a0 = init_vec4f(a0r, a0i, a0r, a0i);

  vec4f_simd_t a4 = init_vec4f(a4r, a4i, a4r, a4i);
  vec4f_simd_t m = init_vec4f(1.0, 1.0, -1.0, -1.0);

  vec4f_simd_t b04 = fma_vec4f(a4,m,a0);

  vec4f_simd_t a2 = init_vec4f(a2r, a2i, a2i, a6r);
  vec4f_simd_t a6 = init_vec4f(a6r, a6i, a6i, a2r);

  vec4f_simd_t b26 = fma_vec4f(a6, m, a2);

  vec4f_simd_t a15= init_vec4f(a1r,a1i, a1r-a5r, a1i - a5i);
  vec4f_simd_t a51 = init_vec4f(a5r,a5i, a1i-a5i, -a1r + a5r);
  vec4f_simd_t b15 = a15 + a51;

  vec4f_simd_t scalar = init_vec4f(1,1,sin_pi_4, sin_pi_4);
  b15 = b15* scalar;

  vec4f_simd_t a37 =  init_vec4f(a3r, a3i, a3i-a7i, -a3r+a7r);
  vec4f_simd_t a73 =  init_vec4f(a7r, a7i, -a3r+a7r, -a3i+a7i);
  vec4f_simd_t b37 = a37+a73;
  b37 = b37 * scalar;

  vec8f_simd_t b0415 = init_vec8f_vec4f(b04, b15); 
  vec8f_simd_t b2637 = init_vec8f_vec4f(b26, b37); 

  // Stage 2
  vec8f_simd_t c0415 =  b0415 + b2637;

  vec8f_simd_t b041357 = blend8f(0,1,2,3,5,12,7,14,b0415, b2637);  
  vec8f_simd_t b263175 = blend8f(0,1,2,3,5,12,7,14,b2637, b0415); 

  vec8f_simd_t c2637 = b041357 - b263175;

  vec8f_simd_t c0426 =  blend8f(0,1,2,3,8,9,10,11, c0415, c2637);
 // init_vec8f_vec4f(lo_vec8f(c0415), lo_vec8f(c2637));// = blend8<0,1,2,3,8,9,10,11>(c0415, c2637);
  vec8f_simd_t c1537 =  blend8f(4,5,6,7,12,13,14,15,c0415, c2637);
// init_vec8f_vec4f(hi_vec8f(c0415), hi_vec8f(c2637)); // = blend8<4,5,6,7,12,13,14,15>(c0415, c2637);

  // Stage 3
  vec8f_simd_t d0123 = c0426 + c1537; 
  vec8f_simd_t d4567 = c0426 - c1537; 

  store_vec8f(d0123, out);
  store_vec8f( d4567, out+8);
}


static
void fft_simd_impl(float * __restrict__   in, float *  __restrict__   out, int len, float *  __restrict__     tw, uint32_t * __restrict__    idx, uint32_t  *  __restrict__  in_idx)
{
  for(int i = 0; i < len >> 3; i += 1){
    fft8_simd(in + in_idx[i], out+16*i, len >> 2);
  }
  int N = 16;
  int out_idx = 32;
  for(int i = len >> 4; i > 0; i = i >> 1, N = N*2,  out_idx = out_idx*2){
    int stride = 2 * i; 
    int const v_idx = __builtin_ctz(stride)-1;
    uint32_t const tw_idx = idx[v_idx]; 
    
    for(int j = 0; j < i; ++j){
     float* out_p = out + out_idx * j;
     for (int k = 0; k < N / 2 ; k+=4) {
        vec8f_simd_t cs1 = load_vec8f(&tw[2*(tw_idx+k)]);
        vec8f_simd_t sc1 = permute8f(1,0,3,2,5,4,7,6,cs1);

        vec8f_simd_t x1357 = load_vec8f(&out_p[2 * k]);
        vec8f_simd_t x2468 = load_vec8f(&out_p[N + 2 * k]);

        vec8f_simd_t aceg = permute8f(0,0,2,2,4,4,6,6,x2468);
        vec8f_simd_t bdfh = permute8f(1,1,3,3,5,5,7,7,x2468);

        vec8f_simd_t tmp =  aceg*cs1;
        vec8f_simd_t tmp2 = bdfh*sc1;

        vec8f_simd_t m = init_vec8f(1.f,-1.f,1.f,-1.f,1.f,-1.f,1.f,-1.f);

        x2468 = fma_vec8f(tmp,m,tmp2);

        vec8f_simd_t sum  = x1357 + x2468 ; 
        vec8f_simd_t diff = x1357 - x2468 ; 

        store_vec8f(sum , &out_p[2 * k] );
        store_vec8f(diff, &out_p[N + 2 * k]);
      }
    }
  }
}

void fft_simd(fft_tw_t const* tw, uint32_t len, const float _Complex* in, float _Complex* out)
{
  assert(tw != NULL);
  assert(tw->len == len && "Incorrectly init twiddle. Length of input/output and twiddle mismatch");

//  int stride = 2;
  fft_simd_impl((float*)in, (float*)out, len, tw->tw, tw->idx, tw->in);
}

void ifft_simd(fft_tw_t const* tw, uint32_t len, const float _Complex* input, float _Complex* output)
{
  assert(tw != NULL);
  assert(tw->len == len && "Incorrectly init twiddle. Length of input/output and twiddle mismatch");

  float* out = (float*)output;
  float* in = (float*)input;

  int stride = 2;
  fft_simd_impl(in, out, len, tw->tw, tw->idx, tw->in);

  int ns = len * stride;
  int i = stride;

  // Reverse the array
  for(; i + 8 < ns / 2; i+=8){
    vec8f_simd_t lo = load_vec8f(&out[i]);
    lo = permute8f(7,6,5,4,3,2,1,0,lo);

    vec8f_simd_t hi = load_vec8f(&out[ns-i-7]); 
    hi = permute8f(7,6,5,4,3,2,1,0,hi);

    store_vec8f(lo,&out[ns-i-7]);
    store_vec8f(hi,&out[i]);
  }

  const float norm = 1.0/len;
  vec8f_simd_t norm8 = init_vec8f_set1(norm);

  i = 0;
  for (; i + 8 < ns ; i += 8){
    vec8f_simd_t tmp = load_vec8f(&out[i]);
    tmp = tmp * norm8; 
    store_vec8f(tmp, &out[i] );
  }

  for (; i < ns ; i ++){
    out[i] *= norm;
  }
}
