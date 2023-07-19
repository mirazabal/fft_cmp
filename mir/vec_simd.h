#ifndef VECTOR_SIMD_MIR_H
#define VECTOR_SIMD_MIR_H

#include <stdint.h>

#if defined(__x86_64__)

#include <immintrin.h>

typedef __m256 vec8f_simd_t;
typedef __m128 vec4f_simd_t;

#else
  _Static_assert(0!=0, "Unknown CPU architecture");
#endif

// Init a vector
#define init_vec4f(f0, f1, f2, f3) ({ \
    vec4f_simd_t dst; \
    dst =  _mm_setr_ps(f0,f1,f2,f3); \
    dst; } )

#define init_vec8f(f0, f1, f2, f3, f4, f5, f6, f7) ({ \
    vec8f_simd_t dst; \
    dst =  _mm256_setr_ps(f0,f1,f2,f3,f4,f5,f6,f7); \
    dst; } )

#define init_vec8f_set1(f0) ({ \
    vec8f_simd_t dst; \
    dst = _mm256_set1_ps(f0); \
    dst; } )

#define init_vec8f_vec4f(v0, v1) ({ \
    vec8f_simd_t dst; \
    dst = _mm256_insertf128_ps(_mm256_castps128_ps256(v0),(v1),1);\
    dst; } )


#define lo_vec8f(v0) ({\
    vec4f_simd_t dst; \
    dst = _mm256_castps256_ps128(v0);\
    dst; } )

#define hi_vec8f(v0) ({\
    vec4f_simd_t dst; \
    dst = _mm256_extractf128_ps(v0,1);\
    dst; } )

// Load unaligned address
#define load_vec8f(addr) ({ \
    vec8f_simd_t dst; \
    dst =  _mm256_loadu_ps(addr); \
    dst; } )

// Store unaligned address
#define store_vec8f(ymm, out)({ \
      _mm256_storeu_ps(out, ymm);\
      })

// Permute vec8f 
#define permute8f(i0, i1, i2, i3,i4, i5, i6, i7, v0) ({ \
    vec8f_simd_t ret_val; \
    __m256i x = _mm256_set_epi32(i7,i6,i5,i4,i3,i2,i1,i0); \
    ret_val = _mm256_permutevar8x32_ps(v0, x); \
    ret_val; })

// Fuse Mul-Add 
// a * b + c


#ifdef __FMA__
#define gen_fma_vec4f(a,b,c) ({ _mm_fmadd_ps(a,b,c); }) 
#define gen_fma_vec8f(a,b,c) ({ _mm256_fmadd_ps(a,b,c); }) 
#elif defined (__FMA4__)
#define gen_fma_vec4f(a,b,c) ({ _mm_macc_ps(a,b,c); });  
#define gen_fma_vec8f(a,b,c) ({ _mm256_macc_ps(a,b,c); });  
#else
#define gen_fma_vec8f(a,b,c) ({ a*b+c; });  
#endif

#define fma_vec8f(a,b,c) ({ \
    vec8f_simd_t ret_val; \
    ret_val = gen_fma_vec8f(a,b,c);\
    ret_val; }) 


#define fma_vec4f(a,b,c) ({ \
    vec4f_simd_t ret_val; \
    ret_val = gen_fma_vec4f(a,b,c);\
    ret_val; }) 




// Blend

#define blend8f_mask(i0, i1, i2, i3, i4, i5, i6, i7)( \
    ((i0 > 7) << 0) | ((i1 > 7) << 1) | \
    ((i2 > 7) << 2) | ((i3 > 7) << 3) | ((i4 > 7) << 4) | \
    ((i5 > 7) << 5) | ((i6 > 7) << 6) | ((i7 > 7) << 7) \
    )

#define blend8f(i0, i1, i2, i3, i4, i5, i6, i7, v0, v1) ({ \
    vec8f_simd_t ret_val; \
    vec8f_simd_t tmp_v0 = permute8f(i0 & 0x7, i1 & 0x7, i2 & 0x7, i3 & 0x7, i4 & 0x7, i5 & 0x7, i6 & 0x7, i7 & 0x7, v0); \
    vec8f_simd_t tmp_v1 = permute8f(i0 - 8, i1 - 8, i2 - 8 , i3 - 8, i4 - 8 , i5 - 8, i6 -8, i7 - 8, v1); \
    ret_val = _mm256_blend_ps(tmp_v0,tmp_v1, blend8f_mask(i0,i1,i2,i3,i4,i5,i6,i7) ); \
    ret_val; }) 

// _mm256_blend_ps

//_mm256_maskz_permutex2var_ps

#endif

