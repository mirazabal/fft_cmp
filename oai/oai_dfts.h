#ifndef OAI_DFTS
#define  OAI_DFTS

#include <stdint.h>

void dft256(int16_t *x,int16_t *y,unsigned char scale);
void dft1024(int16_t *x,int16_t *y,unsigned char scale);
void dft4096(int16_t *x,int16_t *y,unsigned char scale);
void dft16384(int16_t *x,int16_t *y,unsigned char scale);


void idft16384(int16_t *x,int16_t *y,unsigned char scale);
void idft4096(int16_t *x,int16_t *y,unsigned char scale);
void idft1024(int16_t *x,int16_t *y,unsigned char scale);
void idft256(int16_t *x,int16_t *y,unsigned char scale);

#endif
