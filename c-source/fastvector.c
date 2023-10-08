
//#define MAC

#ifdef MAC

#include <Accelerate/Accelerate.h>

float dot_Int4X32(char *x, char *y) {

    const uint8_t * restrict p0 = ((const uint8_t *)x);
    const uint8_t * restrict p1 = ((const uint8_t *)y);

    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int8x16_t  s8b = vdupq_n_s8(0x8);
    
    const uint8x16_t v0 = vld1q_u8(p0);
    const uint8x16_t v1 = vld1q_u8(p1);
    
    // 4-bit -> 8-bit
    const int8x16_t v0l = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
    const int8x16_t v1l = vreinterpretq_s8_u8(vandq_u8(v1, m4b));
    
    const int8x16_t v0h = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
    const int8x16_t v1h = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
    
    // sub 8
    const int8x16_t v0ls = vsubq_s8(v0l, s8b);
    const int8x16_t v1ls = vsubq_s8(v1l, s8b);
    
    const int8x16_t v0hs = vsubq_s8(v0h, s8b);
    const int8x16_t v1hs = vsubq_s8(v1h, s8b);
    
    // dot product into int16x8_t
    int32x4_t p = vdotq_s32(vdupq_n_s32(0), v0ls, v1ls);
    
    p = vdotq_s32(p, v0hs, v1hs);
    
    return vaddvq_s32(p);

}
#else

#include <stdio.h>
#include <stdint.h>

float dot_Int4X32(char *x, char *y) {
  const uint8_t * restrict p0 = (uint8_t *) x;
  const uint8_t * restrict p1 = (uint8_t *) y;

  float sum = 0.0;

  for(int i = 0; i < 16; i++)
    sum += ((p0[i] & 0xf)-8) * ((p1[i] & 0xf)-8) + ((p0[i] >> 4)-8) * ((p1[i] >> 4)-8);
  
  return sum;
}

#endif


void printBytes(char *name, char *val) {
  printf("%s = ", name);
  for(int i=0; i < 40; i++) {
    printf("%02hhx-", val[i]);
  }
  printf("\n");
}

const int floatSize = 4;
const int nibbleSize = 16;
const int blockSize = floatSize + nibbleSize;

float vector_dot(int len, char *x, char *y) {
  float sum0 = 0.0;
  float sum1 = 0.0;

  //printBytes("x", x);
  //printBytes("y", y);

  for(int i = 0; i < len; i += 2) {
    {
      char *px = x+blockSize*i;
      char *py = y+blockSize*i;
      float *fx = (float*) px;
      float *fy = (float *) py;
      float dot = dot_Int4X32(px+floatSize, py+floatSize);
      float val = *fx * *fy * dot;

      //printf("partial sum = %a\n", val);

      sum0 += *fx * *fy * dot;

    }

    {
      char *px = x+blockSize*(i+1);
      char *py = y+blockSize*(i+1);
      float *fx = (float*) px;
      float *fy = (float *) py;
      float dot = dot_Int4X32(px+floatSize, py+floatSize);
      float val = *fx * *fy * dot;
      //printf("partial sum = %a\n", val);

      sum1 += *fx * *fy * dot;;

    }


    
    //printf("partial sum0 = %a\n", sum0);
    //printf("partial sum1 = %a\n", sum1);
  }
  float sum = sum0 + sum1;
  //printf("sum0 = %a\n", sum0);
  //printf("sum1 = %a\n", sum1);
  //printf("sum = %a\n", sum);
  //exit(1);
  return sum;
}

double fusionMultiplySum(float x, float y, double s) {
  double ret = s;
  ret += x*y;
  return ret;
}

float fusionMultiplySumAllFloat(float x, float y, float s) {
  float ret = s;
  ret += x*y;
  return ret;
}
