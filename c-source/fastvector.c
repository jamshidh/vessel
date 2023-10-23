
#if defined(MAC)

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
#elif defined(LINUX)

#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>


static inline __m256i bytesFromNibbles( const uint8_t* rsi )
{
    // Load 16 bytes from memory
    __m128i tmp = _mm_loadu_si128( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m256i bytes = _mm256_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    __m256i high = _mm256_andnot_si256( lowMask, bytes );
    __m256i low = _mm256_and_si256( lowMask, bytes );
    high = _mm256_slli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );
    return bytes;
}

float dot_Int4X32(char *x, char *y) {
  
  // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
  __m256i x_bytes = bytesFromNibbles((uint8_t *) x);
  __m256i y_bytes = bytesFromNibbles((uint8_t *) y);

  // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
  const __m256i off = _mm256_set1_epi8(8);
  x_bytes = _mm256_sub_epi8(x_bytes, off);
  y_bytes = _mm256_sub_epi8(y_bytes, off);

  // Sign-extend first 16 signed bytes into int16_t
  __m256i x16 = _mm256_cvtepi8_epi16( _mm256_castsi256_si128(x_bytes) );
  __m256i y16 = _mm256_cvtepi8_epi16( _mm256_castsi256_si128(y_bytes) );

  // Compute products of int16_t integers, add pairwise
  __m256i i32 = _mm256_madd_epi16( x16, y16 );

  // Sign-extend last 16 signed bytes into int16_t vectors
  x16 = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( x_bytes, 1 ) );
  y16 = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( y_bytes, 1 ) );
  
  // Accumulate products of int16_t integers
  i32 = _mm256_add_epi32( i32, _mm256_madd_epi16( x16, y16 ) );

  // Convert int32_t to float
  __m256 p = _mm256_cvtepi32_ps( i32 );

  __m128 res = _mm256_extractf128_ps( p, 1 );

  res = _mm_add_ps( res, _mm256_castps256_ps128( p ) );
  res = _mm_add_ps( res, _mm_movehl_ps( res, res ) );
  res = _mm_add_ss( res, _mm_movehdup_ps( res ) );

  return  _mm_cvtss_f32(res);
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
