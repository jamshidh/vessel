
#import <Accelerate/Accelerate.h>

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
