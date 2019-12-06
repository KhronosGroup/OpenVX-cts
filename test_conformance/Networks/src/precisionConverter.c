#include "precisionConverter.h"
#include <limits.h>
#include <memory.h>


// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16     0x7C00U

//small helper function to represent uint32_t value as float32
float asfloat(uint32_t v)
{
    unsigned long value = (unsigned long)(void *)&v;
    return *(float*)value;
}


/** @brief Converts FP32 to FP16 with rounding to nearest value to minimize error
*    the denormal values are converted to 0.
*  @param x - value in FP32 format
*  @return value in FP16 format
**************************************************************/
uint16_t FP32ToFP16(float x)
{
    //create minimal positive normal f16 value in f32 format
    //exp:-14,mantissa:0 -> 2^-14 * 1.0
    static uint32_t min16_i = (127 - 14) << 23;
    float min16 = asfloat(min16_i);

    //create maximal positive normal f16 value in f32 and f16 formats
    //exp:15,mantissa:11111 -> 2^15 * 1.(11111)
    static uint32_t max16_i = ((127 + 15) << 23) | 0x007FE000;
    float    max16 = asfloat(max16_i);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    // define and declare variable for intermidiate and output result
    // the union is used to simplify representation changing
    union
    {
        float f;
        uint32_t u;
    } v;
    v.f = x;

    // get sign in 16bit format
    uint32_t    s = (v.u >> 16) & 0x8000; // sign 16:  00000000 00000000 10000000 00000000

                                          // make it abs
    v.u &= 0x7FFFFFFF;                    // abs mask: 01111111 11111111 11111111 11111111

                                          // check NAN and INF
    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32)
    {
        if (v.u & 0x007FFFFF)
            return s | (v.u >> (23 - 10)) | 0x0200; // return NAN f16
        else
            return s | (v.u >> (23 - 10)); // return INF f16
    }

    // to make f32 round to nearest f16
    // create halfULP for f16 and add it to origin value
    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    // if input value is not fit normalized f16 then return 0
    // denormals are not covered by this code and just converted to 0
    if (v.f < min16*0.5F)
        return s;

    // if input value between min16/2 and min16 then return min16
    if (v.f < min16)
        return s | (1 << 10);

    // if input value more than maximal allowed value for f16
    // then return this maximal value
    if (v.f >= max16)
        return max16f16 | s;

    // change exp bias from 127 to 15
    v.u -= ((127 - 15) << 23);

    // round to f16
    v.u >>= (23 - 10);

    return v.u | s;
}

/** @brief Converts FP16 to FP32
*  @param x - value in FP16 format
*  @return value in FP32 format
**************************************************************/
float FP16ToFP32(uint16_t x)
{
    // this is storage for output result
    uint32_t u = x;

    // get sign in 32bit format
    uint32_t s = ((u & 0x8000) << 16);

    // check for NAN and INF
    if ((u & EXP_MASK_F16) == EXP_MASK_F16)
    {
        //keep mantissa only
        u &= 0x03FF;

        // check if it is NAN and raise 10 bit to be align with intrin
        if (u)
            u |= 0x0200;

        u <<= (23 - 10);
        u |= EXP_MASK_F32;
        u |= s;
    }
    // check for zero and denormals. both are converted to zero
    else if ((x & EXP_MASK_F16) == 0)
    {
        u = s;
    }
    else
    {
        //abs
        u = (u & 0x7FFF);

        // shift mantissa and exp from f16 to f32 position
        u <<= (23 - 10);

        //new bias for exp (f16 bias is 15 and f32 bias is 127)
        u += ((127 - 15) << 23);

        //add sign
        u |= s;
    }

    //finaly represent result as float and return
    return asfloat(u);
}

/** @brief Converts S16 (signed int16) to a float
*  @param s16Pixel - A pointer to a value in S16 format.
*  @return float value
***************************************************************/
float S16ToFloat(const char* s16Pixel)
{
    int16_t value = *((int16_t*)s16Pixel);
    return (float)value;
}

/** @brief Converts Q78 to a float
*  @param q78Pixel - A pointer to a value in Q78 format.
*  @return float value
**************************************************************/
float Q78ToFloat(const char* q78Pixel)
{
    int16_t value = *((int16_t*)q78Pixel);
    return ((float)value) / 256.0;
}

/** @brief Converts FP16 to a float
*  @param fp16Pixel - A pointer to a value in FP16 format.
*  @return float value
**************************************************************/
float FP16ToFloat(const char* fp16Pixel)
{
    uint16_t value = *((uint16_t*)fp16Pixel);
    return FP16ToFP32(value);
}

/** @brief Converts FP32 to a float. It just copies the value from the input pointer.
*  @param fp32Pixel - A pointer to a value in FP32 format.
*  @return float value
**************************************************************/
float FP32ToFloat(const char* fp32Pixel)
{
    return *((float*)fp32Pixel);
}

/** @brief Converts float to S16 format and copy it to the input pointer
*  @param s16Pixel - A pointer where to copy the converted value
*  @return void
**************************************************************/
void floatToS16(float floatValue, char* s16Pixel)
{
    int16_t value = (int16_t)floatValue;
    memcpy(s16Pixel, &value, sizeof(int16_t));
}

/** @brief Converts float to Q78 format and copy it to the input pointer
*  @param q78Pixel - A pointer where to copy the converted value
*  @return void
**************************************************************/
void floatToQ78(float floatValue, char* q78Pixel)
{
    float r = floatValue < 0.0 ? -0.5 : 0.5;
    int tmpValue = (int)((floatValue * 256.0 + r));
    int16_t value = tmpValue > SHRT_MAX ? SHRT_MAX : (tmpValue < SHRT_MIN ? SHRT_MIN : (int16_t)tmpValue);
    memcpy(q78Pixel, &value, sizeof(int16_t));
}

/** @brief Converts float to FP16 format and copy it to the input pointer
*  @param fp16Pixel - A pointer where to copy the converted value
*  @return void
**************************************************************/
void floatToFP16(float floatValue, char* fp16Pixel)
{
    uint16_t value = FP32ToFP16(floatValue);
    memcpy(fp16Pixel, &value, sizeof(uint16_t));
}

/** @brief Converts float copy the float value to the input pointer
*  @param fp32Pixel - A pointer where to copy the float value
*  @return void
**************************************************************/
void floatToFP32(float floatValue, char* fp32Pixel)
{
    memcpy(fp32Pixel, &floatValue, sizeof(float));
}

