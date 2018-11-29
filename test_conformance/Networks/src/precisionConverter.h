#pragma once
#include <stdint.h>


/** @brief Converts FP32 to FP16 with rounding to nearest value to minimize error
*    the denormal values are converted to 0.
*  @param x - value in FP32 format
*  @return value in FP16 format
*/
uint16_t FP32ToFP16(float x);

/** @brief Converts FP16 to FP32 
*  @param x - value in FP16 format
*  @return value in FP32 format
*/
float FP16ToFP32(uint16_t x);

/** @brief Converts S16 (signed int16) to a float 
*  @param s16Pixel - A pointer to a value in S16 format.
*  @return float value
*/
float S16ToFloat(const char* s16Pixel);

/** @brief Converts Q78 to a float 
*  @param q78Pixel - A pointer to a value in Q78 format.
*  @return float value
*/
float Q78ToFloat(const char* q78Pixel);

/** @brief Converts FP16 to a float 
*  @param fp16Pixel - A pointer to a value in FP16 format.
*  @return float value
*/
float FP16ToFloat(const char* fp16Pixel);

/** @brief Converts FP32 to a float. It just copies the value from the input pointer.
*  @param fp32Pixel - A pointer to a value in FP32 format.
*  @return float value
*/
float FP32ToFloat(const char* fp32Pixel);

/** @brief Converts float to S16 format and copy it to the input pointer
*  @param s16Pixel - A pointer where to copy the converted value
*  @return void
*/
void floatToS16(float floatValue, char* s16Pixel);

/** @brief Converts float to Q78 format and copy it to the input pointer
*  @param q78Pixel - A pointer where to copy the converted value
*  @return void
*/
void floatToQ78(float floatValue, char* q78Pixel);

/** @brief Converts float to FP16 format and copy it to the input pointer
*  @param fp16Pixel - A pointer where to copy the converted value
*  @return void
*/
void floatToFP16(float floatValue, char* fp16Pixel);

/** @brief Converts float copy the float value to the input pointer
*  @param fp32Pixel - A pointer where to copy the float value
*  @return void
*/
void floatToFP32(float floatValue, char* fp32Pixel);

