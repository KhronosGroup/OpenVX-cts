#pragma once

#include "VX/vx.h"
#include <VX/vx_khr_nn.h>

/** @brief Loads image from a file and converts it to float.
* supported formats: jpeg, png, bmp, psd, tga, gif, hdr, pic, ppm, pgm
*  @param fileName - full path to the image file
*  @param width - image width as it appears in the file (output)
*  @param height - image height as it appears in the file (output)
*  @param channels - the number of the image channels as it appears in the file (output)
*  @return image in float
*/
float* loadImageFromFile(const char* fileName, int* width, int* height, int* channels);


/** @brief Loads image from a file .
* supported formats: jpeg, png, bmp, psd, tga, gif, hdr, pic, ppm, pgm
*  @param fileName - full path to the image file
*  @param width - image width as it appears in the file (output)
*  @param height - image height as it appears in the file (output)
*  @param channels - the number of the image channels as it appears in the file (output)
*  @return image in uint
*/
unsigned char* loadImageFromFileUInt(const char* fileName, int* width, int* height, int* channels);

/** @brief Frees the image loaded by loadImageFromFile
*  @param image - image loaded by loadImageFromFile
*  @return void
*/
void freeImage(float* image);

/** @brief Substracts mean image and then scale the image
*  @param image - image in float
*  @param width - image width
*  @param height - image heigh
*  @param channels - the number of the image channels
*  @param meanValues - mean values to be subtracted from the image
*  @param scale - scale factor
*  @return void
*/
void subtractMeanImageAndScale(float* image, int width, int height, int channels, const float* meanValues, float scale);

/** @brief Substracts mean image
*  @param image - image in float
*  @param width - image width
*  @param height - image heigh
*  @param channels - the number of the image channels
*  @param meanValues - mean values to be subtracted from the image
*  @return void
*/
void subtractMeanImage(float* image, int width, int height, int channels, const float* meanValues);

/** @brief scale image
*  @param image - image in float
*  @param width - image width
*  @param height - image heigh
*  @param channels - the number of the image channels
*  @param scale - scale factor
*  @return void
*/
void scaleImage(float* image, int width, int height, int channels,  float scale);

/** @brief loads image into MDData/Tensor
*  @param mddata - MDData/Tensor where to load the image
*  @param image - image to be loaded into the MDData/Tensor
*  @param width - image width in elements/pixels
*  @param height - image height in elements/pixels
*  @param channels - image channels number
*  @return vx_status - VX_SUCCESS in case of success; other value in case of failure
*/
vx_status imageToMDData(vx_tensor mddata, const float* image, int width, int height, int channels);

/** @brief loads classification text file. The file is a text file and has one class in each line
*  @param fileFullPath - Full path to the classification file
*  @param classesNum - Number of classes (lines) in the file (output).
*  @return array of classes. class number is the index in the array
*/
char** loadClassificationFile(const char* fileFullPath, size_t* classesNum);

/** @brief Deletes the classification data loaded by loadClassificationFile
*  @param classArray - classification data to be deleted
*  @return void
*/
void deleteClassificationContect(char** classArray);

/** @brief Reads the probabilities from MDData/Tensor and store it in a matrix
*  @param mddata - MDData/Tensor from where to read the probabilities
*  @param classesNum - Number of available classes
*  @param probSum - Sum of all the probabilities (output)
*  @return matrix of probabilities 
*/
float** getProbabilitiesFromMDData(vx_tensor mddata, size_t classesNum, float* probSum);

/** @brief Deletes the probabilities matrix created by getProbabilitiesFromMDData
*  @param prob - The probabilities matrix
*  @return void
*/
void deleteProbStructure(float** prob);

/** @brief Moves the highest sortNum probabilities to the beginning of the matrix
*  @param prob - The probabilities matrix
*  @param classesNum - The number of available classes
*  @param sortNum - The number of the highest probabilities to be moved to the beginning
*  @return void
*/
void moveHighestProbToTheBegin(float** prob, size_t classesNum, size_t sortNum);

