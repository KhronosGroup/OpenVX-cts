#include "common.h"
#include "utilities.h"
#include "precisionConverter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define VX_MAX_TENSOR_DIMS_CT 6

//Local function that returns a pointer to a function that converts image data to a float
float(*convertToFloatFunc(vx_enum df))(const char*)
{
    if (df == VX_TYPE_INT16)
        return Q78ToFloat;
#if defined(EXPERIMENTAL_PLATFORM_SUPPORTS_16_FLOAT)
    if (df == VX_TYPE_FLOAT16)
        return FP16ToFloat;
#endif
    if (df == VX_TYPE_FLOAT32)
		return FP32ToFloat;
}

//Local function that returns a pointer to a function that converts float to the image format
void(*convertFromFloatFunc(vx_enum df))(float, char*)
{
    if (df == VX_TYPE_INT16)
        return floatToQ78;
#if defined(EXPERIMENTAL_PLATFORM_SUPPORTS_16_FLOAT)
    if (df == VX_TYPE_FLOAT16)
        return floatToFP16;
#endif
    if (df == VX_TYPE_FLOAT32)
        return floatToFP32;
}

/** @brief Loads image from a file and converts it to float.
* supported formats: jpeg, png, bmp, psd, tga, gif, hdr, pic, ppm, pgm
*  @param fileName - full path to the image file
*  @param width - image width as it appears in the file (output)
*  @param height - image height as it appears in the file (output)
*  @param channels - the number of the image channels as it appears in the file (output)
*  @return image in float
*/
float* loadImageFromFile(const char* fileName, int* width, int* height, int* channels)
{
    WriteLog("Loading image from '%s'...\n", fileName);
    unsigned char* image = stbi_load(fileName, width, height, channels, 0);
    if (!image)
    {
        WriteLog("Failed to load image from file %s", fileName);
        return NULL;
    }
    
    size_t imageSize = (*width) * (*height) * (*channels);
    float* imageF = (float*) malloc((*width) * (*height) * (*channels) * sizeof(float));
    for (size_t i = 0; i < imageSize; ++i)
    {
        imageF[i] = (float)image[i];
    }
    stbi_image_free(image);
    return imageF;
}

/** @brief Loads image from a file.
* supported formats: jpeg, png, bmp, psd, tga, gif, hdr, pic, ppm, pgm
*  @param fileName - full path to the image file
*  @param width - image width as it appears in the file (output)
*  @param height - image height as it appears in the file (output)
*  @param channels - the number of the image channels as it appears in the file (output)
*  @return image in uint
*/
unsigned char* loadImageFromFileUInt(const char* fileName, int* width, int* height, int* channels)
{
    WriteLog("Loading image from '%s'...\n", fileName);
    unsigned char* image = stbi_load(fileName, width, height, channels, 0);
    if (!image)
    {
        WriteLog("Failed to load image from file %s", fileName);
        return NULL;
    }

    return image;
}

/** @brief Frees the image loaded by loadImageFromFile
*  @param image - image loaded by loadImageFromFile
*  @return void
*/
void freeImage(float* image)
{
    if (image != NULL) free(image);
}

/** @brief Substracts mean image and then scale the image
*  @param image - image in float
*  @param width - image width
*  @param height - image heigh
*  @param channels - the number of the image channels
*  @param meanValues - mean values to be subtracted from the image
*  @param scale - scale factor
*  @return void
*/
void subtractMeanImageAndScale(float* image, int width, int height, int channels, const float* meanValues, float scale)
{
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int c = 0; c < channels; ++c)
            {
                int offset = h * width * channels + w * channels + c;
                image[offset] = (image[offset] - meanValues[c]) * scale;
            }
        }
    }
}

/** @brief Substracts mean image
*  @param image - image in float
*  @param width - image width
*  @param height - image heigh
*  @param channels - the number of the image channels
*  @param meanValues - mean values to be subtracted from the image
*  @return void
*/
void subtractMeanImage(float* image, int width, int height, int channels, const float* meanValues)
{
    subtractMeanImageAndScale(image, width, height, channels, meanValues, 1.0);
}

/** @brief scale image
*  @param image - image in float
*  @param width - image width
*  @param height - image heigh
*  @param channels - the number of the image channels
*  @param scale - scale factor
*  @return void
*/
void scaleImage(float* image, int width, int height, int channels,  float scale)
{
    float meanValues[10] = { 0.0 };
    subtractMeanImageAndScale(image, width, height, channels, meanValues, scale);
}

/** @brief loads image into MDData/Tensor
*  @param mddata - MDData/Tensor where to load the image
*  @param image - image to be loaded into the MDData/Tensor
*  @param width - image width in elements/pixels
*  @param height - image height in elements/pixels
*  @param channels - image channels number
*  @return vx_status - VX_SUCCESS in case of success; other value in case of failure
*/
vx_status imageToMDData(vx_tensor mddata, const float* image, int width, int height, int channels)
{
    if (channels != 3 && channels != 1)
	{
	    WriteLog("Trying to load image with %d channels. Currently only images with 1 or 3 channels are supported.\n", channels);
		return VX_FAILURE;
	}
	
	vx_size dims_num = 0;
	vx_size dimensionsArray[VX_MAX_TENSOR_DIMS_CT] = { 0 };
    vx_status status = vxQueryTensor(mddata, VX_TENSOR_NUMBER_OF_DIMS, &dims_num, sizeof(dims_num));
    status |= vxQueryTensor(mddata, VX_TENSOR_DIMS, dimensionsArray, sizeof(dimensionsArray));
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot query MDData!\n");
        return status;
    }

    if (dims_num < 3)
    {
        WriteLog("MDData has less than 3 dimensions. It cannot store an image\n");
        return VX_FAILURE;
    }
	
	if (width != dimensionsArray[1] || height != dimensionsArray[0] || channels != dimensionsArray[2])
	{
	    WriteLog("Image size %dx%dx%d does not suit MDData size %dx%dx%d\n", width, height, channels, dimensionsArray[1], dimensionsArray[0], dimensionsArray[1]);
		return VX_FAILURE;
	}

    vx_enum dt;
    status = vxQueryTensor(mddata, VX_TENSOR_DATA_TYPE, &dt, sizeof(dt));
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot query MDData format!\n");
        return status;
    }

    void* mddataBasePtr = NULL;
	const vx_size viewStart[VX_MAX_TENSOR_DIMS_CT] = { 0 };
    mddataBasePtr = malloc(width*height*3*sizeof(vx_int16));
    if (!mddataBasePtr) { WriteLog("ERROR: malloc failed..."); return VX_FAILURE; }
    
	size_t channelsOrderFix = channels == 1 ? 0 : 2;
    void(*convertFromFloat)(float, char*) = convertFromFloatFunc(dt);
    for (size_t h = 0; h < dimensionsArray[0]; ++h)
    {
        for (size_t w = 0; w < dimensionsArray[1]; ++w)
        {

            for (size_t d = 0; d < dimensionsArray[2]; ++d)
            {
                size_t imageOffset = h * dimensionsArray[1] * dimensionsArray[2] + w * dimensionsArray[2] + (channelsOrderFix - d);

                size_t mddataOffset = sizeof(vx_int16) * (w + width * (h + height * d));
                convertFromFloat(image[imageOffset], (char*)mddataBasePtr + mddataOffset);
            }
        }
    }


    vx_size strides[] = { sizeof(vx_int16), sizeof(vx_int16) * width, sizeof(vx_int16) * width * height };
    status = vxCopyTensorPatch(mddata, 3, viewStart, dimensionsArray, strides, mddataBasePtr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot commit MDData patch!\n");
        return status;
    }

    return VX_SUCCESS;
}

/** @brief loads classification text file. The file is a text file and has one class in each line
*  @param fileFullPath - Full path to the classification file
*  @param classesNum - Number of classes (lines) in the file (output).
*  @return array of classes. class number is the index in the array
*/
char** loadClassificationFile(const char* fileFullPath, size_t* classesNum)
{
    FILE* fp = fopen(fileFullPath, "r");
    if (fp == NULL)
    {
        WriteLog("ERROR: cannot open classification file %s\n", fileFullPath);
        return NULL;
    }
    
    //Get file size
    fseek(fp, 0, SEEK_END); // seek to end of file
    size_t fileSize = ftell(fp); // get current file pointer
    fseek(fp, 0, SEEK_SET); // seek back to beginning of file
    
    char* fileContent = (char*)malloc(fileSize + 1);
    if (fread(fileContent, 1, fileSize, fp) != fileSize)
    {
        WriteLog("Failed to read all data from file %s\n", fileFullPath);
        fclose(fp);
        free(fileContent);
        return NULL;
    }
    fclose(fp);
    
    size_t classCount = 0;
    for (size_t i = 0; i < fileSize; ++i)
    {
        if (fileContent[i] == '\n')
        {
            fileContent[i] = '\0';
            classCount++;
        }
    }
    if (fileContent[fileSize - 1] != '\0')
    {
        fileContent[fileSize] = '\0';
        classCount++;
    }
    
    char** classArray = (char**)malloc(classCount * sizeof(char*));
    size_t classIndex = 0;
    for (size_t i = 0; i < fileSize + 1; ++i)
    {
        if (classIndex < classCount) classArray[classIndex++] = fileContent + i;
        while (i < fileSize && fileContent[i] != '\0') ++i;
    }
    *classesNum = classCount;
    return classArray;
}

/** @brief Deletes the classification data loaded by loadClassificationFile
*  @param classArray - classification data to be deleted
*  @return void
*/
void deleteClassificationContect(char** classArray)
{
    if (classArray == NULL) return;
    if (*classArray != NULL) free(*classArray);
    free(classArray);
}


/** @brief Deletes the probabilities matrix created by getProbabilitiesFromMDData
*  @param prob - The probabilities matrix
*  @return void
*/
void deleteProbStructure(float** prob)
{
    if (prob == NULL) return;
    if (*prob != NULL) free(*prob);
    free(prob);
}

/** @brief Moves the highest sortNum probabilities to the beginning of the matrix
*  @param prob - The probabilities matrix
*  @param classesNum - The number of available classes
*  @param sortNum - The number of the highest probabilities to be moved to the beginning
*  @return void
*/
void moveHighestProbToTheBegin(float** prob, size_t classesNum, size_t sortNum)
{
    if (prob == NULL)
	{
	    WriteLog("Error in moveHighestProbToTheBegin, received NULL pointer\n");
		return;
	}
	
    for (size_t sortIndex = 0; sortIndex < sortNum; ++sortIndex)
    {
        size_t highProbIndex = sortIndex;
        for (size_t probIndex = sortIndex; probIndex < classesNum; ++probIndex)
        {
            if (prob[probIndex][1] > prob[highProbIndex][1]) highProbIndex = probIndex;
        }
        if (highProbIndex != sortIndex)
        {
            float tempClass = prob[sortIndex][0];
            float tempProb = prob[sortIndex][1];
            prob[sortIndex][0] = prob[highProbIndex][0];
            prob[sortIndex][1] = prob[highProbIndex][1];
            prob[highProbIndex][0] = tempClass;
            prob[highProbIndex][1] = tempProb;
        }
    }
}

