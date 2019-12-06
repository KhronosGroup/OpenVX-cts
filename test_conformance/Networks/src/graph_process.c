/** @file graph_process.c
 *  @brief
 *  This file contains the implementation of the graph inputs/outputs processing functions
 */

#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_khr_nn.h>
#include "common.h"
#include "graph_process.h"
#include "utilities.h"
#include "./precisionConverter.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#include <direct.h>
#define mkdir(dir, flags) _mkdir(dir)
#else
#include <sys/stat.h>
#endif

#define VX_MAX_TENSOR_DIMENSIONS    6

float* ResizeImage(vx_size dims[VX_MAX_TENSOR_DIMENSIONS], int chans, int width,
        int height, vx_tensor input, unsigned char* image) {
    float* resized_image = (float*) malloc(
            dims[0] * dims[1] * chans * sizeof(float));
    unsigned char* resized_image_uint = (unsigned char*) malloc(
            dims[0] * dims[1] * chans);
    vx_rectangle_t rect = { 0, 0, 0, 0 };
    rect.end_x = width;
    rect.end_y = height;
    vx_rectangle_t rect_resized = { 0, 0, 0, 0 };
    rect_resized.end_x = dims[0];
    rect_resized.end_y = dims[1];
    vx_context context = vxGetContext((vx_reference) input);
    vx_image images[3];
    vx_image resized_images[3];
    vx_imagepatch_addressing_t addr = { 0 };
    vx_imagepatch_addressing_t addr_resized = { 0 };
    addr.dim_x = width;
    addr.dim_y = height;
    addr.stride_x = 3;
    addr.stride_y = 3 * width;
    addr.step_x = 1;
    addr.step_y = 1;
    addr_resized.dim_x = dims[0];
    addr_resized.dim_y = dims[1];
    addr_resized.stride_x = 3;
    addr_resized.stride_y = 3 * dims[0];
    addr_resized.step_x = 1;
    addr_resized.step_y = 1;
    for (int i = 0; i < 3; i++) {
        images[i] = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        resized_images[i] = vxCreateImage(context, dims[0], dims[1],
                VX_DF_IMAGE_U8);
        vxCopyImagePatch(images[i], &rect, 0, &addr, image + i, VX_WRITE_ONLY,
                VX_MEMORY_TYPE_HOST);
        vxuScaleImage(context, images[i], resized_images[i],
                VX_INTERPOLATION_BILINEAR);
        vxReleaseImage(&images[i]);
        vxCopyImagePatch(resized_images[i], &rect_resized, 0, &addr_resized,
                resized_image_uint + i, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        vxReleaseImage(&resized_images[i]);
    }
    size_t imageSize = dims[0] * dims[1] * chans;
    for (size_t i = 0; i < imageSize; ++i) {
        resized_image[i] = (float) resized_image_uint[i];
    }
    free(resized_image_uint);
    return resized_image;
}

float* CropImage(vx_size dims[VX_MAX_TENSOR_DIMENSIONS], int chans, int width,
        int height, vx_tensor input, unsigned char* image) {
    float* resized_image = (float*) malloc(
            dims[0] * dims[1] * chans * sizeof(float));
    unsigned char* resized_image_uint = (unsigned char*) malloc(
            dims[0] * dims[1] * chans);
    vx_rectangle_t rect = { 0, 0, 0, 0 };
    rect.end_x = width;
    rect.end_y = height;
    vx_rectangle_t rect_resized = { 0, 0, 0, 0 };
    rect_resized.end_x = dims[0];
    rect_resized.end_y = dims[1];
    vx_context context = vxGetContext((vx_reference) input);
    vx_image images[3];
    vx_image resized_images[3];
    vx_imagepatch_addressing_t addr = { 0 };
    vx_imagepatch_addressing_t addr_resized = { 0 };
    addr.dim_x = width;
    addr.dim_y = height;
    addr.stride_x = 3;
    addr.stride_y = 3 * width;
    addr.step_x = 1;
    addr.step_y = 1;
    addr_resized.dim_x = dims[0];
    addr_resized.dim_y = dims[1];
    addr_resized.stride_x = 3;
    addr_resized.stride_y = 3 * dims[0];
    addr_resized.step_x = 1;
    addr_resized.step_y = 1;
    vx_rectangle_t rect_croped;
    rect_croped.start_x = (width - dims[0])/2;
    rect_croped.start_y = (height - dims[1])/2;
    rect_croped.end_x = rect_croped.start_x + dims[0];
    rect_croped.end_y = rect_croped.start_y + dims[1];

    for (int i = 0; i < 3; i++) {
        images[i] = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        vxCopyImagePatch(images[i], &rect, 0, &addr, image + i, VX_WRITE_ONLY,
                VX_MEMORY_TYPE_HOST);
        resized_images[i] = vxCreateImageFromROI(images[i], &rect_croped);
        vxReleaseImage(&images[i]);
        vxCopyImagePatch(resized_images[i], &rect_resized, 0, &addr_resized,
                resized_image_uint + i, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        vxReleaseImage(&resized_images[i]);
    }
    size_t imageSize = dims[0] * dims[1] * chans;
    for (size_t i = 0; i < imageSize; ++i) {
        resized_image[i] = (float) resized_image_uint[i];
    }
    free(resized_image_uint);
    return resized_image;
}

vx_status preprocess(vx_tensor input, const char * fName)
{
    //1. Load image
    //2. Normalize the image pixels the same way you used in the training process (i.e, mean substraction, scaling, etc)
    //3. Scale each pixel with the scale factor ModelOptimizer reported
    //4. Convert each pixel to the required precision (Q78, FP16, etc)
    //5. Fill the input tensor with the pre-processed pixels

    int width, height, chans;
    vx_size dims_num = 0;

    vx_size dims[VX_MAX_TENSOR_DIMENSIONS];


    vx_status status = vxQueryTensor(input, VX_TENSOR_NUMBER_OF_DIMS, &dims_num, sizeof(dims_num));
    status |= vxQueryTensor(input, VX_TENSOR_DIMS, dims, sizeof(dims));
    unsigned char * image = loadImageFromFileUInt(fName, &width, &height, &chans);
    float* resized_image = ResizeImage(dims, chans, width, height, input,
            image);
    float scaleFactor = 1.f / 8;
    float meanValues[] = { 104.f, 117.f, 123.f };

    subtractMeanImageAndScale(resized_image, dims[0], dims[1], chans, meanValues, scaleFactor);


    status = imageToMDData(input, resized_image, dims[0], dims[1], chans);
    freeImage(resized_image);
    free (image);
    return status;


}

vx_status postprocess(vx_tensor output, /*OUT*/ int* detected_class)
{
    //1. Find top-N probabilities indices in the output tensor.
    //2. Probabilities must be converted back to floating point number in order to be interpreted as percentages

    //getProbabilitiesFromMDData(...)
    vx_int16 mem[1000] = {0};

    const vx_size view_start[2] = { 0, 0 };
    const vx_size view_end[2] = { 1000, 1 };
    const vx_size strides[2] = { sizeof(vx_int16), sizeof(vx_int16) * 1000 };
    vx_status status = vxCopyTensorPatch(output, 2, view_start, view_end, strides, &mem[0], VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != VX_SUCCESS) { WriteLog("failed to read out the results form the graph"); return status; }


    {

        float max_prob = 0;
        int res = 0;

        //TODO: is there some define for this 1000??
        for (int i = 0; i < 1000; ++i)
        {
            float prob = Q78ToFloat((char*)&mem[i]);
            if (max_prob < prob)
            {
                max_prob = prob;
                res = i;
            }
        }

        *detected_class = res;

    }

    return VX_SUCCESS;
}

static vx_status saveTensorData(vx_tensor tensor, const char * fn)
{
    if (!tensor) return VX_FAILURE;

    FILE * f = fopen(fn, "wb");

    vx_size dims_num;
    vx_status status = vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &dims_num, sizeof(dims_num));

    vx_size *dims = (vx_size*)malloc(dims_num * sizeof(vx_size));
    if (!dims)
    {
        fclose(f);
        return VX_ERROR_NO_MEMORY;
    }

    status = vxQueryTensor(tensor, VX_TENSOR_DIMS, dims, sizeof(vx_size) * dims_num);
    if (status != VX_SUCCESS)
    {
        fclose(f);
        free(dims);
        return status;
    }

    vx_size count = 1;
    for (vx_size i = 0; i < dims_num; ++i) count *= dims[i];

    vx_size strides[VX_MAX_TENSOR_DIMENSIONS] = { sizeof(vx_int16) };
    for (vx_size i = 0; i < dims_num - 1; ++i)
        strides[i+1] = strides[i] * dims[i];

    vx_int16 * buffer = malloc(count * sizeof(*buffer));

    vx_size view_start[VX_MAX_TENSOR_DIMENSIONS] = { 0 };
    status = vxCopyTensorPatch(tensor, dims_num, view_start, dims, strides, buffer, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if (status == VX_SUCCESS)
    {
        fwrite((vx_int16*)buffer, sizeof(int16_t), count, f);
    }

    fclose(f);
    free(buffer);
    free(dims);
    return status;
}

vx_status debugDumpLayers(ObjectRefContainerType * vxObjectsContainer)
{
    mkdir("output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    FILE * f = fopen("output/layers_dimensions.txt", "wt");

    for (vx_size i = 0; i < vxObjectsContainer->count; ++i)
    {
        if (vxObjectsContainer->pObjects[i].type == VX_TYPE_TENSOR)
        {
            {
                vx_size dims[VX_MAX_TENSOR_DIMENSIONS];
                for (int i = 0; i < VX_MAX_TENSOR_DIMENSIONS; ++i) dims[i] = 0;

                char fileName[256];
                sprintf(fileName, "output/%s.tensor",vxObjectsContainer->pObjects[i].uniqueRef);
                vx_status status = vxQueryTensor((vx_tensor)(vxObjectsContainer->pObjects[i].ref), VX_TENSOR_DIMS, &dims, sizeof(dims));
                if (status != VX_SUCCESS) { printf("unable to get dims from tensor??\n"); return status; }
                fprintf(f, "%s ", vxObjectsContainer->pObjects[i].uniqueRef);
                for (vx_size j = 0; j < VX_MAX_TENSOR_DIMENSIONS; ++j)
                    fprintf(f, "%zu ", dims[j]);
                fprintf(f, "\n");
                saveTensorData((vx_tensor)vxObjectsContainer->pObjects[i].ref, fileName);
            }
        }
    }

    fclose(f);

    return VX_SUCCESS;
}
