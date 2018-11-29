/*
* Copyright (c) 2012-2017 The Khronos Group Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "test_engine/test.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <string.h>

#ifndef _MSC_VER
#define min(a,b) (a<b?a:b)
#endif

TESTCASE(HogCells, CT_VXContext, ct_setup_vx_context, 0)

TEST(HogCells, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0;
    vx_uint32 src_width = 640;
    vx_uint32 src_height = 320;
    vx_int32 cell_width = 8;
    vx_int32 cell_height = 8;
    vx_int32 num_bins = 9;
    const vx_size mag_dims[2] = { 80, 40 };
    const vx_size bins_dims[3] = { 80, 40, 9 };
    vx_tensor magnitudes;
    vx_tensor bins;

    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, src_width, src_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT8, 0), VX_TYPE_TENSOR);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxHOGCellsNode(graph, input, cell_width, cell_height, num_bins, magnitudes, bins), VX_TYPE_NODE);
    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&input));
    VX_CALL(vxReleaseTensor(&magnitudes));
    VX_CALL(vxReleaseTensor(&bins));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(input == 0);
}

static CT_Image hog_read_image(const char *fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}

static vx_status hogcells_ref(CT_Image img, vx_int32 cell_width, vx_int32 cell_height, vx_int32 bins_num, vx_tensor magnitudes, vx_tensor bins)
{
    vx_status status = 0;
    vx_int32 width, height;
    void* p_ct_base = ct_image_get_plane_base(img, 0);
    vx_float32 gx;
    vx_float32 gy;
    vx_float32 orientation;
    vx_float32 magnitude;
    vx_int8 bin;
    
    width = img->width;
    height = img->height;
    vx_int16* mag_ref = (vx_int16 *)malloc(height / cell_height * width / cell_width * sizeof(vx_int16));
    vx_int8* bins_ref = (vx_int8 *)malloc(height / cell_height * width / cell_width * bins_num );
    vx_int16* mag = (vx_int16 *)malloc(height / cell_height * width / cell_width *sizeof(vx_int16));
    vx_int8* bins_p = (vx_int8 *)malloc(height / cell_height * width / cell_width * bins_num);
    memset(mag_ref, 0, height / cell_height * width / cell_width * sizeof(vx_int16));
    memset(bins_ref, 0, height / cell_height * width / cell_width * bins_num);
    float num_div_360 = (float)bins_num / 360.0f;

    vx_size magnitudes_dim_num = 2, magnitudes_dims[6] = { width/cell_width, height/cell_height,0 }, magnitudes_strides[6] = { 2, 2 * width / cell_width };
    vx_size bins_dim_num = 3, bins_dims[6] = { width / cell_width, height / cell_height, bins_num }, bins_strides[6] = { 1,  width / cell_width, height / cell_height * width / cell_width };
    const size_t view_start[6] = { 0 };
    vxCopyTensorPatch(magnitudes, magnitudes_dim_num, view_start, magnitudes_dims, magnitudes_strides, mag, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyTensorPatch(bins, bins_dim_num, view_start, bins_dims, bins_strides, bins_p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    vx_int32 num_cellw = (vx_int32)floor(((vx_float64)width) / ((vx_float64)cell_width));
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int x1 = i - 1 < 0 ? 0 : i - 1;
            int x2 = i + 1 >= width ? width - 1 : i + 1;
            vx_uint8 *gx1 = (vx_uint8*)((vx_uint8*)p_ct_base + j * img->stride + x1);
            vx_uint8 *gx2 = (vx_uint8*)((vx_uint8*)p_ct_base + j * img->stride + x2);
            gx = *gx2 - *gx1;

            int y1 = j - 1 < 0 ? 0 : j - 1;
            int y2 = j + 1 >= height ? height - 1 : j + 1;
            vx_uint8 *gy1 = (vx_uint8*)((vx_uint8*)p_ct_base + y1 * img->stride + i);
            vx_uint8 *gy2 = (vx_uint8*)((vx_uint8*)p_ct_base + y2 * img->stride + i);
            gy = *gy2 - *gy1;

            magnitude = sqrtf(powf(gx, 2) + powf(gy, 2));
            orientation = fmod(atan2f(gy, gx + 0.00000000000001)
                * (180 / 3.14159265), 360);
            if (orientation < 0) {
                orientation += 360;
            }

            bin = (vx_int8)floor(orientation * num_div_360);

            vx_int32 cellx = i / cell_width;
            vx_int32 celly = j / cell_height;
            vx_int32 magnitudes_index = celly * num_cellw + cellx;
            vx_int32 bins_index = (celly * num_cellw + cellx) * bins_num + bin;
            *(mag_ref + magnitudes_index) += magnitude / (cell_width * cell_height);
            *(bins_ref + bins_index) += magnitude / (cell_width * cell_height);
        }
    }
    for (int i = 0; i < height / cell_height * width / cell_width; i++)
    {
        vx_float32 mag_ref_data = *(mag_ref + i);
        vx_float32 mag_data = *(mag + i);
        if (mag_ref_data / mag_data < 0.95 || mag_ref_data / mag_data > 1.05)
        {
            status = VX_FAILURE;
            break;
        }
    }
    if (status == VX_SUCCESS)
    {
        for (int i = 0; i < height / cell_height * width / cell_width * bins_num; i++)
        {
            vx_float32 bins_ref_data = *(bins_ref + i);
            vx_float32 bins_data = *(bins_p + i);
            if (bins_ref_data / bins_data < 0.95 || bins_ref_data / bins_data > 1.05)
            {
                status = VX_FAILURE;
                break;
            }
        }
    }
    
    free(mag_ref);
    free(mag);
    free(bins_ref);
    free(bins_p);
    return status;
}


typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height);
    const char* fileName;
    vx_int32 cell_width;
    vx_int32 cell_height;
    vx_int32 bins_num;
    const char* result_filename;
} Arg;

#define PARAMETERS \
    ARG("case_cells8x8_9_Hogcells", hog_read_image, "lena_gray.bmp", 8, 8, 9, "hogcells_8x8_9.txt"), \
    ARG("case_cells8x8_9_Hogcells", hog_read_image, "lena_gray.bmp", 8, 8, 6, "hogcells_8x8_6.txt"), \
    ARG("case_cells8x8_9_Hogcells", hog_read_image, "lena_gray.bmp", 8, 8, 3, "hogcells_8x8_3.txt"), \
    ARG("case_cells8x8_9_Hogcells", hog_read_image, "lena_gray.bmp", 4, 4, 9, "hogcells_8x8_9.txt"), \
    ARG("case_cells8x8_9_Hogcells", hog_read_image, "lena_gray.bmp", 4, 4, 6, "hogcells_8x8_6.txt"), \

TEST_WITH_ARG(HogCells, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_int32 cell_width = arg_->cell_width;
    vx_int32 cell_height = arg_->cell_height;
    vx_int32 bins_num = arg_->bins_num;
    CT_Image src = NULL;
    vx_status status;

    vx_uint32 src_width;
    vx_uint32 src_height;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0));
    src_width = src->width;
    src_height = src->height;

    const vx_size mag_dims[2] = { src_width / cell_width, src_height / cell_height };
    const vx_size bins_dims[3] = { src_width / cell_width, src_height / cell_height, bins_num };
    vx_tensor magnitudes;
    vx_tensor bins;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT8, 0), VX_TYPE_TENSOR);
    
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxHOGCellsNode(graph, src_image, cell_width, cell_height, bins_num, magnitudes, bins), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(status = hogcells_ref(src, cell_width, cell_height, bins_num, magnitudes, bins));
    
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&src_image));
    VX_CALL(vxReleaseTensor(&magnitudes));
    VX_CALL(vxReleaseTensor(&bins));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(src_image == 0);
}

TEST_WITH_ARG(HogCells, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0;

    vx_int32 cell_width = arg_->cell_width;
    vx_int32 cell_height = arg_->cell_height;
    vx_int32 bins_num = arg_->bins_num;
    CT_Image src = NULL;
    vx_status status;

    vx_uint32 src_width;
    vx_uint32 src_height;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0));
    src_width = src->width;
    src_height = src->height;

    const vx_size mag_dims[2] = { src_width / cell_width, src_height / cell_height };
    const vx_size bins_dims[3] = { src_width / cell_width, src_height / cell_height, bins_num };
    vx_tensor magnitudes;
    vx_tensor bins;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT8, 0), VX_TYPE_TENSOR);

    VX_CALL(vxuHOGCells(context, src_image, cell_width, cell_width, bins_num, magnitudes, bins));
    ASSERT_NO_FAILURE(status = hogcells_ref(src, cell_width, cell_height, bins_num, magnitudes, bins));

    VX_CALL(vxReleaseImage(&src_image));
    VX_CALL(vxReleaseTensor(&magnitudes));
    VX_CALL(vxReleaseTensor(&bins));

    ASSERT(src_image == 0);
}
TESTCASE_TESTS(HogCells,
               testNodeCreation,
               testGraphProcessing,
               testImmediateProcessing)

TESTCASE(HogFeatures, CT_VXContext, ct_setup_vx_context, 0)

TEST(HogFeatures, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0;
    vx_uint32 src_width = 640;
    vx_uint32 src_height = 320;
    vx_int32 cell_width = 8;
    vx_int32 cell_height = 8;
    vx_int32 num_bins = 9;
    const vx_size mag_dims[2] = { 80, 40 };
    const vx_size bins_dims[3] = { 80, 40, 9 };
    vx_hog_t params;
    vx_tensor magnitudes;
    vx_tensor bins;
    vx_tensor features;

    vx_graph graph = 0;
    vx_node cell_node = 0;
    vx_node feature_node = 0;
    params.window_width = 64;
    params.window_height = 32;
    params.block_width = 16;
    params.block_height = 16;
    params.cell_width = 8;
    params.cell_height = 8;
    params.num_bins = 9;
    ASSERT_VX_OBJECT(input = vxCreateImage(context, src_width, src_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT8, 0), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(cell_node = vxHOGCellsNode(graph, input, cell_width, cell_height, num_bins, magnitudes, bins), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(features = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);

    ASSERT_VX_OBJECT(feature_node = vxHOGFeaturesNode(graph, input, magnitudes, bins, &params, 1, features), VX_TYPE_NODE);
    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&cell_node));
    VX_CALL(vxReleaseNode(&feature_node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&input));
    VX_CALL(vxReleaseTensor(&magnitudes));
    VX_CALL(vxReleaseTensor(&bins));
    VX_CALL(vxReleaseTensor(&features));

    ASSERT(cell_node == 0);
    ASSERT(feature_node == 0);
    ASSERT(graph == 0);
    ASSERT(input == 0);
}

static vx_status hogfeatures_ref(CT_Image img, vx_hog_t params, vx_tensor features)
{
    vx_status status = 0;
    vx_int32 width, height;
    void* p_ct_base = ct_image_get_plane_base(img, 0);
    vx_float32 gx;
    vx_float32 gy;
    vx_float32 orientation;
    vx_float32 magnitude;
    vx_int8 bin;

    width = img->width;
    height = img->height;

    vx_int32 cell_height = params.cell_height;
    vx_int32 cell_width = params.cell_width;
    vx_int32 bins_num = params.num_bins;

    vx_int32 num_windowsW = width / params.window_width;
    vx_int32 num_windowsH = height / params.window_height;
    vx_int32 num_blockW = width / params.cell_width - 1;
    vx_int32 num_blockH = height / params.cell_height - 1;
    vx_int32 num_block = num_blockW * num_blockH;
    vx_int32 n_cellsx = width / cell_width;
    vx_int32 cells_per_block_w = params.block_width / cell_width;
    vx_int32 cells_per_block_h = params.block_height / cell_height;
   
    vx_int16* mag_ref = (vx_int16 *)malloc(height / cell_height * width / cell_width * sizeof(vx_int16));
    vx_int8* bins_ref = (vx_int8 *)malloc(height / cell_height * width / cell_width * bins_num );
    vx_int16* features_ref = (vx_int16 *)malloc(num_windowsW * num_windowsH * params.window_width / params.block_stride * 
                                                params.window_height / params.block_stride *bins_num * sizeof(vx_int16));
    vx_int16* features_p = (vx_int16 *)malloc(num_windowsW * num_windowsH * params.window_width / params.block_stride *
                                              params.window_height / params.block_stride *bins_num * sizeof(vx_int16));
    memset(mag_ref, 0, height / cell_height * width / cell_width * sizeof(vx_int16));
    memset(bins_ref, 0, height / cell_height * width / cell_width * bins_num);
    memset(features_ref, 0, num_windowsW * num_windowsH * params.window_width / params.block_stride *
        params.window_height / params.block_stride *bins_num * sizeof(vx_int16));

    float num_div_360 = (float)bins_num / 360.0f;

    vx_size features_dim_num = 3, features_dims[6] = { num_windowsW, num_windowsH, params.window_width / params.block_stride *
        params.window_height / params.block_stride *bins_num }, features_strides[6] = { 2, 2 * num_windowsW, 2 * num_windowsW * num_windowsH};
    const size_t view_start[6] = { 0 };
    vxCopyTensorPatch(features, features_dim_num, view_start, features_dims, features_strides, features_p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    vx_int32 num_cellw = (vx_int32)floor(((vx_float64)width) / ((vx_float64)cell_width));
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int x1 = i - 1 < 0 ? 0 : i - 1;
            int x2 = i + 1 >= width ? width - 1 : i + 1;
            vx_uint8 *gx1 = (vx_uint8*)((vx_uint8*)p_ct_base + j * img->stride + x1);
            vx_uint8 *gx2 = (vx_uint8*)((vx_uint8*)p_ct_base + j * img->stride + x2);
            gx = *gx2 - *gx1;

            int y1 = j - 1 < 0 ? 0 : j - 1;
            int y2 = j + 1 >= height ? height - 1 : j + 1;
            vx_uint8 *gy1 = (vx_uint8*)((vx_uint8*)p_ct_base + y1 * img->stride + i);
            vx_uint8 *gy2 = (vx_uint8*)((vx_uint8*)p_ct_base + y2 * img->stride + i);
            gy = *gy2 - *gy1;

            magnitude = sqrtf(powf(gx, 2) + powf(gy, 2));
            orientation = fmod(atan2f(gy, gx + 0.00000000000001)
                * (180 / 3.14159265), 360);
            if (orientation < 0) {
                orientation += 360;
            }

            bin = (vx_int8)floor(orientation * num_div_360);

            vx_int32 cellx = i / cell_width;
            vx_int32 celly = j / cell_height;
            vx_int32 magnitudes_index = celly * num_cellw + cellx;
            vx_int32 bins_index = (celly * num_cellw + cellx) * bins_num + bin;
            *(mag_ref + magnitudes_index) += magnitude / (cell_width * cell_height);
            *(bins_ref + bins_index) += magnitude / (cell_width * cell_height);
        }
    }
    for (vx_int32 blkH = 0; blkH < num_blockH; blkH++)
    {
        for (vx_int32 blkW = 0; blkW < num_blockW; blkW++)
        {
            vx_float32 sum = 0;
            for (vx_int32 y = 0; y < cells_per_block_h; y++)
            {
                for (vx_int32 x = 0; x < cells_per_block_w; x++)
                {
                    vx_int32 index = (blkH + y)*n_cellsx + (blkW + x);
                    sum += (*(mag_ref + index)) * (*(mag_ref + index));
                }
            }
            sum = sqrtf(sum + 0.00000000000001);
            for (vx_int32 y = 0; y < cells_per_block_h; y++)
            {
                for (vx_int32 x = 0; x < cells_per_block_w; x++)
                {
                    for (vx_int32 k = 0; k < bins_num; k++)
                    {
                        vx_int32 bins_index = (blkH + y)*n_cellsx * bins_num + (blkW + x)*bins_num + k;
                        vx_int32 block_index = blkH * num_blockW * bins_num + blkW * bins_num + k;
                        float hist = min((*(bins_ref + bins_index) / sum), params.threshold);
                        vx_int16 *features_ptr = features_ref + block_index;
                        *features_ptr = *features_ptr + hist;
                    }
                }
            }
        }
    }
    for (int i = 0; i < num_windowsW * num_windowsH * params.window_width / params.block_stride *
        params.window_height / params.block_stride *bins_num; i++)
    {
        vx_float32 features_ref_data = *(features_ref + i);
        vx_float32 features_data = *(features_p + i);
        if (features_ref_data / features_data < 0.95 || features_ref_data / features_data > 1.05)
        {
            status = VX_FAILURE;
            break;
        }
    }
    
    free(mag_ref);
    free(bins_ref);
    free(features_ref);
    free(features_p);
    return status;
}


typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height);
    const char* fileName;
    vx_hog_t hog_params;
} Arg_features;

#define PARAMETERS_FEATURES \
    ARG("case_hogfeature", hog_read_image, "lena_gray.bmp", {8, 8, 16, 16, 8, 9, 32, 32, 32, 0.2}), \
    ARG("case_hogfeature", hog_read_image, "lena_gray.bmp", {4, 4, 8, 8, 4, 9, 32, 32, 32, 0.2}), \
    ARG("case_hogfeature", hog_read_image, "lena_gray.bmp", {8, 8, 16, 16, 8, 6, 32, 32, 32, 0.2}), \
    ARG("case_hogfeature", hog_read_image, "lena_gray.bmp", {4, 4, 8, 8, 4, 6, 32, 32, 32, 0.2}), \
    ARG("case_hogfeature", hog_read_image, "lena_gray.bmp", {8, 8, 16, 16, 8, 6, 32, 32, 32, 0.1}), \

TEST_WITH_ARG(HogFeatures, testGraphProcessing, Arg_features,
    PARAMETERS_FEATURES

)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0;
    vx_graph graph = 0;
    vx_node cell_node = 0;
    vx_node feature_node = 0;
    vx_int32 cell_width = arg_->hog_params.cell_width;
    vx_int32 cell_height = arg_->hog_params.cell_height;
    vx_int32 bins_num = arg_->hog_params.num_bins;
    CT_Image src = NULL;
    vx_status status;

    vx_uint32 src_width;
    vx_uint32 src_height;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0));
    src_width = src->width;
    src_height = src->height;

    const vx_size mag_dims[2] = { src_width / cell_width, src_height / cell_height };
    const vx_size bins_dims[3] = { src_width / cell_width, src_height / cell_height, bins_num };
    const vx_size features_dims[3] = { src_width / arg_->hog_params.window_stride,  src_height / arg_->hog_params.window_stride, 
                                       arg_->hog_params.window_width / arg_->hog_params.block_stride * arg_->hog_params.window_height / arg_->hog_params.block_stride *bins_num };
    vx_tensor magnitudes;
    vx_tensor bins;
    vx_tensor features;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT8, 0), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(features = vxCreateTensor(context, 3, features_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(cell_node = vxHOGCellsNode(graph, src_image, cell_width, cell_height, bins_num, magnitudes, bins), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(feature_node = vxHOGFeaturesNode(graph, src_image, magnitudes, bins, &arg_->hog_params, 1, features), VX_TYPE_NODE);
    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(status = hogfeatures_ref(src, arg_->hog_params, features));
    
    VX_CALL(vxReleaseNode(&cell_node));
    VX_CALL(vxReleaseNode(&feature_node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&src_image));
    VX_CALL(vxReleaseTensor(&magnitudes));
    VX_CALL(vxReleaseTensor(&bins));
    VX_CALL(vxReleaseTensor(&features));

    ASSERT(cell_node == 0);
    ASSERT(feature_node == 0);
    ASSERT(graph == 0);
    ASSERT(src_image == 0);
}

TEST_WITH_ARG(HogFeatures, testImmediateProcessing, Arg_features,
    PARAMETERS_FEATURES
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0;
    vx_graph graph = 0;
    vx_node cell_node = 0;
    vx_node feature_node = 0;
    vx_int32 cell_width = arg_->hog_params.cell_width;
    vx_int32 cell_height = arg_->hog_params.cell_height;
    vx_int32 bins_num = arg_->hog_params.num_bins;
    CT_Image src = NULL;
    vx_status status;

    vx_uint32 src_width;
    vx_uint32 src_height;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0));
    src_width = src->width;
    src_height = src->height;

    const vx_size mag_dims[2] = { src_width / cell_width, src_height / cell_height };
    const vx_size bins_dims[3] = { src_width / cell_width, src_height / cell_height, bins_num };
    const vx_size features_dims[3] = { src_width / arg_->hog_params.window_stride,  src_height / arg_->hog_params.window_stride,
        arg_->hog_params.window_width / arg_->hog_params.block_stride * arg_->hog_params.window_height / arg_->hog_params.block_stride *bins_num };
    vx_tensor magnitudes;
    vx_tensor bins;
    vx_tensor features;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT8, 0), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(features = vxCreateTensor(context, 3, features_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);

    VX_CALL(vxuHOGCells(context, src_image, cell_width, cell_width, bins_num, magnitudes, bins));
    VX_CALL(vxuHOGFeatures(context, src_image, magnitudes, bins, &arg_->hog_params, 1, features));
    ASSERT_NO_FAILURE(status = hogfeatures_ref(src, arg_->hog_params, features));

    VX_CALL(vxReleaseImage(&src_image));
    VX_CALL(vxReleaseTensor(&magnitudes));
    VX_CALL(vxReleaseTensor(&bins));
    VX_CALL(vxReleaseTensor(&features));

    ASSERT(src_image == 0);
}
TESTCASE_TESTS(HogFeatures,
               testNodeCreation,
               testGraphProcessing,
               testImmediateProcessing)
