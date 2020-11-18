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

#ifdef OPENVX_USE_ENHANCED_VISION

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
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);

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

static CT_Image hog_read_image(const vx_char *fileName, vx_int32 width, vx_int32 height)
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
    vx_int16* mag_ref = (vx_int16 *)ct_alloc_mem(height / cell_height * width / cell_width * sizeof(vx_int16));
    vx_int16* bins_ref = (vx_int16 *)ct_alloc_mem(height / cell_height * width / cell_width * bins_num * sizeof(vx_int16));
    vx_int16* mag = (vx_int16 *)ct_alloc_mem(height / cell_height * width / cell_width * sizeof(vx_int16));
    vx_int16* bins_p = (vx_int16 *)ct_alloc_mem(height / cell_height * width / cell_width * bins_num * sizeof(vx_int16));
    memset(mag_ref, 0, height / cell_height * width / cell_width * sizeof(vx_int16));
    memset(bins_ref, 0, height / cell_height * width / cell_width * bins_num * sizeof(vx_int16));
    vx_float32 num_div_360 = (vx_float32)bins_num / 360.0f;

    vx_size magnitudes_dim_num = 2, magnitudes_dims[6] = { width / cell_width, height / cell_height,0 }, magnitudes_strides[6] = { 0 };
    vx_size bins_dim_num = 3, bins_dims[6] = { width / cell_width, height / cell_height, bins_num }, bins_strides[6] = { 0 };
    magnitudes_strides[0] = 2;
    bins_strides[0] = 2;

    for (vx_size i = 1; i < magnitudes_dim_num; i++)
    {
        magnitudes_strides[i] = magnitudes_dims[i - 1] * magnitudes_strides[i - 1];
    }
    for (vx_size i = 1; i < bins_dim_num; i++)
    {
        bins_strides[i] = bins_dims[i - 1] * bins_strides[i - 1];
    }

    const size_t view_start[6] = { 0 };
    vxCopyTensorPatch(magnitudes, magnitudes_dim_num, view_start, magnitudes_dims, magnitudes_strides, mag, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyTensorPatch(bins, bins_dim_num, view_start, bins_dims, bins_strides, bins_p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    vx_int32 num_cellw = (vx_int32)floor(((vx_float64)width) / ((vx_float64)cell_width));

    for (vx_int32 j = 0; j < height; j++)
    {
        for (vx_int32 i = 0; i < width; i++)
        {
            vx_int32 x1 = i - 1 < 0 ? 0 : i - 1;
            vx_int32 x2 = i + 1 >= width ? width - 1 : i + 1;
            vx_uint8 *gx1 = (vx_uint8*)((vx_uint8*)p_ct_base + j * img->stride + x1);
            vx_uint8 *gx2 = (vx_uint8*)((vx_uint8*)p_ct_base + j * img->stride + x2);
            gx = *gx2 - *gx1;

            vx_int32 y1 = j - 1 < 0 ? 0 : j - 1;
            vx_int32 y2 = j + 1 >= height ? height - 1 : j + 1;
            vx_uint8 *gy1 = (vx_uint8*)((vx_uint8*)p_ct_base + y1 * img->stride + i);
            vx_uint8 *gy2 = (vx_uint8*)((vx_uint8*)p_ct_base + y2 * img->stride + i);
            gy = *gy2 - *gy1;

            magnitude = sqrtf(powf(gx, 2) + powf(gy, 2));
            orientation = fmod(atan2f(gy, gx + 0.00000000000001)
                * (180 / 3.14159265), 360);
            if (orientation < 0)
            {
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
    for (vx_int32 i = 0; i < height / cell_height * width / cell_width; i++)
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
        for (vx_int32 i = 0; i < height / cell_height * width / cell_width * bins_num; i++)
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

    ct_free_mem(mag_ref);
    ct_free_mem(mag);
    ct_free_mem(bins_ref);
    ct_free_mem(bins_p);
    return status;
}


typedef struct 
{
    const vx_char* testName;
    CT_Image(*generator)(const vx_char* fileName, vx_int32 width, vx_int32 height);
    const vx_char* fileName;
    vx_int32 cell_width;
    vx_int32 cell_height;
    vx_int32 bins_num;
    const vx_char* result_filename;
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
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);

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
    vx_status status = VX_SUCCESS;

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
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);

    VX_CALL(vxuHOGCells(context, src_image, cell_width, cell_width, bins_num, magnitudes, bins));
    ASSERT_NO_FAILURE(status = hogcells_ref(src, cell_width, cell_height, bins_num, magnitudes, bins));
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, status);

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
    params.window_stride = 64;
    params.block_width = 16;
    params.block_height = 16;
    params.block_stride = 16;
    params.cell_width = 8;
    params.cell_height = 8;
    params.num_bins = 9;

    const vx_size features_dims[3] = { (src_width - params.window_width) / params.window_stride + 1,
                                   (src_height - params.window_height) / params.window_stride + 1,
                                   ((params.window_width - params.block_width) / params.block_stride + 1) *
                                   ((params.window_height - params.block_height) / params.block_stride + 1) *
                                   ((params.block_width * params.block_height) / (params.cell_width * params.cell_height)) * num_bins };

    ASSERT_VX_OBJECT(input = vxCreateImage(context, src_width, src_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(cell_node = vxHOGCellsNode(graph, input, cell_width, cell_height, num_bins, magnitudes, bins), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(features = vxCreateTensor(context, 3, features_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);

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
    vx_int32 block_index_count = 0;

    width = img->width;
    height = img->height;

    vx_int32 cell_height = params.cell_height;
    vx_int32 cell_width = params.cell_width;
    vx_int32 bins_num = params.num_bins;

    vx_int32 num_windowsW = width / params.window_width;
    vx_int32 num_windowsH = height / params.window_height;
    vx_int32 num_blockW = width / params.cell_width - 1;
    vx_int32 num_blockH = height / params.cell_height - 1;
    vx_int32 n_cellsx = width / cell_width;
    vx_int32 cells_per_block_w = params.block_width / cell_width;
    vx_int32 cells_per_block_h = params.block_height / cell_height;
    vx_int32 blocks_per_window_w = (params.window_width - params.block_width) / params.block_stride + 1;
    vx_int32 blocks_per_window_h = (params.window_height - params.block_height) / params.block_stride + 1;

    vx_int16* mag_ref = (vx_int16 *)ct_alloc_mem(height / cell_height * width / cell_width * sizeof(vx_int16));
    vx_int16* bins_ref = (vx_int16 *)ct_alloc_mem(height / cell_height * width / cell_width * bins_num * sizeof(vx_int16));
    vx_size features_dim_num = 3;

    vx_size features_dims[3] = { (width - params.window_width) / params.window_stride + 1,
                                 (height - params.window_height) / params.window_stride + 1,
                                 ((params.window_width - params.block_width) / params.block_stride + 1) *
                                 ((params.window_height - params.block_height) / params.block_stride + 1) *
                                 (params.block_width * params.block_height) / (cell_width * cell_height) * bins_num };

    vx_size features_strides[3] = { sizeof(vx_int16), sizeof(vx_int16) *features_dims[0] , sizeof(vx_int16) *features_dims[0] * features_dims[1] };
    vx_size tensor_data_len = features_dims[0] * features_dims[1] * features_dims[2] * sizeof(vx_int16);

    vx_int16* features_p = (vx_int16 *)ct_alloc_mem(tensor_data_len);
    vx_int16* features_ref = (vx_int16 *)ct_alloc_mem(tensor_data_len);

    memset(mag_ref, 0, (height / cell_height) * (width / cell_width) * sizeof(vx_int16));
    memset(bins_ref, 0, (height / cell_height) * (width / cell_width) * bins_num * sizeof(vx_int16));
    memset(features_ref, 0, tensor_data_len);

    vx_float32 num_div_360 = (vx_float32)bins_num / 360.0f;

    const vx_size view_start[3] = { 0 };
    vxCopyTensorPatch(features, features_dim_num, view_start, features_dims, features_strides, features_p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    vx_int32 num_cellw = (vx_int32)floor(((vx_float64)width) / ((vx_float64)cell_width));

    for (vx_int32 j = 0; j < height; j++)
    {
        for (vx_int32 i = 0; i < width; i++)
        {
            vx_int32 x1 = i - 1 < 0 ? 0 : i - 1;
            vx_int32 x2 = i + 1 >= width ? width - 1 : i + 1;
            vx_uint8 *gx1 = (vx_uint8*)((vx_uint8*)p_ct_base + j * img->stride + x1);
            vx_uint8 *gx2 = (vx_uint8*)((vx_uint8*)p_ct_base + j * img->stride + x2);
            gx = *gx2 - *gx1;

            vx_int32 y1 = j - 1 < 0 ? 0 : j - 1;
            vx_int32 y2 = j + 1 >= height ? height - 1 : j + 1;
            vx_uint8 *gy1 = (vx_uint8*)((vx_uint8*)p_ct_base + y1 * img->stride + i);
            vx_uint8 *gy2 = (vx_uint8*)((vx_uint8*)p_ct_base + y2 * img->stride + i);
            gy = *gy2 - *gy1;

            magnitude = sqrtf(powf(gx, 2) + powf(gy, 2));
            orientation = fmod(atan2f(gy, gx + 0.00000000000001)
                          * (180 / 3.14159265), 360);

            if (orientation < 0) 
            {
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
    // The below for-loop implements the following for each window:
    // 1. Normalizes the histograms at block level using it's cells' magnitudes (L2-Sys)
    // 2. Calculates HoG Descriptors for each window of the image, which is 
    // the concatenated descriptors of all the blocks contained in it.
    //
    // Note: Windows in an image, blocks in a window and cells in a block, 
    // are all processed in a row-major order. Cell bins are addressed in row-major 
    // spanning the entire image. E.g. An image 24x16 has 6 cells of size 8x8. Bin Idx 0-8 
    // for top left cell, 9-17 for next cell to the right, 18-26 for last cell on top row.
    // For next row, cell's Bin Idx are 27-35, 36-44, 44-52.
    for (vx_int32 winH = 0; winH < num_windowsH; winH++)
    {
        for (vx_int32 winW = 0; winW < num_windowsW; winW++)
        {
            // Indexes corresponding to the first cell (top left) of window and block
            vx_uint64 binIdx_blk, binIdx_win;
            vx_uint64 binIdx_cell, magIdx_cell;
            binIdx_win = (winH * (n_cellsx * params.window_stride / params.cell_height) +
                          winW * (params.window_stride / params.cell_width)) * params.num_bins;

            for (vx_int32 blkH = 0; blkH < blocks_per_window_h; blkH++)
            {
                for (vx_int32 blkW = 0; blkW < blocks_per_window_w; blkW++)
                {
                    binIdx_blk = binIdx_win + (blkH * (n_cellsx * params.block_stride / params.cell_height) * params.num_bins) +
                                 (blkW * params.block_stride / params.cell_height) * params.num_bins;

                    vx_float32 sum = 0;
                    vx_float32 renorm_sum = 0;
                    vx_uint32 renorm_block_index_st = block_index_count;

                    // Accumulate squared-magnitudes for all the cells in this block
                    for (vx_int32 y = 0; y < cells_per_block_h; y++)
                    {
                        for (vx_int32 x = 0; x < cells_per_block_w; x++)
                        {
                            magIdx_cell = (binIdx_blk / params.num_bins) + (y * n_cellsx + x);
                            void *mag_ptr = (vx_int16 *)mag_ref + magIdx_cell;
                            sum += (*(vx_int16 *)mag_ptr) * (*(vx_int16 *)mag_ptr);
                        }
                    }
                    // Square root of sum-of-squares of cell magnitudes for L2-Norm
                    // For a block with 4 cells with mag m: sqrt( m1^2 + m2^2 + m3^2 + m4^2)
                    sum = sqrtf(sum + 0.00000000000001f);

                    // Calculate HoG Descriptor for the current block from its cell histograms 
                    for (vx_int32 y = 0; y < cells_per_block_h; y++)
                    {
                        for (vx_int32 x = 0; x < cells_per_block_w; x++)
                        {
                            binIdx_cell = binIdx_blk + (y * n_cellsx + x) * params.num_bins;

                            for (vx_int32 k = 0; k < params.num_bins; k++)
                            {
                                // Bin index for the current cell
                                vx_int32 bins_index = binIdx_cell + k;
                                vx_int32 block_index;

                                block_index = block_index_count;

                                // L2-Sys (at block level) = L2-Norm -> clip at threshold -> renormalize

                                // Normalize each cell histogram bin value using L2-Norm and then clip at threshold
                                // using square root of sum-of-squares of cell magnitudes calculated above
                                vx_float32 hist = min((vx_int16)(*((vx_int16 *)bins_ref + bins_index)) / sum, params.threshold);
                                vx_int16 *features_ptr = (vx_int16 *)features_ref + block_index;
                                hist = hist * powf(2, 8); // Bitshift for storing as INT16, Q78 feature tensor
                                *features_ptr = (vx_int16)hist;
                                block_index_count++;
                            } // End for num_bins
                        } // End for cell_w
                    } // End for cell_h

                    // Renormalize the block histogram after L2-Norm and clipping to get L2-Hys
                    vx_uint32 renorm_block_index_end = block_index_count;

                    // Sum of squares of the block feature vector
                    for (vx_uint32 renorm_count = renorm_block_index_st; renorm_count < renorm_block_index_end; renorm_count++)
                    {
                        vx_int16 *features_ptr = (vx_int16 *)features_ref + renorm_count;
                        vx_float32 feature_val = *features_ptr / powf(2, 8);	// Convert INT16 Q78 tensor value to float
                        renorm_sum += (feature_val * feature_val);
                    }

                    renorm_sum = sqrtf(renorm_sum + 0.00000000000001f);			// Sqrt of 'sum of squares' for renormalization

                    // Renormalize the whole block feature vector
                    for (vx_uint32 renorm_count = renorm_block_index_st; renorm_count < renorm_block_index_end; renorm_count++)
                    {
                        vx_int16 *features_ptr = (vx_int16 *)features_ref + renorm_count;
                        vx_float32 feature_val = ((vx_float32)*features_ptr) / powf(2, 8);	// Convert INT16 Q78 tensor value to float
                        *features_ptr = (vx_int16)((feature_val / renorm_sum) * powf(2, 8));	// Renormalize and Bitshift for INT16, Q78 feature tensor
                    }
                }	// End for BlkW
            }	// End for BlkH
        }	// End for winW
    }	// End for winH
    for (vx_int32 i = 0; i < num_windowsW * num_windowsH * params.window_width / params.block_stride *
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

    ct_free_mem(mag_ref);
    ct_free_mem(bins_ref);
    ct_free_mem(features_ref);
    ct_free_mem(features_p);
    return status;
}


typedef struct 
{
    const vx_char* testName;
    CT_Image(*generator)(const vx_char* fileName, vx_int32 width, vx_int32 height);
    const vx_char* fileName;
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
    const vx_size features_dims[3] = { (src_width - arg_->hog_params.window_width) / arg_->hog_params.window_stride + 1,
                                       (src_height - arg_->hog_params.window_height) / arg_->hog_params.window_stride + 1,
                                       ((arg_->hog_params.window_width - arg_->hog_params.block_width) / arg_->hog_params.block_stride + 1) *
                                       ((arg_->hog_params.window_height - arg_->hog_params.block_height) / arg_->hog_params.block_stride + 1) *
                                       ((arg_->hog_params.block_width * arg_->hog_params.block_height) / (arg_->hog_params.cell_width * arg_->hog_params.cell_height)) * bins_num };

    vx_tensor magnitudes;
    vx_tensor bins;
    vx_tensor features;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
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
    vx_int32 cell_width = arg_->hog_params.cell_width;
    vx_int32 cell_height = arg_->hog_params.cell_height;
    vx_int32 bins_num = arg_->hog_params.num_bins;
    CT_Image src = NULL;
    vx_status status = VX_SUCCESS;

    vx_uint32 src_width;
    vx_uint32 src_height;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0));
    src_width = src->width;
    src_height = src->height;

    const vx_size mag_dims[2] = { src_width / cell_width, src_height / cell_height };
    const vx_size bins_dims[3] = { src_width / cell_width, src_height / cell_height, bins_num };
    const vx_size features_dims[3] = { (src_width - arg_->hog_params.window_width) / arg_->hog_params.window_stride + 1,
                                       (src_height - arg_->hog_params.window_height) / arg_->hog_params.window_stride + 1,
                                       ((arg_->hog_params.window_width - arg_->hog_params.block_width) / arg_->hog_params.block_stride + 1) *
                                       ((arg_->hog_params.window_height - arg_->hog_params.block_height) / arg_->hog_params.block_stride + 1) *
                                       ((arg_->hog_params.block_width * arg_->hog_params.block_height) / (arg_->hog_params.cell_width * arg_->hog_params.cell_height)) * bins_num };

    vx_tensor magnitudes;
    vx_tensor bins;
    vx_tensor features;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(magnitudes = vxCreateTensor(context, 2, mag_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(bins = vxCreateTensor(context, 3, bins_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(features = vxCreateTensor(context, 3, features_dims, VX_TYPE_INT16, 8), VX_TYPE_TENSOR);

    VX_CALL(vxuHOGCells(context, src_image, cell_width, cell_width, bins_num, magnitudes, bins));
    VX_CALL(vxuHOGFeatures(context, src_image, magnitudes, bins, &arg_->hog_params, 1, features));
    ASSERT_NO_FAILURE(status = hogfeatures_ref(src, arg_->hog_params, features));
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, status);

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

#endif //OPENVX_USE_ENHANCED_VISION