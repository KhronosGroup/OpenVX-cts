/*

 * Copyright (c) 2017-2017 The Khronos Group Inc.
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

#include <stdint.h>
#include <VX/vx.h>
#include <VX/vxu.h>

TESTCASE(Nonmaxsuppression, CT_VXContext, ct_setup_vx_context, 0)

TEST(Nonmaxsuppression, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0;
    vx_image mask = 0;
    vx_image output = 0;
    vx_uint32 src_width, src_height;

    vx_int32 wsize = 3;

    vx_graph graph = 0;
    vx_node node = 0;

    src_width = 64;
    src_height = 32;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(input = vxCreateImage(context, src_width, src_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(mask = vxCreateImage(context, src_width, src_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, src_width, src_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(node = vxNonMaxSuppressionNode(graph, input, mask, wsize, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&input));
    VX_CALL(vxReleaseImage(&mask));
    VX_CALL(vxReleaseImage(&output));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(output == 0);
    ASSERT(mask == 0);
    ASSERT(input == 0);
}

static CT_Image nonmaxsuppression_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
        image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_S16, &CT()->seed_, 0, 256));

    return image;
}

static CT_Image nonmaxsuppression_read_image(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}

static CT_Image nonmax_golden(vx_image input, vx_image mask, vx_int32 wsize)
{
    vx_status status = VX_FAILURE;
    vx_int32 rect_start_x, rect_start_y, rect_width, rect_height;
    vx_uint32 full_width, full_height;
    vx_uint8 mask_data = 0;
    vx_df_image format = 0, mask_format = 0;

    status  = vxQueryImage(input, VX_IMAGE_WIDTH,  &full_width,  sizeof(full_width));
    status |= vxQueryImage(input, VX_IMAGE_HEIGHT, &full_height, sizeof(full_height));
    status |= vxQueryImage(input, VX_IMAGE_FORMAT, &format, sizeof(format));

    vx_imagepatch_addressing_t src_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_imagepatch_addressing_t mask_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_rectangle_t src_rect, mask_rect, full_rect = {0, 0, full_width, full_height};
    vx_map_id src_map_id = 0;
    vx_map_id mask_map_id = 0;

    void *src_base  = NULL;
    void *mask_base = NULL;
    void *dst_base  = NULL;

    status |= vxGetValidRegionImage(input, &src_rect);
    status |= vxMapImagePatch(input, &full_rect, 0, &src_map_id, &src_addr, &src_base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

    if (mask != NULL)
    {
        status |= vxGetValidRegionImage(mask, &mask_rect);
        status |= vxMapImagePatch(mask, &full_rect, 0, &mask_map_id, &mask_addr, (void **)&mask_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
        status |= vxQueryImage(mask, VX_IMAGE_FORMAT, &mask_format, sizeof(mask_format));
        if ((mask_rect.start_x > src_rect.start_x) || (mask_rect.start_y > src_rect.start_y) ||
            (mask_rect.end_x   < src_rect.end_x)   || (mask_rect.end_y   < src_rect.end_y))
        {
            status |= vxUnmapImagePatch(mask,  mask_map_id);
            status |= vxUnmapImagePatch(input, src_map_id);
            FAIL_(return NULL, "The mask's valid region didn't cover the entire valid region of the input image.");
        }
    }

    rect_start_x = src_rect.start_x;
    rect_start_y = src_rect.start_y;
    rect_width   = src_rect.end_x - src_rect.start_x;
    rect_height  = src_rect.end_y - src_rect.start_y;

    CT_Image output = ct_allocate_image(full_width, full_height, format);
    dst_base = ct_image_get_plane_base(output, 0);
    vx_int32 border = wsize / 2;

    for (vx_int32 x = rect_start_x + border; x < (rect_start_x + rect_width - border); x++)
    {
        for (vx_int32 y = rect_start_y + border; y < (rect_start_y + rect_height - border); y++)
        {
            vx_uint8 *_mask;
            if (mask != NULL)
            {
                _mask = (vx_uint8 *)vxFormatImagePatchAddress2d(mask_base, x, y, &mask_addr);
                if (mask_format == VX_DF_IMAGE_U1)
                    _mask = (*_mask & (1 << (x % 8))) != 0 ? _mask : &mask_data;
            }
            else
            {
                _mask = &mask_data;
            }
            void *val_p = vxFormatImagePatchAddress2d(src_base, x, y, &src_addr);
            void *dest = (vx_int16*)dst_base + y * output->stride + x;
            vx_int32 src_val = *(vx_int16 *)val_p;
            if (*_mask != 0)
            {
                *(vx_int16 *)dest = (vx_int16)src_val;
            }
            else
            {
                vx_bool flag = 1;
                for (vx_int32 i = -border; i <= border; i++)
                {
                    for (vx_int32 j = -border; j <= border; j++)
                    {
                        void *neighbor = vxFormatImagePatchAddress2d(src_base, x + i, y + j, &src_addr);
                        if (mask != NULL)
                        {
                            _mask = (vx_uint8 *)vxFormatImagePatchAddress2d(mask_base, x + i, y + j, &mask_addr);
                            if (mask_format == VX_DF_IMAGE_U1)
                                _mask = (*_mask & (1 << ((x + i) % 8))) != 0 ? _mask : &mask_data;
                        }
                        else
                        {
                            _mask = &mask_data;
                        }
                        vx_int32 neighbor_val = *(vx_int16 *)neighbor;
                        if ((*_mask == 0)
                            && ( ((j < 0 || (j == 0 && i <= 0)) && (src_val < neighbor_val))
                                 || ((j > 0 || (j == 0 && i > 0)) && (src_val <= neighbor_val)) ))
                        {
                            flag = 0;
                            break;
                        }
                    }
                    if (flag == 0)
                    {
                        break;
                    }
                }

                if (flag)
                {
                    *(vx_int16 *)dest = (vx_int16)src_val;
                }
                else
                {
                    *(vx_int16 *)dest = INT16_MIN;
                }
            }
        }
    }
    status |= vxUnmapImagePatch(input, src_map_id);
    if (mask != NULL)
    {
        status |= vxUnmapImagePatch(mask, mask_map_id);
    }

    return output;
}

typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height);
    const char* fileName;
    vx_int32 wsize;
    vx_bool _mask;
    vx_df_image maskFormat;
    vx_df_image format;
    const char* result_filename;
} Arg;

#define PARAMETERS \
    ARG("case_1_U8_nomask", nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 1, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_1_nomask.bmp"), \
    ARG("case_3_U8_nomask", nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 3, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_3_nomask.bmp"), \
    ARG("case_5_U8_nomask", nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 5, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_5_nomask.bmp"), \
    ARG("case_1_U8_mask",   nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 1, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_1_mask.bmp"), \
    ARG("case_3_U8_mask",   nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 3, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_3_mask.bmp"), \
    ARG("case_5_U8_mask",   nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 5, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_5_mask.bmp"), \
    ARG("case_1_S16_nomask", nonmaxsuppression_generate_random, NULL, 1, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL), \
    ARG("case_3_S16_nomask", nonmaxsuppression_generate_random, NULL, 3, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL), \
    ARG("case_5_S16_nomask", nonmaxsuppression_generate_random, NULL, 5, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL), \
    ARG("case_1_S16_mask",   nonmaxsuppression_generate_random, NULL, 1, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL), \
    ARG("case_3_S16_mask",   nonmaxsuppression_generate_random, NULL, 3, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL), \
    ARG("case_5_S16_mask",   nonmaxsuppression_generate_random, NULL, 5, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL), \
    ARG("_U1_/case_1_U8_mask",  nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 1, vx_true_e, VX_DF_IMAGE_U1, VX_DF_IMAGE_U8, "nms_1_mask.bmp"), \
    ARG("_U1_/case_3_U8_mask",  nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 3, vx_true_e, VX_DF_IMAGE_U1, VX_DF_IMAGE_U8, "nms_3_mask.bmp"), \
    ARG("_U1_/case_5_U8_mask",  nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 5, vx_true_e, VX_DF_IMAGE_U1, VX_DF_IMAGE_U8, "nms_5_mask.bmp"), \
    ARG("_U1_/case_1_S16_mask", nonmaxsuppression_generate_random, NULL, 1, vx_true_e, VX_DF_IMAGE_U1, VX_DF_IMAGE_S16, NULL), \
    ARG("_U1_/case_3_S16_mask", nonmaxsuppression_generate_random, NULL, 3, vx_true_e, VX_DF_IMAGE_U1, VX_DF_IMAGE_S16, NULL), \
    ARG("_U1_/case_5_s16_mask", nonmaxsuppression_generate_random, NULL, 5, vx_true_e, VX_DF_IMAGE_U1, VX_DF_IMAGE_S16, NULL)

TEST_WITH_ARG(Nonmaxsuppression, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;

    vx_image input = 0;
    vx_image mask = 0;
    vx_image output = 0;
    vx_uint32 src_width, src_height;

    vx_int32 wsize = arg_->wsize;
    vx_int32 border = wsize/2;
    CT_Image src = NULL, ct_output = NULL;

    vx_status status;

    void *mask_base = NULL;
    vx_imagepatch_addressing_t mask_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_rectangle_t mask_rect;
    vx_map_id mask_map_id = 0;
    if (arg_->format == VX_DF_IMAGE_U8)
    {
        ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0));
    }
    else
    {
        ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 640, 480));
    }
    src_width = src->width;
    src_height = src->height;

    ASSERT_VX_OBJECT(input = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, src_width, src_height, arg_->format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    if (arg_->_mask)
    {
        ASSERT_VX_OBJECT(mask = vxCreateImage(context, src_width, src_height, arg_->maskFormat), VX_TYPE_IMAGE);
        status = vxGetValidRegionImage(mask, &mask_rect);
        status |= vxMapImagePatch(mask, &mask_rect, 0, &mask_map_id, &mask_addr, (void **)&mask_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
        for (vx_uint32 i = 0; i < src_width; i++)
        {
            for (vx_uint32 j = 0; j < src_height; j++)
            {
                vx_uint8 val;
                void *mask_ptr = vxFormatImagePatchAddress2d(mask_base, i, j, &mask_addr);
                if (i % 2 == 0 && j % 2 == 0)
                {
                    val = 1;
                }
                else
                {
                    val = 0;
                }
                *(vx_uint8 *)mask_ptr = (arg_->maskFormat == VX_DF_IMAGE_U1)
                                        ? (*(vx_uint8 *)mask_ptr & ~(1 << (i % 8))) | (val << (i % 8)) : val;
            }
        }
        status |= vxUnmapImagePatch(mask, mask_map_id);
    }

    ASSERT_VX_OBJECT(node = vxNonMaxSuppressionNode(graph, input, mask, wsize, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(ct_output = ct_image_from_vx_image(output));
    CT_Image golden_image;
    if (arg_->format == VX_DF_IMAGE_U8)
    {
        golden_image = arg_->generator(arg_->result_filename, 0, 0);
    }
    else
    {
        golden_image = nonmax_golden(input, mask, wsize);
    }
    ct_adjust_roi(ct_output, border, border, border, border);
    ct_adjust_roi(golden_image, border, border, border, border);
    EXPECT_EQ_CTIMAGE(golden_image, ct_output);

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&input));
    if (arg_->_mask)
    {
        VX_CALL(vxReleaseImage(&mask));
    }
    VX_CALL(vxReleaseImage(&output));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(output == 0);
    ASSERT(mask == 0);
    ASSERT(input == 0);
}

TEST_WITH_ARG(Nonmaxsuppression, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_image input = 0;
    vx_image mask = 0;
    vx_image output = 0;
    vx_uint32 src_width, src_height;

    vx_int32 wsize = arg_->wsize;
    vx_int32 border = wsize/2;
    CT_Image src = NULL, ct_output = NULL;

    vx_status status;

    void *mask_base = NULL;
    vx_imagepatch_addressing_t mask_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_rectangle_t mask_rect;
    vx_map_id mask_map_id = 0;

    if (arg_->format == VX_DF_IMAGE_U8)
    {
        ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0));
    }
    else
    {
        ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 640, 480));
    }
    src_width = src->width;
    src_height = src->height;

    ASSERT_VX_OBJECT(input = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, src_width, src_height, arg_->format), VX_TYPE_IMAGE);

    if (arg_->_mask)
    {
        ASSERT_VX_OBJECT(mask = vxCreateImage(context, src_width, src_height, arg_->maskFormat), VX_TYPE_IMAGE);
        status = vxGetValidRegionImage(mask, &mask_rect);
        status |= vxMapImagePatch(mask, &mask_rect, 0, &mask_map_id, &mask_addr, (void **)&mask_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
        for (vx_uint32 i = 0; i < src_width; i++)
        {
            for (vx_uint32 j = 0; j < src_height; j++)
            {
                vx_uint8 val;
                void *mask_ptr = vxFormatImagePatchAddress2d(mask_base, i, j, &mask_addr);
                if (i % 2 == 0 && j % 2 == 0)
                {
                    val = 1;
                }
                else
                {
                    val = 0;
                }
                *(vx_uint8 *)mask_ptr = (arg_->maskFormat == VX_DF_IMAGE_U1)
                                        ? (*(vx_uint8 *)mask_ptr & ~(1 << (i % 8))) | (val << (i % 8)) : val;
            }
        }
        status |= vxUnmapImagePatch(mask, mask_map_id);
    }
    VX_CALL(vxuNonMaxSuppression(context, input, mask, wsize, output));

    ASSERT_NO_FAILURE(ct_output = ct_image_from_vx_image(output));

    CT_Image golden_image;
    if (arg_->format == VX_DF_IMAGE_U8)
    {
        golden_image = arg_->generator(arg_->result_filename, 0, 0);
    }
    else
    {
        golden_image = nonmax_golden(input, mask, wsize);
    }

    ct_adjust_roi(ct_output, border, border, border, border);
    ct_adjust_roi(golden_image, border, border, border, border);
    EXPECT_EQ_CTIMAGE(golden_image, ct_output);

    VX_CALL(vxReleaseImage(&input));
    if (arg_->_mask)
    {
        VX_CALL(vxReleaseImage(&mask));
    }
    VX_CALL(vxReleaseImage(&output));

    ASSERT(output == 0);
    ASSERT(mask == 0);
    ASSERT(input == 0);
}

typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height);
    const char* fileName;
    vx_int32 wsize;
    vx_bool _mask;
    vx_df_image maskFormat;
    vx_df_image format;
    const char* result_filename;
    vx_rectangle_t region_shift;
} ValidRegionTest_Arg;

#define REGION_PARAMETERS \
    ARG("case_1_U8_nomask_region_shrink=1", nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 1, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_1_nomask.bmp", {1, 1, -1, -1}), \
    ARG("case_1_U8_nomask_region_shrink=7", nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 1, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_1_nomask.bmp", {7, 7, -7, -7}), \
    ARG("case_3_U8_mask_region_shrink=1",   nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 3, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_3_mask.bmp",   {1, 1, -1, -1}), \
    ARG("case_3_U8_mask_region_shrink=7",   nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 3, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, "nms_3_mask.bmp",   {7, 7, -7, -7}), \
    ARG("case_1_S16_nomask_region_shrink=1", nonmaxsuppression_generate_random, NULL, 1, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL, {1, 1, -1, -1}), \
    ARG("case_1_S16_nomask_region_shrink=7", nonmaxsuppression_generate_random, NULL, 1, vx_false_e, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL, {7, 7, -7, -7}), \
    ARG("case_3_S16_mask_region_shrink=1",   nonmaxsuppression_generate_random, NULL, 3, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL, {1, 1, -1, -1}), \
    ARG("case_3_S16_mask_region_shrink=7",   nonmaxsuppression_generate_random, NULL, 3, vx_true_e,  VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, NULL, {7, 7, -7, -7}), \
    ARG("_U1_/case_3_U8_mask_region_shrink=1",  nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 3, vx_true_e,  VX_DF_IMAGE_U1, VX_DF_IMAGE_U8, "nms_3_mask.bmp",   {1, 1, -1, -1}), \
    ARG("_U1_/case_3_U8_mask_region_shrink=7",  nonmaxsuppression_read_image, "blurred_lena_gray.bmp", 3, vx_true_e,  VX_DF_IMAGE_U1, VX_DF_IMAGE_U8, "nms_3_mask.bmp",   {7, 7, -7, -7}), \
    ARG("_U1_/case_3_S16_mask_region_shrink=1", nonmaxsuppression_generate_random, NULL, 3, vx_true_e,  VX_DF_IMAGE_U1, VX_DF_IMAGE_S16, NULL, {1, 1, -1, -1}), \
    ARG("_U1_/case_3_S16_mask_region_shrink=7", nonmaxsuppression_generate_random, NULL, 3, vx_true_e,  VX_DF_IMAGE_U1, VX_DF_IMAGE_S16, NULL, {7, 7, -7, -7})

TEST_WITH_ARG(Nonmaxsuppression, testWithValidRegion, ValidRegionTest_Arg,
    REGION_PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_image input = 0, mask = 0, output = 0;
    vx_uint32 src_width, src_height;

    vx_int32 wsize = arg_->wsize;
    vx_int32 border = wsize / 2;
    CT_Image ct_input = NULL, ct_output = NULL, golden_image = NULL;

    vx_status status;

    void *mask_base = NULL;
    vx_imagepatch_addressing_t mask_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_rectangle_t mask_rect, src_rect, rect_shft = arg_->region_shift;
    vx_map_id mask_map_id = 0;

    if (arg_->format == VX_DF_IMAGE_U8)
    {
        ASSERT_NO_FAILURE(ct_input = arg_->generator(arg_->fileName, 0, 0));
    }
    else    // format == VX_DF_IMAGE_S16
    {
        ASSERT_NO_FAILURE(ct_input = arg_->generator(arg_->fileName, 640, 480));
    }
    src_width  = ct_input->width;
    src_height = ct_input->height;

    ASSERT_VX_OBJECT(input = ct_image_to_vx_image(ct_input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, src_width, src_height, arg_->format), VX_TYPE_IMAGE);

    if (arg_->_mask)
    {
        ASSERT_VX_OBJECT(mask = vxCreateImage(context, src_width, src_height, arg_->maskFormat), VX_TYPE_IMAGE);
        status = vxGetValidRegionImage(mask, &mask_rect);
        status |= vxMapImagePatch(mask, &mask_rect, 0, &mask_map_id, &mask_addr, (void **)&mask_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
        for (vx_uint32 i = 0; i < src_width; i++)
        {
            for (vx_uint32 j = 0; j < src_height; j++)
            {
                vx_uint8 val;
                void *mask_ptr = vxFormatImagePatchAddress2d(mask_base, i, j, &mask_addr);
                if (i % 2 == 0 && j % 2 == 0)
                {
                    val = 1;
                }
                else
                {
                    val = 0;
                }
                *(vx_uint8 *)mask_ptr = (arg_->maskFormat == VX_DF_IMAGE_U1)
                                        ? (*(vx_uint8 *)mask_ptr & ~(1 << (i % 8))) | (val << (i % 8)) : val;
            }
        }
        status |= vxUnmapImagePatch(mask, mask_map_id);
    }

    ASSERT_NO_FAILURE(vxGetValidRegionImage(input, &src_rect));
    ALTERRECTANGLE(src_rect, rect_shft.start_x, rect_shft.start_y, rect_shft.end_x, rect_shft.end_y);
    ASSERT_NO_FAILURE(vxSetImageValidRectangle(input, &src_rect));

    VX_CALL(vxuNonMaxSuppression(context, input, mask, wsize, output));

    ASSERT_NO_FAILURE(ct_output = ct_image_from_vx_image(output));

    if (arg_->format == VX_DF_IMAGE_U8)
    {
        golden_image = arg_->generator(arg_->result_filename, 0, 0);
    }
    else
    {
        golden_image = nonmax_golden(input, mask, wsize);
    }

    ct_adjust_roi(ct_output,    rect_shft.start_x, rect_shft.start_y, -rect_shft.end_x, -rect_shft.end_y);
    ct_adjust_roi(golden_image, rect_shft.start_x, rect_shft.start_y, -rect_shft.end_x, -rect_shft.end_y);
    ct_adjust_roi(ct_output,    border, border, border, border);
    ct_adjust_roi(golden_image, border, border, border, border);
    EXPECT_EQ_CTIMAGE(golden_image, ct_output);

    VX_CALL(vxReleaseImage(&input));
    if (arg_->_mask)
    {
        VX_CALL(vxReleaseImage(&mask));
    }
    VX_CALL(vxReleaseImage(&output));

    ASSERT(output == 0);
    ASSERT(mask == 0);
    ASSERT(input == 0);
}

TESTCASE_TESTS(Nonmaxsuppression,
               testNodeCreation,
               testGraphProcessing,
               testImmediateProcessing,
               testWithValidRegion)

#endif //OPENVX_USE_ENHANCED_VISION
