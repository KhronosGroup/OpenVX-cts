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

#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION

#include "test_engine/test.h"
#include <VX/vx.h>
#include <VX/vxu.h>

TESTCASE(Median3x3, CT_VXContext, ct_setup_vx_context, 0)

TEST(Median3x3, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxMedian3x3Node(graph, src_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&dst_image));
    VX_CALL(vxReleaseImage(&src_image));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

// Generate input to cover these requirements:
// There should be a image with randomly generated pixel intensities.
static CT_Image median3x3_generate_random(const char* fileName, int width, int height, vx_df_image format)
{
    CT_Image image;

    ASSERT_(return 0, format == VX_DF_IMAGE_U1 || format == VX_DF_IMAGE_U8);

    if (format == VX_DF_IMAGE_U1)
        ASSERT_NO_FAILURE_(return 0, image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 2));
    else
        ASSERT_NO_FAILURE_(return 0, image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 256));

    return image;
}

static CT_Image median3x3_read_image(const char* fileName, int width, int height, vx_df_image format)
{
    CT_Image image_load = NULL, image_ret = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    ASSERT_(return 0, format == VX_DF_IMAGE_U1 || format == VX_DF_IMAGE_U8);

    image_load = ct_read_image(fileName, 1);
    ASSERT_(return 0, image_load);
    ASSERT_(return 0, image_load->format == VX_DF_IMAGE_U8);

    if (format == VX_DF_IMAGE_U1)
    {
        ASSERT_NO_FAILURE_(return 0, threshold_U8_ct_image(image_load, 127));   // Threshold to make the U1 image less trivial
        ASSERT_NO_FAILURE_(return 0, image_ret = ct_allocate_image(image_load->width, image_load->height, VX_DF_IMAGE_U1));
        ASSERT_NO_FAILURE_(return 0, U8_ct_image_to_U1_ct_image(image_load, image_ret));
    }
    else
    {
        image_ret = image_load;
    }

    ASSERT_(return 0, image_ret);
    ASSERT_(return 0, image_ret->format == format);

    return image_ret;
}

static int compare_for_median_get(const void * a, const void * b)
{
    return *(int*)a - *(int*)b;
}

static int32_t median_get_U1(int32_t values[9][2])
{
    int i;
    int32_t v_acc = 0;
    for (i = 0; i < 9; i++)     // Find median value by counting number of pixels == 1 and checking if sum > 4
    {
        v_acc += (values[i][0] & (1 << (values[i][1] % 8))) >> (values[i][1] % 8);
    }
    return (v_acc > 4) ? 1 : 0;
}

static int32_t median_get_U8(int32_t *values)
{
    qsort(values, 9, sizeof(values[0]), compare_for_median_get);
    return values[4];
}

static uint8_t median3x3_calculate(CT_Image src, uint32_t x, uint32_t y)
{
    if (src->format == VX_DF_IMAGE_U1)
    {
        int32_t values[9][2] = {
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x + 0, y + 0), (int32_t)x + 0},
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x - 1, y + 0), (int32_t)x - 1},
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x + 1, y + 0), (int32_t)x + 1},
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x + 0, y - 1), (int32_t)x + 0},
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x - 1, y - 1), (int32_t)x - 1},
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x + 1, y - 1), (int32_t)x + 1},
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x + 0, y + 1), (int32_t)x + 0},
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x - 1, y + 1), (int32_t)x - 1},
            {(int32_t)*CT_IMAGE_DATA_PTR_1U(src, x + 1, y + 1), (int32_t)x + 1}
        };
        return (uint8_t)median_get_U1(values);
    }
    else
    {
        int32_t values[9] = {
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 0, y + 0),
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 1, y + 0),
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 1, y + 0),
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 0, y - 1),
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 1, y - 1),
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 1, y - 1),
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 0, y + 1),
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 1, y + 1),
            (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 1, y + 1)
        };
        return (uint8_t)median_get_U8(values);
    }
}

static uint8_t median3x3_calculate_replicate(CT_Image src, uint32_t x_, uint32_t y_)
{
    int32_t x = (int)x_;
    int32_t y = (int)y_;
    if (src->format == VX_DF_IMAGE_U1)
    {
        int32_t values[9] = {
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x + 0, y + 0),
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x - 1, y + 0),
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x + 1, y + 0),
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x + 0, y - 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x - 1, y - 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x + 1, y - 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x + 0, y + 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x - 1, y + 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_1U(src, x + 1, y + 1)
        };
        return (uint8_t)median_get_U8(values);
    }
    else
    {
        int32_t values[9] = {
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y + 0),
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y + 0),
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y + 0),
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y - 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y - 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y - 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y + 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y + 1),
            (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y + 1)
        };
        return (uint8_t)median_get_U8(values);
    }
}

static uint8_t median3x3_calculate_constant(CT_Image src, uint32_t x_, uint32_t y_, vx_uint32 constant_value)
{
    int32_t x = (int)x_;
    int32_t y = (int)y_;
    if (src->format == VX_DF_IMAGE_U1)
    {
        vx_bool const_val_bool = (constant_value == 0) ? vx_false_e : vx_true_e;
        int32_t values[9] = {
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x + 0, y + 0, const_val_bool),
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x - 1, y + 0, const_val_bool),
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x + 1, y + 0, const_val_bool),
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x + 0, y - 1, const_val_bool),
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x - 1, y - 1, const_val_bool),
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x + 1, y - 1, const_val_bool),
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x + 0, y + 1, const_val_bool),
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x - 1, y + 1, const_val_bool),
            (int32_t)CT_IMAGE_DATA_CONSTANT_1U(src, x + 1, y + 1, const_val_bool)
        };
        return (uint8_t)median_get_U8(values);
    }
    else
    {
        int32_t values[9] = {
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y + 0, constant_value),
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y + 0, constant_value),
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y + 0, constant_value),
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y - 1, constant_value),
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y - 1, constant_value),
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y - 1, constant_value),
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y + 1, constant_value),
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y + 1, constant_value),
            (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y + 1, constant_value)
        };
        return (uint8_t)median_get_U8(values);
    }
}


static CT_Image median3x3_create_reference_image(CT_Image src, vx_border_t border)
{
    CT_Image dst;

    CT_ASSERT_(return NULL, src->format == VX_DF_IMAGE_U1 || src->format == VX_DF_IMAGE_U8);

    dst = ct_allocate_image(src->width, src->height, src->format);

    if (border.mode == VX_BORDER_UNDEFINED)
    {
        if (src->format == VX_DF_IMAGE_U1)
        {
            CT_FILL_IMAGE_1U(return 0, dst,
                    if (x >= 1 && y >= 1 && x < src->width - 1 && y < src->height - 1)
                    {
                        uint32_t xShftdSrc = x + src->roi.x % 8;
                        uint8_t res = median3x3_calculate(src, xShftdSrc, y);
                        *dst_data = (*dst_data & ~(1 << offset)) | (res << offset);
                    });
        }
        else
        {
            CT_FILL_IMAGE_8U(return 0, dst,
                    if (x >= 1 && y >= 1 && x < src->width - 1 && y < src->height - 1)
                    {
                        uint8_t res = median3x3_calculate(src, x, y);
                        *dst_data = res;
                    });
        }
    }
    else if (border.mode == VX_BORDER_REPLICATE)
    {
        if (src->format == VX_DF_IMAGE_U1)
        {
            CT_FILL_IMAGE_1U(return 0, dst,
                    {
                        uint32_t xShftdSrc = x + src->roi.x % 8;
                        uint8_t res = median3x3_calculate_replicate(src, xShftdSrc, y);
                        *dst_data = (*dst_data & ~(1 << offset)) | (res << offset);
                    });
        }
        else
        {
            CT_FILL_IMAGE_8U(return 0, dst,
                    {
                        uint8_t res = median3x3_calculate_replicate(src, x, y);
                        *dst_data = res;
                    });
        }
    }
    else if (border.mode == VX_BORDER_CONSTANT)
    {
        vx_uint32 constant_value = border.constant_value.U32;
        if (src->format == VX_DF_IMAGE_U1)
        {
            CT_FILL_IMAGE_1U(return 0, dst,
                    {
                        uint32_t xShftdSrc = x + src->roi.x % 8;
                        uint8_t res = median3x3_calculate_constant(src, xShftdSrc, y, constant_value);
                        *dst_data = (*dst_data & ~(1 << offset)) | (res << offset);
                    });
        }
        else
        {
            CT_FILL_IMAGE_8U(return 0, dst,
                    {
                        uint8_t res = median3x3_calculate_constant(src, x, y, constant_value);
                        *dst_data = res;
                    });
        }
    }
    else
    {
        ASSERT_(return 0, 0);
    }
    return dst;
}


static void median3x3_check(CT_Image src, CT_Image dst, vx_border_t border)
{
    CT_Image dst_ref = NULL;

    ASSERT(src && dst);

    ASSERT_NO_FAILURE(dst_ref = median3x3_create_reference_image(src, border));

    ASSERT_NO_FAILURE(
        if (border.mode == VX_BORDER_UNDEFINED)
        {
            ct_adjust_roi(dst,  1, 1, 1, 1);
            ct_adjust_roi(dst_ref, 1, 1, 1, 1);
        }
    );

    EXPECT_EQ_CTIMAGE(dst_ref, dst);
#if 0
    if (CT_HasFailure())
    {
        printf("=== SRC ===\n");
        ct_dump_image_info(src);
        printf("=== DST ===\n");
        ct_dump_image_info(dst);
        printf("=== EXPECTED ===\n");
        ct_dump_image_info(dst_ref);
    }
#endif
}

typedef struct {
    const char* testName;
    CT_Image (*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    vx_border_t border;
    int width, height;
    vx_df_image format;
} Filter_Arg;

#define MEDIAN_PARAMETERS \
    CT_GENERATE_PARAMETERS("randomInput", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_SMALL_SET, ADD_TYPE_U8, ARG, median3x3_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ADD_TYPE_U8, ARG, median3x3_read_image, "lena.bmp"), \
    CT_GENERATE_PARAMETERS("_U1_/randomInput", ADD_VX_BORDERS_U1_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_SMALL_SET, ADD_TYPE_U1, ARG, median3x3_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("_U1_/lena", ADD_VX_BORDERS_U1_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ADD_TYPE_U1, ARG, median3x3_read_image, "lena.bmp")

TEST_WITH_ARG(Median3x3, testGraphProcessing, Filter_Arg,
    MEDIAN_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    CT_Image src = NULL, dst = NULL;
    vx_border_t border = arg_->border;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = ct_create_similar_image(src_image), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxMedian3x3Node(graph, src_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(median3x3_check(src, dst, border));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(node == 0);
    ASSERT(graph == 0);

    VX_CALL(vxReleaseImage(&dst_image));
    VX_CALL(vxReleaseImage(&src_image));

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

TEST_WITH_ARG(Median3x3, testImmediateProcessing, Filter_Arg,
    MEDIAN_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;

    CT_Image src = NULL, dst = NULL;
    vx_border_t border = arg_->border;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = ct_create_similar_image(src_image), VX_TYPE_IMAGE);

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));

    VX_CALL(vxuMedian3x3(context, src_image, dst_image));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(median3x3_check(src, dst, border));

    VX_CALL(vxReleaseImage(&dst_image));
    VX_CALL(vxReleaseImage(&src_image));

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

typedef struct {
    const char* testName;
    CT_Image (*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    vx_border_t border;
    int width, height;
    vx_df_image format;
    vx_rectangle_t regionShift;
} ValidRegionTest_Arg;

#ifdef MEDIAN_PARAMETERS
#undef MEDIAN_PARAMETERS
#endif
#define MEDIAN_PARAMETERS \
    CT_GENERATE_PARAMETERS("lena", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ADD_TYPE_U8, ADD_VALID_REGION_SHRINKS, ARG, median3x3_read_image, "lena.bmp"), \
    CT_GENERATE_PARAMETERS("_U1_/lena", ADD_VX_BORDERS_U1_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ADD_TYPE_U1, ADD_VALID_REGION_SHRINKS, ARG, median3x3_read_image, "lena.bmp")

TEST_WITH_ARG(Median3x3, testWithValidRegion, ValidRegionTest_Arg,
    MEDIAN_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;

    CT_Image src = NULL, dst = NULL;
    vx_border_t border = arg_->border;
    vx_rectangle_t rect = {0, 0, 0, 0}, rect_shft = arg_->regionShift;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = ct_create_similar_image(src_image), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(vxGetValidRegionImage(src_image, &rect));
    ALTERRECTANGLE(rect, rect_shft.start_x, rect_shft.start_y, rect_shft.end_x, rect_shft.end_y);
    ASSERT_NO_FAILURE(vxSetImageValidRectangle(src_image, &rect));

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));

    VX_CALL(vxuMedian3x3(context, src_image, dst_image));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));
    ASSERT_NO_FAILURE(ct_adjust_roi(dst, rect_shft.start_x, rect_shft.start_y, -rect_shft.end_x, -rect_shft.end_y));

    ASSERT_NO_FAILURE(ct_adjust_roi(src, rect_shft.start_x, rect_shft.start_y, -rect_shft.end_x, -rect_shft.end_y));
    ASSERT_NO_FAILURE(median3x3_check(src, dst, border));

    VX_CALL(vxReleaseImage(&dst_image));
    VX_CALL(vxReleaseImage(&src_image));

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

TESTCASE_TESTS(Median3x3, testNodeCreation, testGraphProcessing, testImmediateProcessing, testWithValidRegion)

#endif //OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION
