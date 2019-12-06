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

#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"

#define VALID_SHIFT_MIN 0
// #define VALID_SHIFT_MIN -64
#define VALID_SHIFT_MAX 7

#define CT_EXECUTE_ASYNC

static void referenceConvertDepth(CT_Image src, CT_Image dst, int shift, vx_enum policy)
{
    uint32_t i, j;

    ASSERT(src && dst);
    ASSERT(src->width == dst->width);
    ASSERT(src->height == dst->height);
    ASSERT((src->format == VX_DF_IMAGE_U1  && dst->format == VX_DF_IMAGE_U8)  ||
           (src->format == VX_DF_IMAGE_U1  && dst->format == VX_DF_IMAGE_S16) ||
           (src->format == VX_DF_IMAGE_U8  && dst->format == VX_DF_IMAGE_U1)  ||
           (src->format == VX_DF_IMAGE_U8  && dst->format == VX_DF_IMAGE_S16) ||
           (src->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_U1)  ||
           (src->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_U8));
    ASSERT(policy == VX_CONVERT_POLICY_WRAP || policy == VX_CONVERT_POLICY_SATURATE);

    if (shift > 16) shift = 16;
    if (shift < -16) shift = -16;

    if (src->format == VX_DF_IMAGE_U1)
    {
        // Up-convert policy from U1 doesn't take policy and shift into account
        if (dst->format == VX_DF_IMAGE_U8)
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                    {
                        uint32_t xShftd = j + src->roi.x % 8;    // U1 ROI offset
                        uint8_t  pixel  = (src->data.y[i * ct_stride_bytes(src) + xShftd / 8] & (1 << (xShftd % 8))
                                          ) != 0 ? 255 : 0;
                        dst->data.y[i * dst->stride + j] = pixel;
                    }
        }
        else    // dst->format == VX_DF_IMAGE_S16
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                    {
                        uint32_t xShftd = j + src->roi.x % 8;    // U1 ROI offset
                        int16_t  pixel  = (src->data.y[i * ct_stride_bytes(src) + xShftd / 8] & (1 << (xShftd % 8))
                                          ) != 0 ? -1 : 0;
                        dst->data.s16[i * dst->stride + j] = pixel;
                    }
        }
    }
    else if (src->format == VX_DF_IMAGE_U8)
    {
        if (dst->format == VX_DF_IMAGE_U1)
        {
            // Down-convert policy to U1 doesn't take policy and shift into account
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                {
                    uint32_t xShftd = j + src->roi.x % 8;        // U1 ROI offset
                    uint8_t  pixel  = (src->data.y[i * src->stride + j] != 0) ? 1 << (xShftd % 8) : 0;
                    dst->data.y[i * ct_stride_bytes(dst) + xShftd / 8] =
                        (dst->data.y[i * ct_stride_bytes(dst) + xShftd / 8] & ~(1 << (xShftd % 8))) | pixel;
                }
        }
        else    // dst->format == VX_DF_IMAGE_S16
        {
            // Up-converting U8 to S16 doesn't take policy into account
            if (shift < 0)
            {
                for (i = 0; i < dst->height; ++i)
                    for (j = 0; j < dst->width; ++j)
                        dst->data.s16[i * dst->stride + j] = ((unsigned)src->data.y[i * src->stride + j]) >> (-shift);
            }
            else
            {
                for (i = 0; i < dst->height; ++i)
                    for (j = 0; j < dst->width; ++j)
                        dst->data.s16[i * dst->stride + j] = ((unsigned)src->data.y[i * src->stride + j]) << shift;
            }
        }
    }
    // src->format == VX_DF_IMAGE_S16
    else if (dst->format == VX_DF_IMAGE_U1)
    {
        // Down-convert policy to U1 doesn't take policy and shift into account
        for (i = 0; i < dst->height; ++i)
            for (j = 0; j < dst->width; ++j)
            {
                uint32_t xShftd = j + src->roi.x % 8;            // U1 ROI offset
                uint8_t  pixel  = src->data.s16[i * src->stride + j] != 0 ? 1 << (xShftd % 8) : 0;
                dst->data.y[i * ct_stride_bytes(dst) + xShftd / 8] =
                    (dst->data.y[i * ct_stride_bytes(dst) + xShftd / 8] & ~(1 << (xShftd % 8))) | pixel;
            }
    }
    // dst->format == VX_DF_IMAGE_U8
    else if (policy == VX_CONVERT_POLICY_WRAP)
    {
        // Down-conversion (S16 to U8) + wrap
        if (shift < 0)
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                    dst->data.y[i * dst->stride + j] = src->data.s16[i * src->stride + j] << (-shift);
        }
        else
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                    dst->data.y[i * dst->stride + j] = src->data.s16[i * src->stride + j] >> shift;
        }
    }
    else // policy == VX_CONVERT_POLICY_SATURATE
    {
        // Down-conversion (S16 to U8) + saturate
        if (shift < 0)
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                {
                    int32_t v = src->data.s16[i * src->stride + j] << (-shift);
                    if (v > 255) v = 255;
                    if (v < 0) v = 0;
                    dst->data.y[i * dst->stride + j] = v;
                }
        }
        else
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                {
                    int32_t v = src->data.s16[i * src->stride + j] >> shift;
                    if (v > 255) v = 255;
                    if (v < 0) v = 0;
                    dst->data.y[i * dst->stride + j] = v;
                }
        }
    }
}

static void fillSequence(CT_Image dst, uint32_t seq_init)
{
    uint32_t i, j;
    uint32_t val = seq_init;

    ASSERT(dst);
    ASSERT(dst->format == VX_DF_IMAGE_U1 || dst->format == VX_DF_IMAGE_U8 || dst->format == VX_DF_IMAGE_S16);

    if (dst->format == VX_DF_IMAGE_U1)
    {
        for (i = 0; i < dst->height; ++i)
            for (j = 0; j < dst->width; ++j)
            {
                uint32_t xShftd = j + dst->roi.x % 8;            // U1 ROI offset
                uint8_t  pixel  = (++val % 2) << (xShftd % 8);
                dst->data.y[i * ct_stride_bytes(dst) + xShftd / 8] =
                    (dst->data.y[i * ct_stride_bytes(dst) + xShftd / 8] & ~(1 << (xShftd % 8))) | pixel;
            }
    }
    else if (dst->format == VX_DF_IMAGE_U8)
    {
        for (i = 0; i < dst->height; ++i)
            for (j = 0; j < dst->width; ++j)
                dst->data.y[i * dst->stride + j] = ++val;
    }
    else
    {
        for (i = 0; i < dst->height; ++i)
            for (j = 0; j < dst->width; ++j)
                dst->data.s16[i * dst->stride + j] = ++val;
    }
}

TESTCASE(vxuConvertDepth, CT_VXContext, ct_setup_vx_context, 0)
TESTCASE(vxConvertDepth,  CT_VXContext, ct_setup_vx_context, 0)

TEST(vxuConvertDepth, NegativeSizes)
{
    vx_image img88x88, img88x40, img40x40;
    vx_int32 shift_zero = 0;
    vx_int32 shift_one = 1;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(img88x88 = vxCreateImage(context, 88, 88, VX_DF_IMAGE_U8),  VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img88x40 = vxCreateImage(context, 88, 40, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img40x40 = vxCreateImage(context, 40, 40, VX_DF_IMAGE_U8),  VX_TYPE_IMAGE);

    // initialize to guarantee that images are allocated
    ASSERT_NO_FAILURE(ct_fill_image_random(img88x88, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(img88x40, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(img40x40, &CT()->seed_));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x88, img88x40, VX_CONVERT_POLICY_SATURATE, shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x88, img88x40, VX_CONVERT_POLICY_WRAP,     shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x88, img88x40, VX_CONVERT_POLICY_SATURATE, shift_one));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x88, img88x40, VX_CONVERT_POLICY_WRAP,     shift_one));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x40, img40x40, VX_CONVERT_POLICY_SATURATE, shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x40, img40x40, VX_CONVERT_POLICY_WRAP,     shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x40, img40x40, VX_CONVERT_POLICY_SATURATE, shift_one));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x40, img40x40, VX_CONVERT_POLICY_WRAP,     shift_one));

    VX_CALL(vxReleaseImage(&img88x88));
    VX_CALL(vxReleaseImage(&img88x40));
    VX_CALL(vxReleaseImage(&img40x40));
}

TEST(vxuConvertDepth, NegativeSizes_U1_)
{
    vx_image img88x88, img88x40, img40x40, img16x40, img16x16;
    vx_int32 shift_zero = 0;
    vx_int32 shift_one = 1;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(img88x88 = vxCreateImage(context, 88, 88, VX_DF_IMAGE_U1),  VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img88x40 = vxCreateImage(context, 88, 40, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img40x40 = vxCreateImage(context, 40, 40, VX_DF_IMAGE_U1),  VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img16x40 = vxCreateImage(context, 16, 40, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img16x16 = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U1),  VX_TYPE_IMAGE);

    // initialize to guarantee that images are allocated
    ASSERT_NO_FAILURE(ct_fill_image_random(img88x88, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(img88x40, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(img40x40, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(img16x40, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(img16x16, &CT()->seed_));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x88, img88x40, VX_CONVERT_POLICY_SATURATE, shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x88, img88x40, VX_CONVERT_POLICY_WRAP,     shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x88, img88x40, VX_CONVERT_POLICY_SATURATE, shift_one));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x88, img88x40, VX_CONVERT_POLICY_WRAP,     shift_one));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x40, img40x40, VX_CONVERT_POLICY_SATURATE, shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x40, img40x40, VX_CONVERT_POLICY_WRAP,     shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x40, img40x40, VX_CONVERT_POLICY_SATURATE, shift_one));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x40, img40x40, VX_CONVERT_POLICY_WRAP,     shift_one));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img40x40, img16x40, VX_CONVERT_POLICY_SATURATE, shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img40x40, img16x40, VX_CONVERT_POLICY_WRAP,     shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img40x40, img16x40, VX_CONVERT_POLICY_SATURATE, shift_one));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img40x40, img16x40, VX_CONVERT_POLICY_WRAP,     shift_one));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img16x40, img16x16, VX_CONVERT_POLICY_SATURATE, shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img16x40, img16x16, VX_CONVERT_POLICY_WRAP,     shift_zero));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img16x40, img16x16, VX_CONVERT_POLICY_SATURATE, shift_one));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img16x40, img16x16, VX_CONVERT_POLICY_WRAP,     shift_one));

    VX_CALL(vxReleaseImage(&img88x88));
    VX_CALL(vxReleaseImage(&img88x40));
    VX_CALL(vxReleaseImage(&img40x40));
    VX_CALL(vxReleaseImage(&img16x40));
    VX_CALL(vxReleaseImage(&img16x16));
}

TEST(vxConvertDepth, NegativeSizes)
{
    vx_image img88x88, img88x40, img40x40;
    vx_graph graph;
    vx_node node;
    vx_scalar shift;
    vx_int32 sh = 1;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(shift = vxCreateScalar(context, VX_TYPE_INT32, &sh), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(img88x88 = vxCreateImage(context, 88, 88, VX_DF_IMAGE_U8),  VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img88x40 = vxCreateImage(context, 88, 40, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img40x40 = vxCreateImage(context, 40, 40, VX_DF_IMAGE_U8),  VX_TYPE_IMAGE);

    /* U8 -> S16 */
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x88, img88x40, VX_CONVERT_POLICY_SATURATE, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x88, img88x40, VX_CONVERT_POLICY_WRAP, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    /* S16 -> U8 */
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x40, img40x40, VX_CONVERT_POLICY_SATURATE, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x40, img40x40, VX_CONVERT_POLICY_WRAP, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    VX_CALL(vxReleaseImage(&img88x88));
    VX_CALL(vxReleaseImage(&img88x40));
    VX_CALL(vxReleaseImage(&img40x40));
    VX_CALL(vxReleaseScalar(&shift));
}

TEST(vxConvertDepth, NegativeSizes_U1_)
{
    vx_image img88x88, img88x40, img40x40, img16x40, img16x16;
    vx_graph graph;
    vx_node node;
    vx_scalar shift;
    vx_int32 sh = 1;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(shift = vxCreateScalar(context, VX_TYPE_INT32, &sh), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(img88x88 = vxCreateImage(context, 88, 88, VX_DF_IMAGE_U1),  VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img88x40 = vxCreateImage(context, 88, 40, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img40x40 = vxCreateImage(context, 40, 40, VX_DF_IMAGE_U1),  VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img16x40 = vxCreateImage(context, 16, 40, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img16x16 = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U1),  VX_TYPE_IMAGE);

    /* U1 -> U8 */
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x88, img88x40, VX_CONVERT_POLICY_SATURATE, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x88, img88x40, VX_CONVERT_POLICY_WRAP, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    /* U8 -> U1 */
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x40, img40x40, VX_CONVERT_POLICY_SATURATE, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x40, img40x40, VX_CONVERT_POLICY_WRAP, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    /* U1 -> S16 */
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img40x40, img16x40, VX_CONVERT_POLICY_SATURATE, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img40x40, img16x40, VX_CONVERT_POLICY_WRAP, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    /* S16 -> U1 */
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img16x40, img16x16, VX_CONVERT_POLICY_SATURATE, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img16x40, img16x16, VX_CONVERT_POLICY_WRAP, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    VX_CALL(vxReleaseImage(&img88x88));
    VX_CALL(vxReleaseImage(&img88x40));
    VX_CALL(vxReleaseImage(&img40x40));
    VX_CALL(vxReleaseImage(&img16x40));
    VX_CALL(vxReleaseImage(&img16x16));
    VX_CALL(vxReleaseScalar(&shift));
}

typedef struct {
    const char* name;
    uint32_t width;
    uint32_t height;
    vx_df_image format_from;
    vx_df_image format_to;
    vx_enum policy;
} cvt_depth_arg;

#define CVT_ARG(w,h,from,to,p)    \
    ARG(        #p "/" #w "x" #h " " #from "->" #to, w, h, VX_DF_IMAGE_##from, VX_DF_IMAGE_##to, VX_CONVERT_POLICY_##p)
#define CVT_ARG_U1(w,h,from,to,p) \
    ARG("_U1_/" #p "/" #w "x" #h " " #from "->" #to, w, h, VX_DF_IMAGE_##from, VX_DF_IMAGE_##to, VX_CONVERT_POLICY_##p)

#define PREPEND_SIZE(macro, ...)                \
    CT_EXPAND(macro(1, 1, __VA_ARGS__)),        \
    CT_EXPAND(macro(15, 17, __VA_ARGS__)),      \
    CT_EXPAND(macro(32, 32, __VA_ARGS__)),      \
    CT_EXPAND(macro(640, 480, __VA_ARGS__)),    \
    CT_EXPAND(macro(1231, 1234, __VA_ARGS__))

    /*,
    CT_EXPAND(macro(1280, 720, __VA_ARGS__)),
    CT_EXPAND(macro(1920, 1080, __VA_ARGS__))*/

#define CVT_ARGS                                    \
    PREPEND_SIZE(CVT_ARG,     U8, S16, SATURATE),   \
    PREPEND_SIZE(CVT_ARG,     U8, S16, WRAP),       \
    PREPEND_SIZE(CVT_ARG,    S16,  U8, SATURATE),   \
    PREPEND_SIZE(CVT_ARG,    S16,  U8, WRAP),       \
    PREPEND_SIZE(CVT_ARG_U1,  U1,  U8, SATURATE),   \
    PREPEND_SIZE(CVT_ARG_U1,  U1,  U8, WRAP),       \
    PREPEND_SIZE(CVT_ARG_U1,  U8,  U1, SATURATE),   \
    PREPEND_SIZE(CVT_ARG_U1,  U8,  U1, WRAP),       \
    PREPEND_SIZE(CVT_ARG_U1,  U1, S16, SATURATE),   \
    PREPEND_SIZE(CVT_ARG_U1,  U1, S16, WRAP),       \
    PREPEND_SIZE(CVT_ARG_U1, S16,  U1, SATURATE),   \
    PREPEND_SIZE(CVT_ARG_U1, S16,  U1, WRAP)

TEST_WITH_ARG(vxuConvertDepth, BitExact, cvt_depth_arg, CVT_ARGS)
{
    vx_image src, dst;
    CT_Image ref_src, ref_dst, vx_dst;
    vx_int32 shift_val;
    vx_context context = context_->vx_context_;

    ASSERT_NO_FAILURE({
        ref_src = ct_allocate_image(arg_->width, arg_->height, arg_->format_from);
        fillSequence(ref_src, (uint32_t)CT()->seed_);
        src = ct_image_to_vx_image(ref_src, context);
    });

    ASSERT_VX_OBJECT(dst = vxCreateImage(context, arg_->width, arg_->height, arg_->format_to), VX_TYPE_IMAGE);

    ref_dst = ct_allocate_image(arg_->width, arg_->height, arg_->format_to);
    vx_dst = ct_allocate_image(arg_->width, arg_->height, arg_->format_to);
    for (shift_val = VALID_SHIFT_MIN; shift_val <= VALID_SHIFT_MAX; ++shift_val)
    {
        ct_update_progress(shift_val - VALID_SHIFT_MIN, VALID_SHIFT_MAX - VALID_SHIFT_MIN + 1);
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, src, dst, arg_->policy, shift_val));

        ASSERT_NO_FAILURE({
            ct_image_copyfrom_vx_image(vx_dst, dst);
            referenceConvertDepth(ref_src, ref_dst, shift_val, arg_->policy);
        });

        EXPECT_EQ_CTIMAGE(ref_dst, vx_dst);
        if (CT_HasFailure())
        {
            printf("Shift value is %d\n", shift_val);
            break;
        }
    }

    // checked release vx images
    VX_CALL(vxReleaseImage(&dst));
    VX_CALL(vxReleaseImage(&src));
    EXPECT_EQ_PTR(NULL, dst);
    EXPECT_EQ_PTR(NULL, src);
}

TEST_WITH_ARG(vxConvertDepth, BitExact, cvt_depth_arg, CVT_ARGS)
{
    vx_image src, dst;
    CT_Image ref_src, ref_dst, vx_dst;
    vx_graph graph;
    vx_node node;
    vx_scalar scalar_shift;
    vx_int32 shift = 0;
    vx_int32 tmp = 0;
    vx_context context = context_->vx_context_;

    ASSERT_NO_FAILURE({
        ref_src = ct_allocate_image(arg_->width, arg_->height, arg_->format_from);
        fillSequence(ref_src, (uint32_t)CT()->seed_);
        src = ct_image_to_vx_image(ref_src, context);
    });

    ASSERT_VX_OBJECT(dst = vxCreateImage(context, arg_->width, arg_->height, arg_->format_to), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(scalar_shift = vxCreateScalar(context, VX_TYPE_INT32, &tmp), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, src, dst, arg_->policy, scalar_shift), VX_TYPE_NODE);

    ref_dst = ct_allocate_image(arg_->width, arg_->height, arg_->format_to);
    vx_dst = ct_allocate_image(arg_->width, arg_->height, arg_->format_to);
    for (shift = VALID_SHIFT_MIN; shift <= VALID_SHIFT_MAX; ++shift)
    {
        ct_update_progress(shift - VALID_SHIFT_MIN, VALID_SHIFT_MAX - VALID_SHIFT_MIN + 1);
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCopyScalar(scalar_shift, &shift, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

        // run graph
#ifdef CT_EXECUTE_ASYNC
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxScheduleGraph(graph));
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxWaitGraph(graph));
#else
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));
#endif

        ASSERT_NO_FAILURE({
            ct_image_copyfrom_vx_image(vx_dst, dst);
            referenceConvertDepth(ref_src, ref_dst, shift, arg_->policy);
        });

        EXPECT_EQ_CTIMAGE(ref_dst, vx_dst);
        if (CT_HasFailure())
        {
            printf("Shift value is %d\n", shift);
            break;
        }
    }

    VX_CALL(vxReleaseImage(&dst));
    VX_CALL(vxReleaseImage(&src));
    VX_CALL(vxReleaseScalar(&scalar_shift));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
}

TESTCASE_TESTS(vxuConvertDepth, DISABLED_NegativeSizes, DISABLED_NegativeSizes_U1_, BitExact)
TESTCASE_TESTS(vxConvertDepth,  DISABLED_NegativeSizes, DISABLED_NegativeSizes_U1_, BitExact)

#endif //OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION
