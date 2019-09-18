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

#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"

#ifdef _MSC_VER
#define ONE_255 (1.0f/255)
#else
#define ONE_255 0x1.010102p-8f
#endif
#define ONE_2_0 1.0f
#define ONE_2_1 (1.0f/(1<<1))
#define ONE_2_2 (1.0f/(1<<2))
#define ONE_2_3 (1.0f/(1<<3))
#define ONE_2_4 (1.0f/(1<<4))
#define ONE_2_5 (1.0f/(1<<5))
#define ONE_2_6 (1.0f/(1<<6))
#define ONE_2_7 (1.0f/(1<<7))
#define ONE_2_8 (1.0f/(1<<8))
#define ONE_2_9 (1.0f/(1<<9))
#define ONE_2_10 (1.0f/(1<<10))
#define ONE_2_11 (1.0f/(1<<11))
#define ONE_2_12 (1.0f/(1<<12))
#define ONE_2_13 (1.0f/(1<<13))
#define ONE_2_14 (1.0f/(1<<14))
#define ONE_2_15 (1.0f/(1<<15))

#define ONE_255_STR "(1/255)"
#define ONE_2_0_STR "(1/2^0)"
#define ONE_2_1_STR "(1/2^1)"
#define ONE_2_2_STR "(1/2^2)"
#define ONE_2_3_STR "(1/2^3)"
#define ONE_2_4_STR "(1/2^4)"
#define ONE_2_5_STR "(1/2^5)"
#define ONE_2_6_STR "(1/2^6)"
#define ONE_2_7_STR "(1/2^7)"
#define ONE_2_8_STR "(1/2^8)"
#define ONE_2_9_STR "(1/2^9)"
#define ONE_2_10_STR "(1/2^10)"
#define ONE_2_11_STR "(1/2^11)"
#define ONE_2_12_STR "(1/2^12)"
#define ONE_2_13_STR "(1/2^13)"
#define ONE_2_14_STR "(1/2^14)"
#define ONE_2_15_STR "(1/2^15)"

static void referenceWeightedAverage(CT_Image src0, CT_Image src1, vx_float32 scale, CT_Image dst)
{
    uint32_t i, j;
    ASSERT(src0 && src1 && dst);
    ASSERT(src0->width == src1->width  && src0->width == dst->width);
    ASSERT(src0->height == src1->height && src0->height == dst->height);

#define WEIGHTED_AVERAGE_LOOP(s0, s1, r)                                                                        \
    do{                                                                                                         \
        for (i = 0; i < dst->height; ++i)                                                                       \
            for (j = 0; j < dst->width; ++j)                                                                    \
            {                                                                                                   \
                vx_int32 val0 = (vx_uint8)src0->data.s0[i * src0->stride + j];                                  \
                vx_int32 val1 = (vx_uint8)src1->data.s1[i * src1->stride + j];                                  \
                vx_int32 res0 = (vx_int32)((1 - scale)* (vx_float32)(val1)+ scale * (vx_float32)(val0)) ;       \
                dst->data.r[i * dst->stride + j] = (vx_uint8)res0;                                              \
            }                                                                                                   \
    }while(0)

    if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_U8)
        WEIGHTED_AVERAGE_LOOP(y, y, y);
    else
        FAIL("Unsupported combination of argument formats: %.4s + %.4s = %.4s", &src0->format, &src1->format, &dst->format);

#undef WEIGHTED_AVERAGE_LOOP
}

typedef struct {
    const char* name;
    vx_df_image format;
    int width, height;
    vx_float32 scale;
} fuzzy_arg;

#define FUZZY_ARG(owp, w, h, scale)                  \
    ARG(#owp "/" #w "x" #h " " scale##_STR "=" ,     \
        VX_DF_IMAGE_##owp, w, h, scale)

#define APPEND_SCALE(macro, ...)                       \
    CT_EXPAND(macro(__VA_ARGS__, ONE_255)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_0)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_1)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_2)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_3)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_4)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_5)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_6)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_7)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_8)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_9)),            \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_10)),           \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_11)),           \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_12)),           \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_13)),           \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_14)),           \
    CT_EXPAND(macro(__VA_ARGS__, ONE_2_15))

#define WEIGHTED_AVERAGE_TEST_CASE(owp)         \
    APPEND_SCALE(FUZZY_ARG, owp, 640, 480),     \
    APPEND_SCALE(FUZZY_ARG, owp, 15, 15),       \
    APPEND_SCALE(FUZZY_ARG, owp, 320, 240)

TESTCASE(WeightedAverage, CT_VXContext, ct_setup_vx_context, 0)

TEST_WITH_ARG(WeightedAverage, testvxWeightedAverage, fuzzy_arg,
    WEIGHTED_AVERAGE_TEST_CASE(U8))
{
    int format = arg_->format;
    vx_scalar scale = 0;
    vx_image src_in0;
    vx_image src_in1;
    vx_image out;
    CT_Image ref1, ref2, vxout, refdst;
    CT_Image output = NULL;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(out = vxCreateImage(context, arg_->width, arg_->height, format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &arg_->scale), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(src_in0 = vxCreateImage(context, arg_->width, arg_->height, format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src_in1 = vxCreateImage(context, arg_->width, arg_->height, format), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(ct_fill_image_random(src_in0, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src_in1, &CT()->seed_));

    ASSERT_VX_OBJECT(vxWeightedAverageImageNode(graph, src_in0, scale, src_in1, out), VX_TYPE_NODE);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));

    ref1 = ct_image_from_vx_image(src_in0);
    ref2 = ct_image_from_vx_image(src_in1);
    vxout = ct_image_from_vx_image(out);
    refdst = ct_allocate_image(arg_->width, arg_->height, format);

    referenceWeightedAverage(ref1, ref2, arg_->scale, refdst);
    
    EXPECT_EQ_CTIMAGE(refdst, vxout);

    if (node)
        VX_CALL(vxReleaseNode(&node));
    if (graph)
        VX_CALL(vxReleaseGraph(&graph));
    ASSERT(node == 0 && graph == 0);
    VX_CALL(vxReleaseImage(&src_in0));
    VX_CALL(vxReleaseImage(&src_in1));
    VX_CALL(vxReleaseScalar(&scale));
    VX_CALL(vxReleaseImage(&out));
}

TEST_WITH_ARG(WeightedAverage, testvxuWeightedAverage, fuzzy_arg,
    WEIGHTED_AVERAGE_TEST_CASE(U8))
{
    int format = arg_->format;
    vx_scalar scale = 0;
    vx_image src_in0;
    vx_image src_in1;
    vx_image out;
    CT_Image ref1, ref2, vxout, refdst;
    CT_Image output = NULL;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(out = vxCreateImage(context, arg_->width, arg_->height, format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &arg_->scale), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(src_in0 = vxCreateImage(context, arg_->width, arg_->height, format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src_in1 = vxCreateImage(context, arg_->width, arg_->height, format), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(ct_fill_image_random(src_in0, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src_in1, &CT()->seed_));

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuWeightedAverageImage(context, src_in0, scale, src_in1, out));

    ref1 = ct_image_from_vx_image(src_in0);
    ref2 = ct_image_from_vx_image(src_in1);
    vxout = ct_image_from_vx_image(out);
    refdst = ct_allocate_image(arg_->width, arg_->height, format);

    referenceWeightedAverage(ref1, ref2, arg_->scale, refdst);

    EXPECT_EQ_CTIMAGE(refdst, vxout);

    if (node)
        VX_CALL(vxReleaseNode(&node));
    if (graph)
        VX_CALL(vxReleaseGraph(&graph));
    ASSERT(node == 0 && graph == 0);
    VX_CALL(vxReleaseImage(&src_in0));
    VX_CALL(vxReleaseImage(&src_in1));
    VX_CALL(vxReleaseScalar(&scale));
    VX_CALL(vxReleaseImage(&out));
}

TESTCASE_TESTS(WeightedAverage, testvxWeightedAverage, testvxuWeightedAverage)
