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

#include <VX/vx.h>
#include <VX/vxu.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static void reference_threshold(CT_Image src, CT_Image dst, vx_enum ttype, int ta, int tb, int true_val, int false_val)
{
    uint32_t x, y, width, height, srcstride, dststride;

    ASSERT(src && dst);
    ASSERT((src->format == VX_DF_IMAGE_U8 || src->format == VX_DF_IMAGE_S16) && dst->format == VX_DF_IMAGE_U8);
    ASSERT(src->width > 0 && src->height > 0 &&
           src->width == dst->width && src->height == dst->height);
    width = src->width;
    height = src->height;
    srcstride = ct_stride_bytes(src);
    dststride = ct_stride_bytes(dst);

    if(src->format == VX_DF_IMAGE_U8)
    {
        for( y = 0; y < height; y++ )
        {
            const uint8_t* srcptr = src->data.y + y*srcstride;
            uint8_t* dstptr = dst->data.y + y*dststride;
            if( ttype == VX_THRESHOLD_TYPE_BINARY )
            {
                for( x = 0; x < width; x++ )
                    dstptr[x] = srcptr[x] > ta ? true_val : false_val;
            }
            else
            {
                for( x = 0; x < width; x++ )
                    dstptr[x] = srcptr[x] < ta || srcptr[x] > tb ? false_val : true_val;
            }
        }
    }
    else
    {
        for( y = 0; y < height; y++ )
        {
            int16_t dst_value = 0;
            const int16_t* srcptr = (const int16_t*)(src->data.y + y*srcstride);
            uint8_t* dstptr = dst->data.y + y*dststride;
            if( ttype == VX_THRESHOLD_TYPE_BINARY )
            {
                for( x = 0; x < width; x++ )
                {
                    dst_value = srcptr[x] > ta ? true_val : false_val;

                    dst_value = (dst_value < 0 ? 0 : dst_value);
                    dst_value = (dst_value > UINT8_MAX ? UINT8_MAX : dst_value);
                    dstptr[x] = (uint8_t)dst_value;
                }
            }
            else
            {
                for( x = 0; x < width; x++ )
                {
                    dst_value = srcptr[x] < ta || srcptr[x] > tb ? false_val : true_val;

                    dst_value = (dst_value < 0 ? 0 : dst_value);
                    dst_value = (dst_value > UINT8_MAX ? UINT8_MAX : dst_value);
                    dstptr[x] = (uint8_t)dst_value;
                }
            }
        }
    }
}


TESTCASE(Threshold, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    int mode;
    vx_enum ttype;
    int format;
} format_arg;

#define THRESHOLD_CASE(imm, ttype, format) { #imm "/" #ttype "/" #format "/U8", CT_##imm##_MODE, VX_THRESHOLD_TYPE_##ttype, VX_DF_IMAGE_##format}

#define CT_THRESHOLD_TRUE_VALUE  255
#define CT_THRESHOLD_FALSE_VALUE 0

TEST_WITH_ARG(Threshold, testOnRandom, format_arg,
              THRESHOLD_CASE(Immediate, BINARY, U8),
              THRESHOLD_CASE(Immediate, BINARY, S16),
              THRESHOLD_CASE(Immediate, RANGE, U8),
              THRESHOLD_CASE(Immediate, RANGE, S16),
              THRESHOLD_CASE(Graph, BINARY, U8),
              THRESHOLD_CASE(Graph, BINARY, S16),
              THRESHOLD_CASE(Graph, RANGE, U8),
              THRESHOLD_CASE(Graph, RANGE, S16),
              )
{
    int format = arg_->format;
    int ttype = arg_->ttype;
    int mode = arg_->mode;
    vx_image src, dst;
    vx_threshold vxt;
    CT_Image src0, dst0, dst1;
    vx_node node = 0;
    vx_graph graph = 0;
    vx_context context = context_->vx_context_;
    int iter, niters = 100;
    uint64_t rng;
    int a = 0, b = 256;
    int true_val = CT_THRESHOLD_TRUE_VALUE;
    int false_val = CT_THRESHOLD_FALSE_VALUE;

    rng = CT()->seed_;

    for( iter = 0; iter < niters; iter++ )
    {
        int width, height;

        uint8_t _ta = CT_RNG_NEXT_INT(rng, 0, 256), _tb = CT_RNG_NEXT_INT(rng, 0, 256);
        vx_int32 ta = CT_MIN(_ta, _tb), tb = CT_MAX(_ta, _tb);

        if( ct_check_any_size() )
        {
            width = ct_roundf(ct_log_rng(&rng, 0, 10));
            height = ct_roundf(ct_log_rng(&rng, 0, 10));
            width = CT_MAX(width, 1);
            height = CT_MAX(height, 1);
        }
        else
        {
            width = 640;
            height = 480;
        }

        ct_update_progress(iter, niters);

        ASSERT_NO_FAILURE(src0 = ct_allocate_ct_image_random(width, height, format, &rng, a, b));
        if( iter % 20 == 0 )
        {
            uint8_t val = (uint8_t)CT_RNG_NEXT_INT(rng, a, b);
            ct_memset(src0->data.y, val, ct_stride_bytes(src0)*src0->height);
        }
        ASSERT_NO_FAILURE(dst0 = ct_allocate_image(width, height, VX_DF_IMAGE_U8));
        ASSERT_NO_FAILURE(reference_threshold(src0, dst0, ttype, ta, tb, true_val, false_val));

        src = ct_image_to_vx_image(src0, context);
        dst = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        ASSERT_VX_OBJECT(dst, VX_TYPE_IMAGE);
        vxt = vxCreateThresholdForImage(context,  ttype, format, VX_DF_IMAGE_U8);
        if( ttype == VX_THRESHOLD_TYPE_BINARY )
        {
              vx_pixel_value_t pa;
              pa.S32 = ta;
              ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCopyThresholdValue(vxt, &pa, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }
        else
        {
            vx_pixel_value_t pa, pb;
            pa.S32 = ta;
            pb.S32 = tb;
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCopyThresholdRange(vxt, &pa, &pb, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        vx_pixel_value_t ptrue, pfalse;
        ptrue.S32 = true_val;
        pfalse.S32 = false_val;
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCopyThresholdOutput(vxt, &ptrue, &pfalse, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

        if( mode == CT_Immediate_MODE )
        {
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuThreshold(context, src, vxt, dst));
        }
        else
        {
            graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
            node = vxThresholdNode(graph, src, vxt, dst);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            if (vxIsGraphVerified(graph))
            {
                /* do not expect graph to be in verified state before vxGraphVerify call */
                CT_FAIL("check for vxIsGraphVerified() failed\n");
            }
            VX_CALL(vxVerifyGraph(graph));
            if(!vxIsGraphVerified(graph))
            {
                /* expect graph to be in verified state before vxGraphVerify call */
                /* NB, according to the spec, vxProcessGraph may also do graph verification */
                CT_FAIL("check for vxIsGraphVerified() failed\n");
            }
            VX_CALL(vxProcessGraph(graph));
        }

        dst1 = ct_image_from_vx_image(dst);

        ASSERT_CTIMAGE_NEAR(dst0, dst1, 0);
        VX_CALL(vxReleaseImage(&src));
        VX_CALL(vxReleaseImage(&dst));
        VX_CALL(vxReleaseThreshold(&vxt));
        if(node)
            VX_CALL(vxReleaseNode(&node));
        if(graph)
            VX_CALL(vxReleaseGraph(&graph));
        ASSERT(node == 0 && graph == 0);
        CT_CollectGarbage(CT_GC_IMAGE);
    }
}


typedef struct {
    const char* name;
    vx_enum src_type;
    vx_enum dst_type;
} threshold_create_arg;

#define THRESHOLD_CREATE(src_type, dst_type) {#src_type "/" #dst_type, VX_DF_IMAGE_##src_type,  VX_DF_IMAGE_##dst_type}

TEST_WITH_ARG(Threshold, testThresholdCreation, threshold_create_arg,
              THRESHOLD_CREATE(U8, U8),
              THRESHOLD_CREATE(S16, U8),
              )
{
    vx_context context = context_->vx_context_;
    vx_enum thresh_type = VX_THRESHOLD_TYPE_BINARY;

    vx_threshold threshold;
    ASSERT_VX_OBJECT(threshold = vxCreateThresholdForImage(context, thresh_type, arg_->src_type, arg_->dst_type), VX_TYPE_THRESHOLD);

    vx_df_image input_type, output_type;
    vxQueryThreshold(threshold,  VX_THRESHOLD_INPUT_FORMAT, &input_type, sizeof(vx_df_image));
    vxQueryThreshold(threshold,  VX_THRESHOLD_OUTPUT_FORMAT, &output_type, sizeof(vx_df_image));
    ASSERT_EQ_INT(arg_->src_type, input_type);
    ASSERT_EQ_INT(arg_->dst_type, output_type);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxReleaseThreshold(&threshold));
}

TEST_WITH_ARG(Threshold, testVirtualThresholdCreation, threshold_create_arg,
              THRESHOLD_CREATE(U8, U8),
              THRESHOLD_CREATE(S16, U8),
              )
{
    vx_context context = context_->vx_context_;
    vx_enum thresh_type = VX_THRESHOLD_TYPE_BINARY;
    vx_graph graph = vxCreateGraph(context);

    vx_threshold threshold;
    ASSERT_VX_OBJECT(threshold = vxCreateVirtualThresholdForImage(graph, thresh_type, arg_->src_type, arg_->dst_type), VX_TYPE_THRESHOLD);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxReleaseThreshold(&threshold));

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxReleaseGraph(&graph));
}

TESTCASE_TESTS(Threshold,
               testThresholdCreation,
               testVirtualThresholdCreation,
               testOnRandom
               )
