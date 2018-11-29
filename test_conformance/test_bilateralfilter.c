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

#include <VX/vx_types.h>
#include <VX/vx_khr_nn.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


#define DEBUG_TEST_TENSOR_ENABLE_PRINTF 0
#define DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR 0
#define DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS 1

#define TEST_TENSOR_NUM_ITERATIONS              10
#define TEST_TENSOR_MIN_DIM_SZ                  1
#define TEST_TENSOR_MAX_DIM_SZ                  20

#define Q78_FIXED_POINT_POSITION 8
#define MAX_TENSOR_DIMS 3

#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define MAX(a, b)   ((a) < (b) ? (b) : (a))
#define CLAMP(v, lower, upper) MAX((lower), MIN((v), (upper)))

#define MIN_TOLERANCE  (0.95)

enum TestTensorDF
{
    TT_Q78,
    TT_U8
};

static size_t ownGetFlatByteOffset(
        size_t index,
        vx_size dim_num,
        const vx_size * in_dims,
        const vx_size * in_strides)
{
    size_t res = 0;

    for (vx_size d = 0; d < dim_num; ++d)
    {
        res += in_strides[d] * (index % in_dims[d]);
        index /= in_dims[d];
    }

    return res;
}

static void ownUnpackFormat(
        enum TestTensorDF fmt,
        /*OUT*/ vx_enum * data_type,
        /*OUT*/ vx_uint8 * fixed_point_position,
        /*out*/ vx_size * sizeof_data_type)
{
    switch(fmt)
    {
        case TT_Q78:
            *data_type = VX_TYPE_INT16;
            *fixed_point_position = Q78_FIXED_POINT_POSITION;
            *sizeof_data_type = sizeof(vx_int16);
            break;
        case TT_U8:
            *data_type = VX_TYPE_UINT8;
            *fixed_point_position = 0;
            *sizeof_data_type = sizeof(vx_uint8);
            break;
        default:
            assert(0);
    }
}

static void ownFillRandData(
        enum TestTensorDF fmt,
        uint64_t * rng,
        size_t count,
        /*OUT*/ void * data)
{
    switch(fmt)
    {
        case TT_Q78:
            for(size_t i = 0; i < count; ++i)
                ((vx_int16*)data)[i] = (vx_int16)CT_RNG_NEXT_INT(*rng, INT16_MIN, INT16_MAX+1);
            break;
        case TT_U8:
            for(size_t i = 0; i < count; ++i)
                ((vx_uint8*)data)[i] = (vx_uint8)CT_RNG_NEXT_INT(*rng, 0, UINT8_MAX+1);
            break;
        default:
            assert(0);
    }
}

static void ownCheckBilateralFilterResult(
        const void * in_ptr, const vx_size * in_dims, const vx_size * in_strides,
        enum TestTensorDF fmt,
        vx_size dim_num,
        vx_size out_count,
        int   diameter,
        float sigmaSpace,
        float sigmaValues,
        void * out_ptr, const vx_size * out_dims, const vx_size * out_strides)
{
    vx_float32 tolerance = 0.0;
    vx_float32 total_num = 0.0;
    vx_float32 equal_num = 0.0;
    vx_int32 radius = diameter/2;
    vx_float32 color_weight_8[256];
    vx_float32 color_weight_16[256*256];
    vx_float32 sum = 0, wsum = 0, w = 0;
    vx_float32 gauss_color_coeff = -0.5/(sigmaValues*sigmaValues);
    vx_float32 gauss_space_coeff = -0.5/(sigmaSpace*sigmaSpace);

    vx_float32 *space_weight = (vx_float32*)malloc((radius * 2 + 1) * sizeof(vx_float32));

    for(vx_int32 i = 0; i < 256; i++)
    {
        color_weight_8[i] = (vx_float32)exp(i*i*gauss_color_coeff);
    }

    for(vx_int32 i = 0; i < 256*256; i++)
    {
        color_weight_16[i] = (vx_float32)exp(i*i*gauss_color_coeff);
    }

    for(vx_int32 i = -radius; i <= radius; i++ )
    {
        space_weight[i+radius] = (vx_float32)exp(i*i*gauss_space_coeff);
    }

    for (size_t index = radius; index < out_count-radius; ++index)
    {
        const size_t in_byte_offset = ownGetFlatByteOffset(index, dim_num, in_dims, in_strides);
        const size_t out_byte_offset = ownGetFlatByteOffset(index, dim_num, out_dims, out_strides);

        const char * in_b_ptr = (char*)in_ptr + in_byte_offset;
        const char * out_b_ptr = (char*)out_ptr + out_byte_offset;

        switch (fmt)
        {
            case TT_Q78:
            {
                const vx_int16 in = *(vx_int16*)in_b_ptr;
                const vx_int16 out = *(vx_int16*)out_b_ptr;
                int16_t ref;

                sum = 0, wsum = 0;

                for(vx_int32 j = -radius; j <= radius; j++)
                {
                    vx_size nei_byte_offset = ownGetFlatByteOffset(index + j, dim_num, in_dims, in_strides);
                    const char *nei_b_ptr = (char*)in_ptr + nei_byte_offset;
                    const vx_int16 nei = *(vx_int16*)nei_b_ptr;
                    w = space_weight[j+radius]*color_weight_16[abs(nei - in)];
                    sum += nei*w;
                    wsum += w;
                }
                ref = (vx_int16)round(sum/wsum);
                
                total_num += 1; 

                if (ref == out)
                {
                     equal_num += 1;
                } 
            }
            break;
            case TT_U8:
            {
                const vx_uint8 in = *(vx_uint8*)in_b_ptr;
                const vx_uint8 out = *(vx_uint8*)out_b_ptr;
                uint8_t ref;

                sum = 0, wsum = 0;

                for(vx_int32 j = -radius; j <= radius; j++)
                {
                    vx_size nei_byte_offset = ownGetFlatByteOffset(index + j, dim_num, in_dims, in_strides);
                    const char *nei_b_ptr = (char*)in_ptr + nei_byte_offset;
                    const vx_uint8 nei = *(vx_uint8*)nei_b_ptr;
                    w = space_weight[j+radius]*color_weight_8[abs(nei - in)];
                    sum += nei*w;
                    wsum += w;
                }
                ref = (vx_uint8)round(sum/wsum);

                total_num += 1; 

                if (ref == out)
                {
                     equal_num += 1;
                } 
            }
            break;
            default: assert(0);
        }
    }

    tolerance = (equal_num / total_num);

    free(space_weight);

    ASSERT(tolerance >= MIN_TOLERANCE);
}

/****************************************************************************
 *                                                                          *
 *                          Test vxBilateralFilterNode                      *
 *                                                                          *
 ***************************************************************************/

TESTCASE(BilateralFilter, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char * name;
    enum TestTensorDF src_fmt;
    enum TestTensorDF dst_fmt;
    int   diameter;
    float sigmaSpace;
    float sigmaValues;
} test_bilateral_filter_op_arg;

TEST_WITH_ARG(BilateralFilter, testBilateralFilterOp, test_bilateral_filter_op_arg,
        ARG("BILATERAL_FILTER_Q78", TT_Q78, TT_Q78, 5, 1, 1),
        ARG("BILATERAL_FILTER_U8", TT_U8, TT_U8, 5, 1, 1),
)
{
    const vx_context context = context_->vx_context_;
    const enum TestTensorDF src_fmt = arg_->src_fmt;
    const enum TestTensorDF dst_fmt = arg_->dst_fmt;
    assert(src_fmt == TT_Q78 || src_fmt == TT_U8);
    assert(dst_fmt == TT_Q78 || dst_fmt == TT_U8);

    const int diameter = arg_->diameter;
    const float sigmaSpace = arg_->sigmaSpace;
    const float sigmaValues = arg_->sigmaValues;

    vx_size max_dims = 0;
    {
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum src_data_type;
    vx_enum dst_data_type;
    vx_uint8 src_fixed_point_position;
    vx_uint8 dst_fixed_point_position;
    vx_size src_sizeof_data_type;
    vx_size dst_sizeof_data_type;
    ownUnpackFormat(src_fmt, &src_data_type, &src_fixed_point_position, &src_sizeof_data_type);
    ownUnpackFormat(dst_fmt, &dst_data_type, &dst_fixed_point_position, &dst_sizeof_data_type);

    size_t * const tensor_dims = malloc(sizeof(*tensor_dims) * max_dims);
    size_t * const src_tensor_strides = malloc(sizeof(*src_tensor_strides) * max_dims);
    size_t * const dst_tensor_strides = malloc(sizeof(*dst_tensor_strides) * max_dims);
    ASSERT(tensor_dims && src_tensor_strides && dst_tensor_strides);

    // The input data a vx_tensor. maximum 3 dimension and minimum 2.
    for (vx_size dims = 2; dims <= max_dims; ++dims)
    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims #: %zu,\titer #: %d\n", dims, iter);
            fflush(stdout);
        }

        for (vx_size i = 0; i < dims; ++i)
        {
            tensor_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

            src_tensor_strides[i] = i ? src_tensor_strides[i-1] * tensor_dims[i-1] : src_sizeof_data_type;
            dst_tensor_strides[i] = i ? dst_tensor_strides[i-1] * tensor_dims[i-1] : dst_sizeof_data_type;
        }

        vx_tensor src_tensor = vxCreateTensor(context, dims, tensor_dims, src_data_type, src_fixed_point_position);
        vx_tensor dst_tensor = vxCreateTensor(context, dims, tensor_dims, dst_data_type, dst_fixed_point_position);
        ASSERT_VX_OBJECT(src_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(dst_tensor, VX_TYPE_TENSOR);

        const size_t src_tensor_bytes = tensor_dims[dims-1] * src_tensor_strides[dims-1];
        const size_t dst_tensor_bytes = tensor_dims[dims-1] * dst_tensor_strides[dims-1];
        const size_t count = src_tensor_bytes / src_sizeof_data_type;

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          tensor_dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", tensor_dims[i]); } printf(" }, \n");
            printf("\t        }\n");
        }

        void * const src_data = malloc(src_tensor_bytes);
        void * const dst_data = malloc(dst_tensor_bytes);
        ASSERT(src_data && dst_data);

        {
            ownFillRandData(src_fmt, &rng, count, src_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(src_tensor, dims, view_start, tensor_dims, src_tensor_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            vx_graph graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = vxBilateralFilterNode(graph, src_tensor, diameter,  sigmaSpace, sigmaValues, dst_tensor);

            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxReleaseNode(&node));
            EXPECT_EQ_PTR(NULL, node);

            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));

            VX_CALL(vxReleaseGraph(&graph));
            EXPECT_EQ_PTR(NULL, graph);
        }

        // Verify the reuslts
        {
            const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(dst_tensor, dims, view_start, tensor_dims, dst_tensor_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            ownCheckBilateralFilterResult(
                    src_data, tensor_dims, src_tensor_strides,
                    dst_fmt,
                    dims,
                    count,
                    diameter,
                    sigmaSpace,
                    sigmaValues,
                    dst_data, tensor_dims, dst_tensor_strides);
        }

        VX_CALL(vxReleaseTensor(&src_tensor));
        VX_CALL(vxReleaseTensor(&dst_tensor));
        EXPECT_EQ_PTR(NULL, src_tensor);
        EXPECT_EQ_PTR(NULL, dst_tensor);

        free(src_data);
        free(dst_data);
    }

    free(tensor_dims);
    free(src_tensor_strides);
    free(dst_tensor_strides);
}


TEST(BilateralFilter, testNodeCreation)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF src_fmt = TT_Q78;
    const enum TestTensorDF dst_fmt = TT_Q78;

    const int diameter = 5;
    const float sigmaSpace = 1;
    const float sigmaValues = 1;

    vx_size max_dims = 0;
    {
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum src_data_type;
    vx_enum dst_data_type;
    vx_uint8 src_fixed_point_position;
    vx_uint8 dst_fixed_point_position;
    vx_size src_sizeof_data_type;
    vx_size dst_sizeof_data_type;
    ownUnpackFormat(src_fmt, &src_data_type, &src_fixed_point_position, &src_sizeof_data_type);
    ownUnpackFormat(dst_fmt, &dst_data_type, &dst_fixed_point_position, &dst_sizeof_data_type);

    size_t * const tensor_dims = malloc(sizeof(*tensor_dims) * max_dims);
    size_t * const src_tensor_strides = malloc(sizeof(*src_tensor_strides) * max_dims);
    size_t * const dst_tensor_strides = malloc(sizeof(*dst_tensor_strides) * max_dims);
    ASSERT(tensor_dims && src_tensor_strides && dst_tensor_strides);

    // The input data a vx_tensor. maximum 3 dimension and minimum 2.
    for (vx_size dims = 2; dims <= max_dims; ++dims)
    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims #: %zu,\titer #: %d\n", dims, iter);
            fflush(stdout);
        }

        for (vx_size i = 0; i < dims; ++i)
        {
            tensor_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

            src_tensor_strides[i] = i ? src_tensor_strides[i-1] * tensor_dims[i-1] : src_sizeof_data_type;
            dst_tensor_strides[i] = i ? dst_tensor_strides[i-1] * tensor_dims[i-1] : dst_sizeof_data_type;
        }

        vx_tensor src_tensor = vxCreateTensor(context, dims, tensor_dims, src_data_type, src_fixed_point_position);
        vx_tensor dst_tensor = vxCreateTensor(context, dims, tensor_dims, dst_data_type, dst_fixed_point_position);
        ASSERT_VX_OBJECT(src_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(dst_tensor, VX_TYPE_TENSOR);

        const size_t src_tensor_bytes = tensor_dims[dims-1] * src_tensor_strides[dims-1];
        const size_t dst_tensor_bytes = tensor_dims[dims-1] * dst_tensor_strides[dims-1];
        const size_t count = src_tensor_bytes / src_sizeof_data_type;

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          tensor_dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", tensor_dims[i]); } printf(" }, \n");
            printf("\t        }\n");
        }

        void * const src_data = malloc(src_tensor_bytes);
        void * const dst_data = malloc(dst_tensor_bytes);
        ASSERT(src_data && dst_data);

        {
            ownFillRandData(src_fmt, &rng, count, src_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(src_tensor, dims, view_start, tensor_dims, src_tensor_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            vx_graph graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = vxBilateralFilterNode(graph, src_tensor, diameter,  sigmaSpace, sigmaValues, dst_tensor);

            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxReleaseNode(&node));
            EXPECT_EQ_PTR(NULL, node);

            VX_CALL(vxVerifyGraph(graph));

            VX_CALL(vxReleaseGraph(&graph));
            EXPECT_EQ_PTR(NULL, graph);
        }

        VX_CALL(vxReleaseTensor(&src_tensor));
        VX_CALL(vxReleaseTensor(&dst_tensor));
        EXPECT_EQ_PTR(NULL, src_tensor);
        EXPECT_EQ_PTR(NULL, dst_tensor);

        free(src_data);
        free(dst_data);
    }

    free(tensor_dims);
    free(src_tensor_strides);
    free(dst_tensor_strides);
}

TESTCASE_TESTS(BilateralFilter,
    testNodeCreation,
    testBilateralFilterOp
);
