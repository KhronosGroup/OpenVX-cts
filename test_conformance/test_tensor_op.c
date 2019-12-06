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

#include <VX/vx.h>
#include <VX/vxu.h>
#include "test_tensor_util.h"


TESTCASE(TensorOp, CT_VXContext, ct_setup_vx_context, 0)


/****************************************************************************
 *                                                                          *
 *                          Test vxTensorAddNode                            *
 *                          Test vxTensorMultiplyNode                       *
 *                          Test vxTensorSubtractNode                       *
 *                                                                          *
 ***************************************************************************/

enum TestTensorOp
{
    TT_ADD,
    TT_SUB,
    TT_MUL,
};

static void ownCheckAddSubMulResult(
        const void * in0_ptr, const vx_size * in0_dims, const vx_size * in0_strides,
        const void * in1_ptr, const vx_size * in1_dims, const vx_size * in1_strides,
        enum TestTensorDF fmt,
        enum TestTensorOp op,
        vx_size dim_num,
        vx_size out_count,
        bool wrap,  // true for WRAP, else SATURATE
        bool to_ne, // true for ROUND_TO_NE, else ROUND_TO_ZERO (only used for fmt == TT_MUL)
        vx_float32 scale,   // only used for fmt == TT_MUL
        void * out_ptr, const vx_size * out_dims, const vx_size * out_strides)
{
    double q78_scale = (double)scale / Q78_SCALE;

    for (size_t index = 0; index < out_count; ++index)
    {
        const size_t in0_byte_offset = ownGetFlatByteOffsetWithBroadcast(index, dim_num, in0_dims, in0_strides, out_dims);
        const size_t in1_byte_offset = ownGetFlatByteOffsetWithBroadcast(index, dim_num, in1_dims, in1_strides, out_dims);
        const size_t out_byte_offset = ownGetFlatByteOffset(index, dim_num, out_dims, out_strides);

        const char * in0_b_ptr = (char*)in0_ptr + in0_byte_offset;
        const char * in1_b_ptr = (char*)in1_ptr + in1_byte_offset;
        const char * out_b_ptr = (char*)out_ptr + out_byte_offset;

        switch (fmt)
        {
        case TT_Q78:
            {
                const vx_int16 in0 = *(vx_int16*)in0_b_ptr;
                const vx_int16 in1 = *(vx_int16*)in1_b_ptr;
                const vx_int16 out = *(vx_int16*)out_b_ptr;
                int16_t ref = 0;

                switch (op)
                {
                case TT_ADD:
                    {
                        int32_t tmp = in0 + in1;
                        ref = wrap ? trunc_to_int16(tmp) : CLAMP(tmp, INT16_MIN, INT16_MAX);
                    }
                    break;
                case TT_SUB:
                    {
                        int32_t tmp = in0 - in1;
                        ref = wrap ? trunc_to_int16(tmp) : CLAMP(tmp, INT16_MIN, INT16_MAX);
                    }
                    break;
                case TT_MUL:
                    {
                        double tmp = in0 * in1 * q78_scale;
                        tmp = to_ne ? nearbyint(tmp) : trunc(tmp);
                        ref = wrap ? trunc_to_int16(tmp) : CLAMP(tmp, INT16_MIN, INT16_MAX);
                    }
                    break;
                default: assert(0);
                }

                const int max_raw_int_diff = 1;
                if (I64_ABS_DIFF(ref, out) > max_raw_int_diff) {
                    printf("DIFF!!! { idx: %zu, in0: %f, in0_raw: 0x%04X, in1: %f, in1_raw: 0x%04X, out: %f, out_raw: 0x%04X, ref: %f, ref_raw: 0x%04X\n }\n",
                            index, in0 / 256.f, in0, in1 / 256.f, in1, out / 256.f, out, ref / 256.f, ref);
                    if (max_raw_int_diff)
                    {
                        if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                            ASSERT_EQ_INT(I64_ABS_DIFF(ref, out) > max_raw_int_diff, 0);
                        else
                            EXPECT_EQ_INT(I64_ABS_DIFF(ref, out) > max_raw_int_diff, 0);
                    }
                    else
                    {
                        if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                            ASSERT_EQ_INT(ref, out);
                        else
                            EXPECT_EQ_INT(ref, out);
                    }
                }
            }
            break;
        case TT_U8:
            {
                const vx_uint8 in0 = *(vx_uint8*)in0_b_ptr;
                const vx_uint8 in1 = *(vx_uint8*)in1_b_ptr;
                const vx_uint8 out = *(vx_uint8*)out_b_ptr;
                uint8_t ref = 0;

                switch (op)
                {
                case TT_ADD:
                    {
                        int32_t tmp = in0 + in1;
                        ref = wrap ? tmp : CLAMP(tmp, 0, UINT8_MAX);
                    }
                    break;
                case TT_SUB:
                    {
                        int32_t tmp = in0 - in1;
                        ref = wrap ? tmp : CLAMP(tmp, 0, UINT8_MAX);
                    }
                    break;
                case TT_MUL:
                    {
                        double tmp = in0 * in1 * scale;
                        tmp = to_ne ? nearbyint(tmp) : trunc(tmp);
                        ref = wrap ? tmp : CLAMP(tmp, 0, UINT8_MAX);
                    }
                    break;
                default: assert(0);
                }

                if (ref != out)
                    printf("DIFF!!! { idx: %zu, in0: %d, in1: %d, out: %d, ref: %d }\n", index, in0, in1, out, ref);
                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT_EQ_INT(out, ref); else EXPECT_EQ_INT(out, ref);
            }
            break;
        case TT_S8:
            {
                const vx_int8 in0 = *(vx_int8*)in0_b_ptr;
                const vx_int8 in1 = *(vx_int8*)in1_b_ptr;
                const vx_int8 out = *(vx_int8*)out_b_ptr;
                int8_t ref = 0;

                switch (op)
                {
                case TT_ADD:
                    {
                        int32_t tmp = in0 + in1;
                        ref = wrap ? trunc_to_int8(tmp) : CLAMP(tmp, INT8_MIN, INT8_MAX);
                    }
                    break;
                case TT_SUB:
                    {
                        int32_t tmp = in0 - in1;
                        ref = wrap ? trunc_to_int8(tmp) : CLAMP(tmp, INT8_MIN, INT8_MAX);
                    }
                    break;
                case TT_MUL:
                    {
                        double tmp = in0 * in1 * scale;
                        tmp = to_ne ? nearbyint(tmp) : trunc(tmp);
                        ref = wrap ? trunc_to_int8(tmp) : CLAMP(tmp, INT8_MIN, INT8_MAX);
                    }
                    break;
                default: assert(0);
                }

                if (ref != out)
                    printf("DIFF!!! { idx: %zu, in0: %d, in1: %d, out: %d, ref: %d }\n", index, in0, in1, out, ref);
                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT_EQ_INT(out, ref); else EXPECT_EQ_INT(out, ref);
            }
            break;
        default: assert(0);
        }
    }
}

typedef struct
{
    const char * name;

    enum TestTensorDF fmt;
    enum TestTensorOp op;

    enum vx_convert_policy_e convert_policy;
    enum vx_round_policy_e rounding_policy;
    vx_float32 scale;
} test_tensor_elementwise_op_arg;

#define TT_ELEMENTWISE_OP_BASE(NAME_,FMT_,OP_,OF_,ROUND_,SCALE_)                            \
    ARG(NAME_,TT_##FMT_,TT_##OP_,VX_CONVERT_POLICY_##OF_,VX_ROUND_POLICY_TO_##ROUND_,SCALE_),

#define TT_ELEMENTWISE_OP_MUL2(NAME_,FMT_,OF_,ROUND_)                           \
    TT_ELEMENTWISE_OP_BASE(NAME_"_1f",FMT_,MUL,OF_,ROUND_,1.f)                  \
    TT_ELEMENTWISE_OP_BASE(NAME_"_1_255f",FMT_,MUL,OF_,ROUND_,(1.f/255))        \
    TT_ELEMENTWISE_OP_BASE(NAME_"_1_32768f",FMT_,MUL,OF_,ROUND_,(1.f/(1<<15)))

#define TT_ELEMENTWISE_OP_MUL(NAME_,FMT_,OF_)                   \
    TT_ELEMENTWISE_OP_MUL2(NAME_"_ZERO",FMT_,OF_,ZERO)          \
    TT_ELEMENTWISE_OP_MUL2(NAME_"_NE",FMT_,OF_,NEAREST_EVEN)

#define TT_ELEMENTWISE_OP1(NAME_,FMT_,OF_)                      \
    TT_ELEMENTWISE_OP_BASE(NAME_"_ADD",FMT_,ADD,OF_,ZERO, 1)    \
    TT_ELEMENTWISE_OP_BASE(NAME_"_SUB",FMT_,SUB,OF_,ZERO, 1)    \
    TT_ELEMENTWISE_OP_MUL(NAME_"_MUL",FMT_,OF_)

#define TT_ELEMENTWISE_OP0(FMT_)                    \
    TT_ELEMENTWISE_OP1(#FMT_"_WRAP",FMT_,WRAP)      \
    TT_ELEMENTWISE_OP1(#FMT_"_SAT",FMT_,SATURATE)

#define TT_ELEMENTWISE_OP_ALL() \
    TT_ELEMENTWISE_OP0(Q78)     \
    TT_ELEMENTWISE_OP0(U8)      \
    TT_ELEMENTWISE_OP0(S8)

TEST_WITH_ARG(TensorOp, testvxTensorElementwiseOp, test_tensor_elementwise_op_arg,
        TT_ELEMENTWISE_OP_ALL()
)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF fmt = arg_->fmt;
    const enum TestTensorOp op = arg_->op;

    const enum vx_convert_policy_e overflow_policy = arg_->convert_policy;
    const enum vx_round_policy_e rounding_policy = arg_->rounding_policy;
    const vx_float32 scale = arg_->scale;

    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);
    assert(op == TT_ADD || op == TT_SUB || op == TT_MUL);
    assert(overflow_policy == VX_CONVERT_POLICY_WRAP || overflow_policy == VX_CONVERT_POLICY_SATURATE);
    assert(rounding_policy == VX_ROUND_POLICY_TO_ZERO || rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN);

    // Only MUL supports rounding_policy and scale, we chose not to allow anything but default values for other ops
    assert(TT_MUL || (rounding_policy == VX_ROUND_POLICY_TO_ZERO && scale == 1.f));

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    size_t * const in0_dims = ct_alloc_mem(sizeof(*in0_dims) * max_dims);
    size_t * const in1_dims = ct_alloc_mem(sizeof(*in1_dims) * max_dims);
    size_t * const out_dims = ct_alloc_mem(sizeof(*out_dims) * max_dims);
    ASSERT(in0_dims && in1_dims && out_dims);

    size_t * const in0_strides = ct_alloc_mem(sizeof(*in0_strides) * max_dims);
    size_t * const in1_strides = ct_alloc_mem(sizeof(*in1_strides) * max_dims);
    size_t * const out_strides = ct_alloc_mem(sizeof(*out_strides) * max_dims);
    ASSERT(in0_strides && in1_strides && out_strides);

    // The test strategy is a simple one: For each of the 1..max_dims supported
    // we test a TEST_TENSOR_NUM_ITERATIONS of random dim and broadcast
    // configurations. This approach may have issues if the implementation
    // supports a lot of dimensions since their random size being up to
    // TEST_TENSOR_MAX_DIM_SZ, could result in a huge memory requirement.
    // However from previous experience we expect this to typically be 4-6 dims.
    // The other issue is that we do not test huge dimensions.
    // Further limitations include lack of virtual/ view inputs and outputs as -
    // well as lack of modified stride testing etc.

    // Note that iter is the inner loop, so that if two implementations support
    // D1 and D2 dims resp. the same (pseudo-random) values would be used when
    // testing the common min(D1, D2) dimensions.
    for (vx_size dims = 1; dims <= max_dims; ++dims)
    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims #: %zu,\titer #: %d\n", dims, iter);
            fflush(stdout);
        }

        // First step is to get some random dim sizes, calc the strides and create the tensors.

        for (vx_size i = 0; i < dims; ++i)
        {
            const size_t new_dim = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

            const int mask0 = !!CT_RNG_NEXT_INT(rng, 0, TEST_TENSOR_INVERSE_MASK_PROBABILITY);
            const int mask1 = !!CT_RNG_NEXT_INT(rng, 0, TEST_TENSOR_INVERSE_MASK_PROBABILITY);

            // Note: Broadcasting is described as for each dim, either in0 and in1 have the same
            // size or "1" for a broadcasted value. And the output is strictly determined by them
            // so that the implementation is required to support
            // { in0, in1, out } = { 1, 5, 5 } but not { in0, in1, out } = { 1, 1, 5 }
            // even though the KHR sample implementation currently supports both.
            in0_dims[i] = mask0 ? new_dim : 1;
            in1_dims[i] = mask1 ? new_dim : 1;
            out_dims[i] = mask0 || mask1 ? new_dim : 1;

            in0_strides[i] = i ? in0_strides[i - 1] * in0_dims[i - 1] : sizeof_data_type;
            in1_strides[i] = i ? in1_strides[i - 1] * in1_dims[i - 1] : sizeof_data_type;
            out_strides[i] = i ? out_strides[i - 1] * out_dims[i - 1] : sizeof_data_type;
        }

        vx_tensor in0_tensor = vxCreateTensor(context, dims, in0_dims, data_type, fixed_point_position);
        vx_tensor in1_tensor = vxCreateTensor(context, dims, in1_dims, data_type, fixed_point_position);
        vx_tensor out_tensor = vxCreateTensor(context, dims, out_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(in0_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(in1_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        const size_t in0_bytes = in0_dims[dims - 1] * in0_strides[dims - 1];
        const size_t in1_bytes = in1_dims[dims - 1] * in1_strides[dims - 1];
        const size_t out_bytes = out_dims[dims - 1] * out_strides[dims - 1];

        const size_t in0_count = in0_bytes / sizeof_data_type;
        const size_t in1_count = in1_bytes / sizeof_data_type;
        const size_t out_count = out_bytes / sizeof_data_type;

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          dim_num: %zu,\n", dims);
            printf("\t          in0 : { dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", in0_dims[i]); } printf(" }, count: %zu, bytes: %zu },\n", in0_count, in0_bytes);
            printf("\t          in1 : { dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", in1_dims[i]); } printf(" }, count: %zu, bytes: %zu },\n", in1_count, in1_bytes);
            printf("\t          out : { dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", out_dims[i]); } printf(" }, count: %zu, bytes: %zu },\n", out_count, out_bytes);
            printf("\t        }\n");
        }

        //TODO: This is pretty wasteful as it's repeating a lot of work per iteration:
        //      Both in the repeated malloc + free and inefficient data population
        //      which discards much of the random data, only using a part of it.

        // Second step is to allocate the input and output data locations and populate the inputs.

        void * const in0_data = ct_alloc_mem(in0_bytes);
        void * const in1_data = ct_alloc_mem(in1_bytes);
        void * const out_data = ct_alloc_mem(out_bytes);
        ASSERT(in0_data && in1_data && out_data);

        {
            ownFillRandData(fmt, &rng, in0_count, in0_data);
            ownFillRandData(fmt, &rng, in1_count, in1_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(in0_tensor, dims, view_start, in0_dims, in0_strides, in0_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            VX_CALL(vxCopyTensorPatch(in1_tensor, dims, view_start, in1_dims, in1_strides, in1_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            vx_graph graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = NULL;

            switch (op)
            {
            case TT_ADD:
                node = vxTensorAddNode(graph, in0_tensor, in1_tensor, overflow_policy, out_tensor);
                break;
            case TT_SUB:
                node = vxTensorSubtractNode(graph, in0_tensor, in1_tensor, overflow_policy, out_tensor);
                break;
            case TT_MUL:
            {
                vx_scalar scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &scale);
                ASSERT_VX_OBJECT(scalar, VX_TYPE_SCALAR);

                node = vxTensorMultiplyNode(graph, in0_tensor, in1_tensor, scalar, overflow_policy, rounding_policy, out_tensor);

                VX_CALL(vxReleaseScalar(&scalar));
                EXPECT_EQ_PTR(NULL, scalar);
                break;
            }
            default:
                ASSERT(0);
                // Not implemented;
            }

            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);

            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));
            VX_CALL(vxReleaseNode(&node));
            EXPECT_EQ_PTR(NULL, node);

            VX_CALL(vxReleaseGraph(&graph));
            EXPECT_EQ_PTR(NULL, graph);
        }

        // Verify the results
        {
            const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, dims, view_start, out_dims, out_strides, out_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            ownCheckAddSubMulResult(
                    in0_data, in0_dims, in0_strides,
                    in1_data, in1_dims, in1_strides,
                    fmt,
                    op,
                    dims,
                    out_count,
                    overflow_policy == VX_CONVERT_POLICY_WRAP,
                    rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN,
                    scale,
                    out_data, out_dims, out_strides);
        }

        VX_CALL(vxReleaseTensor(&in0_tensor));
        VX_CALL(vxReleaseTensor(&in1_tensor));
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, in0_tensor);
        EXPECT_EQ_PTR(NULL, in1_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        ct_free_mem(in0_data);
        ct_free_mem(in1_data);
        ct_free_mem(out_data);
    }

    ct_free_mem(in0_dims);
    ct_free_mem(in1_dims);
    ct_free_mem(out_dims);

    ct_free_mem(in0_strides);
    ct_free_mem(in1_strides);
    ct_free_mem(out_strides);
}

TEST_WITH_ARG(TensorOp, testvxuTensorElementwiseOp, test_tensor_elementwise_op_arg,
        TT_ELEMENTWISE_OP_ALL()
)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF fmt = arg_->fmt;
    const enum TestTensorOp op = arg_->op;

    const enum vx_convert_policy_e overflow_policy = arg_->convert_policy;
    const enum vx_round_policy_e rounding_policy = arg_->rounding_policy;
    const vx_float32 scale = arg_->scale;

    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);
    assert(op == TT_ADD || op == TT_SUB || op == TT_MUL);
    assert(overflow_policy == VX_CONVERT_POLICY_WRAP || overflow_policy == VX_CONVERT_POLICY_SATURATE);
    assert(rounding_policy == VX_ROUND_POLICY_TO_ZERO || rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN);

    // Only MUL supports rounding_policy and scale, we chose not to allow anything but default values for other ops
    assert(TT_MUL || (rounding_policy == VX_ROUND_POLICY_TO_ZERO && scale == 1.f));

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    size_t * const in0_dims = ct_alloc_mem(sizeof(*in0_dims) * max_dims);
    size_t * const in1_dims = ct_alloc_mem(sizeof(*in1_dims) * max_dims);
    size_t * const out_dims = ct_alloc_mem(sizeof(*out_dims) * max_dims);
    ASSERT(in0_dims && in1_dims && out_dims);

    size_t * const in0_strides = ct_alloc_mem(sizeof(*in0_strides) * max_dims);
    size_t * const in1_strides = ct_alloc_mem(sizeof(*in1_strides) * max_dims);
    size_t * const out_strides = ct_alloc_mem(sizeof(*out_strides) * max_dims);
    ASSERT(in0_strides && in1_strides && out_strides);

    // The test strategy is a simple one: For each of the 1..max_dims supported
    // we test a TEST_TENSOR_NUM_ITERATIONS of random dim and broadcast
    // configurations. This approach may have issues if the implementation
    // supports a lot of dimensions since their random size being up to
    // TEST_TENSOR_MAX_DIM_SZ, could result in a huge memory requirement.
    // However from previous experience we expect this to typically be 4-6 dims.
    // The other issue is that we do not test huge dimensions.
    // Further limitations include lack of virtual/ view inputs and outputs as -
    // well as lack of modified stride testing etc.

    // Note that iter is the inner loop, so that if two implementations support
    // D1 and D2 dims resp. the same (pseudo-random) values would be used when
    // testing the common min(D1, D2) dimensions.
    for (vx_size dims = 1; dims <= max_dims; ++dims)
    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims #: %zu,\titer #: %d\n", dims, iter);
            fflush(stdout);
        }

        // First step is to get some random dim sizes, calc the strides and create the tensors.

        for (vx_size i = 0; i < dims; ++i)
        {
            const size_t new_dim = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

            const int mask0 = !!CT_RNG_NEXT_INT(rng, 0, TEST_TENSOR_INVERSE_MASK_PROBABILITY);
            const int mask1 = !!CT_RNG_NEXT_INT(rng, 0, TEST_TENSOR_INVERSE_MASK_PROBABILITY);

            // Note: Broadcasting is described as for each dim, either in0 and in1 have the same
            // size or "1" for a broadcasted value. And the output is strictly determined by them
            // so that the implementation is required to support
            // { in0, in1, out } = { 1, 5, 5 } but not { in0, in1, out } = { 1, 1, 5 }
            // even though the KHR sample implementation currently supports both.
            in0_dims[i] = mask0 ? new_dim : 1;
            in1_dims[i] = mask1 ? new_dim : 1;
            out_dims[i] = mask0 || mask1 ? new_dim : 1;

            in0_strides[i] = i ? in0_strides[i - 1] * in0_dims[i - 1] : sizeof_data_type;
            in1_strides[i] = i ? in1_strides[i - 1] * in1_dims[i - 1] : sizeof_data_type;
            out_strides[i] = i ? out_strides[i - 1] * out_dims[i - 1] : sizeof_data_type;
        }

        vx_tensor in0_tensor = vxCreateTensor(context, dims, in0_dims, data_type, fixed_point_position);
        vx_tensor in1_tensor = vxCreateTensor(context, dims, in1_dims, data_type, fixed_point_position);
        vx_tensor out_tensor = vxCreateTensor(context, dims, out_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(in0_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(in1_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        const size_t in0_bytes = in0_dims[dims - 1] * in0_strides[dims - 1];
        const size_t in1_bytes = in1_dims[dims - 1] * in1_strides[dims - 1];
        const size_t out_bytes = out_dims[dims - 1] * out_strides[dims - 1];

        const size_t in0_count = in0_bytes / sizeof_data_type;
        const size_t in1_count = in1_bytes / sizeof_data_type;
        const size_t out_count = out_bytes / sizeof_data_type;

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          dim_num: %zu,\n", dims);
            printf("\t          in0 : { dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", in0_dims[i]); } printf(" }, count: %zu, bytes: %zu },\n", in0_count, in0_bytes);
            printf("\t          in1 : { dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", in1_dims[i]); } printf(" }, count: %zu, bytes: %zu },\n", in1_count, in1_bytes);
            printf("\t          out : { dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", out_dims[i]); } printf(" }, count: %zu, bytes: %zu },\n", out_count, out_bytes);
            printf("\t        }\n");
        }

        //TODO: This is pretty wasteful as it's repeating a lot of work per iteration:
        //      Both in the repeated malloc + free and inefficient data population
        //      which discards much of the random data, only using a part of it.

        // Second step is to allocate the input and output data locations and populate the inputs.

        void * const in0_data = ct_alloc_mem(in0_bytes);
        void * const in1_data = ct_alloc_mem(in1_bytes);
        void * const out_data = ct_alloc_mem(out_bytes);
        ASSERT(in0_data && in1_data && out_data);

        {
            ownFillRandData(fmt, &rng, in0_count, in0_data);
            ownFillRandData(fmt, &rng, in1_count, in1_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(in0_tensor, dims, view_start, in0_dims, in0_strides, in0_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            VX_CALL(vxCopyTensorPatch(in1_tensor, dims, view_start, in1_dims, in1_strides, in1_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            switch (op)
            {
            case TT_ADD:
                VX_CALL(vxuTensorAdd(context, in0_tensor, in1_tensor, overflow_policy, out_tensor));
                break;
            case TT_SUB:
                VX_CALL(vxuTensorSubtract(context, in0_tensor, in1_tensor, overflow_policy, out_tensor));
                break;
            case TT_MUL:
            {
                vx_scalar scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &scale);
                ASSERT_VX_OBJECT(scalar, VX_TYPE_SCALAR);

                VX_CALL(vxuTensorMultiply(context, in0_tensor, in1_tensor, scalar, overflow_policy, rounding_policy, out_tensor));

                VX_CALL(vxReleaseScalar(&scalar));
                EXPECT_EQ_PTR(NULL, scalar);
                break;
            }
            default:
                ASSERT(0);
                // Not implemented;
            }
        }

        // Verify the results
        {
            const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, dims, view_start, out_dims, out_strides, out_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            ownCheckAddSubMulResult(
                    in0_data, in0_dims, in0_strides,
                    in1_data, in1_dims, in1_strides,
                    fmt,
                    op,
                    dims,
                    out_count,
                    overflow_policy == VX_CONVERT_POLICY_WRAP,
                    rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN,
                    scale,
                    out_data, out_dims, out_strides);
        }

        VX_CALL(vxReleaseTensor(&in0_tensor));
        VX_CALL(vxReleaseTensor(&in1_tensor));
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, in0_tensor);
        EXPECT_EQ_PTR(NULL, in1_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        ct_free_mem(in0_data);
        ct_free_mem(in1_data);
        ct_free_mem(out_data);
    }

    ct_free_mem(in0_dims);
    ct_free_mem(in1_dims);
    ct_free_mem(out_dims);

    ct_free_mem(in0_strides);
    ct_free_mem(in1_strides);
    ct_free_mem(out_strides);
}


/****************************************************************************
 *                                                                          *
 *                              LUT Test                                    *
 *                                                                          *
 ***************************************************************************/

static void ownFillRandDataForLUT(
        enum TestTensorDF fmt,
        uint64_t * rng,
        size_t index_count,
        size_t lut_count,
        size_t lut_offset,
        /*OUT*/ void * data)
{
    switch(fmt)
    {
        case TT_Q78:
            for(size_t i = 0; i < index_count; ++i)
                ((vx_int16*)data)[i] = (vx_int16)(CT_RNG_NEXT_INT(*rng, 0, lut_count) - lut_offset);
            break;
        case TT_U8:
            for(size_t i = 0; i < index_count; ++i)
                ((vx_uint8*)data)[i] = (vx_uint8)(CT_RNG_NEXT_INT(*rng, 0, lut_count) - lut_offset);
            break;
//        case TT_S8:
//            for(size_t i = 0; i < index_count; ++i)
//                ((vx_int8*)data)[i] = (vx_int8)(CT_RNG_NEXT_INT(*rng, 0, lut_count) - lut_offset);
//            break;
        default:
            assert(0);
    }
}

static void ownUnpackFormatForLUT(
        enum TestTensorDF fmt,
        /*OUT*/ vx_size * lut_max_count,
        /*OUT*/ vx_enum * lut_data_type)
{
    switch(fmt)
    {
        case TT_Q78:
            *lut_max_count = UINT16_MAX;
            *lut_data_type = VX_TYPE_INT16;
            break;
        case TT_U8:
//        case TT_S8:
            *lut_max_count = UINT8_MAX;
            *lut_data_type = VX_TYPE_UINT8;
            break;
        default:
            assert(0);
    }
}

//TODO: test_tensor_lut_op_arg and test_tensor_transpose_op_arg are basically identical - unify them?
typedef struct
{
    const char * name;

    enum TestTensorDF fmt;
} test_tensor_lut_op_arg;

TEST_WITH_ARG(TensorOp, testvxTensorTableLookup, test_tensor_lut_op_arg,
        ARG("Q78_TABLELOOKUP", TT_Q78),
        ARG("U8_TABLELOOKUP", TT_U8),
//        ARG("S8_TABLELOOKUP", TT_S8),
)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8);// || fmt == TT_S8);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    vx_size lut_max_count = 0;
    vx_enum lut_data_type = 0;
    ownUnpackFormatForLUT(fmt, &lut_max_count, &lut_data_type);

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);
    size_t * const tensor_strides = ct_alloc_mem(sizeof(*tensor_strides) * max_dims);
    ASSERT(tensor_dims && tensor_strides);

    // The strategy is a simple one: For each of the 1..max_dims supported,
    // we test a TEST_TENSOR_NUM_ITERATIONS of random tensor and LUT configs.
    // While LUT should be (Not verified) sufficiently tested by the Non NN
    // tests, we preffer to use random LUT dims and data each iteration anyway
    // since the whole test should't take long and this can be used as a
    // standalone conformance part for our own tests.
    //TODO: @Tomer, should we rather use a single (per fmt) randomly populated
    //      LUT, instead, anyway? Or is this acceptable?

    // Note that iter is the inner loop, so that if two implementations support
    // D1 and D2 dims resp. the same (pseudo-random) values would be used when
    // testing the common min(D1, D2) dimensions.
    for (vx_size dims = 1; dims <= max_dims; ++dims)
    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims #: %zu,\titer #: %d\n", dims, iter);
            fflush(stdout);
        }

        // First step is to get some random dim sizes, calc the strides and create the tensors.

        for (vx_size i = 0; i < dims; ++i)
        {
            tensor_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

            tensor_strides[i] = i ? tensor_strides[i-1] * tensor_dims[i-1] : sizeof_data_type;
        }

        vx_tensor src_tensor = vxCreateTensor(context, dims, tensor_dims, data_type, fixed_point_position);
        vx_tensor dst_tensor = vxCreateTensor(context, dims, tensor_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(src_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(dst_tensor, VX_TYPE_TENSOR);

        const size_t tensor_bytes = tensor_dims[dims-1] * tensor_strides[dims-1];
        const size_t tensor_count = tensor_bytes / sizeof_data_type;

        const vx_size lut_count = (vx_size)CT_RNG_NEXT_INT(rng, 1, lut_max_count+1);
        const vx_uint32 lut_offset = (lut_data_type == VX_TYPE_INT16) ? (vx_uint32)(lut_count / 2) : 0;

        vx_lut lut = vxCreateLUT(context, lut_data_type, lut_count);
        ASSERT_VX_OBJECT(lut, VX_TYPE_LUT);

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          tensor_dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", tensor_dims[i]); } printf(" }, \n");
            printf("\t          LUT_count: %zu,", lut_count);
            printf("\t        }\n");
        }

        void * const src_data = ct_alloc_mem(tensor_bytes);
        void * const dst_data = ct_alloc_mem(tensor_bytes);
        void * const lut_data = ct_alloc_mem(sizeof_data_type * lut_count);
        ASSERT(src_data && dst_data && lut_data);

        {   //TODO: ownTestInitTensors(..) ?
            ownFillRandDataForLUT(fmt, &rng, tensor_count, lut_count, lut_offset, src_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(src_tensor, dims, view_start, tensor_dims, tensor_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            for (size_t i = 0; i < lut_count; ++i)
            {
                switch (fmt)
                {
                case TT_Q78:
                    ((vx_int16*)lut_data)[i] = (vx_int16)(CT_RNG_NEXT_INT(rng, INT16_MIN, INT16_MAX + 1));
                    break;
                case TT_U8:
//                case TT_S8:
                    ((vx_uint8*)lut_data)[i] = (vx_uint8)(CT_RNG_NEXT_INT(rng, 0, UINT8_MAX + 1));
                    break;
                default: assert(0);
                }
            }

            VX_CALL(vxCopyLUT(lut, lut_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            vx_graph graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = vxTensorTableLookupNode(graph, src_tensor, lut, dst_tensor);
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
            VX_CALL(vxCopyTensorPatch(dst_tensor, dims, view_start, tensor_dims, tensor_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            for (size_t index = 0; index < tensor_count; ++index)
            {
                const size_t tensor_byte_offset = ownGetFlatByteOffset(index, dims, tensor_dims, tensor_strides);

                switch(fmt)
                {
                case TT_Q78:
                {
                    const vx_int16 res = *(vx_int16*)((char*)dst_data + tensor_byte_offset);
                    const vx_int16 val = *(vx_int16*)((char*)src_data + tensor_byte_offset);
                    const int16_t ref = *((vx_int16*)lut_data + (size_t)((int32_t)lut_offset + (int32_t)val));

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu] : %f (raw: %d), LUT[%d + %u]: %f (raw: %d), res[%zu]: %f (raw: %d) }\n",
                            tensor_byte_offset / sizeof(vx_int16), val / 256.f, val,
                            val, lut_offset, ref / 256.f, ref,
                            tensor_byte_offset / sizeof(vx_int16), res / 256.f, res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                case TT_U8:
                {
                    const vx_uint8 res = *(vx_uint8*)((char*)dst_data + tensor_byte_offset);
                    const vx_uint8 val = *(vx_uint8*)((char*)src_data + tensor_byte_offset);
                    const uint8_t ref = *((vx_uint8*)lut_data + (size_t)((int32_t)lut_offset + (int32_t)val));

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu] : %d, LUT[%d + %u]: %d, res[%zu]: %d }\n",
                            tensor_byte_offset / sizeof(vx_uint8), val,
                            val, lut_offset, ref,
                            tensor_byte_offset / sizeof(vx_uint8), res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
//                case TT_S8:
//                {
//                    const vx_int8 res = *(vx_int8*)(dst_data + tensor_byte_offset);
//                    const uint8_t val = *(uint8_t*)(src_data + tensor_byte_offset);
//                    const vx_int8 ref = *((vx_int8*)lut_data + (size_t)((int32_t)lut_offset + (int32_t)val));
//
//                    if (res != ref)
//                    {
//                        printf("DIFF!!!\t\t{ src[%zu] : %d, LUT[%d + %u]: %d, res[%zu]: %d }\n",
//                            tensor_byte_offset / sizeof(vx_int8), val,
//                            val, lut_offset, ref,
//                            tensor_byte_offset / sizeof(vx_int8), res);
//                    }
//                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
//                    {
//                        ASSERT_EQ_INT(res, ref);
//                    }
//                    else
//                    {
//                        EXPECT_EQ_INT(res, ref);
//                    }
//                }
//                break;
                default: assert(0);
                }
            }
        }

        VX_CALL(vxReleaseTensor(&src_tensor));
        VX_CALL(vxReleaseTensor(&dst_tensor));
        VX_CALL(vxReleaseLUT(&lut));

        EXPECT_EQ_PTR(NULL, src_tensor);
        EXPECT_EQ_PTR(NULL, dst_tensor);
        EXPECT_EQ_PTR(NULL, lut);

        ct_free_mem(src_data);
        ct_free_mem(dst_data);
    }

    ct_free_mem(tensor_dims);
    ct_free_mem(tensor_strides);
}

TEST_WITH_ARG(TensorOp, testvxuTensorTableLookup, test_tensor_lut_op_arg,
        ARG("Q78_TABLELOOKUP", TT_Q78),
        ARG("U8_TABLELOOKUP", TT_U8)
)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    vx_size lut_max_count = 0;
    vx_enum lut_data_type = 0;
    ownUnpackFormatForLUT(fmt, &lut_max_count, &lut_data_type);

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);
    size_t * const tensor_strides = ct_alloc_mem(sizeof(*tensor_strides) * max_dims);
    ASSERT(tensor_dims && tensor_strides);

    // The strategy is a simple one: For each of the 1..max_dims supported,
    // we test a TEST_TENSOR_NUM_ITERATIONS of random tensor and LUT configs.
    // While LUT should be (Not verified) sufficiently tested by the Non NN
    // tests, we preffer to use random LUT dims and data each iteration anyway
    // since the whole test should't take long and this can be used as a
    // standalone conformance part for our own tests.
    //TODO: @Tomer, should we rather use a single (per fmt) randomly populated
    //      LUT, instead, anyway? Or is this acceptable?

    // Note that iter is the inner loop, so that if two implementations support
    // D1 and D2 dims resp. the same (pseudo-random) values would be used when
    // testing the common min(D1, D2) dimensions.
    for (vx_size dims = 1; dims <= max_dims; ++dims)
    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims #: %zu,\titer #: %d\n", dims, iter);
            fflush(stdout);
        }

        // First step is to get some random dim sizes, calc the strides and create the tensors.

        for (vx_size i = 0; i < dims; ++i)
        {
            tensor_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

            tensor_strides[i] = i ? tensor_strides[i-1] * tensor_dims[i-1] : sizeof_data_type;
        }

        vx_tensor src_tensor = vxCreateTensor(context, dims, tensor_dims, data_type, fixed_point_position);
        vx_tensor dst_tensor = vxCreateTensor(context, dims, tensor_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(src_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(dst_tensor, VX_TYPE_TENSOR);

        const size_t tensor_bytes = tensor_dims[dims-1] * tensor_strides[dims-1];
        const size_t tensor_count = tensor_bytes / sizeof_data_type;

        const vx_size lut_count = (vx_size)CT_RNG_NEXT_INT(rng, 1, lut_max_count+1);
        const vx_uint32 lut_offset = (lut_data_type == VX_TYPE_INT16) ? (vx_uint32)(lut_count / 2) : 0;

        vx_lut lut = vxCreateLUT(context, lut_data_type, lut_count);
        ASSERT_VX_OBJECT(lut, VX_TYPE_LUT);

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          tensor_dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", tensor_dims[i]); } printf(" }, \n");
            printf("\t          LUT_count: %zu,", lut_count);
            printf("\t        }\n");
        }

        void * const src_data = ct_alloc_mem(tensor_bytes);
        void * const dst_data = ct_alloc_mem(tensor_bytes);
        void * const lut_data = ct_alloc_mem(sizeof_data_type * lut_count);
        ASSERT(src_data && dst_data && lut_data);

        {   //TODO: ownTestInitTensors(..) ?
            ownFillRandDataForLUT(fmt, &rng, tensor_count, lut_count, lut_offset, src_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(src_tensor, dims, view_start, tensor_dims, tensor_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            for (size_t i = 0; i < lut_count; ++i)
            {
                switch (fmt)
                {
                case TT_Q78:
                    ((vx_int16*)lut_data)[i] = (vx_int16)(CT_RNG_NEXT_INT(rng, INT16_MIN, INT16_MAX + 1));
                    break;
                case TT_U8:
                    ((vx_uint8*)lut_data)[i] = (vx_uint8)(CT_RNG_NEXT_INT(rng, 0, UINT8_MAX + 1));
                    break;
                default: assert(0);
                }
            }

            VX_CALL(vxCopyLUT(lut, lut_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            VX_CALL(vxuTensorTableLookup(context, src_tensor, lut, dst_tensor));
        }

        // Verify the reuslts
        {
            const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(dst_tensor, dims, view_start, tensor_dims, tensor_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            for (size_t index = 0; index < tensor_count; ++index)
            {
                const size_t tensor_byte_offset = ownGetFlatByteOffset(index, dims, tensor_dims, tensor_strides);

                switch(fmt)
                {
                case TT_Q78:
                {
                    const vx_int16 res = *(vx_int16*)((char*)dst_data + tensor_byte_offset);
                    const vx_int16 val = *(vx_int16*)((char*)src_data + tensor_byte_offset);
                    const int16_t ref = *((vx_int16*)lut_data + (size_t)((int32_t)lut_offset + (int32_t)val));

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu] : %f (raw: %d), LUT[%d + %u]: %f (raw: %d), res[%zu]: %f (raw: %d) }\n",
                            tensor_byte_offset / sizeof(vx_int16), val / 256.f, val,
                            val, lut_offset, ref / 256.f, ref,
                            tensor_byte_offset / sizeof(vx_int16), res / 256.f, res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                case TT_U8:
                {
                    const vx_uint8 res = *(vx_uint8*)((char*)dst_data + tensor_byte_offset);
                    const vx_uint8 val = *(vx_uint8*)((char*)src_data + tensor_byte_offset);
                    const uint8_t ref = *((vx_uint8*)lut_data + (size_t)((int32_t)lut_offset + (int32_t)val));

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu] : %d, LUT[%d + %u]: %d, res[%zu]: %d }\n",
                            tensor_byte_offset / sizeof(vx_uint8), val,
                            val, lut_offset, ref,
                            tensor_byte_offset / sizeof(vx_uint8), res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                default: assert(0);
                }
            }
        }

        VX_CALL(vxReleaseTensor(&src_tensor));
        VX_CALL(vxReleaseTensor(&dst_tensor));
        VX_CALL(vxReleaseLUT(&lut));

        EXPECT_EQ_PTR(NULL, src_tensor);
        EXPECT_EQ_PTR(NULL, dst_tensor);
        EXPECT_EQ_PTR(NULL, lut);

        ct_free_mem(src_data);
        ct_free_mem(dst_data);
    }

    ct_free_mem(tensor_dims);
    ct_free_mem(tensor_strides);
}


/****************************************************************************
 *                                                                          *
 *                          Test vxTensorTransposeNode                      *
 *                                                                          *
 ***************************************************************************/

typedef struct
{
    const char * name;

    enum TestTensorDF fmt;
} test_tensor_transpose_op_arg;

TEST_WITH_ARG(TensorOp, testvxTensorTranspose, test_tensor_transpose_op_arg,
        ARG("Q78_TRANSPOSE", TT_Q78),
        ARG("U8_TRANSPOSE", TT_U8),
        ARG("S8_TRANSPOSE", TT_S8),
)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    size_t * const src_dims = ct_alloc_mem(sizeof(*src_dims) * max_dims);
    size_t * const dst_dims = ct_alloc_mem(sizeof(*dst_dims) * max_dims);
    ASSERT(src_dims && dst_dims);

    //TODO: fix the following comment after its settlted :)
    // The way we implement the transposed query is simply swapping 2 of the
    // relevant dims strides in swizzled strides compared to original ones
    size_t * const src_strides = ct_alloc_mem(sizeof(*src_strides) * max_dims);
    size_t * const dst_strides = ct_alloc_mem(sizeof(*dst_strides) * max_dims);
    size_t * const ref_strides = ct_alloc_mem(sizeof(*ref_strides) * max_dims);
    ASSERT(src_strides && dst_strides && ref_strides);

    //TODO: @Tomer, should swapping a dim with itself be acceptable?

    // The strategy is a simple one: For each of the 1..max_dims supported,
    // we test all n^2 possible 2 dim combos for transposition.
    // We choose to do so since $sum_{n=1}^{max_dims} n ^2 ~ O(n^3)# which
    // isn't much for any practical number of supported dimensions.
    // An alternative method could be similar to the one used in the
    // Elementwise Op tests, where we ran TEST_TENSOR_NUM_ITERATIONS iters
    // with random 2 dim choice. But for practical values of max_dims (~6?)
    // it's hardly any different.
    //TODO: @Tomer, do you preffer the ranom approach?
    //
    // Not that we still chose to use the psuedo random data rather than
    // sequential values since the S8/U8 types would force a short repeating
    // pattern making it slightly harder to debug and possibly missing
    // perfectly aligned tranpose cases, however unlikely... :)

    // Note that iter is the inner loop, so that if two implementations support
    // D1 and D2 dims resp. the same (pseudo-random) values would be used when
    // testing the common min(D1, D2) dimensions.
    //TODO: If a single dim cannot be "transposed with itself" (copy/NOP for virt), start from 2
    for (vx_size dims = 1; dims <= max_dims; ++dims)
    for (vx_size transpose_dim0 = 0; transpose_dim0 < dims; ++transpose_dim0)
    for (vx_size transpose_dim1 = 1; transpose_dim1 < dims; ++transpose_dim1)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims: %zu, transpose_dim0: %zu, transpose_dim1: %zu\n", dims, transpose_dim0, transpose_dim1);
            fflush(stdout);
        }

        // First step is to get some random dim sizes, calc the strides and create the tensors.

        {
            for (vx_size i = 0; i < dims; ++i)
            {
                src_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
                dst_dims[i] = src_dims[i];

                src_strides[i] = i ? src_strides[i - 1] * src_dims[i - 1] : sizeof_data_type;
                ref_strides[i] = src_strides[i];
            }

            dst_dims[transpose_dim1] = src_dims[transpose_dim0];
            dst_dims[transpose_dim0] = src_dims[transpose_dim1];
            ref_strides[transpose_dim1] = src_strides[transpose_dim0];
            ref_strides[transpose_dim0] = src_strides[transpose_dim1];

            for (vx_size i = 0; i < dims; ++i)
            {
                dst_strides[i] = i ? dst_strides[i - 1] * dst_dims[i - 1] : sizeof_data_type;
            }
        }

        vx_tensor src_tensor = vxCreateTensor(context, dims, src_dims, data_type, fixed_point_position);
        vx_tensor dst_tensor = vxCreateTensor(context, dims, dst_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(src_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(dst_tensor, VX_TYPE_TENSOR);

        const size_t bytes = src_dims[dims - 1] * src_strides[dims - 1];
        const size_t count = bytes / sizeof_data_type;

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          src_dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", src_dims[i]); } printf(" },\n");
            printf("\t          dst_dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", dst_dims[i]); } printf(" },\n");
            printf("            count: %zu, bytes: %zu,\n", count, bytes);
            printf("            tensor_transpose_dims: { %zu, %zu }\n", transpose_dim0, transpose_dim1);
            printf("\t        }\n");
        }

        //TODO: This is pretty wasteful as it's repeating a lot of work per iteration:
        //      Both in the repeated malloc + free and inefficient data population
        //      which discards much of the random data, only using a part of it.

        // Second step is to allocate the input and output data locations and populate the inputs.

        void * const src_data = ct_alloc_mem(bytes);
        void * const dst_data = ct_alloc_mem(bytes);
        ASSERT(src_data && dst_data);

        {   //TODO: ownTestInitTensors(..) ?
            ownFillRandData(fmt, &rng, count, src_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(src_tensor, dims, view_start, src_dims, src_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            vx_graph graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = vxTensorTransposeNode(graph, src_tensor, dst_tensor, transpose_dim0, transpose_dim1);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxReleaseNode(&node));
            EXPECT_EQ_PTR(NULL, node);

            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));

            VX_CALL(vxReleaseGraph(&graph));
            EXPECT_EQ_PTR(NULL, graph);
        }

        // Verify the results
        {
            const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(dst_tensor, dims, view_start, dst_dims, dst_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            for (size_t index = 0; index < count; ++index)
            {
                const size_t res_byte_offset = ownGetFlatByteOffset(index, dims, dst_dims, dst_strides);
                const size_t ref_byte_offset = ownGetFlatByteOffset(index, dims, dst_dims, ref_strides);

                //TODO: can unify the following to avoid the copy pasta...

                switch(fmt)
                {
                case TT_Q78:
                {
                    const vx_int16 res = *(vx_int16*)((char*)dst_data + res_byte_offset);
                    const vx_int16 ref = *(vx_int16*)((char*)src_data + ref_byte_offset);

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu]: %f (raw: %d), dst[%zu]: %f (raw: %d) }\n",
                            ref_byte_offset / sizeof(vx_int16), ref / 256.f, ref,
                            res_byte_offset / sizeof(vx_int16), res / 256.f, res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                case TT_U8:
                {
                    const vx_uint8 res = *(vx_uint8*)((char*)dst_data + res_byte_offset);
                    const vx_uint8 ref = *(vx_uint8*)((char*)src_data + ref_byte_offset);

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu]: %d, dst[%zu]: %d }\n",
                            ref_byte_offset / sizeof(vx_uint8), ref,
                            res_byte_offset / sizeof(vx_uint8), res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                case TT_S8:
                {
                    const vx_int8 res = *(vx_int8*)((char*)dst_data + res_byte_offset);
                    const vx_int8 ref = *(vx_int8*)((char*)src_data + ref_byte_offset);

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu]: %d, dst[%zu]: %d }\n",
                            ref_byte_offset / sizeof(vx_int8), ref,
                            res_byte_offset / sizeof(vx_int8), res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                default: assert(0);
                }
            }
        }

        VX_CALL(vxReleaseTensor(&src_tensor));
        VX_CALL(vxReleaseTensor(&dst_tensor));
        EXPECT_EQ_PTR(NULL, src_tensor);
        EXPECT_EQ_PTR(NULL, dst_tensor);

        ct_free_mem(src_data);
        ct_free_mem(dst_data);
    }

    ct_free_mem(src_dims);
    ct_free_mem(dst_dims);

    ct_free_mem(src_strides);
    ct_free_mem(dst_strides);
    ct_free_mem(ref_strides);
}

TEST_WITH_ARG(TensorOp, testvxuTensorTranspose, test_tensor_transpose_op_arg,
        ARG("Q78_TRANSPOSE", TT_Q78),
        ARG("U8_TRANSPOSE", TT_U8),
        ARG("S8_TRANSPOSE", TT_S8),
)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    size_t * const src_dims = ct_alloc_mem(sizeof(*src_dims) * max_dims);
    size_t * const dst_dims = ct_alloc_mem(sizeof(*dst_dims) * max_dims);
    ASSERT(src_dims && dst_dims);

    //TODO: fix the following comment after its settlted :)
    // The way we implement the transposed query is simply swapping 2 of the
    // relevant dims strides in swizzled strides compared to original ones
    size_t * const src_strides = ct_alloc_mem(sizeof(*src_strides) * max_dims);
    size_t * const dst_strides = ct_alloc_mem(sizeof(*dst_strides) * max_dims);
    size_t * const ref_strides = ct_alloc_mem(sizeof(*ref_strides) * max_dims);
    ASSERT(src_strides && dst_strides && ref_strides);

    //TODO: @Tomer, should swapping a dim with itself be acceptable?

    // The strategy is a simple one: For each of the 1..max_dims supported,
    // we test all n^2 possible 2 dim combos for transposition.
    // We choose to do so since $sum_{n=1}^{max_dims} n ^2 ~ O(n^3)# which
    // isn't much for any practical number of supported dimensions.
    // An alternative method could be similar to the one used in the
    // Elementwise Op tests, where we ran TEST_TENSOR_NUM_ITERATIONS iters
    // with random 2 dim choice. But for practical values of max_dims (~6?)
    // it's hardly any different.
    //TODO: @Tomer, do you preffer the ranom approach?
    //
    // Not that we still chose to use the psuedo random data rather than
    // sequential values since the S8/U8 types would force a short repeating
    // pattern making it slightly harder to debug and possibly missing
    // perfectly aligned tranpose cases, however unlikely... :)

    // Note that iter is the inner loop, so that if two implementations support
    // D1 and D2 dims resp. the same (pseudo-random) values would be used when
    // testing the common min(D1, D2) dimensions.
    //TODO: If a single dim cannot be "transposed with itself" (copy/NOP for virt), start from 2
    for (vx_size dims = 1; dims <= max_dims; ++dims)
    for (vx_size transpose_dim0 = 0; transpose_dim0 < dims; ++transpose_dim0)
    for (vx_size transpose_dim1 = 1; transpose_dim1 < dims; ++transpose_dim1)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims: %zu, transpose_dim0: %zu, transpose_dim1: %zu\n", dims, transpose_dim0, transpose_dim1);
            fflush(stdout);
        }

        // First step is to get some random dim sizes, calc the strides and create the tensors.

        {
            for (vx_size i = 0; i < dims; ++i)
            {
                src_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
                dst_dims[i] = src_dims[i];

                src_strides[i] = i ? src_strides[i - 1] * src_dims[i - 1] : sizeof_data_type;
                ref_strides[i] = src_strides[i];
            }

            dst_dims[transpose_dim1] = src_dims[transpose_dim0];
            dst_dims[transpose_dim0] = src_dims[transpose_dim1];
            ref_strides[transpose_dim1] = src_strides[transpose_dim0];
            ref_strides[transpose_dim0] = src_strides[transpose_dim1];

            for (vx_size i = 0; i < dims; ++i)
            {
                dst_strides[i] = i ? dst_strides[i - 1] * dst_dims[i - 1] : sizeof_data_type;
            }
        }

        vx_tensor src_tensor = vxCreateTensor(context, dims, src_dims, data_type, fixed_point_position);
        vx_tensor dst_tensor = vxCreateTensor(context, dims, dst_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(src_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(dst_tensor, VX_TYPE_TENSOR);

        const size_t bytes = src_dims[dims - 1] * src_strides[dims - 1];
        const size_t count = bytes / sizeof_data_type;

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          src_dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", src_dims[i]); } printf(" },\n");
            printf("\t          dst_dims: { "); for (size_t i = 0; i < dims; ++i) { printf("%zu, ", dst_dims[i]); } printf(" },\n");
            printf("            count: %zu, bytes: %zu,\n", count, bytes);
            printf("            tensor_transpose_dims: { %zu, %zu }\n", transpose_dim0, transpose_dim1);
            printf("\t        }\n");
        }

        //TODO: This is pretty wasteful as it's repeating a lot of work per iteration:
        //      Both in the repeated malloc + free and inefficient data population
        //      which discards much of the random data, only using a part of it.

        // Second step is to allocate the input and output data locations and populate the inputs.

        void * const src_data = ct_alloc_mem(bytes);
        void * const dst_data = ct_alloc_mem(bytes);
        ASSERT(src_data && dst_data);

        {   //TODO: ownTestInitTensors(..) ?
            ownFillRandData(fmt, &rng, count, src_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(src_tensor, dims, view_start, src_dims, src_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            VX_CALL(vxuTensorTranspose(context, src_tensor, dst_tensor, transpose_dim0, transpose_dim1));
        }

        // Verify the results
        {
            const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(dst_tensor, dims, view_start, dst_dims, dst_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            for (size_t index = 0; index < count; ++index)
            {
                const size_t res_byte_offset = ownGetFlatByteOffset(index, dims, dst_dims, dst_strides);
                const size_t ref_byte_offset = ownGetFlatByteOffset(index, dims, dst_dims, ref_strides);

                //TODO: can unify the following to avoid the copy pasta...

                switch(fmt)
                {
                case TT_Q78:
                {
                    const vx_int16 res = *(vx_int16*)((char*)dst_data + res_byte_offset);
                    const vx_int16 ref = *(vx_int16*)((char*)src_data + ref_byte_offset);

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu]: %f (raw: %d), dst[%zu]: %f (raw: %d) }\n",
                            ref_byte_offset / sizeof(vx_int16), ref / 256.f, ref,
                            res_byte_offset / sizeof(vx_int16), res / 256.f, res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                case TT_U8:
                {
                    const vx_uint8 res = *(vx_uint8*)((char*)dst_data + res_byte_offset);
                    const vx_uint8 ref = *(vx_uint8*)((char*)src_data + ref_byte_offset);

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu]: %d, dst[%zu]: %d }\n",
                            ref_byte_offset / sizeof(vx_uint8), ref,
                            res_byte_offset / sizeof(vx_uint8), res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                case TT_S8:
                {
                    const vx_int8 res = *(vx_int8*)((char*)dst_data + res_byte_offset);
                    const vx_int8 ref = *(vx_int8*)((char*)src_data + ref_byte_offset);

                    if (res != ref)
                    {
                        printf("DIFF!!!\t\t{ src[%zu]: %d, dst[%zu]: %d }\n",
                            ref_byte_offset / sizeof(vx_int8), ref,
                            res_byte_offset / sizeof(vx_int8), res);
                    }
                    if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR)
                    {
                        ASSERT_EQ_INT(res, ref);
                    }
                    else
                    {
                        EXPECT_EQ_INT(res, ref);
                    }
                }
                break;
                default: assert(0);
                }
            }
        }

        VX_CALL(vxReleaseTensor(&src_tensor));
        VX_CALL(vxReleaseTensor(&dst_tensor));
        EXPECT_EQ_PTR(NULL, src_tensor);
        EXPECT_EQ_PTR(NULL, dst_tensor);

        ct_free_mem(src_data);
        ct_free_mem(dst_data);
    }

    ct_free_mem(src_dims);
    ct_free_mem(dst_dims);

    ct_free_mem(src_strides);
    ct_free_mem(dst_strides);
    ct_free_mem(ref_strides);
}

/****************************************************************************
 *                                                                          *
 *                          Test vxTensorConvertDepthNode                   *
 *                                                                          *
 ***************************************************************************/

typedef struct
{
    const char * name;

    enum vx_convert_policy_e policy;
    enum TestTensorDF src_fmt;
    enum TestTensorDF dst_fmt;

    float offset;
    float norm;
} test_tensor_convert_depth_op_arg;

//TODO: what kind of configs do we want to test? It doesn't have to be full width conversions
TEST_WITH_ARG(TensorOp, testvxTensorConvertDepth, test_tensor_convert_depth_op_arg,
        ARG("DEPTH_CONVERT_SAT_Q78_TO_Q78_FULL", VX_CONVERT_POLICY_SATURATE, TT_Q78, TT_Q78, 0.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_Q78_TO_U8_FULL", VX_CONVERT_POLICY_SATURATE, TT_Q78, TT_U8, 128.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_Q78_TO_S8_FULL", VX_CONVERT_POLICY_SATURATE, TT_Q78, TT_S8, 0.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_U8_TO_Q78_FULL", VX_CONVERT_POLICY_SATURATE, TT_U8, TT_Q78, -128.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_U8_TO_U8_FULL", VX_CONVERT_POLICY_SATURATE, TT_U8, TT_U8, 0.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_U8_TO_S8_FULL", VX_CONVERT_POLICY_SATURATE, TT_U8, TT_S8, -128.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_S8_TO_Q78_FULL", VX_CONVERT_POLICY_SATURATE, TT_S8, TT_Q78, 0.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_S8_TO_U8_FULL", VX_CONVERT_POLICY_SATURATE, TT_S8, TT_U8, 128.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_S8_TO_S8_FULL", VX_CONVERT_POLICY_SATURATE, TT_S8, TT_S8, 0.f, 1.f),

        ARG("DEPTH_CONVERT_WRAP_Q78_TO_Q78_FULL", VX_CONVERT_POLICY_WRAP, TT_Q78, TT_Q78, 0.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_Q78_TO_U8_FULL", VX_CONVERT_POLICY_WRAP, TT_Q78, TT_U8, 128.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_Q78_TO_S8_FULL", VX_CONVERT_POLICY_WRAP, TT_Q78, TT_S8, 0.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_U8_TO_Q78_FULL", VX_CONVERT_POLICY_WRAP, TT_U8, TT_Q78, -128.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_U8_TO_U8_FULL", VX_CONVERT_POLICY_WRAP, TT_U8, TT_U8, 0.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_U8_TO_S8_FULL", VX_CONVERT_POLICY_WRAP, TT_U8, TT_S8, -128.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_S8_TO_Q78_FULL", VX_CONVERT_POLICY_WRAP, TT_S8, TT_Q78, 0.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_S8_TO_U8_FULL", VX_CONVERT_POLICY_WRAP, TT_S8, TT_U8, 128.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_S8_TO_S8_FULL", VX_CONVERT_POLICY_WRAP, TT_S8, TT_S8, 0.f, 1.f),
)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF src_fmt = arg_->src_fmt;
    const enum TestTensorDF dst_fmt = arg_->dst_fmt;
    assert(src_fmt == TT_Q78 || src_fmt == TT_U8 || src_fmt == TT_S8);
    assert(dst_fmt == TT_Q78 || dst_fmt == TT_U8 || dst_fmt == TT_S8);

    const enum vx_convert_policy_e policy = arg_->policy;
    assert(policy == VX_CONVERT_POLICY_SATURATE || policy == VX_CONVERT_POLICY_WRAP);

    const float offset = arg_->offset;
    const float norm = arg_->norm;

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum src_data_type = 0;
    vx_enum dst_data_type = 0;
    vx_uint8 src_fixed_point_position = 0;
    vx_uint8 dst_fixed_point_position = 0;
    vx_size src_sizeof_data_type = 0;
    vx_size dst_sizeof_data_type = 0;
    ownUnpackFormat(src_fmt, &src_data_type, &src_fixed_point_position, &src_sizeof_data_type);
    ownUnpackFormat(dst_fmt, &dst_data_type, &dst_fixed_point_position, &dst_sizeof_data_type);

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);
    size_t * const src_tensor_strides = ct_alloc_mem(sizeof(*src_tensor_strides) * max_dims);
    size_t * const dst_tensor_strides = ct_alloc_mem(sizeof(*dst_tensor_strides) * max_dims);
    ASSERT(tensor_dims && src_tensor_strides && dst_tensor_strides);

    //TODO: what's the testing strategy here? missing desc.

    // Note that iter is the inner loop, so that if two implementations support
    // D1 and D2 dims resp. the same (pseudo-random) values would be used when
    // testing the common min(D1, D2) dimensions.
    for (vx_size dims = 1; dims <= max_dims; ++dims)
    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims #: %zu,\titer #: %d\n", dims, iter);
            fflush(stdout);
        }

        // First step is to get some random dim sizes, calc the strides and create the tensors.

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

        void * const src_data = ct_alloc_mem(src_tensor_bytes);
        void * const dst_data = ct_alloc_mem(dst_tensor_bytes);
        ASSERT(src_data && dst_data);

        {   //TODO: ownTestInitTensors(..) ?
            ownFillRandData(src_fmt, &rng, count, src_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(src_tensor, dims, view_start, tensor_dims, src_tensor_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            vx_graph graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_scalar norm_sc = vxCreateScalar(context, VX_TYPE_FLOAT32, &norm);
            vx_scalar offset_sc = vxCreateScalar(context, VX_TYPE_FLOAT32, &offset);
            ASSERT_VX_OBJECT(norm_sc, VX_TYPE_SCALAR);
            ASSERT_VX_OBJECT(offset_sc, VX_TYPE_SCALAR);

            vx_node node = vxTensorConvertDepthNode(graph, src_tensor, policy, norm_sc, offset_sc, dst_tensor);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxReleaseNode(&node));
            EXPECT_EQ_PTR(NULL, node);

            VX_CALL(vxReleaseScalar(&norm_sc));
            VX_CALL(vxReleaseScalar(&offset_sc));
            EXPECT_EQ_PTR(NULL, norm_sc);
            EXPECT_EQ_PTR(NULL, offset_sc);

            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));

            VX_CALL(vxReleaseGraph(&graph));
            EXPECT_EQ_PTR(NULL, graph);
        }

        // Verify the reuslts
        {
            const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(dst_tensor, dims, view_start, tensor_dims, dst_tensor_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            const float scale = 1.f / norm;
            const bool wrap = policy == VX_CONVERT_POLICY_WRAP;

            for (size_t index = 0; index < count; ++index)
            {
                const size_t src_tensor_byte_offset = ownGetFlatByteOffset(index, dims, tensor_dims, src_tensor_strides);
                const size_t dst_tensor_byte_offset = ownGetFlatByteOffset(index, dims, tensor_dims, dst_tensor_strides);

                float tmp = .0f;

                switch(src_fmt)
                {
                case TT_Q78:
                  tmp = *(vx_int16*)((char*)src_data + src_tensor_byte_offset);
                    tmp /= Q78_SCALE;
                    break;
                case TT_U8:
                    tmp = *(vx_uint8*)((char*)src_data + src_tensor_byte_offset);
                    break;
                case TT_S8:
                    tmp = *(vx_int8*)((char*)src_data + src_tensor_byte_offset);
                    break;
                default: assert(0);
                }

                tmp = (tmp - offset) * scale;

                //TODO: missing allowed eps
                //TODO: missing diff printf
                switch(dst_fmt)
                {
                case TT_Q78:
                    {
                        tmp *= Q78_SCALE;
                        vx_int16 ref = wrap ? (vx_int16)tmp : CLAMP(tmp, INT16_MIN, INT16_MAX); //TODO: cast issue?
                        vx_int16 res = *(vx_int16*)((char*)dst_data + dst_tensor_byte_offset);
                        if (res != ref) printf("DIFF!!!\n");
                        if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT_EQ_INT(res, ref); else EXPECT_EQ_INT(res, ref);
                    }
                    break;
                case TT_U8:
                    {
                        vx_uint8 ref = wrap ? (vx_uint8)tmp : CLAMP(tmp, 0, UINT8_MAX);  // CLAMP not really needed
                        vx_uint8 res = *(vx_uint8*)((char*)dst_data + dst_tensor_byte_offset);
                        if (res != ref) printf("DIFF!!!\n");
                        if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT_EQ_INT(res, ref); else EXPECT_EQ_INT(res, ref);
                    }
                    break;
                case TT_S8:
                    {
                        vx_int8 ref = wrap ? (vx_int8)tmp : (vx_int8)CLAMP(tmp, INT8_MIN, INT8_MAX); //TODO: cast issue?
                        vx_int8 res = *(vx_int8*)((char*)dst_data + dst_tensor_byte_offset);
                        if (res != ref) printf("DIFF!!!\n");
                        if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT_EQ_INT(res, ref); else EXPECT_EQ_INT(res, ref);
                    }
                    break;
                default: assert(0);
                }
            }
        }

        VX_CALL(vxReleaseTensor(&src_tensor));
        VX_CALL(vxReleaseTensor(&dst_tensor));
        EXPECT_EQ_PTR(NULL, src_tensor);
        EXPECT_EQ_PTR(NULL, dst_tensor);

        ct_free_mem(src_data);
        ct_free_mem(dst_data);
    }

    ct_free_mem(tensor_dims);
    ct_free_mem(src_tensor_strides);
    ct_free_mem(dst_tensor_strides);
}

TEST_WITH_ARG(TensorOp, testvxuTensorConvertDepth, test_tensor_convert_depth_op_arg,
        ARG("DEPTH_CONVERT_SAT_Q78_TO_Q78_FULL", VX_CONVERT_POLICY_SATURATE, TT_Q78, TT_Q78, 0.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_Q78_TO_U8_FULL", VX_CONVERT_POLICY_SATURATE, TT_Q78, TT_U8, 128.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_Q78_TO_S8_FULL", VX_CONVERT_POLICY_SATURATE, TT_Q78, TT_S8, 0.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_U8_TO_Q78_FULL", VX_CONVERT_POLICY_SATURATE, TT_U8, TT_Q78, -128.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_U8_TO_U8_FULL", VX_CONVERT_POLICY_SATURATE, TT_U8, TT_U8, 0.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_U8_TO_S8_FULL", VX_CONVERT_POLICY_SATURATE, TT_U8, TT_S8, -128.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_S8_TO_Q78_FULL", VX_CONVERT_POLICY_SATURATE, TT_S8, TT_Q78, 0.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_S8_TO_U8_FULL", VX_CONVERT_POLICY_SATURATE, TT_S8, TT_U8, 128.f, 1.f),
        ARG("DEPTH_CONVERT_SAT_S8_TO_S8_FULL", VX_CONVERT_POLICY_SATURATE, TT_S8, TT_S8, 0.f, 1.f),

        ARG("DEPTH_CONVERT_WRAP_Q78_TO_Q78_FULL", VX_CONVERT_POLICY_WRAP, TT_Q78, TT_Q78, 0.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_Q78_TO_U8_FULL", VX_CONVERT_POLICY_WRAP, TT_Q78, TT_U8, 128.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_Q78_TO_S8_FULL", VX_CONVERT_POLICY_WRAP, TT_Q78, TT_S8, 0.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_U8_TO_Q78_FULL", VX_CONVERT_POLICY_WRAP, TT_U8, TT_Q78, -128.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_U8_TO_U8_FULL", VX_CONVERT_POLICY_WRAP, TT_U8, TT_U8, 0.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_U8_TO_S8_FULL", VX_CONVERT_POLICY_WRAP, TT_U8, TT_S8, -128.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_S8_TO_Q78_FULL", VX_CONVERT_POLICY_WRAP, TT_S8, TT_Q78, 0.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_S8_TO_U8_FULL", VX_CONVERT_POLICY_WRAP, TT_S8, TT_U8, 128.f, 1.f),
        ARG("DEPTH_CONVERT_WRAP_S8_TO_S8_FULL", VX_CONVERT_POLICY_WRAP, TT_S8, TT_S8, 0.f, 1.f),
)
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF src_fmt = arg_->src_fmt;
    const enum TestTensorDF dst_fmt = arg_->dst_fmt;
    assert(src_fmt == TT_Q78 || src_fmt == TT_U8 || src_fmt == TT_S8);
    assert(dst_fmt == TT_Q78 || dst_fmt == TT_U8 || dst_fmt == TT_S8);

    const enum vx_convert_policy_e policy = arg_->policy;
    assert(policy == VX_CONVERT_POLICY_SATURATE || policy == VX_CONVERT_POLICY_WRAP);

    const float offset = arg_->offset;
    const float norm = arg_->norm;

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum src_data_type = 0;
    vx_enum dst_data_type = 0;
    vx_uint8 src_fixed_point_position = 0;
    vx_uint8 dst_fixed_point_position = 0;
    vx_size src_sizeof_data_type = 0;
    vx_size dst_sizeof_data_type = 0;
    ownUnpackFormat(src_fmt, &src_data_type, &src_fixed_point_position, &src_sizeof_data_type);
    ownUnpackFormat(dst_fmt, &dst_data_type, &dst_fixed_point_position, &dst_sizeof_data_type);

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);
    size_t * const src_tensor_strides = ct_alloc_mem(sizeof(*src_tensor_strides) * max_dims);
    size_t * const dst_tensor_strides = ct_alloc_mem(sizeof(*dst_tensor_strides) * max_dims);
    ASSERT(tensor_dims && src_tensor_strides && dst_tensor_strides);

    //TODO: what's the testing strategy here? missing desc.

    // Note that iter is the inner loop, so that if two implementations support
    // D1 and D2 dims resp. the same (pseudo-random) values would be used when
    // testing the common min(D1, D2) dimensions.
    for (vx_size dims = 1; dims <= max_dims; ++dims)
    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("dims #: %zu,\titer #: %d\n", dims, iter);
            fflush(stdout);
        }

        // First step is to get some random dim sizes, calc the strides and create the tensors.

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

        void * const src_data = ct_alloc_mem(src_tensor_bytes);
        void * const dst_data = ct_alloc_mem(dst_tensor_bytes);
        ASSERT(src_data && dst_data);

        {   //TODO: ownTestInitTensors(..) ?
            ownFillRandData(src_fmt, &rng, count, src_data);

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(src_tensor, dims, view_start, tensor_dims, src_tensor_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            vx_scalar norm_sc = vxCreateScalar(context, VX_TYPE_FLOAT32, &norm);
            vx_scalar offset_sc = vxCreateScalar(context, VX_TYPE_FLOAT32, &offset);
            ASSERT_VX_OBJECT(norm_sc, VX_TYPE_SCALAR);
            ASSERT_VX_OBJECT(offset_sc, VX_TYPE_SCALAR);

            VX_CALL(vxuTensorConvertDepth(context, src_tensor, policy, norm_sc, offset_sc, dst_tensor));

            VX_CALL(vxReleaseScalar(&norm_sc));
            VX_CALL(vxReleaseScalar(&offset_sc));
            EXPECT_EQ_PTR(NULL, norm_sc);
            EXPECT_EQ_PTR(NULL, offset_sc);
        }

        // Verify the reuslts
        {
            const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(dst_tensor, dims, view_start, tensor_dims, dst_tensor_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            const float scale = 1.f / norm;
            const bool wrap = policy == VX_CONVERT_POLICY_WRAP;

            for (size_t index = 0; index < count; ++index)
            {
                const size_t src_tensor_byte_offset = ownGetFlatByteOffset(index, dims, tensor_dims, src_tensor_strides);
                const size_t dst_tensor_byte_offset = ownGetFlatByteOffset(index, dims, tensor_dims, dst_tensor_strides);

                float tmp = .0f;

                switch(src_fmt)
                {
                case TT_Q78:
                  tmp = *(vx_int16*)((char*)src_data + src_tensor_byte_offset);
                    tmp /= Q78_SCALE;
                    break;
                case TT_U8:
                    tmp = *(vx_uint8*)((char*)src_data + src_tensor_byte_offset);
                    break;
                case TT_S8:
                    tmp = *(vx_int8*)((char*)src_data + src_tensor_byte_offset);
                    break;
                default: assert(0);
                }

                tmp = (tmp - offset) * scale;

                //TODO: missing allowed eps
                //TODO: missing diff printf
                switch(dst_fmt)
                {
                case TT_Q78:
                    {
                        tmp *= Q78_SCALE;
                        vx_int16 ref = wrap ? (vx_int16)tmp : CLAMP(tmp, INT16_MIN, INT16_MAX); //TODO: cast issue?
                        vx_int16 res = *(vx_int16*)((char*)dst_data + dst_tensor_byte_offset);
                        if (res != ref) printf("DIFF!!!\n");
                        if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT_EQ_INT(res, ref); else EXPECT_EQ_INT(res, ref);
                    }
                    break;
                case TT_U8:
                    {
                        vx_uint8 ref = wrap ? (vx_uint8)tmp : CLAMP(tmp, 0, UINT8_MAX);  // CLAMP not really needed
                        vx_uint8 res = *(vx_uint8*)((char*)dst_data + dst_tensor_byte_offset);
                        if (res != ref) printf("DIFF!!!\n");
                        if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT_EQ_INT(res, ref); else EXPECT_EQ_INT(res, ref);
                    }
                    break;
                case TT_S8:
                    {
                        vx_int8 ref = wrap ? (vx_int8)tmp : (vx_int8)CLAMP(tmp, INT8_MIN, INT8_MAX); //TODO: cast issue?
                        vx_int8 res = *(vx_int8*)((char*)dst_data + dst_tensor_byte_offset);
                        if (res != ref) printf("DIFF!!!\n");
                        if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT_EQ_INT(res, ref); else EXPECT_EQ_INT(res, ref);
                    }
                    break;
                default: assert(0);
                }
            }
        }

        VX_CALL(vxReleaseTensor(&src_tensor));
        VX_CALL(vxReleaseTensor(&dst_tensor));
        EXPECT_EQ_PTR(NULL, src_tensor);
        EXPECT_EQ_PTR(NULL, dst_tensor);

        ct_free_mem(src_data);
        ct_free_mem(dst_data);
    }

    ct_free_mem(tensor_dims);
    ct_free_mem(src_tensor_strides);
    ct_free_mem(dst_tensor_strides);
}


/****************************************************************************
 *                                                                          *
 *                     Test vxTensorMatrixMultiplyNode                      *
 *                                                                          *
 ***************************************************************************/

static void ownTensorMatrixMultiply(
        enum TestTensorDF fmt,
        const void * a_data, const vx_size * a_dims, const vx_size * a_strides, bool a_transposed,
        const void * b_data, const vx_size * b_dims, const vx_size * b_strides, bool b_transposed,
        const void * c_data, const vx_size * c_dims, const vx_size * c_strides, bool c_transposed,
        void * ref_data, const vx_size * out_dims, const vx_size * out_strides)
{
    const vx_size aa_strides[2] = { a_strides[a_transposed], a_strides[1-a_transposed] };
    const vx_size bb_strides[2] = { b_strides[b_transposed], b_strides[1-b_transposed] };
    const vx_size cc_strides[2] = { c_strides[c_transposed], c_strides[1-c_transposed] };
    size_t common_dim = a_dims[a_transposed];

    switch(fmt)
    {
        case TT_Q78:
            {
                for (size_t y = 0; y < out_dims[1]; ++y)
                for (size_t x = 0; x < out_dims[0]; ++x)
                {
                    int_fast32_t accum = 0;

                    for (size_t i = 0; i < common_dim; ++i)
                    {
                        int_fast32_t a_val = *(vx_int16*)((char*)a_data + aa_strides[1] * y + aa_strides[0] * i);
                        int_fast32_t b_val = *(vx_int16*)((char*)b_data + bb_strides[1] * i + bb_strides[0] * x);

                        accum += a_val * b_val;
                    }

                    if (c_data)
                    {
                        accum += *(vx_int16*)((char*)c_data + cc_strides[1] * y + cc_strides[0] * x) * Q78_SCALE;
                    }

                    *(vx_int16*)((char*)ref_data + out_strides[1] * y + out_strides[0] * x) =
                        CLAMP((accum + Q78_HALF) / Q78_SCALE, INT16_MIN, INT16_MAX);
                }
            }
            break;
        case TT_U8:
            {
                for (size_t y = 0; y < out_dims[1]; ++y)
                for (size_t x = 0; x < out_dims[0]; ++x)
                {
                    int_fast32_t accum = 0;

                    for (size_t i = 0; i < common_dim; ++i)
                    {
                        int_fast32_t a_val = *(vx_uint8*)((char*)a_data + aa_strides[1] * y + aa_strides[0] * i);
                        int_fast32_t b_val = *(vx_uint8*)((char*)b_data + bb_strides[1] * i + bb_strides[0] * x);

                        accum += a_val * b_val;
                    }

                    if (c_data)
                    {
                        accum += *(vx_uint8*)((char*)c_data + cc_strides[1] * y + cc_strides[0] * x);
                    }

                    *(vx_uint8*)((char*)ref_data + out_strides[1] * y + out_strides[0] * x) =
                        CLAMP(accum, 0, UINT8_MAX);
                }
            }
            break;
        case TT_S8:
            {
                for (size_t y = 0; y < out_dims[1]; ++y)
                for (size_t x = 0; x < out_dims[0]; ++x)
                {
                    int_fast32_t accum = 0;

                    for (size_t i = 0; i < common_dim; ++i)
                    {
                        int_fast32_t a_val = *(vx_int8*)((char*)a_data + aa_strides[1] * y + aa_strides[0] * i);
                        int_fast32_t b_val = *(vx_int8*)((char*)b_data + bb_strides[1] * i + bb_strides[0] * x);

                        accum += a_val * b_val;
                    }

                    if (c_data)
                    {
                        accum += *(vx_int8*)((char*)c_data + cc_strides[1] * y + cc_strides[0] * x);
                    }

                    *(vx_int8*)((char*)ref_data + out_strides[1] * y + out_strides[0] * x) =
                        CLAMP(accum, INT8_MIN, INT8_MAX);
                }
            }
            break;
        default:
            assert(0);
    }
}

typedef struct
{
    const char * name;

    enum TestTensorDF fmt;
    bool c_present;

    bool a_transposed;
    bool b_transposed;
    bool c_transposed;  // ingored unless c_present
} test_tensor_matrix_multiply_op_arg;

//TODO: what kind of configs do we want to test? It doesn't have to be full width conversions
#define TT_TENSOR_MAD_BASE(S_,FMT_,C_PRESENT_,A_T_,B_T_,C_T_)   \
    ARG("MAD_"S_, TT_ ## FMT_,C_PRESENT_,A_T_,B_T_,C_T_),

#define TT_TENSOR_MAD_B(S_,FMT_,A_,B_)                  \
    TT_TENSOR_MAD_BASE(S_,FMT_,false,A_,B_,false)       \
    TT_TENSOR_MAD_BASE(S_"_C",FMT_,true,A_,B_,false)    \
    TT_TENSOR_MAD_BASE(S_"_CT",FMT_,true,A_,B_,true)

#define TT_TENSOR_MAD_A(S_,FMT_,A_)         \
    TT_TENSOR_MAD_B(S_ "_B",FMT_,A_,false)  \
    TT_TENSOR_MAD_B(S_ "_BT",FMT_,A_,true)

#define TT_TENSOR_MAD_0(FMT_)               \
    TT_TENSOR_MAD_A(#FMT_"_A", FMT_,false)  \
    TT_TENSOR_MAD_A(#FMT_"_AT",FMT_,true)

#define TT_TENSOR_MAD_ALL() \
    TT_TENSOR_MAD_0(Q78)    \
    TT_TENSOR_MAD_0(U8)     \
    TT_TENSOR_MAD_0(S8)

TEST_WITH_ARG(TensorOp, testvxTensorMatrixMultiply, test_tensor_matrix_multiply_op_arg,
        TT_TENSOR_MAD_ALL()
)
{
    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter); fflush(stdout);
        }

        const vx_size m = (vx_size)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        const vx_size n = (vx_size)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        const vx_size k = (vx_size)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

        // Not that unlike common GEMM, here we do not update an existing c but
        // output to a different tensor!

        const vx_size a_dims[2] = { arg_->a_transposed ? m : n, arg_->a_transposed ? n : m };
        const vx_size b_dims[2] = { arg_->b_transposed ? n : k, arg_->b_transposed ? k : n };
        const vx_size c_dims[2] = { arg_->c_transposed ? m : k, arg_->c_transposed ? k : m };
        const vx_size out_dims[2] = { k, m };

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          a_dims: { %zu, %zu },\n", a_dims[0], a_dims[1]);
            printf("\t          b_dims: { %zu, %zu },\n", b_dims[0], b_dims[1]);
            if (arg_->c_present)
                printf("\t          c_dims: { %zu, %zu },\n", c_dims[0], c_dims[1]);
            printf("\t          out_dims: { %zu, %zu },\n", out_dims[0], out_dims[1]);
            printf("\t        }\n");
        }

        const vx_size a_strides[2] = { sizeof_data_type, sizeof_data_type * a_dims[0] };
        const vx_size b_strides[2] = { sizeof_data_type, sizeof_data_type * b_dims[0] };
        const vx_size c_strides[2] = { sizeof_data_type, sizeof_data_type * c_dims[0] };
        const vx_size out_strides[2] = { sizeof_data_type, sizeof_data_type * out_dims[0] };

        void * a_data = ct_alloc_mem(m * n * sizeof_data_type);
        void * b_data = ct_alloc_mem(n * k * sizeof_data_type);
        void * c_data = arg_->c_present ? ct_alloc_mem(m * k * sizeof_data_type) : NULL;
        void * out_data = ct_alloc_mem(m * k * sizeof_data_type);
        void * ref_data = ct_alloc_mem(m * k * sizeof_data_type);
        ASSERT(a_data && b_data && (!arg_->c_present || c_data) && out_data && ref_data);

        // Since we check the sum of products here, and te accumulator is only
        // supposed to be 32 bits, we need smaller values so that the intermidiate
        // results don't exceed it.
        ownFillSmallRandData(arg_->fmt, &rng, m * n, a_dims[0] + 1, a_data);
        ownFillSmallRandData(arg_->fmt, &rng, n * k, a_dims[0] + 1, b_data);
        if (arg_->c_present) { ownFillSmallRandData(arg_->fmt, &rng, m * k, a_dims[0] + 1, c_data); }

        vx_tensor a_tensor = vxCreateTensor(context_->vx_context_, 2, a_dims, data_type, fixed_point_position);
        vx_tensor b_tensor = vxCreateTensor(context_->vx_context_, 2, b_dims, data_type, fixed_point_position);
        vx_tensor c_tensor = arg_->c_present ? vxCreateTensor(context_->vx_context_, 2, c_dims, data_type, fixed_point_position) : NULL;
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, 2, out_dims, data_type, fixed_point_position);

        ASSERT_VX_OBJECT(a_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(b_tensor, VX_TYPE_TENSOR);
        if (arg_->c_present)
        {
            ASSERT_VX_OBJECT(c_tensor, VX_TYPE_TENSOR);
        }
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        vx_size view_start[2] = { 0, 0 };
        VX_CALL(vxCopyTensorPatch(a_tensor, 2, view_start, a_dims, a_strides, a_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        VX_CALL(vxCopyTensorPatch(b_tensor, 2, view_start, b_dims, b_strides, b_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        if (arg_->c_present)
        {
            VX_CALL(vxCopyTensorPatch(c_tensor, 2, view_start, c_dims, c_strides, c_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Create, run and dispose of the graph
        {
            vx_graph graph = vxCreateGraph(context_->vx_context_);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_tensor_matrix_multiply_params_t params = { arg_->a_transposed, arg_->b_transposed, arg_->c_transposed };
            vx_node node = vxTensorMatrixMultiplyNode(graph, a_tensor, b_tensor, c_tensor, &params, out_tensor);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxReleaseNode(&node));
            EXPECT_EQ_PTR(NULL, node);

            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));

            VX_CALL(vxReleaseGraph(&graph));
            EXPECT_EQ_PTR(NULL, graph);
        }

        {
            ownTensorMatrixMultiply(
                    arg_->fmt,
                    a_data, a_dims, a_strides, arg_->a_transposed,
                    b_data, b_dims, b_strides, arg_->b_transposed,
                    c_data, c_dims, c_strides, arg_->c_transposed,
                    ref_data, out_dims, out_strides);

            VX_CALL(vxCopyTensorPatch(out_tensor, 2, view_start, out_dims, out_strides, out_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out_data, out_dims, 2, out_strides,
                        ref_data, out_dims, 2, out_strides,
                        8, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out_data + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)ref_data + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&a_tensor));
        VX_CALL(vxReleaseTensor(&b_tensor));
        if (arg_->c_present) VX_CALL(vxReleaseTensor(&c_tensor));
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, a_tensor);
        EXPECT_EQ_PTR(NULL, b_tensor);
        EXPECT_EQ_PTR(NULL, c_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        ct_free_mem(a_data);
        ct_free_mem(b_data);
        ct_free_mem(c_data);
        ct_free_mem(out_data);
    }
}

TEST_WITH_ARG(TensorOp, testvxuTensorMatrixMultiply, test_tensor_matrix_multiply_op_arg,
        TT_TENSOR_MAD_ALL()
)
{
    const vx_context context = context_->vx_context_;
    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if(!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position= 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter); fflush(stdout);
        }

        const vx_size m = (vx_size)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        const vx_size n = (vx_size)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        const vx_size k = (vx_size)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

        // Not that unlike common GEMM, here we do not update an existing c but
        // output to a different tensor!

        const vx_size a_dims[2] = { arg_->a_transposed ? m : n, arg_->a_transposed ? n : m };
        const vx_size b_dims[2] = { arg_->b_transposed ? n : k, arg_->b_transposed ? k : n };
        const vx_size c_dims[2] = { arg_->c_transposed ? m : k, arg_->c_transposed ? k : m };
        const vx_size out_dims[2] = { k, m };

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          a_dims: { %zu, %zu },\n", a_dims[0], a_dims[1]);
            printf("\t          b_dims: { %zu, %zu },\n", b_dims[0], b_dims[1]);
            if (arg_->c_present)
                printf("\t          c_dims: { %zu, %zu },\n", c_dims[0], c_dims[1]);
            printf("\t          out_dims: { %zu, %zu },\n", out_dims[0], out_dims[1]);
            printf("\t        }\n");
        }

        const vx_size a_strides[2] = { sizeof_data_type, sizeof_data_type * a_dims[0] };
        const vx_size b_strides[2] = { sizeof_data_type, sizeof_data_type * b_dims[0] };
        const vx_size c_strides[2] = { sizeof_data_type, sizeof_data_type * c_dims[0] };
        const vx_size out_strides[2] = { sizeof_data_type, sizeof_data_type * out_dims[0] };

        void * a_data = ct_alloc_mem(m * n * sizeof_data_type);
        void * b_data = ct_alloc_mem(n * k * sizeof_data_type);
        void * c_data = arg_->c_present ? ct_alloc_mem(m * k * sizeof_data_type) : NULL;
        void * out_data = ct_alloc_mem(m * k * sizeof_data_type);
        void * ref_data = ct_alloc_mem(m * k * sizeof_data_type);
        ASSERT(a_data && b_data && (!arg_->c_present || c_data) && out_data && ref_data);

        // Since we check the sum of products here, and te accumulator is only
        // supposed to be 32 bits, we need smaller values so that the intermidiate
        // results don't exceed it.
        ownFillSmallRandData(arg_->fmt, &rng, m * n, a_dims[0] + 1, a_data);
        ownFillSmallRandData(arg_->fmt, &rng, n * k, a_dims[0] + 1, b_data);
        if (arg_->c_present) { ownFillSmallRandData(arg_->fmt, &rng, m * k, a_dims[0] + 1, c_data); }

        vx_tensor a_tensor = vxCreateTensor(context_->vx_context_, 2, a_dims, data_type, fixed_point_position);
        vx_tensor b_tensor = vxCreateTensor(context_->vx_context_, 2, b_dims, data_type, fixed_point_position);
        vx_tensor c_tensor = arg_->c_present ? vxCreateTensor(context_->vx_context_, 2, c_dims, data_type, fixed_point_position) : NULL;
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, 2, out_dims, data_type, fixed_point_position);

        ASSERT_VX_OBJECT(a_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(b_tensor, VX_TYPE_TENSOR);
        if (arg_->c_present)
        {
            ASSERT_VX_OBJECT(c_tensor, VX_TYPE_TENSOR);
        }
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        vx_size view_start[2] = { 0, 0 };
        VX_CALL(vxCopyTensorPatch(a_tensor, 2, view_start, a_dims, a_strides, a_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        VX_CALL(vxCopyTensorPatch(b_tensor, 2, view_start, b_dims, b_strides, b_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        if (arg_->c_present)
        {
            VX_CALL(vxCopyTensorPatch(c_tensor, 2, view_start, c_dims, c_strides, c_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Create, run vxuTensorMatrixMultiply
        {
            vx_tensor_matrix_multiply_params_t params = { arg_->a_transposed, arg_->b_transposed, arg_->c_transposed };
            VX_CALL(vxuTensorMatrixMultiply(context, a_tensor, b_tensor, c_tensor, &params, out_tensor));
        }

        {
            ownTensorMatrixMultiply(
                    arg_->fmt,
                    a_data, a_dims, a_strides, arg_->a_transposed,
                    b_data, b_dims, b_strides, arg_->b_transposed,
                    c_data, c_dims, c_strides, arg_->c_transposed,
                    ref_data, out_dims, out_strides);

            VX_CALL(vxCopyTensorPatch(out_tensor, 2, view_start, out_dims, out_strides, out_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out_data, out_dims, 2, out_strides,
                        ref_data, out_dims, 2, out_strides,
                        8, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out_data + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)ref_data + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&a_tensor));
        VX_CALL(vxReleaseTensor(&b_tensor));
        if (arg_->c_present) VX_CALL(vxReleaseTensor(&c_tensor));
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, a_tensor);
        EXPECT_EQ_PTR(NULL, b_tensor);
        EXPECT_EQ_PTR(NULL, c_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        ct_free_mem(a_data);
        ct_free_mem(b_data);
        ct_free_mem(c_data);
        ct_free_mem(out_data);
    }
}


TESTCASE_TESTS(TensorOp,
    /* vx_nodes.h function tests */
    testvxTensorElementwiseOp,
    testvxuTensorElementwiseOp,
    testvxTensorTableLookup,
    testvxuTensorTableLookup,
    testvxTensorTranspose,
    testvxuTensorTranspose,
    testvxTensorConvertDepth,
    testvxuTensorConvertDepth,
    testvxTensorMatrixMultiply,
    testvxuTensorMatrixMultiply
    /* minigraph tests */
    /*, testTensorOpSanity*/
)

#endif //OPENVX_USE_ENHANCED_VISION
