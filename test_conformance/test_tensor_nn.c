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

#ifdef OPENVX_USE_NN

#include "test_tensor_util.h"

TESTCASE(TensorNN, CT_VXContext, ct_setup_vx_context, 0)


/****************************************************************************
 *                                                                          *
 *                          Test vxConvolutionLayer                         *
 *                                                                          *
 ***************************************************************************/

static void ownGetConvPoolRandParams(
        uint64_t * rng,
        size_t pad_sz, size_t kernel_sz,
        size_t dilation, // 0 in pooling
        bool use_ceil,
        /*OUT*/ size_t * input_sz,
        /*OUT*/ size_t * stride,
        /*OUT*/ size_t * output_sz)
{
    const int min_input = MAX(kernel_sz + (kernel_sz - 1) * dilation - 2 * pad_sz, 1);
    const int max_input = MIN(min_input, TEST_TENSOR_MAX_DIM_SZ) + 5;
    *input_sz = (size_t)CT_RNG_NEXT_INT(*rng, min_input, max_input);

    const size_t stride_candidate = (size_t)CT_RNG_NEXT_INT(*rng, 1, 4);
    const size_t round_addition = use_ceil ? stride_candidate - 1 : 0;
    const size_t numerator = *input_sz + 2 * pad_sz - (kernel_sz + (kernel_sz - 1) * dilation);
    *output_sz = (numerator + round_addition) / stride_candidate + 1;

    // There's an ambiguity in the stride determination for ex.
    //      2 = (6 + 2 * 0 - 1) / stride + 1
    // would be correct with a stride of 3, 4 or, 5.
    // We therefore pick the smallest stride satisying the equation.
    size_t t = stride_candidate;
    while (t > 1 && (numerator + (use_ceil ? t - 2 : 0)) / (t - 1) + 1 == *output_sz) --t;
    *stride = t;
}

static void ownConvolution(
        enum TestTensorDF fmt,
        const void * input_ptr, tensor_desc_t input,
        const void * weight_ptr, tensor_desc_t weight,
        const void * bias_ptr, tensor_desc_t bias,
        vx_size pad_x, vx_size pad_y,
        vx_size stride_x, vx_size stride_y,
        bool wrap,  // true for WRAP, else SATURATE
        bool to_ne, // true for ROUND_TO_NE, else ROUND_TO_ZERO
        vx_size dilation_x, vx_size dilation_y,
        void * output_ptr, tensor_desc_t output)
{
    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    assert(input.dim_num == 3 || input.dim_num == 4);
    assert(weight.dim_num == 4);
    assert(bias.dim_num == 0 || bias.dim_num == 1 || bias.dim_num == 3);
    assert(output.dim_num == input.dim_num);

    const size_t input_w = input.dims[0];
    const size_t input_h = input.dims[1];
    const size_t input_c = input.dims[2];
    const size_t input_b = input.dim_num > 3 ? input.dims[3] : 1;

    const size_t weight_w = weight.dims[0];
    const size_t weight_h = weight.dims[1];
    const size_t weight_ifm = weight.dims[2];
    const size_t weight_ofm = weight.dims[3];

    const bool bias_present = !!bias.dim_num;
    const bool bias_shared = bias.dim_num == 1;
    const size_t bias_w = bias.dim_num > 0 ? bias.dims[0] : 0;
    const size_t bias_h = bias.dim_num > 1 ? bias.dims[1] : 1;
    const size_t bias_ofm = bias.dim_num > 2 ? bias.dims[2] : 1;

    const size_t output_w = output.dims[0];
    const size_t output_h = output.dims[1];
    const size_t output_c = output.dims[2];
    const size_t output_b = output.dim_num > 3 ? output.dims[3] : 1;

    assert(weight_w + (weight_w - 1) * dilation_x <= input_w + 2 * pad_x);
    assert(weight_h + (weight_h - 1) * dilation_y <= input_h + 2 * pad_y);
    assert(weight_ifm == input_c);
    assert(weight_ofm == output_c);

    if (bias_shared)
    {
        assert(bias_w == weight_ofm);
    }
    else if (bias_present)
    {
        assert(bias_w == output_w);
        assert(bias_h == output_h);
        assert(bias_ofm == output_c);
    }

    assert(output_b == input_b);

    ownAssertStridesModSizeof(fmt, input);
    ownAssertStridesModSizeof(fmt, weight);
    ownAssertStridesModSizeof(fmt, bias);
    ownAssertStridesModSizeof(fmt, output);

    // Input and output pointers for the current batch being processed,
    // Note: The compiler should've been able to hoist this out... And
    // there's a bunch of other possible hoising iopportunities here.
    const char * in_b_ptr = input_ptr;
    char * out_b_ptr = output_ptr;

    for (size_t b = 0; b < output_b; ++b)
    for (size_t ofm = 0; ofm < output_c; ++ofm)
    for (size_t y = 0; y < output_h; ++y)
    for (size_t x = 0; x < output_w; ++x)
    {
        int32_t sum = 0;
        if (bias_present)
        {
            const size_t bias_byte_offset =
                bias_shared
                ? (bias.strides[0] * ofm)
                : (bias.strides[2] * ofm + bias.strides[1] * y + bias.strides[0] * x);

            sum = ownLoadValueAsRawInt(fmt, (char *)bias_ptr + bias_byte_offset);
        }
        
        const size_t xx = x * stride_x;
        const size_t yy = y * stride_y;

        for (size_t ifm = 0; ifm < input_c; ++ifm)
        {
            for (size_t w_y = 0; w_y < weight_h; ++w_y)
            for (size_t w_x = 0; w_x < weight_w; ++w_x)
            {
                const size_t tmp_x = xx + w_x * (dilation_x + 1) + dilation_x;
                const size_t tmp_y = yy + w_y * (dilation_y + 1) + dilation_y;

                if (tmp_x >= pad_x && tmp_x < input_w + pad_x &&
                    tmp_y >= pad_y && tmp_y < input_h + pad_y)
                {
                    const size_t input_byte_offset =
                        (b ? input.strides[3] * b : 0) +
                        input.strides[2] * ifm +
                        input.strides[1] * (tmp_y - pad_y) +
                        input.strides[0] * (tmp_x - pad_x);
                    const size_t weight_byte_offset =
                        weight.strides[3] * ofm +
                        weight.strides[2] * ifm +
                        weight.strides[1] * w_y +
                        weight.strides[0] * w_x;

                    const int_fast32_t i_val = ownLoadValueAsRawInt(fmt, in_b_ptr + input_byte_offset);
                    const int_fast32_t w_val = ownLoadValueAsRawInt(fmt, (char *)weight_ptr + weight_byte_offset);

                    // This is ok since all of them fit into int32_t
                    sum = ownApplyWrapRoundingToAccum(fmt, i_val * w_val, wrap, to_ne) + sum;
                }
            }
            sum = ownWrapOrSat(fmt, sum, wrap);
        }

        // The step here could be added to the loops instead of recalcing
        // if, but does the compiler fail to hoist them out???
        const size_t output_byte_offset =
            (b ? output.strides[3] * b : 0) +
            output.strides[2] * ofm +
            output.strides[1] * y +
            output.strides[0] * x;
        ownStoreRawIntValue(fmt, sum, out_b_ptr + output_byte_offset);
    }
}

enum TT_CONVOLUTION_BIAS_TYPE
{
    BIAS_NONE,
    BIAS_SHARED,
    BIAS_PER_LOC,
};

typedef struct
{
    const char * name;
    enum TestTensorDF fmt;
    vx_size weight_w;
    vx_size weight_h;

    vx_size padding_x;
    vx_size padding_y;
    enum vx_convert_policy_e convert_policy;
    enum vx_round_policy_e rounding_policy;
    enum vx_nn_rounding_type_e down_scale_size_rounding;
    vx_size dilation_x;
    vx_size dilation_y;

    int batching_dim;
    enum TT_CONVOLUTION_BIAS_TYPE bias_type;
} test_convolution_layer_arg;

#define TT_CONVOLUTION_CASES_BASE(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,D_X_,D_Y_,BATCH_,BIAS_) \
    ARG(NAME_"_SZ_X"#SZ_X_"_Y"#SZ_Y_"_PAD_X"#PAD_X_"_Y"#PAD_Y_"_DILATION_X"#D_X_"_Y"#D_Y_,                          \
        TT_##FMT_, SZ_X_, SZ_Y_, PAD_X_, PAD_Y_, VX_CONVERT_POLICY_##OF_, VX_ROUND_POLICY_TO_##ROUND_,              \
        VX_NN_DS_SIZE_ROUNDING_##DS_ROUND_, D_X_, D_Y_, BATCH_, BIAS_),

#define TT_CONVOLUTION_CASES_5(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,D_X_,D_Y_,BATCH_)                          \
    TT_CONVOLUTION_CASES_BASE(NAME_"_NOBIAS",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,D_X_,D_Y_,BATCH_,BIAS_NONE)        \
    TT_CONVOLUTION_CASES_BASE(NAME_"_SHAREDBIAS",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,D_X_,D_Y_,BATCH_,BIAS_SHARED)  \
    TT_CONVOLUTION_CASES_BASE(NAME_"_PERLOCBIAS",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,D_X_,D_Y_,BATCH_,BIAS_PER_LOC)

#define TT_CONVOLUTION_CASES_4(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,D_X_,D_Y_)         \
    TT_CONVOLUTION_CASES_5(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,D_X_,D_Y_,0)           \
    TT_CONVOLUTION_CASES_5(NAME_"_BATCH",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,D_X_,D_Y_,1)

#define TT_CONVOLUTION_CASES_3(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_)   \
    TT_CONVOLUTION_CASES_4(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,0,0) \
    TT_CONVOLUTION_CASES_4(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,0,1) \
    TT_CONVOLUTION_CASES_4(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,DS_ROUND_,1,0)

#define TT_CONVOLUTION_CASES_2(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_)             \
    TT_CONVOLUTION_CASES_3(NAME_"_FLOOR",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,FLOOR)   \
    TT_CONVOLUTION_CASES_3(NAME_"_CEIL",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,CEILING)

#define TT_CONVOLUTION_CASES_1(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_)                \
    TT_CONVOLUTION_CASES_2(NAME_"_ZERO",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ZERO)        \
    TT_CONVOLUTION_CASES_2(NAME_"_NE",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,NEAREST_EVEN)

#define TT_CONVOLUTION_CASES_0(FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_)                  \
    TT_CONVOLUTION_CASES_1(#FMT_"_WRAP",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,WRAP)    \
    TT_CONVOLUTION_CASES_1(#FMT_"_SAT",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,SATURATE)

#define TT_CONVOLUTION_CASES_EXTRA(FMT_)    \
     TT_CONVOLUTION_CASES_0(FMT_,3,4,1,2)

#define TT_CONVOLUTION_CASES_ALEXNET(FMT_)  \
    TT_CONVOLUTION_CASES_0(FMT_,11,11,0,0)  \
    TT_CONVOLUTION_CASES_0(FMT_,6,6,0,0)    \
    TT_CONVOLUTION_CASES_0(FMT_,5,5,0,0)    \
    TT_CONVOLUTION_CASES_0(FMT_,3,3,0,0)

#define TT_CONVOLUTION_CASES_ALL()      \
    TT_CONVOLUTION_CASES_ALEXNET(U8)    \
    TT_CONVOLUTION_CASES_EXTRA(U8)

TEST_WITH_ARG(TensorNN, testConvolutionLayer, test_convolution_layer_arg,
    TT_CONVOLUTION_CASES_ALL()
)
{
    assert (arg_->fmt == TT_Q78 || arg_->fmt == TT_U8 || arg_->fmt == TT_S8);
    assert (arg_->batching_dim >= 0);
    assert (arg_->bias_type == BIAS_NONE || arg_->bias_type == BIAS_SHARED || arg_->bias_type == BIAS_PER_LOC);
    assert (arg_->convert_policy == VX_CONVERT_POLICY_WRAP ||
            arg_->convert_policy == VX_CONVERT_POLICY_SATURATE);
    assert (arg_->rounding_policy == VX_ROUND_POLICY_TO_ZERO ||
            arg_->rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN);
    assert (arg_->down_scale_size_rounding == VX_NN_DS_SIZE_ROUNDING_FLOOR ||
            arg_->down_scale_size_rounding == VX_NN_DS_SIZE_ROUNDING_CEILING);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims >= (size_t)(3 + arg_->batching_dim));
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type;
    vx_uint8 fixed_point_position;
    vx_size sizeof_data_type;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    const size_t inout_dim_num = 3 + arg_->batching_dim;
    const size_t weight_dim_num = 4;
    const size_t bias_dim_num =
        arg_->bias_type == BIAS_NONE ? 0 :
        arg_->bias_type == BIAS_SHARED ? 1 : 3;

    size_t in_dims[4];
    size_t weight_dims[4];
    size_t bias_dims[3];
    size_t out_dims[4];

    size_t in_strides[4];
    size_t weight_strides[4];
    size_t bias_strides[3];
    size_t out_strides[4];

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter);
            fflush(stdout);
        }

        size_t input_w, stride_x, output_w;
        ownGetConvPoolRandParams(
                &rng,
                arg_->padding_x, arg_->weight_w,
                arg_->dilation_x,
                arg_->down_scale_size_rounding == VX_NN_DS_SIZE_ROUNDING_CEILING,
                &input_w, &stride_x, &output_w);

        size_t input_h, stride_y, output_h;
        ownGetConvPoolRandParams(
                &rng,
                arg_->padding_y, arg_->weight_h,
                arg_->dilation_y,
                arg_->down_scale_size_rounding == VX_NN_DS_SIZE_ROUNDING_CEILING,
                &input_h, &stride_y, &output_h);

        in_dims[0] = input_w;
        in_dims[1] = input_h;
        for (vx_size i = 2; i < inout_dim_num; ++i)
        {
            in_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        }

        out_dims[0] = output_w;
        out_dims[1] = output_h;
        out_dims[2] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        for (vx_size i = 3; i < inout_dim_num; ++i)
        {
            out_dims[i] = in_dims[i];
        }

        weight_dims[0] = arg_->weight_w;
        weight_dims[1] = arg_->weight_h;
        weight_dims[2] = in_dims[2];
        weight_dims[3] = out_dims[2];

        if (bias_dim_num == 1) { bias_dims[0] = out_dims[2]; }
        else if (bias_dim_num == 3)
        {
            bias_dims[0] = out_dims[0];
            bias_dims[1] = out_dims[1];
            bias_dims[2] = out_dims[2];
        }

        vx_tensor in_tensor = vxCreateTensor(context_->vx_context_, inout_dim_num, in_dims, data_type, fixed_point_position);
        vx_tensor weight_tensor = vxCreateTensor(context_->vx_context_, weight_dim_num, weight_dims, data_type, fixed_point_position);
        vx_tensor bias_tensor = bias_dim_num ? vxCreateTensor(context_->vx_context_, bias_dim_num, bias_dims, data_type, fixed_point_position) : NULL;
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, inout_dim_num, out_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(in_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(weight_tensor, VX_TYPE_TENSOR);
        if (bias_dim_num) { ASSERT_VX_OBJECT(in_tensor, VX_TYPE_TENSOR); }
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        ownGetFlatByteStrides(arg_->fmt, in_dims, inout_dim_num, in_strides);
        ownGetFlatByteStrides(arg_->fmt, weight_dims, weight_dim_num, weight_strides);
        ownGetFlatByteStrides(arg_->fmt, bias_dims, bias_dim_num, bias_strides);
        ownGetFlatByteStrides(arg_->fmt, out_dims, inout_dim_num, out_strides);

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          in_dims: { "); for (size_t i = 0; i < inout_dim_num; ++i) { printf("%zu, ", in_dims[i]); } printf(" }, \n");
            printf("\t          weight_dims: { "); for (size_t i = 0; i < weight_dim_num; ++i) { printf("%zu, ", weight_dims[i]); } printf(" }, \n");
            if (bias_dim_num)
            {
                printf("\t          bias_dims: { "); for (size_t i = 0; i < bias_dim_num; ++i) { printf("%zu, ", bias_dims[i]); } printf(" }, \n");
            }
            printf("\t          out_dims: { "); for (size_t i = 0; i < inout_dim_num; ++i) { printf("%zu, ", out_dims[i]); } printf(" }, \n");
            printf("\t        }\n");
        }

        const size_t in_bytes = in_dims[inout_dim_num-1] * in_strides[inout_dim_num-1];
        const size_t weight_bytes = weight_dims[weight_dim_num-1] * weight_strides[weight_dim_num-1];
        const size_t bias_bytes = bias_dim_num ? bias_dims[bias_dim_num-1] * bias_strides[bias_dim_num-1] : 0;
        const size_t out_bytes = out_dims[inout_dim_num-1] * out_strides[inout_dim_num-1];

        const size_t in_count = in_bytes / sizeof_data_type;
        const size_t weight_count = weight_bytes / sizeof_data_type;
        const size_t bias_count = bias_bytes / sizeof_data_type;

        void * const in = malloc(in_bytes);
        void * const weight = malloc(weight_bytes);
        void * const bias = bias_dim_num ? malloc(bias_bytes) : NULL;
        void * const out = malloc(out_bytes);
        void * const refs = malloc(out_bytes);
        ASSERT(in && weight && (!bias_count || bias) && out && refs);

        {
            const int conv_prod_count = arg_->weight_w * arg_->weight_h * in_dims[2];

            ownFillSmallRandData(arg_->fmt, &rng, in_count, conv_prod_count, in);
            ownFillSmallRandData(arg_->fmt, &rng, weight_count, conv_prod_count, weight);
            if (bias_dim_num) { ownFillRandData(arg_->fmt, &rng, bias_count, bias); }

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(in_tensor, inout_dim_num, view_start, in_dims, in_strides, in, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            VX_CALL(vxCopyTensorPatch(weight_tensor, weight_dim_num, view_start, weight_dims, weight_strides, weight, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            if (bias_dim_num)
            {
                VX_CALL(vxCopyTensorPatch(bias_tensor, bias_dim_num, view_start, bias_dims, bias_strides, bias, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            }
            VX_CALL(vxCopyTensorPatch(out_tensor, inout_dim_num, view_start, out_dims, out_strides, out, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            vx_graph graph = vxCreateGraph(context_->vx_context_);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            const vx_nn_convolution_params_t params =
            {
                arg_->padding_x, arg_->padding_y, arg_->convert_policy, arg_->rounding_policy,
                arg_->down_scale_size_rounding, arg_->dilation_x, arg_->dilation_y
            };
            vx_node node = vxConvolutionLayer(graph, in_tensor, weight_tensor, bias_tensor, &params, sizeof(params), out_tensor);
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
            tensor_desc_t in_td = { inout_dim_num, in_dims, in_strides };
            tensor_desc_t weight_td = { weight_dim_num, weight_dims, weight_strides };
            tensor_desc_t bias_td = { bias_dim_num, bias_dims, bias_strides };
            tensor_desc_t out_td = { inout_dim_num, out_dims, out_strides };

            ownConvolution(
                    arg_->fmt,
                    in, in_td,
                    weight, weight_td,
                    bias, bias_td,
                    arg_->padding_x, arg_->padding_y,
                    stride_x, stride_y,
                    arg_->convert_policy == VX_CONVERT_POLICY_WRAP,
                    arg_->rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN,
                    arg_->dilation_x, arg_->dilation_y,
                    refs, out_td);

            const vx_size view_start[5] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, inout_dim_num, view_start, out_dims, out_strides, out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out, out_dims, inout_dim_num, out_strides,
                        refs, out_dims, inout_dim_num, out_strides,
                        8, //0, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)refs + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&in_tensor));
        VX_CALL(vxReleaseTensor(&weight_tensor));
        if (bias_dim_num) { VX_CALL(vxReleaseTensor(&bias_tensor)); }
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, in_tensor);
        EXPECT_EQ_PTR(NULL, weight_tensor);
        EXPECT_EQ_PTR(NULL, bias_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        free(in);
        free(weight);
        free(bias);
        free(out);
        free(refs);
    }
}


/****************************************************************************
 *                                                                          *
 *                          Test vxFullyConnectedLayer                      *
 *                                                                          *
 ***************************************************************************/

static void ownFullyConnected(
        enum TestTensorDF fmt,
        const void * input_ptr, tensor_desc_t input,
        const void * weight_ptr, tensor_desc_t weight,
        const void * bias_ptr, tensor_desc_t bias,
        bool wrap,  // true for WRAP, else SATURATE
        bool to_ne, // true for ROUND_TO_NE, else ROUND_TO_ZERO
        void * output_ptr, tensor_desc_t output)
{
    assert (fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    const size_t batch_dim_num = output.dim_num - 1;
    assert (batch_dim_num >= 0 && batch_dim_num <= 3);

    const size_t core_dim_num = input.dim_num - batch_dim_num;
    assert ((core_dim_num == 1 && weight.dim_num == 2) ||
            (core_dim_num == 3 && (weight.dim_num == 2 || weight.dim_num == 4)));

    assert (bias.dim_num == !!bias_ptr);
    const bool bias_present = !!bias.dim_num;

    if (core_dim_num == 1)
    {
        assert (weight.dims[0] == input.dims[0]);
    }
    else if (weight.dim_num == 2)
    {
        assert (weight.dims[0] == input.dims[0] * input.dims[1] * input.dims[2]);
    }
    else
    {
        assert (weight.dims[0] == input.dims[0]);
        assert (weight.dims[1] == input.dims[1]);
        assert (weight.dims[2] == input.dims[2]);
    }

    assert (weight.dims[weight.dim_num - 1] == output.dims[0]);
    assert (!bias_present || bias.dims[0] == output.dims[0]);

    for (size_t i = 0; i < batch_dim_num; ++i)
    {
        assert (output.dims[i + 1] == input.dims[i + core_dim_num]);
    }

    ownAssertStridesModSizeof(fmt, input);
    ownAssertStridesModSizeof(fmt, weight);
    ownAssertStridesModSizeof(fmt, bias);
    ownAssertStridesModSizeof(fmt, output);

    const size_t tmp_batch_dims[3] =
    {
        (batch_dim_num > 0 ? output.dims[1] : 1),
        (batch_dim_num > 1 ? output.dims[2] : 1),
        (batch_dim_num > 2 ? output.dims[3] : 1),
    };

    const size_t tmp_input_dims[3] =
    {
        (core_dim_num == 3 ? input.dims[0] : 1),
        (core_dim_num == 3 ? input.dims[1] : 1),
        input.dims[core_dim_num - 1],
    };

    const size_t ofm_num = output.dims[0];

    for (size_t b2 = 0; b2 < tmp_batch_dims[2]; ++b2)
    for (size_t b1 = 0; b1 < tmp_batch_dims[1]; ++b1)
    for (size_t b0 = 0; b0 < tmp_batch_dims[0]; ++b0)
    for (size_t ofm = 0; ofm < ofm_num; ++ofm)
    {
        int_fast32_t sum =
            bias_present ? ownLoadValueAsRawInt(fmt, (char *)bias_ptr + bias.strides[0] * ofm) : 0;

        for (size_t ifm = 0; ifm < tmp_input_dims[2]; ++ifm)
        for (size_t y = 0; y < tmp_input_dims[1]; ++y)
        for (size_t x = 0; x < tmp_input_dims[0]; ++x)
        {
            size_t weight_byte_offset = weight.strides[weight.dim_num-1] * ofm;
            if (core_dim_num == 1)
            {
                weight_byte_offset += weight.strides[0] * ifm;
            }
            else if (weight.dim_num == 2)
            {
                const size_t count = x + tmp_input_dims[0] * (y + tmp_input_dims[1] * ifm);
                weight_byte_offset += weight.strides[0] * count;
            }
            else
            {
                weight_byte_offset +=
                    weight.strides[2] * ifm +
                    weight.strides[1] * y +
                    weight.strides[0] * x;
            }

            const size_t input_byte_offset =
                (batch_dim_num > 2 ? input.strides[core_dim_num + 2] * b2 : 0) +
                (batch_dim_num > 1 ? input.strides[core_dim_num + 1] * b1 : 0) +
                (batch_dim_num > 0 ? input.strides[core_dim_num + 0] * b0 : 0) +
                input.strides[core_dim_num - 1] * ifm +
                (core_dim_num == 3 ? input.strides[1] * y : 0) +
                (core_dim_num == 3 ? input.strides[0] * x : 0);

            const int_fast32_t w_val = ownLoadValueAsRawInt(fmt, (char *)weight_ptr + weight_byte_offset);
            const int_fast32_t i_val = ownLoadValueAsRawInt(fmt, (char *)input_ptr + input_byte_offset);

            // This is ok since all of them fit into int32_t
            sum = ownApplyWrapRoundingToAccum(fmt, i_val * w_val, wrap, to_ne) + sum;
        }

        sum = ownWrapOrSat(fmt, sum, wrap);

        const size_t output_byte_offset =
            (batch_dim_num > 2 ? output.strides[3] * b2 : 0) +
            (batch_dim_num > 1 ? output.strides[2] * b1 : 0) +
            (batch_dim_num > 0 ? output.strides[1] * b0 : 0) +
            output.strides[0] * ofm;

        ownStoreRawIntValue(fmt, sum, (char *)output_ptr + output_byte_offset);
    }
}

typedef struct
{
    const char * name;
    enum TestTensorDF fmt;

    enum vx_convert_policy_e overflow_policy;
    enum vx_round_policy_e rounding_policy;

    vx_size core_dim;
    vx_size weight_dim;
    bool bias_present;
    vx_size batch_dim;
} test_fully_connected_layer_arg;

#define TT_FULLYCONNECTED_CASES_BASE(NAME_,FMT_,OF_,ROUND_,CORE_DIMS_,W_DIMS_,BATCH_,BIAS_) \
    ARG(NAME_"_COREDIMS_"#CORE_DIMS_"_WEIGHTDIMS_"#W_DIMS_"_BATCHDIMS_"#BATCH_,             \
        TT_##FMT_,VX_CONVERT_POLICY_##OF_, VX_ROUND_POLICY_TO_##ROUND_,                     \
        CORE_DIMS_, W_DIMS_, BIAS_, BATCH_),

#define TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,CORE_DIMS_,W_DIM_,BATCH_)           \
    TT_FULLYCONNECTED_CASES_BASE(NAME_"_NOBIAS",FMT_,OF_,ROUND_,CORE_DIMS_,W_DIM_,BATCH_,0) \
    TT_FULLYCONNECTED_CASES_BASE(NAME_"_BIAS",FMT_,OF_,ROUND_,CORE_DIMS_,W_DIM_,BATCH_,1)

#define TT_FULLYCONNECTED_CASES_2(NAME_,FMT_,OF_,ROUND_)    \
    TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,1,2,0)  \
    TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,1,2,1)  \
    TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,1,2,2)  \
    TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,1,2,3)  \
    TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,3,2,0)  \
    TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,3,2,1)  \
    TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,3,4,0)  \
    TT_FULLYCONNECTED_CASES_3(NAME_,FMT_,OF_,ROUND_,3,4,1)

#define TT_FULLYCONNECTED_CASES_1(NAME_,FMT_,OF_)               \
    TT_FULLYCONNECTED_CASES_2(NAME_"_ZERO",FMT_,OF_,ZERO)       \
    TT_FULLYCONNECTED_CASES_2(NAME_"_NE",FMT_,OF_,NEAREST_EVEN)

#define TT_FULLYCONNECTED_CASES_0(FMT_)                     \
    TT_FULLYCONNECTED_CASES_1(#FMT_"_WRAP",FMT_,WRAP)       \
    TT_FULLYCONNECTED_CASES_1(#FMT_"_SAT",FMT_,SATURATE)    \

#define TT_FULLYCONNECTED_CASES_ALL()   \
    TT_FULLYCONNECTED_CASES_0(U8)

TEST_WITH_ARG(TensorNN, testFullyConnectedLayer, test_fully_connected_layer_arg,
        TT_FULLYCONNECTED_CASES_ALL()
)
{
    assert (arg_->fmt == TT_Q78 || arg_->fmt == TT_U8 || arg_->fmt == TT_S8);
    assert (arg_->overflow_policy == VX_CONVERT_POLICY_WRAP ||
            arg_->overflow_policy == VX_CONVERT_POLICY_SATURATE);
    assert (arg_->rounding_policy == VX_ROUND_POLICY_TO_ZERO ||
            arg_->rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN);
    assert ((arg_->core_dim == 1 && arg_->weight_dim == 2) ||
            (arg_->core_dim == 3 && (arg_->weight_dim == 2 || arg_->weight_dim == 4)));
    assert (arg_->batch_dim >= 0 && arg_->core_dim + arg_->batch_dim <= 4);

    {   // TODO: ownTestGetMaxDims() ?
        vx_size max_dims = 0;
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims >= 4);
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type;
    vx_uint8 fixed_point_position;
    vx_size sizeof_data_type;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    const size_t in_dim_num = arg_->core_dim + arg_->batch_dim;
    const size_t weight_dim_num = arg_->weight_dim;
    const size_t bias_dim_num = arg_->bias_present;
    const size_t out_dim_num = 1 + arg_->batch_dim;

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter);
            fflush(stdout);
        }

        vx_size in_dims[4];
        vx_size weight_dims[4];
        vx_size bias_dims[1];
        vx_size out_dims[4];
        {
            for (size_t i = 0; i < in_dim_num; ++i) 
            {
                in_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
            }

            out_dims[0] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
            for (size_t i = 0; i < arg_->batch_dim; ++i)
            {
                out_dims[i + 1] = in_dims[i + arg_->core_dim];
            }

            weight_dims[weight_dim_num-1] = out_dims[0];
            if (arg_->core_dim == 1)
            {
                weight_dims[0] = in_dims[0];
            }
            else if (arg_->weight_dim == 2)
            {
                weight_dims[0] = in_dims[0] * in_dims[1] * in_dims[2];
            }
            else
            {
                weight_dims[0] = in_dims[0];
                weight_dims[1] = in_dims[1];
                weight_dims[2] = in_dims[2];
            } 

            if (bias_dim_num) bias_dims[0] = out_dims[0];
        }

        vx_size in_strides[4];
        vx_size weight_strides[4];
        vx_size bias_strides[1];
        vx_size out_strides[4];
        ownGetFlatByteStrides(arg_->fmt, in_dims, in_dim_num, in_strides);
        ownGetFlatByteStrides(arg_->fmt, weight_dims, weight_dim_num, weight_strides);
        ownGetFlatByteStrides(arg_->fmt, bias_dims, bias_dim_num, bias_strides);
        ownGetFlatByteStrides(arg_->fmt, out_dims, out_dim_num, out_strides);

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          in_dims: { "); for (size_t i = 0; i < in_dim_num; ++i) { printf("%zu, ", in_dims[i]); } printf(" }, \n");
            printf("\t          weight_dims: { "); for (size_t i = 0; i < weight_dim_num; ++i) { printf("%zu, ", weight_dims[i]); } printf(" }, \n");
            if (bias_dim_num)
            {
                printf("\t          bias_dims: { "); for (size_t i = 0; i < bias_dim_num; ++i) { printf("%zu, ", bias_dims[i]); } printf(" }, \n");
            }
            printf("\t          out_dims: { "); for (size_t i = 0; i < out_dim_num; ++i) { printf("%zu, ", out_dims[i]); } printf(" }, \n");
            printf("\t        }\n");
        }

        const size_t in_bytes = in_dims[in_dim_num-1] * in_strides[in_dim_num-1];
        const size_t weight_bytes = weight_dims[weight_dim_num-1] * weight_strides[weight_dim_num-1];
        const size_t bias_bytes = bias_dim_num ? bias_dims[bias_dim_num-1] * bias_strides[bias_dim_num-1] : 0;
        const size_t out_bytes = out_dims[out_dim_num-1] * out_strides[out_dim_num-1];

        const size_t in_count = in_bytes / sizeof_data_type;
        const size_t weight_count = bias_bytes / sizeof_data_type;
        const size_t bias_count = bias_bytes / sizeof_data_type;

        void * const in = malloc(in_bytes);
        void * const weight = malloc(weight_bytes);
        void * const bias = bias_dim_num ? malloc(bias_bytes) : NULL;
        void * const out = malloc(out_bytes);
        void * const refs = malloc(out_bytes);
        ASSERT(in && weight && (!bias_dim_num || bias) && out && refs);

        vx_tensor in_tensor = vxCreateTensor(context_->vx_context_, in_dim_num, in_dims, data_type, fixed_point_position);
        vx_tensor weight_tensor = vxCreateTensor(context_->vx_context_, weight_dim_num, weight_dims, data_type, fixed_point_position);
        vx_tensor bias_tensor =
            bias_dim_num
            ? vxCreateTensor(context_->vx_context_, bias_dim_num, bias_dims, data_type, fixed_point_position)
            : NULL;
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, out_dim_num, out_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(in_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(weight_tensor, VX_TYPE_TENSOR);
        if (bias_dim_num) { ASSERT_VX_OBJECT(bias_tensor, VX_TYPE_TENSOR); }
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        {
            size_t fc_prod_count = 1;
            for (size_t i = 0; i < weight_dim_num - 1; ++ i)
            {
                fc_prod_count *= weight_dims[i];
            }

            ownFillSmallRandData(arg_->fmt, &rng, in_count, fc_prod_count, in);
            ownFillSmallRandData(arg_->fmt, &rng, weight_count, fc_prod_count, weight);
            if (bias_dim_num) { ownFillRandData(arg_->fmt, &rng, bias_count, bias); }

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(in_tensor, in_dim_num, view_start, in_dims, in_strides, in, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            VX_CALL(vxCopyTensorPatch(weight_tensor, weight_dim_num, view_start, weight_dims, weight_strides, weight, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            if (bias_dim_num)
            {
                VX_CALL(vxCopyTensorPatch(bias_tensor, bias_dim_num, view_start, bias_dims, bias_strides, bias, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            }
            VX_CALL(vxCopyTensorPatch(out_tensor, out_dim_num, view_start, out_dims, out_strides, out, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            vx_graph graph = vxCreateGraph(context_->vx_context_);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = vxFullyConnectedLayer(
                    graph,
                    in_tensor, weight_tensor, bias_tensor,
                    arg_->overflow_policy,
                    arg_->rounding_policy,
                    out_tensor);
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
            tensor_desc_t in_td = { in_dim_num, in_dims, in_strides };
            tensor_desc_t weight_td = { weight_dim_num, weight_dims, weight_strides };
            tensor_desc_t bias_td = { bias_dim_num, bias_dims, bias_strides };
            tensor_desc_t out_td = { out_dim_num, out_dims, out_strides };

            ownFullyConnected(
                    arg_->fmt,
                    in, in_td,
                    weight, weight_td,
                    bias, bias_td,
                    arg_->overflow_policy == VX_CONVERT_POLICY_WRAP,
                    arg_->rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN,
                    refs, out_td);

            const vx_size view_start[4] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, out_dim_num, view_start, out_dims, out_strides, out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out, out_dims, out_dim_num, out_strides,
                        refs, out_dims, out_dim_num, out_strides,
                        8, //0, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)refs + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&in_tensor));
        VX_CALL(vxReleaseTensor(&weight_tensor));
        if (bias_dim_num) { VX_CALL(vxReleaseTensor(&bias_tensor)); }
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, in_tensor);
        EXPECT_EQ_PTR(NULL, weight_tensor);
        EXPECT_EQ_PTR(NULL, bias_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        free(in);
        free(weight);
        free(bias);
        free(out);
        free(refs);
    }
}


/****************************************************************************
 *                                                                          *
 *                          Test vxPoolingLayer                             *
 *                                                                          *
 ***************************************************************************/

static void ownPooling(
        enum TestTensorDF fmt,
        const void * input_ptr, tensor_desc_t input,
        bool max_pooling,   // MAX vs AVG pooling
        size_t size_x, size_t size_y,
        size_t pad_x, size_t pad_y,
        size_t stride_x, size_t stride_y,
        void * output_ptr, tensor_desc_t output)
{
    assert(input.dim_num == 3 || input.dim_num == 4);
    assert(output.dim_num == input.dim_num);

    const size_t input_w = input.dims[0];
    const size_t input_h = input.dims[1];
    const size_t input_c = input.dims[2];
    const size_t input_b = input.dim_num > 3 ? input.dims[3] : 1;

    const size_t output_w = output.dims[0];
    const size_t output_h = output.dims[1];
    const size_t output_c = output.dims[2];
    const size_t output_b = output.dim_num > 3 ? output.dims[3] : 1;

    assert(input_w + 2 * pad_x >= size_x);
    assert(input_h + 2 * pad_y >= size_y);
//    assert(missing_div_with_round_mode((input_w + 2 * pad_x - size_x), stride_x) + 1 == output_w);
//    assert(missing_div_with_round_mode((input_h + 2 * pad_y - size_y), stride_y) + 1 == output_h);

    //TODO: verify this is enforced by the input/output validators
    assert(output_c == input_c);
    assert(output_b == input_b);

    // Since we calc offsets manually and cast to (int16_t *), we expect the-
    // alignment to be correct already
    ownAssertStridesModSizeof (fmt, input);
    ownAssertStridesModSizeof (fmt, output);

    //TODO: previously there was a 1d/3d stride for ofm but there's no 1D pool, right?

    // Input and output pointers for the current batch being processed,
    // Note: The compiler should've been able to hoist this out... And
    // there's a bunch of other possible hoising iopportunities here.
    const char * in_b_ptr = input_ptr;
    char * out_b_ptr = output_ptr;

    for (size_t b = 0; b < output_b; ++b, in_b_ptr += input.strides[3], out_b_ptr += output.strides[3])
    for (size_t c = 0; c < output_c; ++c)
    for (size_t y = 0; y < output_h; ++y)
    for (size_t x = 0; x < output_w; ++x)
    {
        int32_t result = max_pooling ? ownGetMinValue(fmt) : 0;

        const size_t xx_start = CLAMP(x * stride_x,          pad_x, input_w + pad_x) - pad_x;
        const size_t xx_after = CLAMP(x * stride_x + size_x, pad_x, input_w + pad_x) - pad_x;

        const size_t yy_start = CLAMP(y * stride_y,          pad_y, input_h + pad_y) - pad_y;
        const size_t yy_after = CLAMP(y * stride_y + size_y, pad_y, input_h + pad_y) - pad_y;

        for (size_t yy = yy_start; yy < yy_after; ++yy)
        for (size_t xx = xx_start; xx < xx_after; ++xx)
        {
            const size_t input_byte_offset =
                input.strides[2] * c +
                input.strides[1] * yy +
                input.strides[0] * xx;
            const int32_t i_val = ownLoadValueAsRawInt(fmt, in_b_ptr + input_byte_offset);

            result = max_pooling? MAX(result, i_val) : (result + i_val);
        }

        if (!max_pooling)
        {
            //result = conversion_24_8(result / (int16_t)(size_x * size_y));
          result = CLAMP(result / (int32_t)(size_x * size_y), ownGetMinValue(fmt), ownGetMaxValue(fmt));
        }

        const size_t output_byte_offset =
            output.strides[2] * c +
            output.strides[1] * y +
            output.strides[0] * x;
        ownStoreRawIntValue(fmt, result, out_b_ptr + output_byte_offset);
    }
}

typedef struct
{
    const char * name;
    enum TestTensorDF fmt;
    enum vx_nn_pooling_type_e pooling_type;

    vx_size size_x;
    vx_size size_y;
    vx_size padding_x;
    vx_size padding_y;
    enum vx_nn_rounding_type_e down_scale_size_rounding;

    bool batching_dim;
} test_pooling_layer_arg;

#define TT_POOLING_CASES_BASE(NAME_,FMT_,TYPE_,ROUND_,SX_,SY_,PX_,PY_,BATCH_)   \
    ARG(#FMT_"_SIZE_X"#SX_"_Y"#SY_"_PAD_X"#PX_"_Y"#PY_"_"#TYPE_""NAME_,         \
        TT_##FMT_, VX_NN_POOLING_##TYPE_, SX_, SY_, PX_, PY_,                   \
        VX_NN_DS_SIZE_ROUNDING_##ROUND_, BATCH_),

#define TT_POOLING_CASES_2(FMT_,TYPE_,ROUND_,SX_,SY_,PX_,PY_)           \
    TT_POOLING_CASES_BASE("",FMT_,TYPE_,ROUND_,SX_,SY_,PX_,PY_,0)       \
    TT_POOLING_CASES_BASE("_BATCH",FMT_,TYPE_,ROUND_,SX_,SY_,PX_,PY_,1)

#define TT_POOLING_CASES_1(FMT_,TYPE_,ROUND_)   \
    TT_POOLING_CASES_2(FMT_,TYPE_,ROUND_,3,4,1,2)

#define TT_POOLING_CASES_0(FMT_,TYPE_)      \
    TT_POOLING_CASES_1(FMT_,TYPE_,FLOOR)    \
    TT_POOLING_CASES_1(FMT_,TYPE_,CEILING)

#define TT_POOLING_CASES_EXTRA(FMT_)    \
    TT_POOLING_CASES_0(FMT_,MAX)        \
    TT_POOLING_CASES_0(FMT_,AVG)

#define TT_POOLING_CASES_ALEXNET(FMT_)          \
    TT_POOLING_CASES_2(FMT_,MAX,FLOOR,3,3,0,0)

#define TT_POOLING_CASES_ALL()      \
    TT_POOLING_CASES_ALEXNET(U8)    \
    TT_POOLING_CASES_EXTRA(U8)

TEST_WITH_ARG(TensorNN, testPoolingLayer, test_pooling_layer_arg,
        TT_POOLING_CASES_ALL()
)
{
    assert (arg_->fmt == TT_Q78 || arg_->fmt == TT_U8 || arg_->fmt == TT_S8);
    assert (arg_->pooling_type == VX_NN_POOLING_MAX ||
            arg_->pooling_type == VX_NN_POOLING_AVG);
    assert (arg_->down_scale_size_rounding == VX_NN_DS_SIZE_ROUNDING_FLOOR ||
            arg_->down_scale_size_rounding == VX_NN_DS_SIZE_ROUNDING_CEILING);
    assert (arg_->batching_dim == 0 || arg_->batching_dim == 1);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims >= (size_t)(3 + arg_->batching_dim));
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type;
    vx_uint8 fixed_point_position;
    vx_size sizeof_data_type;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    const size_t dim_num = 3 + arg_->batching_dim;

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter);
            fflush(stdout);
        }

        size_t input_w, stride_x, output_w;
        ownGetConvPoolRandParams(
                &rng,
                arg_->padding_x, arg_->size_x,
                0 /* there's no dilation in pooling */,
                arg_->down_scale_size_rounding == VX_NN_DS_SIZE_ROUNDING_CEILING,
                &input_w, &stride_x, &output_w);

        size_t input_h, stride_y, output_h;
        ownGetConvPoolRandParams(
                &rng,
                arg_->padding_y, arg_->size_y,
                0 /* there's no dilation in pooling */,
                arg_->down_scale_size_rounding == VX_NN_DS_SIZE_ROUNDING_CEILING,
                &input_h, &stride_y, &output_h);

        const size_t chan = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        const size_t batch = arg_->batching_dim ? (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1) : 0;

        const vx_size in_dims[4] = { input_w, input_h, chan, batch };
        const vx_size out_dims[4] = { output_w, output_h, chan, batch };

        vx_size in_strides[4];
        vx_size out_strides[4];
        ownGetFlatByteStrides(arg_->fmt, in_dims, dim_num, in_strides);
        ownGetFlatByteStrides(arg_->fmt, out_dims, dim_num, out_strides);

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          in_dims: { "); for (size_t i = 0; i < dim_num; ++i) { printf("%zu, ", in_dims[i]); } printf(" }, \n");
            printf("\t          out_dims: { "); for (size_t i = 0; i < dim_num; ++i) { printf("%zu, ", out_dims[i]); } printf(" }, \n");
            printf("\t        }\n");
        }

        const size_t in_bytes = in_dims[dim_num-1] * in_strides[dim_num-1];
        const size_t out_bytes = out_dims[dim_num-1] * out_strides[dim_num-1];

        const size_t in_count = in_bytes / sizeof_data_type;

        void * const in = malloc(in_bytes);
        void * const out = malloc(out_bytes);
        void * const refs = malloc(out_bytes);
        ASSERT(in && out && refs);

        vx_tensor in_tensor = vxCreateTensor(context_->vx_context_, dim_num, in_dims, data_type, fixed_point_position);
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, dim_num, out_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(in_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        {
            // No real need to fo ownFillSmallRandData here because of the
            // guranteed 32bit accum and our data counts being small.
            ownFillRandData(arg_->fmt, &rng, in_count, in); 

            const vx_size view_start[4] = { 0 };
            VX_CALL(vxCopyTensorPatch(in_tensor, dim_num, view_start, in_dims, in_strides, in, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            vx_graph graph = vxCreateGraph(context_->vx_context_);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = vxPoolingLayer(
                    graph, in_tensor, arg_->pooling_type,
                    arg_->size_x, arg_->size_y,
                    arg_->padding_x, arg_->padding_y,
                    arg_->down_scale_size_rounding,
                    out_tensor);
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
            tensor_desc_t in_td = { dim_num, in_dims, in_strides };
            tensor_desc_t out_td = { dim_num, out_dims, out_strides };

            ownPooling(
                    arg_->fmt,
                    in, in_td,
                    arg_->pooling_type == VX_NN_POOLING_MAX,
                    arg_->size_x, arg_->size_y,
                    arg_->padding_x, arg_->padding_y,
                    stride_x, stride_y,
                    refs, out_td);

            const vx_size view_start[4] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, dim_num, view_start, out_dims, out_strides, out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out, out_dims, dim_num, out_strides,
                        refs, out_dims, dim_num, out_strides,
                        8, //0, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)refs + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&in_tensor));
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, in_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        free(in);
        free(out);
        free(refs);
    }
}


/****************************************************************************
 *                                                                          *
 *                          Test vxSoftmaxLayer                             *
 *                                                                          *
 ***************************************************************************/

static void ownSoftmax(
        enum TestTensorDF fmt,
        const void * input_ptr, tensor_desc_t input,
        void * output_ptr, tensor_desc_t output)
{
//TODO: @Tomer, should we allow extra batch dims beyond 4? conv and poll have upto 3 of them! if not we can just discard this define and its usage
#define SOFTMAX_ALLOW_EXTRA_DIMS
    
#ifdef SOFTMAX_ALLOW_EXTRA_DIMS
    assert(input.dim_num >= 1 && input.dim_num <= 4);
#else
    assert(input.dim_num >= 1 && input.dim_num < MAX_NUM_OF_DIMENSIONS);
#endif

    assert(input.dim_num == output.dim_num);

    // Since we calc offsets manually and cast to (int16_t *), we expect the-
    // alignment to be correct already
    ownAssertStridesModSizeof (fmt, input);
    ownAssertStridesModSizeof (fmt, output);

    // We precalc and store the key (summation) index and the rest of the dims
    // which describe batching before the main loop for clarity, since the later may be partially shifted depending on the key dim.

    size_t key_sz = 0;
    size_t key_in_stride = 0;
    
#ifdef SOFTMAX_ALLOW_EXTRA_DIMS
    size_t batch_sz[5] = { 1, 1, 1, 1, 1 };
    size_t batch_in_strides[5] = { 0 };
    size_t batch_out_strides[5] = { 0 };
#else
    size_t batch_sz[3] = { 1, 1, 1 };
    size_t batch_in_strides[3] = { 0 };
    size_t batch_out_strides[3] = { 0 };
#endif

#if 1
    {
        size_t key = input.dim_num > 2 ? 2 : 0;

        key_sz = input.dims[key];
        key_in_stride = input.strides[key];

        for (size_t i = 0; i < input.dim_num - 1; ++i)
        {
            size_t idx = i < key ? i : i + 1;

            batch_sz[i] = input.dims[idx];
            batch_in_strides[i] = input.strides[idx];
            batch_out_strides[i] = output.strides[idx];
        }
    }
#else
    switch (input.dim_num)
    {
#ifdef SOFTMAX_ALLOW_EXTRA_DIMS
    case 6:
        batch_sz[4] = input.dims[5];
        batch_in_strides[4] = input.strides[5];
        batch_out_strides[4] = output.strides[5];
        /* fallthrough */
    case 5:
        batch_sz[3] = input.dims[4];
        batch_in_strides[3] = input.strides[4];
        batch_out_strides[3] = output.strides[4];
        /* fallthrough */
#endif
    case 4:
        batch_sz[2] = input.dims[3];
        batch_in_strides[2] = input.strides[3];
        batch_out_strides[2] = output.strides[3];
        /* fallthrough */
    case 3:
        key_sz = input.dims[2];
        key_in_stride = input.strides[2];

        batch_sz[1] = input.dims[1];
        batch_in_strides[1] = input.strides[1];
        batch_out_strides[1] = output.strides[1];

        batch_sz[0] = input.dims[0];
        batch_in_strides[0] = input.strides[0];
        batch_out_strides[0] = output.strides[0];
        break;
    case 2:
        batch_sz[0] = input.dims[1];
        batch_in_strides[0] = input.strides[1];
        batch_out_strides[0] = output.strides[1];
        /* fallthrough */
    case 1:
        key_sz = input.dims[0];
        key_in_stride = input.strides[0];
        break;
    default:
        assert(0);
    }
#endif

// The main loop calculation can be done with a double accumulator, float with
// value normalization (exp(val-max_val)) to avoid getting to inf or plain -
// float. Leaving all 3 here for result comparision, since the spec has nothing
// about required accumulator width.
//
// Note: For U8, S8 all 3 will result in the same results. But for Q78, because
//       summing exp(127) is quite large for a single precision float, using it
//       may already result in inf within the summation causing all values to
//       0 after softmax! And obviously for F32, the change of getting there is
//       even higher.
//
// Set to 0 for float, 1 for double, 2 for float with norm.
#define SOFTMAX_ACCUM_TYPE 0

#ifdef SOFTMAX_ALLOW_EXTRA_DIMS
    for (size_t b4 = 0; b4 < batch_sz[4]; ++b4)
    for (size_t b3 = 0; b3 < batch_sz[3]; ++b3)
#endif
    for (size_t b2 = 0; b2 < batch_sz[2]; ++b2)
    for (size_t b1 = 0; b1 < batch_sz[1]; ++b1)
    for (size_t b0 = 0; b0 < batch_sz[0]; ++b0)
    {
        // Input and output pointers for the current batch being processed.
        const char * in_b_ptr = (char*)input_ptr +
            batch_in_strides[2] * b2 +
            batch_in_strides[1] * b1 +
            batch_in_strides[0] * b0;
        char * out_b_ptr = (char*)output_ptr +
            batch_out_strides[2] * b2 +
            batch_out_strides[1] * b1 +
            batch_out_strides[0] * b0;

#ifdef SOFTMAX_ALLOW_EXTRA_DIMS
            in_b_ptr += batch_in_strides[4] * b4 + batch_in_strides[3] * b3;
            out_b_ptr += batch_out_strides[4] * b4 + batch_out_strides[3] * b3;
#endif

#if SOFTMAX_ACCUM_TYPE == 0
        float sum = 0.f;

        for (size_t i = 0; i < key_sz; ++i)
        {
            const int_fast32_t in = ownLoadValueAsRawInt(fmt, in_b_ptr + key_in_stride * i);
            float in_val = ownUnquantize(fmt, in);

            sum += expf(in_val);
        }

        for (size_t i = 0; i < key_sz; ++i)
        {
            const int_fast32_t in = ownLoadValueAsRawInt(fmt, in_b_ptr + key_in_stride * i);
            float in_val = ownUnquantize(fmt, in);

            ownStoreRawIntValue(fmt, ownQuantize(fmt, expf(in_val) / sum), out_b_ptr + key_in_stride * i);
        }
#elif SOFTMAX_ACCUM_TYPE == 1
        double sum = 0.;

        for (size_t i = 0; i < key_sz; ++i)
        {
            const int16_t * in_ptr = (int16_t *)(in_b_ptr + key_in_stride * i);
            float in_val = UNQUANTIZE(*in_ptr);

            sum += exp(in_val);
        }

        for (size_t i = 0; i < key_sz; ++i)
        {
            const int16_t * in_ptr = (int16_t *)(in_b_ptr + key_in_stride * i);
            float in_val = UNQUANTIZE(*in_ptr);

            int16_t * out_ptr = (int16_t *)(out_b_ptr + key_in_stride * i);
            *out_ptr = QUANTIZE(exp(in_val) / sum);
        }
#elif SOFTMAX_ACCUM_TYPE == 2
        float max_val = -FLT_MAX;
        float sum = 0.f;

        for (size_t i = 0; i < key_sz; ++i)
        {
            const int16_t * in_ptr = (int16_t *)(in_b_ptr + key_in_stride * i);
            float in_val = UNQUANTIZE(*in_ptr);

            max_val = MAX(max_val, in_val);
        }
        
        // Note: It may be benificial to cache the exponents
        for (size_t i = 0; i < key_sz; ++i)
        {
            const int16_t * in_ptr = (int16_t *)(in_b_ptr + key_in_stride * i);
            float in_val = UNQUANTIZE(*in_ptr);

            sum += expf(in_val - max_val);
        }

        for (size_t i = 0; i < key_sz; ++i)
        {
            const int16_t * in_ptr = (int16_t *)(in_b_ptr + key_in_stride * i);
            float in_val = UNQUANTIZE(*in_ptr);

            int16_t * out_ptr = (int16_t *)(out_b_ptr + key_in_stride * i);
            *out_ptr = QUANTIZE(expf(in_val - max_val) / sum);
        }
#else
#error SOFTMAX_ACCUM_TYPE must be 0..2
#endif
    }
}

typedef struct
{
    const char * name;

    enum TestTensorDF fmt;
    vx_size dim_num;
} test_softmax_layer_arg;

#define TT_SOFTMAX_CASES_BASE(FMT_,DIMS_)       \
    ARG(#FMT_"_DIMS"#DIMS_, TT_##FMT_, DIMS_),

#define TT_SOFTMAX_CASES_0(FMT_)    \
    TT_SOFTMAX_CASES_BASE(FMT_,1)   \
    TT_SOFTMAX_CASES_BASE(FMT_,2)   \
    TT_SOFTMAX_CASES_BASE(FMT_,3)   \
    TT_SOFTMAX_CASES_BASE(FMT_,4)

#define TT_SOFTMAX_CASES_ALL()  \
    TT_SOFTMAX_CASES_0(U8)

TEST_WITH_ARG(TensorNN, testSoftmaxLayer, test_softmax_layer_arg,
        TT_SOFTMAX_CASES_ALL()
)
{
    assert (arg_->fmt == TT_Q78 || arg_->fmt == TT_U8 || arg_->fmt == TT_S8);
    assert (arg_->dim_num >= 1 && arg_->dim_num <=4);

    {   // TODO: ownTestGetMaxDims() ?
        vx_size max_dims = 0;
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims >= arg_->dim_num); 
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type;
    vx_uint8 fixed_point_position;
    vx_size sizeof_data_type;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter);
            fflush(stdout);
        }

        size_t dims[4];
        for (size_t i = 0; i < arg_->dim_num; ++i)
        {
            dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        }

        size_t strides[4];
        ownGetFlatByteStrides(arg_->fmt, dims, arg_->dim_num, strides);

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: { dims: { ");
            for (size_t i = 0; i < arg_->dim_num; ++i) { printf("%zu, ", dims[i]); }
            printf(" } }\n");
        }

        const size_t bytes = dims[arg_->dim_num-1] * strides[arg_->dim_num-1];
        const size_t count = bytes / sizeof_data_type;

        void * const in = malloc(bytes);
        void * const out = malloc(bytes);
        void * const refs = malloc(bytes);
        ASSERT(in && out && refs);

        vx_tensor in_tensor = vxCreateTensor(context_->vx_context_, arg_->dim_num, dims, data_type, fixed_point_position);
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, arg_->dim_num, dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(in_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        {
            // No real need to fo ownFillSmallRandData here because of the
            // guranteed 32bit accum and our data counts being small.
            ownFillRandData(arg_->fmt, &rng, count, in); 

            const vx_size view_start[4] = { 0 };
            VX_CALL(vxCopyTensorPatch(in_tensor, arg_->dim_num, view_start, dims, strides, in, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            vx_graph graph = vxCreateGraph(context_->vx_context_);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = vxSoftmaxLayer(graph, in_tensor, out_tensor);
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
            tensor_desc_t td = { arg_->dim_num, dims, strides };
            ownSoftmax(arg_->fmt, in, td, refs, td);

            const vx_size view_start[4] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, arg_->dim_num, view_start, dims, strides, out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out, dims, arg_->dim_num, strides,
                        refs, dims, arg_->dim_num, strides,
                        8, //0, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)refs + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&in_tensor));
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, in_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        free(in);
        free(out);
        free(refs);
    }
}


/****************************************************************************
 *                                                                          *
 *                          test vxNormalizationlayer                       *
 *                                                                          *
 ***************************************************************************/

//static void ownNormalization() { /*TODO*/ }
//
//typedef struct
//{
//    const char * name;
//
//    enum TestTensorDF fmt;
//} test_normalization_layer_arg;
//
//TEST_WITH_ARG(TensorNN, testNormalizationLayer, test_normalization_layer_arg,
//        ARG("Q78", TT_Q78),
//)
//{
//}


/****************************************************************************
 *                                                                          *
 *                          test vxActivationLayer                          *
 *                                                                          *
 ***************************************************************************/

static void ownActivation(
        enum TestTensorDF fmt,
        const void * input_ptr, tensor_desc_t input,
        enum vx_nn_activation_function_e func,
        float a, float b,
        void * output_ptr, tensor_desc_t output)
{
    assert (func == VX_NN_ACTIVATION_LOGISTIC ||
            func == VX_NN_ACTIVATION_HYPERBOLIC_TAN ||
            func == VX_NN_ACTIVATION_RELU ||
            func == VX_NN_ACTIVATION_BRELU ||
            func == VX_NN_ACTIVATION_SOFTRELU ||
            func == VX_NN_ACTIVATION_ABS ||
            func == VX_NN_ACTIVATION_SQUARE ||
            func == VX_NN_ACTIVATION_SQRT ||
            func == VX_NN_ACTIVATION_LINEAR);
    
    assert (input.dim_num == output.dim_num);
    assert (input.dim_num > 0 && input.dim_num <= 4);

    for (size_t i = 0; i < input.dim_num; ++i)
    {
        assert (input.dims[i] == output.dims[i]);
    }

    ownAssertStridesModSizeof(fmt, input);
    ownAssertStridesModSizeof(fmt, output);

    const size_t dim0 = output.dims[0];
    const size_t dim1 = output.dim_num > 1 ? output.dims[1]: 1;
    const size_t dim2 = output.dim_num > 2 ? output.dims[2]: 1;
    const size_t dim3 = output.dim_num > 3 ? output.dims[3]: 1;

    for (size_t i3 = 0; i3 < dim3; ++i3)
    for (size_t i2 = 0; i2 < dim2; ++i2)
    for (size_t i1 = 0; i1 < dim1; ++i1)
    for (size_t i0 = 0; i0 < dim0; ++i0)
    {
        const size_t input_byte_offset =
            (input.dim_num > 3 ? input.strides[3] * i3 : 0) +
            (input.dim_num > 2 ? input.strides[2] * i2 : 0) +
            (input.dim_num > 1 ? input.strides[1] * i1 : 0) +
            input.strides[0] * i0;

        int_fast32_t val = ownLoadValueAsRawInt(fmt, (char*)input_ptr + input_byte_offset);

        //TODO: should we check that val is a legal input for the functoin?

        switch(func)
        {
            case VX_NN_ACTIVATION_LOGISTIC:
                val = (int)(1 / (1 + exp(val)));    //TODO: conversion issue?
                break;
            case VX_NN_ACTIVATION_HYPERBOLIC_TAN:
                val = (int)(a * tanh(b * val));     //TODO: conversion issue?
                break;
            case VX_NN_ACTIVATION_RELU:
                val = MAX(0, val);
                break;
            case VX_NN_ACTIVATION_BRELU:
                val = MIN(a, MAX(0, val));          //TODO: conversion issue?
                break;
            case VX_NN_ACTIVATION_SOFTRELU:
                val = log(1 + exp(val));            //TODO: conversion issue?
                break;
            case VX_NN_ACTIVATION_ABS:
                val = val < 0 ? - val : val;
                break;
            case VX_NN_ACTIVATION_SQUARE:
                val = val * val;
                break;
            case VX_NN_ACTIVATION_SQRT:
                val = sqrt(val);                    //TODO: conversoin issue?
                break;
            case VX_NN_ACTIVATION_LINEAR:
                val = a * val + b;
                break;
            default:
                assert(0);
        }

        const size_t output_byte_offset =
            (output.dim_num > 3 ? output.strides[3] * i3 : 0) +
            (output.dim_num > 2 ? output.strides[2] * i2 : 0) +
            (output.dim_num > 1 ? output.strides[1] * i1 : 0) +
            output.strides[0] * i0;

        val = ownWrapOrSat(fmt, val, false);    //TODO: what should be done here??
        ownStoreRawIntValue(fmt, val, (char*)output_ptr + output_byte_offset);
    }
}

typedef struct
{
    const char * name;

    enum TestTensorDF fmt;
    vx_size dim_num;

    enum vx_nn_activation_function_e func;
    vx_float32 a;
    vx_float32 b;
} test_activation_layer_arg;

#define TT_ACTIVATION_CASES_BASE(NAME_,FMT_,DIMS_,FUNC_,A_,B_)      \
    ARG(#FMT_"_DIMS"#DIMS_"_"#FUNC_""NAME_,                         \
            TT_##FMT_, DIMS_, VX_NN_ACTIVATION_##FUNC_, A_, B_),

//TODO: what do we want to test here???
#define TT_ACTIVATION_CASES_1(FMT_,DIM_)                            \
    TT_ACTIVATION_CASES_BASE("",FMT_,DIM_,LOGISTIC,0,0)             \
    TT_ACTIVATION_CASES_BASE("_A1_B1",FMT_,DIM_,HYPERBOLIC_TAN,1,1) \
    TT_ACTIVATION_CASES_BASE("_A2_B2",FMT_,DIM_,HYPERBOLIC_TAN,2,2) \
    TT_ACTIVATION_CASES_BASE("",FMT_,DIM_,RELU,0,0)
#ifdef ACTIVATION_EXTRA
    TT_ACTIVATION_CASES_BASE("_A50",FMT_,DIM_,BRELU,50,0)           \
    TT_ACTIVATION_CASES_BASE("",FMT_,DIM_,SOFTRELU,0,0)             \
    TT_ACTIVATION_CASES_BASE("",FMT_,DIM_,ABS,0,0)                  \
    TT_ACTIVATION_CASES_BASE("",FMT_,DIM_,SQUARE,0,0)               \
    TT_ACTIVATION_CASES_BASE("",FMT_,DIM_,SQRT,0,0)                 \
    TT_ACTIVATION_CASES_BASE("",FMT_,DIM_,LINEAR,1,0)               \
    TT_ACTIVATION_CASES_BASE("_Ahalf_B2",FMT_,DIM_,LINEAR,.5f,2)
#endif //ACTIVATION_EXTRA

#define TT_ACTIVATION_CASES_0(FMT_) \
    TT_ACTIVATION_CASES_1(FMT_,1)   \
    TT_ACTIVATION_CASES_1(FMT_,2)   \
    TT_ACTIVATION_CASES_1(FMT_,3)   \
    TT_ACTIVATION_CASES_1(FMT_,4)

#define TT_ACTIVATION_CASES_ALL()   \
    TT_ACTIVATION_CASES_0(U8)

TEST_WITH_ARG(TensorNN, testActivationLayer, test_activation_layer_arg,
        TT_ACTIVATION_CASES_ALL()
)
{
    assert (arg_->fmt == TT_Q78 || arg_->fmt == TT_U8 || arg_->fmt == TT_S8);
    assert (arg_->dim_num >= 1 && arg_->dim_num <= 4);
    assert (arg_->func == VX_NN_ACTIVATION_LOGISTIC ||
            arg_->func == VX_NN_ACTIVATION_HYPERBOLIC_TAN ||
            arg_->func == VX_NN_ACTIVATION_RELU ||
            arg_->func == VX_NN_ACTIVATION_BRELU ||
            arg_->func == VX_NN_ACTIVATION_SOFTRELU ||
            arg_->func == VX_NN_ACTIVATION_ABS ||
            arg_->func == VX_NN_ACTIVATION_SQUARE ||
            arg_->func == VX_NN_ACTIVATION_SQRT ||
            arg_->func == VX_NN_ACTIVATION_LINEAR);
    assert (arg_->a >= 0.f && arg_->b >= 0.f);

    {   // TODO: ownTestGetMaxDims() ?
        vx_size max_dims = 0;
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims >= arg_->dim_num); 
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type;
    vx_uint8 fixed_point_position;
    vx_size sizeof_data_type;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter);
            fflush(stdout);
        }

        size_t dims[4];
        for (size_t i = 0; i < arg_->dim_num; ++i)
        {
            dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        }

        size_t strides[4];
        ownGetFlatByteStrides(arg_->fmt, dims, arg_->dim_num, strides);

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: { dims: { ");
            for (size_t i = 0; i < arg_->dim_num; ++i) { printf("%zu, ", dims[i]); }
            printf(" } }\n");
        }

        const size_t bytes = dims[arg_->dim_num-1] * strides[arg_->dim_num-1];
        const size_t count = bytes / sizeof_data_type;

        void * const in = malloc(bytes);
        void * const out = malloc(bytes);
        void * const refs = malloc(bytes);
        ASSERT(in && out && refs);

        vx_tensor in_tensor = vxCreateTensor(context_->vx_context_, arg_->dim_num, dims, data_type, fixed_point_position);
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, arg_->dim_num, dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(in_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        {
            ownFillRandData(arg_->fmt, &rng, count, in); 

            const vx_size view_start[4] = { 0 };
            VX_CALL(vxCopyTensorPatch(in_tensor, arg_->dim_num, view_start, dims, strides, in, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            vx_graph graph = vxCreateGraph(context_->vx_context_);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            vx_node node = vxActivationLayer(graph, in_tensor, arg_->func, arg_->a, arg_->b, out_tensor);
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
            tensor_desc_t td = { arg_->dim_num, dims, strides };
            ownActivation(arg_->fmt, in, td, arg_->func, arg_->a, arg_->b, refs, td);

            const vx_size view_start[4] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, arg_->dim_num, view_start, dims, strides, out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out, dims, arg_->dim_num, strides,
                        refs, dims, arg_->dim_num, strides,
                        8, //0, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)refs + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&in_tensor));
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, in_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        free(in);
        free(out);
        free(refs);
    }
}


/****************************************************************************
 *                                                                          *
 *                          Test vxROIPoolingLayer                          *
 *                                                                          *
 ***************************************************************************/

static void ownROIPooling(
        enum TestTensorDF fmt,
        const void * data, vx_size data_dim_num, const vx_size * data_dims, const vx_size * data_strides,
        const void * rois, vx_size rois_dim_num, const vx_size * rois_dims, const vx_size * rois_strides,
        void * out, vx_size out_dim_num, const vx_size * out_dims, const vx_size * out_strides)
{
    assert ((data_dim_num == 3 && rois_dim_num == 2 && out_dim_num == 4) ||
            (data_dim_num == 4 && rois_dim_num == 3 && out_dim_num == 5));

    // format: [batch][channels][height][width]
    const size_t data_w = data_dims[0];
    const size_t data_h = data_dims[1];
    const size_t data_c = data_dims[2];
    const size_t data_b = data_dim_num == 4 ? data_dims[3] : 1;

    // format: [batch][roi_count][4]
    const size_t rois_d = rois_dims[0];
    const size_t rois_r = rois_dims[1];
    const size_t rois_b = rois_dim_num == 3 ? rois_dims[2] : 1;

    // format: [batch][roi_count][channels][height][width]
    const size_t out_w = out_dims[0];
    const size_t out_h = out_dims[1];
    const size_t out_c = out_dims[2];
    const size_t out_r = out_dims[3];
    const size_t out_b = out_dim_num == 5 ? out_dims[4] : 1;

    assert(data_c == out_c);
    assert(data_b == rois_b && data_b == out_b);
    assert(rois_d == 4);
    assert(rois_r == out_r);

    {
        size_t sizeof_data_type = 0;
        switch(fmt)
        {
            case TT_Q78: sizeof_data_type = sizeof(vx_int16); break;
            case TT_U8: sizeof_data_type = sizeof(vx_uint8); break;
            case TT_S8: sizeof_data_type = sizeof(vx_int8); break;
            default: assert(0);
        }
        for (size_t i = 0; i < data_dim_num; ++i) { assert(data_strides[i] % sizeof_data_type == 0); }
        for (size_t i = 0; i < rois_dim_num; ++i) { assert(rois_strides[i] % sizeof_data_type == 0); }
        for (size_t i = 0; i < out_dim_num; ++i) { assert(out_strides[i] % sizeof_data_type == 0); }
    }

    const int_fast32_t lowest_val = ownGetMinValue(fmt);

    for (size_t b = 0; b < out_b; ++b)
    for (size_t r = 0; r < out_r; ++r)
    for (size_t c = 0; c < out_c; ++c)
    for (size_t y = 0; y < out_h; ++y)
    for (size_t x = 0; x < out_w; ++x)
    {
        const char * roi_b_ptr = (char*)rois + rois_strides[1] * r + (b ? rois_strides[2] * b : 0);

        const int roi_x0 = ownLoadValueAsRawInt(fmt, roi_b_ptr + rois_strides[0] * 0);
        const int roi_y0 = ownLoadValueAsRawInt(fmt, roi_b_ptr + rois_strides[0] * 1);
        const int roi_x1 = ownLoadValueAsRawInt(fmt, roi_b_ptr + rois_strides[0] * 2);
        const int roi_y1 = ownLoadValueAsRawInt(fmt, roi_b_ptr + rois_strides[0] * 3);

        // The final coordinate is within the ROI => +1
        // And we treat malformed dimensions as 1
        const int roi_w = MAX(roi_x1 - roi_x0, 0) + 1;
        const int roi_h = MAX(roi_y1 - roi_y0, 0) + 1;

        // Note that "after" is rounded up else we get the last cell,
        // instead of the cell beyond.
        //
        // For ex. with src being a 6 cell row and dst being a 4 cell one:
        // >>> [((x + 0) * 6) // 4 for x in range(4)]   # "begin" values
        // [0, 1, 3, 4]                                 # as expected
        // >>> [((x + 1) * 6) // 4 for x in range(4)]   # "after" values
        // [1, 3, 4, 6]                                 # [2, 3, 5, 6] expected!
        const int dx_begin = ((x + 0) * roi_w) / out_w;
        const int dy_begin = ((y + 0) * roi_h) / out_h;
        const int dx_after = ((x + 1) * roi_w + (out_w - 1)) / out_w;
        const int dy_after = ((y + 1) * roi_h + (out_h - 1)) / out_h;

        // clamp in case roi_x or roi_y were unreasonable
        const int x_begin = CLAMP((size_t)(roi_x0 + dx_begin), 0, data_w);
        const int y_begin = CLAMP((size_t)(roi_y0 + dy_begin), 0, data_h);
        const int x_after = CLAMP((size_t)(roi_x0 + dx_after), 0, data_w);
        const int y_after = CLAMP((size_t)(roi_y0 + dy_after), 0, data_h);

        const char * data_b_ptr = (char*)data + data_strides[3] * b + data_strides[2] * c;

        // If there's no values for the current roi, we default to 0
        const bool non_empty = (x_begin < x_after && y_begin < y_after);
        int res = non_empty ? lowest_val : 0;

        for (int yy = y_begin; yy < y_after; ++yy)
        for (int xx = x_begin; xx < x_after; ++xx)
        {
            const void * val_ptr = data_b_ptr + data_strides[1] * yy + data_strides[0] * xx;
            int val = ownLoadValueAsRawInt(fmt, val_ptr);

            res = MAX(res, val);
        }

        const size_t output_byte_offset =
            out_strides[4] * b +
            out_strides[3] * r + out_strides[2] * c +
            out_strides[1] * y + out_strides[0] * x;
        ownStoreRawIntValue(fmt, res, (char*)out + output_byte_offset);
    }
}

typedef struct
{
    const char * name;

    enum TestTensorDF fmt;
    bool with_batching;
} test_roi_pooling_arg;

TEST_WITH_ARG(TensorNN, testROIPoolingLayer, test_roi_pooling_arg,
        ARG("Q78", TT_Q78, false),
        ARG("U8", TT_U8, false),
        ARG("S8", TT_S8, false),

        ARG("Q78_Bathcing", TT_Q78, true),
        ARG("U8_Batching", TT_U8, true),
        ARG("S8_Batching", TT_S8, true),
)
{
    assert(arg_->fmt == TT_Q78 || arg_->fmt == TT_U8 || arg_->fmt == TT_S8);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims >= (size_t)(arg_->with_batching ? 5 : 4));
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type;
    vx_uint8 fixed_point_position;
    vx_size sizeof_data_type;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    const size_t data_dim_num = arg_->with_batching ? 4 : 3;
    const size_t rois_dim_num = arg_->with_batching ? 3 : 2;
    const size_t out_dim_num = arg_->with_batching ? 5 : 4;

    size_t * const data_dims = malloc(sizeof(*data_dims) * data_dim_num);
    size_t * const rois_dims = malloc(sizeof(*rois_dims) * rois_dim_num);
    size_t * const out_dims = malloc(sizeof(*out_dims) * out_dim_num);
    ASSERT(data_dims && rois_dims && out_dims);
    
    size_t * const data_strides = malloc(sizeof(*data_strides) * data_dim_num);
    size_t * const rois_strides = malloc(sizeof(*rois_strides) * rois_dim_num);
    size_t * const out_strides = malloc(sizeof(*out_strides) * out_dim_num);
    ASSERT(data_strides && rois_strides && out_strides);

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter);
            fflush(stdout);
        }

        for (vx_size i = 0; i < data_dim_num; ++i)
        {
            data_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
            data_strides[i] = i ? data_strides[i-1] * data_dims[i-1] : sizeof_data_type;
        }

        rois_dims[0] = 4;
        rois_dims[1] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

        out_dims[0] = data_dims[0];
        out_dims[1] = data_dims[1];
        out_dims[2] = data_dims[2];
        out_dims[3] = rois_dims[1];

        if (arg_->with_batching)
        {
            out_dims[4] = rois_dims[2] = data_dims[3];
        }

        vx_tensor data_tensor = vxCreateTensor(context_->vx_context_, data_dim_num, data_dims, data_type, fixed_point_position);
        vx_tensor rois_tensor = vxCreateTensor(context_->vx_context_, rois_dim_num, rois_dims, data_type, fixed_point_position);
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, out_dim_num, out_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(data_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(rois_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        for (size_t i = 0; i < rois_dim_num; ++i)
        {
            rois_strides[i] = i ? rois_strides[i-1] * rois_dims[i-1] : sizeof_data_type;
        }
        for (size_t i = 0; i < out_dim_num; ++i)
        {
            out_strides[i] = i ? out_strides[i-1] * out_dims[i-1] : sizeof_data_type;
        }

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          data_dims: { "); for (size_t i = 0; i < data_dim_num; ++i) { printf("%zu, ", data_dims[i]); } printf(" }, \n");
            printf("\t          rois_dims: { "); for (size_t i = 0; i < rois_dim_num; ++i) { printf("%zu, ", rois_dims[i]); } printf(" }, \n");
            printf("\t          out_dims: { "); for (size_t i = 0; i < out_dim_num; ++i) { printf("%zu, ", out_dims[i]); } printf(" }, \n");
            printf("\t        }\n");
        }

        const size_t data_bytes = data_dims[data_dim_num-1] * data_strides[data_dim_num-1];
        const size_t rois_bytes = rois_dims[rois_dim_num-1] * rois_strides[rois_dim_num-1];
        const size_t out_bytes = out_dims[out_dim_num-1] * out_strides[out_dim_num-1];

        const size_t data_count = data_bytes / sizeof_data_type;

        void * const data = malloc(data_bytes);
        void * const rois = malloc(rois_bytes);
        void * const out = malloc(out_bytes);
        void * const refs = malloc(out_bytes);
        ASSERT(data && rois && out && refs);

        {
            ownFillRandData(arg_->fmt, &rng, data_count, data);
            for (size_t i = 0; i < rois_dims[1]; ++i)
            {
                switch(arg_->fmt)
                {
                    case TT_Q78:
                        ((vx_int16*)rois)[4*i+0] = (vx_int16)CT_RNG_NEXT_INT(rng, -2, data_dims[0] + 2);
                        ((vx_int16*)rois)[4*i+1] = (vx_int16)CT_RNG_NEXT_INT(rng, -2, data_dims[1] + 2);
                        ((vx_int16*)rois)[4*i+2] = (vx_int16)CT_RNG_NEXT_INT(rng, -2, data_dims[0] + 2);
                        ((vx_int16*)rois)[4*i+3] = (vx_int16)CT_RNG_NEXT_INT(rng, -2, data_dims[1] + 2);
                        break;
                    case TT_U8:
                        ((vx_uint8*)rois)[4*i+0] = (vx_uint8)CT_RNG_NEXT_INT(rng, 0, data_dims[0] + 2);
                        ((vx_uint8*)rois)[4*i+1] = (vx_uint8)CT_RNG_NEXT_INT(rng, 0, data_dims[1] + 2);
                        ((vx_uint8*)rois)[4*i+2] = (vx_uint8)CT_RNG_NEXT_INT(rng, 0, data_dims[0] + 2);
                        ((vx_uint8*)rois)[4*i+3] = (vx_uint8)CT_RNG_NEXT_INT(rng, 0, data_dims[1] + 2);
                        break;
                    case TT_S8:
                        ((vx_int8*)rois)[4*i+0] = (vx_int8)CT_RNG_NEXT_INT(rng, -2, data_dims[0] + 2);
                        ((vx_int8*)rois)[4*i+1] = (vx_int8)CT_RNG_NEXT_INT(rng, -2, data_dims[1] + 2);
                        ((vx_int8*)rois)[4*i+2] = (vx_int8)CT_RNG_NEXT_INT(rng, -2, data_dims[0] + 2);
                        ((vx_int8*)rois)[4*i+3] = (vx_int8)CT_RNG_NEXT_INT(rng, -2, data_dims[1] + 2);
                        break;
                    default:
                        assert(0);
                }
            }

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(data_tensor, data_dim_num, view_start, data_dims, data_strides, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            VX_CALL(vxCopyTensorPatch(rois_tensor, rois_dim_num, view_start, rois_dims, rois_strides, rois, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        // Third step is creating, running and disposing of the graph.
        {
            vx_graph graph = vxCreateGraph(context_->vx_context_);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);


            const vx_nn_roi_pool_params_t roi_pool_params = { VX_NN_POOLING_MAX };
            vx_node node = vxROIPoolingLayer(graph, data_tensor, rois_tensor, &roi_pool_params, sizeof(roi_pool_params), out_tensor);

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
            ownROIPooling(
                    arg_->fmt,
                    data, data_dim_num, data_dims, data_strides,
                    rois, rois_dim_num, rois_dims, rois_strides,
                    refs, out_dim_num, out_dims, out_strides);

            const size_t view_start[5] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, out_dim_num, view_start, out_dims, out_strides, out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out, out_dims, 2, out_strides,
                        refs, out_dims, 2, out_strides,
                        8, //0, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)refs + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&data_tensor));
        VX_CALL(vxReleaseTensor(&rois_tensor));
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, data_tensor);
        EXPECT_EQ_PTR(NULL, rois_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        free(data);
        free(rois);
        free(out);
        free(refs);
    }

    free(data_dims);
    free(rois_dims);
    free(out_dims);

    free(data_strides);
    free(rois_strides);
    free(out_strides);
}


/****************************************************************************
 *                                                                          *
 *                          test vxDeconvolutionLayer                       *
 *                                                                          *
 ***************************************************************************/

static void ownGetDeconvRandParams(
        uint64_t * rng,
        size_t pad_sz, size_t kernel_sz,
        size_t a,
        /*OUT*/ size_t * input_sz,
        /*OUT*/ size_t * upscale,
        /*OUT*/ size_t * output_sz)
{
    *upscale = (size_t)CT_RNG_NEXT_INT(*rng, a + 1, a + 3);

    const int tmp = (2 * (int)pad_sz - (int)kernel_sz - (int)a + ((int)*upscale - 1)) /(int)*upscale;

    const int min_input = 2 + MAX(tmp, 0);  //TODO: can we lower this?
    const int max_input = MIN(min_input, TEST_TENSOR_MAX_DIM_SZ) + 5;
    *input_sz = (size_t)CT_RNG_NEXT_INT(*rng, min_input, max_input);

    *output_sz = (*input_sz - 1) * *upscale + kernel_sz + a - 2 * pad_sz;
}

static void ownDeconvolution(
        enum TestTensorDF fmt,
        const void * input_ptr, tensor_desc_t input,
        const void * weight_ptr, tensor_desc_t weight,
        const void * bias_ptr, tensor_desc_t bias,
        vx_size pad_x, vx_size pad_y,
        vx_size upscale_x, vx_size upscale_y,
        bool wrap,  // true for WRAP, else SATURATE
        bool to_ne, // true for ROUND_TO_NE, else ROUND_TO_ZERO
        vx_size a_x, vx_size a_y,
        void * output_ptr, tensor_desc_t output)
{
    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    assert(input.dim_num == 3 || input.dim_num == 4);
    assert(weight.dim_num == 4);
    assert(bias.dim_num == 0 || bias.dim_num == 1 || bias.dim_num == 3);
    assert(output.dim_num == input.dim_num);

    const size_t input_w = input.dims[0];
    const size_t input_h = input.dims[1];
    const size_t input_c = input.dims[2];
    const size_t input_b = input.dim_num > 3 ? input.dims[3] : 1;

    const size_t weight_w = weight.dims[0];
    const size_t weight_h = weight.dims[1];
    const size_t weight_ifm = weight.dims[2];
    const size_t weight_ofm = weight.dims[3];

    const bool bias_present = !!bias.dim_num;
    const bool bias_shared = bias.dim_num == 1;
    const size_t bias_w = bias.dim_num > 0 ? bias.dims[0] : 0;
    const size_t bias_h = bias.dim_num > 1 ? bias.dims[1] : 1;
    const size_t bias_ofm = bias.dim_num > 2 ? bias.dims[2] : 1;

    const size_t output_w = output.dims[0];
    const size_t output_h = output.dims[1];
    const size_t output_c = output.dims[2];
    const size_t output_b = output.dim_num > 3 ? output.dims[3] : 1;

    assert(weight_ifm == input_c);
    assert(weight_ofm == output_c);

    assert(upscale_x > 0 && upscale_y > 0);
    assert(a_x < upscale_x && a_y < upscale_y);
    assert((input_w - 1) * upscale_x + weight_w + a_x > 2 * pad_x);
    assert((input_h - 1) * upscale_y + weight_h + a_y > 2 * pad_y);
    assert(output_w == (input_w - 1) * upscale_x + weight_w + a_x - 2 * pad_x);
    assert(output_h == (input_h - 1) * upscale_y + weight_h + a_y - 2 * pad_y);

    assert(weight_w >= pad_x + 1);
    assert(weight_h >= pad_y + 1);
    const size_t start_x_pad = weight_w - pad_x - 1;
    const size_t start_y_pad = weight_h - pad_y - 1;

    // NOTE:
    // The complete input width being sampled is,
    //  start_x_pad + ((input_w - 1) * upscale_x + 1) + after_x_pad
    // which is
    //  (input_w - 1) * upscale_x + 2 * weight_w - 2 * pad_x - 1 + a_x
    // and the stride being 1, the output width comes down to this plus 1 - weight_w
    // which ends up being
    //  (input_w - 1) * upscale_x + weight_w - 2 * pad_x + a_x
    // which is exactly output_w

    if (bias_shared)
    {
        assert(bias_w == weight_ofm);
    }
    else if (bias_present)
    {
        assert(bias_w == output_w);
        assert(bias_h == output_h);
        assert(bias_ofm == output_c);
    }

    assert(output_b == input_b);

    ownAssertStridesModSizeof(fmt, input);
    ownAssertStridesModSizeof(fmt, weight);
    ownAssertStridesModSizeof(fmt, bias);
    ownAssertStridesModSizeof(fmt, output);

    // Input and output pointers for the current batch being processed,
    // Note: The compiler should've been able to hoist this out... And
    // there's a bunch of other possible hoising iopportunities here.
    const char * in_b_ptr = input_ptr;
    char * out_b_ptr = output_ptr;

    for (size_t b = 0; b < output_b; ++b)
    for (size_t ofm = 0; ofm < output_c; ++ofm)
    for (size_t y = 0; y < output_h; ++y)
    for (size_t x = 0; x < output_w; ++x)
    {
        int32_t sum = 0;
        if (bias_present)
        {
            const size_t bias_byte_offset =
                bias_shared
                ? (bias.strides[0] * ofm)
                : (bias.strides[2] * ofm + bias.strides[1] * y + bias.strides[0] * x);

            sum = ownLoadValueAsRawInt(fmt, (char *)bias_ptr + bias_byte_offset);
        }
        
        for (size_t ifm = 0; ifm < input_c; ++ifm)
        {
            for (size_t w_y = 0; w_y < weight_h; ++w_y)
            for (size_t w_x = 0; w_x < weight_w; ++w_x)
            {
                if (x + w_x >= start_x_pad && x + w_x < input_w + start_x_pad &&
                    y + w_y >= start_y_pad && y + w_y < input_h + start_y_pad)
                {
                    const size_t xx = x + w_x - start_x_pad;
                    const size_t yy = y + w_y - start_y_pad;

                    if (xx % upscale_x == 0 && yy % upscale_y == 0)
                    {
                        const size_t input_byte_offset =
                            (b ? input.strides[3] * b : 0) +
                            input.strides[2] * ifm +
                            input.strides[1] * (yy / upscale_y) +
                            input.strides[0] * (xx / upscale_x);
                        const size_t weight_byte_offset =
                            weight.strides[3] * ofm +
                            weight.strides[2] * ifm +
                            weight.strides[1] * w_y +
                            weight.strides[0] * w_x;

                        const int_fast32_t i_val = ownLoadValueAsRawInt(fmt, in_b_ptr + input_byte_offset);
                        const int_fast32_t w_val = ownLoadValueAsRawInt(fmt, (char *)weight_ptr + weight_byte_offset);

                        // This is ok since all of them fit into int32_t
                        sum = ownApplyWrapRoundingToAccum(fmt, i_val * w_val, wrap, to_ne) + sum;
                    }
                }
            }
            sum = ownWrapOrSat(fmt, sum, wrap);
        }

        // The step here could be added to the loops instead of recalcing
        // if, but does the compiler fail to hoist them out???
        const size_t output_byte_offset =
            (b ? output.strides[3] * b : 0) +
            output.strides[2] * ofm +
            output.strides[1] * y +
            output.strides[0] * x;
        ownStoreRawIntValue(fmt, sum, out_b_ptr + output_byte_offset);
    }
}

typedef struct
{
    const char * name;
    enum TestTensorDF fmt;
    vx_size weight_w;
    vx_size weight_h;

    vx_size padding_x;
    vx_size padding_y;
    enum vx_convert_policy_e convert_policy;
    enum vx_round_policy_e rounding_policy;
    vx_size a_x;
    vx_size a_y;

    int batching_dim;
    enum TT_CONVOLUTION_BIAS_TYPE bias_type;
} test_deconvolution_layer_arg;

//TODO: take a more thorugh look at these, taken form conv
#define TT_DECONVOLUTION_CASES_BASE(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,A_X_,A_Y_,BATCH_,BIAS_) \
    ARG(NAME_"_SZ_X"#SZ_X_"_Y"#SZ_Y_"_PAD_X"#PAD_X_"_Y"#PAD_Y_"_A_X"#A_X_"_Y"#A_Y_,                         \
        TT_##FMT_, SZ_X_, SZ_Y_, PAD_X_, PAD_Y_, VX_CONVERT_POLICY_##OF_, VX_ROUND_POLICY_TO_##ROUND_,      \
        A_X_, A_Y_, BATCH_, BIAS_),

#define TT_DECONVOLUTION_CASES_4(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,A_X_,A_Y_,BATCH_)                          \
    TT_DECONVOLUTION_CASES_BASE(NAME_"_NOBIAS",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,A_X_,A_Y_,BATCH_,BIAS_NONE)        \
    TT_DECONVOLUTION_CASES_BASE(NAME_"_SHAREDBIAS",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,A_X_,A_Y_,BATCH_,BIAS_SHARED)  \
    TT_DECONVOLUTION_CASES_BASE(NAME_"_PERLOCBIAS",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,A_X_,A_Y_,BATCH_,BIAS_PER_LOC)

#define TT_DECONVOLUTION_CASES_3(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,A_X_,A_Y_)         \
    TT_DECONVOLUTION_CASES_4(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,A_X_,A_Y_,0)           \
    TT_DECONVOLUTION_CASES_4(NAME_"_BATCH",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,A_X_,A_Y_,1)

#define TT_DECONVOLUTION_CASES_2(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_)   \
    TT_DECONVOLUTION_CASES_3(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,0,0)   \
    TT_DECONVOLUTION_CASES_3(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,0,1)   \
    TT_DECONVOLUTION_CASES_3(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ROUND_,1,0)

#define TT_DECONVOLUTION_CASES_1(NAME_,FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_)              \
    TT_DECONVOLUTION_CASES_2(NAME_"_ZERO",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,ZERO)      \
    TT_DECONVOLUTION_CASES_2(NAME_"_NE",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,OF_,NEAREST_EVEN)

#define TT_DECONVOLUTION_CASES_0(FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_)                \
    TT_DECONVOLUTION_CASES_1(#FMT_"_WRAP",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,WRAP)  \
    TT_DECONVOLUTION_CASES_1(#FMT_"_SAT",FMT_,SZ_X_,SZ_Y_,PAD_X_,PAD_Y_,SATURATE)

#define TT_DECONVOLUTION_CASES_EXTRA(FMT_)      \
    TT_DECONVOLUTION_CASES_0(FMT_,11,11,0,0)    \
    TT_DECONVOLUTION_CASES_0(FMT_,6,6,0,0)      \
    TT_DECONVOLUTION_CASES_0(FMT_,5,5,0,0)      \
    TT_DECONVOLUTION_CASES_0(FMT_,3,3,0,0)      \
    TT_DECONVOLUTION_CASES_0(FMT_,3,4,1,2)

#define TT_DECONVOLUTION_CASES_ALL()    \
    TT_DECONVOLUTION_CASES_EXTRA(U8)

TEST_WITH_ARG(TensorNN, testDeconvolutionLayer, test_deconvolution_layer_arg,
        TT_DECONVOLUTION_CASES_ALL()
)
{
    assert (arg_->fmt == TT_Q78 || arg_->fmt == TT_U8 || arg_->fmt == TT_S8);
    assert (arg_->batching_dim >= 0);
    assert (arg_->bias_type == BIAS_NONE || arg_->bias_type == BIAS_SHARED || arg_->bias_type == BIAS_PER_LOC);
    assert (arg_->convert_policy == VX_CONVERT_POLICY_WRAP ||
            arg_->convert_policy == VX_CONVERT_POLICY_SATURATE);
    assert (arg_->rounding_policy == VX_ROUND_POLICY_TO_ZERO ||
            arg_->rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context_->vx_context_, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims >= (size_t)(3 + arg_->batching_dim));
    }

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type;
    vx_uint8 fixed_point_position;
    vx_size sizeof_data_type;
    ownUnpackFormat(arg_->fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    const size_t inout_dim_num = 3 + arg_->batching_dim;
    const size_t weight_dim_num = 4;
    const size_t bias_dim_num =
        arg_->bias_type == BIAS_NONE ? 0 :
        arg_->bias_type == BIAS_SHARED ? 1 : 3;

    size_t in_dims[4];
    size_t weight_dims[4];
    size_t bias_dims[3];
    size_t out_dims[4];

    size_t in_strides[4];
    size_t weight_strides[4];
    size_t bias_strides[3];
    size_t out_strides[4];

    for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
    {
        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("iter #: %d\n", iter);
            fflush(stdout);
        }

        size_t input_w, upscale_x, output_w;
        ownGetDeconvRandParams(
                &rng,
                arg_->padding_x, arg_->weight_w,
                arg_->a_x,
                &input_w, &upscale_x, &output_w);

        size_t input_h, upscale_y, output_h;
        ownGetDeconvRandParams(
                &rng,
                arg_->padding_y, arg_->weight_h,
                arg_->a_y,
                &input_h, &upscale_y, &output_h);

        in_dims[0] = input_w;
        in_dims[1] = input_h;
        for (vx_size i = 2; i < inout_dim_num; ++i)
        {
            in_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        }

        out_dims[0] = output_w;
        out_dims[1] = output_h;
        out_dims[2] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
        for (vx_size i = 3; i < inout_dim_num; ++i)
        {
            out_dims[i] = in_dims[i];
        }

        weight_dims[0] = arg_->weight_w;
        weight_dims[1] = arg_->weight_h;
        weight_dims[2] = in_dims[2];
        weight_dims[3] = out_dims[2];

        if (bias_dim_num == 1) { bias_dims[0] = out_dims[2]; }
        else if (bias_dim_num == 3)
        {
            bias_dims[0] = out_dims[0];
            bias_dims[1] = out_dims[1];
            bias_dims[2] = out_dims[2];
        }

        vx_tensor in_tensor = vxCreateTensor(context_->vx_context_, inout_dim_num, in_dims, data_type, fixed_point_position);
        vx_tensor weight_tensor = vxCreateTensor(context_->vx_context_, weight_dim_num, weight_dims, data_type, fixed_point_position);
        vx_tensor bias_tensor = bias_dim_num ? vxCreateTensor(context_->vx_context_, bias_dim_num, bias_dims, data_type, fixed_point_position) : NULL;
        vx_tensor out_tensor = vxCreateTensor(context_->vx_context_, inout_dim_num, out_dims, data_type, fixed_point_position);
        ASSERT_VX_OBJECT(in_tensor, VX_TYPE_TENSOR);
        ASSERT_VX_OBJECT(weight_tensor, VX_TYPE_TENSOR);
        if (bias_dim_num) { ASSERT_VX_OBJECT(in_tensor, VX_TYPE_TENSOR); }
        ASSERT_VX_OBJECT(out_tensor, VX_TYPE_TENSOR);

        ownGetFlatByteStrides(arg_->fmt, in_dims, inout_dim_num, in_strides);
        ownGetFlatByteStrides(arg_->fmt, weight_dims, weight_dim_num, weight_strides);
        ownGetFlatByteStrides(arg_->fmt, bias_dims, bias_dim_num, bias_strides);
        ownGetFlatByteStrides(arg_->fmt, out_dims, inout_dim_num, out_strides);

        if (DEBUG_TEST_TENSOR_ENABLE_PRINTF)
        {
            printf("\tconfig: {\n");
            printf("\t          in_dims: { "); for (size_t i = 0; i < inout_dim_num; ++i) { printf("%zu, ", in_dims[i]); } printf(" }, \n");
            printf("\t          weight_dims: { "); for (size_t i = 0; i < weight_dim_num; ++i) { printf("%zu, ", weight_dims[i]); } printf(" }, \n");
            if (bias_dim_num)
            {
                printf("\t          bias_dims: { "); for (size_t i = 0; i < bias_dim_num; ++i) { printf("%zu, ", bias_dims[i]); } printf(" }, \n");
            }
            printf("\t          out_dims: { "); for (size_t i = 0; i < inout_dim_num; ++i) { printf("%zu, ", out_dims[i]); } printf(" }, \n");
            printf("\t        }\n");
        }

        const size_t in_bytes = in_dims[inout_dim_num-1] * in_strides[inout_dim_num-1];
        const size_t weight_bytes = weight_dims[weight_dim_num-1] * weight_strides[weight_dim_num-1];
        const size_t bias_bytes = bias_dim_num ? bias_dims[bias_dim_num-1] * bias_strides[bias_dim_num-1] : 0;
        const size_t out_bytes = out_dims[inout_dim_num-1] * out_strides[inout_dim_num-1];

        const size_t in_count = in_bytes / sizeof_data_type;
        const size_t weight_count = weight_bytes / sizeof_data_type;
        const size_t bias_count = bias_bytes / sizeof_data_type;

        void * const in = malloc(in_bytes);
        void * const weight = malloc(weight_bytes);
        void * const bias = bias_dim_num ? malloc(bias_bytes) : NULL;
        void * const out = malloc(out_bytes);
        void * const refs = malloc(out_bytes);
        ASSERT(in && weight && (!bias_count || bias) && out && refs);

        {
            const int deconv_prod_count = arg_->weight_w * arg_->weight_h * in_dims[2];

            ownFillSmallRandData(arg_->fmt, &rng, in_count, deconv_prod_count, in);
            ownFillSmallRandData(arg_->fmt, &rng, weight_count, deconv_prod_count, weight);
            if (bias_dim_num) { ownFillRandData(arg_->fmt, &rng, bias_count, bias); }

            vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
            VX_CALL(vxCopyTensorPatch(in_tensor, inout_dim_num, view_start, in_dims, in_strides, in, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            VX_CALL(vxCopyTensorPatch(weight_tensor, weight_dim_num, view_start, weight_dims, weight_strides, weight, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            if (bias_dim_num)
            {
                VX_CALL(vxCopyTensorPatch(bias_tensor, bias_dim_num, view_start, bias_dims, bias_strides, bias, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            }
            VX_CALL(vxCopyTensorPatch(out_tensor, inout_dim_num, view_start, out_dims, out_strides, out, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        }

        {
            vx_graph graph = vxCreateGraph(context_->vx_context_);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

            const vx_nn_deconvolution_params_t params =
            {
                arg_->padding_x, arg_->padding_y, arg_->convert_policy, arg_->rounding_policy,
//                arg_->down_scale_size_rounding,
                arg_->a_x, arg_->a_y
            };
            vx_node node = vxDeconvolutionLayer(graph, in_tensor, weight_tensor, bias_tensor, &params, sizeof(params), out_tensor);
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
            tensor_desc_t in_td = { inout_dim_num, in_dims, in_strides };
            tensor_desc_t weight_td = { weight_dim_num, weight_dims, weight_strides };
            tensor_desc_t bias_td = { bias_dim_num, bias_dims, bias_strides };
            tensor_desc_t out_td = { inout_dim_num, out_dims, out_strides };

            ownDeconvolution(
                    arg_->fmt,
                    in, in_td,
                    weight, weight_td,
                    bias, bias_td,
                    arg_->padding_x, arg_->padding_y,
                    upscale_x, upscale_y,
                    arg_->convert_policy == VX_CONVERT_POLICY_WRAP,
                    arg_->rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN,
                    arg_->a_x, arg_->a_y,
                    refs, out_td);

            const vx_size view_start[5] = { 0 };
            VX_CALL(vxCopyTensorPatch(out_tensor, inout_dim_num, view_start, out_dims, out_strides, out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

            size_t first_diff_index;
            size_t first_diff_byte_offset0;
            size_t first_diff_byte_offset1;
            if (!ownExpectIdenticalData(
                        arg_->fmt,
                        out, out_dims, inout_dim_num, out_strides,
                        refs, out_dims, inout_dim_num, out_strides,
                        8, //0, //(arg_->fmt == TT_Q78 ? 1 : 0),
                        &first_diff_index,
                        &first_diff_byte_offset0,
                        &first_diff_byte_offset1))
            {
                printf("DIFF! { idx: %zu, out: ", first_diff_index);
                ownPrettyPrintVal(arg_->fmt, (char*)out + first_diff_byte_offset0);
                printf(", ref: ");
                ownPrettyPrintVal(arg_->fmt, (char*)refs + first_diff_byte_offset1);
                printf(" }\n");

                if (!DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR) ASSERT(0);
            }
        }

        VX_CALL(vxReleaseTensor(&in_tensor));
        VX_CALL(vxReleaseTensor(&weight_tensor));
        if (bias_dim_num) { VX_CALL(vxReleaseTensor(&bias_tensor)); }
        VX_CALL(vxReleaseTensor(&out_tensor));
        EXPECT_EQ_PTR(NULL, in_tensor);
        EXPECT_EQ_PTR(NULL, weight_tensor);
        EXPECT_EQ_PTR(NULL, bias_tensor);
        EXPECT_EQ_PTR(NULL, out_tensor);

        free(in);
        free(weight);
        free(bias);
        free(out);
        free(refs);
    }
}

TESTCASE_TESTS(TensorNN,
    /* vx_khr_nn.h function tests */
    testConvolutionLayer,
    testFullyConnectedLayer,
    testPoolingLayer,
    testSoftmaxLayer,
//    testNormalizationLayer,
    testActivationLayer,
    testROIPoolingLayer,
    testDeconvolutionLayer
)
#endif
