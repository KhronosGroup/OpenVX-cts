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

#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_NEURAL_NETWORKS || OPENVX_CONFORMANCE_NNEF_IMPORT

#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"
#include "test_tensor_util.h"

/* ***************************************************************************
//  Tensor tests
*/
TESTCASE(Tensor, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char * name;

    enum TestTensorDF fmt;
} test_tensor_arg;

TEST_WITH_ARG(Tensor, testvxCreateTensorFromHandle, test_tensor_arg,
    ARG("Q78_vxCreateTensorFromHandle", TT_Q78),
    ARG("U8_vxCreateTensorFromHandle", TT_U8),
    ARG("S8_vxCreateTensorFromHandle", TT_S8),
    )
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if (!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
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
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);
    size_t * const tensor_strides = ct_alloc_mem(sizeof(*tensor_strides) * max_dims);
    ASSERT(tensor_dims && tensor_strides);

    void * ptr = NULL;

    for (vx_size dims = 1; dims <= max_dims; ++dims)
    {
        for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
        {
            for (vx_size i = 0; i < dims; ++i)
            {
                tensor_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ + 1);

                tensor_strides[i] = i ? tensor_strides[i - 1] * tensor_dims[i - 1] : sizeof_data_type;
            }

            vx_tensor src_tensor = vxCreateTensor(context, dims, tensor_dims, data_type, fixed_point_position);
            ASSERT_VX_OBJECT(src_tensor, VX_TYPE_TENSOR);
            vx_tensor dst_tensor = vxCreateTensorFromHandle(context, dims, tensor_dims, data_type, fixed_point_position,
                tensor_strides, ptr, VX_MEMORY_TYPE_HOST);
            ASSERT_VX_OBJECT(dst_tensor, VX_TYPE_TENSOR);

            //check
            vx_size src_check_size, dst_check_size;
            void *src_check_ptr = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);
            void *dst_check_ptr = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);

            //VX_TENSOR_NUMBER_OF_DIMS
            src_check_size = dst_check_size = sizeof(vx_size);
            vxQueryTensor(src_tensor, VX_TENSOR_NUMBER_OF_DIMS, src_check_ptr, src_check_size);
            vxQueryTensor(dst_tensor, VX_TENSOR_NUMBER_OF_DIMS, dst_check_ptr, dst_check_size);
            EXPECT_EQ_INT((*(vx_size *)src_check_ptr), (*(vx_size *)dst_check_ptr));

            //VX_TENSOR_DIMS
            src_check_size = dst_check_size = sizeof(vx_size) * dims;
            vxQueryTensor(src_tensor, VX_TENSOR_DIMS, src_check_ptr, src_check_size);
            vxQueryTensor(dst_tensor, VX_TENSOR_DIMS, dst_check_ptr, dst_check_size);
            EXPECT_EQ_INT((*(vx_size *)src_check_ptr), (*(vx_size *)dst_check_ptr));

            //VX_TENSOR_DATA_TYPE
            src_check_size = dst_check_size = sizeof(vx_enum);
            vxQueryTensor(src_tensor, VX_TENSOR_DATA_TYPE, src_check_ptr, src_check_size);
            vxQueryTensor(dst_tensor, VX_TENSOR_DATA_TYPE, dst_check_ptr, dst_check_size);
            EXPECT_EQ_INT((*(vx_enum *)src_check_ptr), (*(vx_enum *)dst_check_ptr));

            //VX_TENSOR_FIXED_POINT_POSITION
            src_check_size = dst_check_size = sizeof(vx_int8);
            vxQueryTensor(src_tensor, VX_TENSOR_FIXED_POINT_POSITION, src_check_ptr, src_check_size);
            vxQueryTensor(dst_tensor, VX_TENSOR_FIXED_POINT_POSITION, dst_check_ptr, dst_check_size);
            EXPECT_EQ_INT((*(vx_int8 *)src_check_ptr), (*(vx_int8 *)dst_check_ptr));

            ct_free_mem(src_check_ptr);
            ct_free_mem(dst_check_ptr);

            VX_CALL(vxReleaseTensor(&src_tensor));
            VX_CALL(vxReleaseTensor(&dst_tensor));

            EXPECT_EQ_PTR(NULL, src_tensor);
            EXPECT_EQ_PTR(NULL, dst_tensor);
        }
    }

    ct_free_mem(tensor_dims);
    ct_free_mem(tensor_strides);
}

TEST_WITH_ARG(Tensor, testvxSwapTensorHandle, test_tensor_arg,
    ARG("Q78_vxSwapTensorHandle", TT_Q78),
    ARG("U8_vxSwapTensorHandle", TT_U8),
    ARG("S8_vxSwapTensorHandle", TT_S8),
    )
{
    const vx_context context = context_->vx_context_;

    vx_status ret;

    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    vx_size max_dims = 0;
    {
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if (!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    uint64_t rng2;
    {
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng2, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    size_t * const in0_dims = ct_alloc_mem(sizeof(*in0_dims) * max_dims);
    ASSERT(in0_dims);

    size_t * const in0_strides = ct_alloc_mem(sizeof(*in0_strides) * max_dims);
    ASSERT(in0_strides);

    for (vx_size i = 0; i < max_dims; ++i)
    {
        in0_dims[i] = CLAMP(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ / 2);
        in0_strides[i] = i ? in0_strides[i - 1] * in0_dims[i - 1] : sizeof_data_type;
    }

    vx_tensor in0_tensor = vxCreateTensor(context, max_dims, in0_dims, data_type, fixed_point_position);
    vx_tensor in1_tensor = vxCreateTensor(context, max_dims, in0_dims, data_type, fixed_point_position);
    ASSERT_VX_OBJECT(in0_tensor, VX_TYPE_TENSOR);
    ASSERT_VX_OBJECT(in1_tensor, VX_TYPE_TENSOR);

    size_t in0_bytes = 1;
    for (vx_size i = 0; i < max_dims; ++i)
    {
        in0_bytes *= in0_dims[i];
    }
    size_t malloc_bytes = in0_bytes * sizeof_data_type;

    void * in0_data = ct_alloc_mem(malloc_bytes);
    void * in1_data = ct_alloc_mem(malloc_bytes);
    void * out0_data = ct_alloc_mem(malloc_bytes);
    void * out1_data = ct_alloc_mem(malloc_bytes);
    ASSERT(in0_data && in1_data && out0_data &&out1_data);

    {
        ownFillRandData(fmt, &rng, in0_bytes, in0_data);
        ownFillRandData(fmt, &rng2, in0_bytes, in1_data);

        vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
        ret = vxCopyTensorPatch(in0_tensor, max_dims, view_start, in0_dims, in0_strides, in0_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, ret);
        ret = vxCopyTensorPatch(in1_tensor, max_dims, view_start, in0_dims, in0_strides, in1_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, ret);
    }

    //SWAP
    void* prev_ptrs[6] = { 0, 0, 0, 0, 0, 0 };
    ret = vxSwapTensorHandle(in0_tensor, in1_data, prev_ptrs);
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, ret);

    {
        const size_t view_start[MAX_TENSOR_DIMS] = { 0 };
        VX_CALL(vxCopyTensorPatch(in0_tensor, max_dims, view_start, in0_dims, in0_strides, out0_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        VX_CALL(vxCopyTensorPatch(in1_tensor, max_dims, view_start, in0_dims, in0_strides, out1_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

        // Verify the results for new_ptr
        for (size_t index = 0; index < in0_bytes; ++index)
        {
            size_t out_byte_offset = 0;
            vx_size index_leftover = index;
            int divisor = 1;
            for (vx_size i = 0; i < max_dims; i++)
            {
                divisor = (vx_uint32)(in0_dims[i]);
                vx_size curr_dim_index = index_leftover%divisor;
                out_byte_offset += in0_strides[i] * (curr_dim_index);
                index_leftover = index_leftover / divisor;
            }

            const char * out_b_ptr = (char*)out0_data + out_byte_offset;
            const char * ref_b_ptr = (char*)out1_data + out_byte_offset;

            switch (fmt)
            {
            case TT_Q78:
                {
                    const vx_int16 out = *(vx_int16*)out_b_ptr;
                    int16_t ref = *(vx_int16*)ref_b_ptr;
                    EXPECT_EQ_INT(ref, out);
                    break;
                }
            case TT_U8:
                {
                    const vx_uint8 out = *(vx_uint8*)out_b_ptr;
                    const uint8_t ref = *(vx_uint8*)ref_b_ptr;
                    EXPECT_EQ_INT(ref, out);
                    break;
                }
            case TT_S8:
                {
                    const vx_int8 out = *(vx_int8*)out_b_ptr;
                    const vx_int8 ref = *(vx_int8*)ref_b_ptr;
                    EXPECT_EQ_INT(ref, out);
                    break;
                }
            default: assert(0);
            }
        }

        //verify the result for prev_ptr
        vx_tensor tmp = vxCreateTensorFromHandle(context, max_dims, in0_dims, data_type, fixed_point_position, in0_strides, *prev_ptrs, VX_MEMORY_TYPE_HOST);
        VX_CALL(vxCopyTensorPatch(tmp, max_dims, view_start, in0_dims, in0_strides, out1_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        for (size_t index = 0; index < in0_bytes; ++index)
        {
            size_t out_byte_offset = 0;
            vx_size index_leftover = index;
            int divisor = 1;
            for (vx_size i = 0; i < max_dims; i++)
            {
                divisor = (vx_uint32)(in0_dims[i]);
                vx_size curr_dim_index = index_leftover%divisor;
                out_byte_offset += in0_strides[i] * (curr_dim_index);
                index_leftover = index_leftover / divisor;
            }
            const char * out_b_ptr = (char*)out1_data + out_byte_offset;
            const char * ref_b_ptr = (char*)in0_data + out_byte_offset;

            switch (fmt)
            {
            case TT_Q78:
            {
                const vx_int16 out = *(vx_int16*)out_b_ptr;
                int16_t ref = *(vx_int16*)ref_b_ptr;
                EXPECT_EQ_INT(ref, out);
                break;
            }
            case TT_U8:
            {
                const vx_uint8 out = *(vx_uint8*)out_b_ptr;
                const uint8_t ref = *(vx_uint8*)ref_b_ptr;
                EXPECT_EQ_INT(ref, out);
                break;
            }
            case TT_S8:
            {
                const vx_int8 out = *(vx_int8*)out_b_ptr;
                const vx_int8 ref = *(vx_int8*)ref_b_ptr;
                EXPECT_EQ_INT(ref, out);
                break;
            }
            default: assert(0);
            }
        }

        VX_CALL(vxReleaseTensor(&tmp));
        EXPECT_EQ_PTR(NULL, tmp);
    }

    VX_CALL(vxReleaseTensor(&in0_tensor));
    VX_CALL(vxReleaseTensor(&in1_tensor));
    EXPECT_EQ_PTR(NULL, in0_tensor);
    EXPECT_EQ_PTR(NULL, in1_tensor);

    ct_free_mem(in0_data);
    //No need free in1_data, as free by vxReleaseTensor(&in0_tensor)
    ct_free_mem(out0_data);
    ct_free_mem(out1_data);

    ct_free_mem(in0_dims);
    ct_free_mem(in0_strides);
}

TEST_WITH_ARG(Tensor, testMapandUnMapTensorPatch, test_tensor_arg,
    ARG("Q78_testMapandUnMapTensorPatch", TT_Q78),
    ARG("U8_testMapandUnMapTensorPatch", TT_U8),
    ARG("S8_testMapandUnMapTensorPatch", TT_S8),
    )
{
    const vx_context context = context_->vx_context_;

    vx_status ret;

    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    vx_size max_dims = 0;
    {
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if (!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
    }

    uint64_t rng;
    {
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    size_t * const in0_dims = ct_alloc_mem(sizeof(*in0_dims) * max_dims);
    ASSERT(in0_dims);

    size_t * const in0_strides = ct_alloc_mem(sizeof(*in0_strides) * max_dims);
    ASSERT(in0_strides);

    for (vx_size i = 0; i < max_dims; ++i)
    {
        in0_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ + 1);
        in0_strides[i] = i ? in0_strides[i - 1] * in0_dims[i - 1] : sizeof_data_type;
    }

    vx_tensor in0_tensor = vxCreateTensor(context, max_dims, in0_dims, data_type, fixed_point_position);
    ASSERT_VX_OBJECT(in0_tensor, VX_TYPE_TENSOR);

    size_t in0_bytes = 1;
    for (vx_size i = 0; i < max_dims; ++i)
    {
        in0_bytes *= in0_dims[i];
    }
    size_t malloc_bytes = in0_bytes * sizeof_data_type;

    void * in0_data = ct_alloc_mem(malloc_bytes);
    ASSERT(in0_data);
    //init in0_data
    ownFillRandData(fmt, &rng, in0_bytes, in0_data);
    vx_size view_start[MAX_TENSOR_DIMS] = { 0 };
    ret = vxCopyTensorPatch(in0_tensor, max_dims, view_start, in0_dims, in0_strides, in0_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, ret);

    vx_map_id map_id;
    void* plane_ptr = 0;

    ret = vxMapTensorPatch(in0_tensor, max_dims, view_start, in0_dims,
        &map_id, in0_strides, &plane_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, ret);

    for (size_t index = 0; index < in0_bytes; ++index)
    {
        const char * out_b_ptr = (char*)plane_ptr + index;
        const char * ref_b_ptr = (char*)in0_data + index;

        switch (fmt)
        {
        case TT_Q78:
        {
            const vx_int16 out = *(vx_int16*)out_b_ptr;
            int16_t ref = *(vx_int16*)ref_b_ptr;
            EXPECT_EQ_INT(ref, out);
            break;
        }
        case TT_U8:
        {
            const vx_uint8 out = *(vx_uint8*)out_b_ptr;
            const uint8_t ref = *(vx_uint8*)ref_b_ptr;
            EXPECT_EQ_INT(ref, out);
            break;
        }
        case TT_S8:
        {
            const vx_int8 out = *(vx_int8*)out_b_ptr;
            const vx_int8 ref = *(vx_int8*)ref_b_ptr;
            EXPECT_EQ_INT(ref, out);
            break;
        }
        default: assert(0);
        }
    }

    ret = vxUnmapTensorPatch(in0_tensor, map_id);
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, ret);

    VX_CALL(vxReleaseTensor(&in0_tensor));
    EXPECT_EQ_PTR(NULL, in0_tensor);

    ct_free_mem(in0_data);
    ct_free_mem(in0_dims);
    ct_free_mem(in0_strides);
}

TEST_WITH_ARG(Tensor, testvxCreateVirtualTensor, test_tensor_arg,
    ARG("Q78_vxCreateVirtualTensor", TT_Q78),
    ARG("U8_vxCreateVirtualTensor", TT_U8),
    ARG("S8_vxCreateVirtualTensor", TT_S8),
    )
{
    const vx_context context = context_->vx_context_;

    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8 || fmt == TT_S8);

    vx_size max_dims = 0;
    {   // TODO: ownTestGetMaxDims() ?
        VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
        ASSERT(max_dims > 3);
        if (!DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS) max_dims = 4; else max_dims = MIN(max_dims, MAX_TENSOR_DIMS);
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

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);
    ASSERT(tensor_dims);

    for (vx_size dims = 1; dims <= max_dims; ++dims)
    {
        for (int iter = 0; iter < TEST_TENSOR_NUM_ITERATIONS; ++iter)
        {
            for (vx_size i = 0; i < dims; ++i)
            {
                tensor_dims[i] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ + 1);
            }

            vx_graph graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
            vx_tensor src_tensor = vxCreateVirtualTensor(graph, dims, tensor_dims, data_type, fixed_point_position);
            ASSERT_VX_OBJECT(src_tensor, VX_TYPE_TENSOR);

            //check
            vx_size src_check_size = 0;
            vx_size expect_num_of_dims = 0;
            vx_enum expect_data_type = 0;
            vx_int8 expect_fixed_point_position = 0;
            size_t *src_check_ptr = ct_alloc_mem(sizeof(size_t) * max_dims);

            //VX_TENSOR_NUMBER_OF_DIMS
            src_check_size = sizeof(vx_size);
            vxQueryTensor(src_tensor, VX_TENSOR_NUMBER_OF_DIMS, (void *)(&(expect_num_of_dims)), src_check_size);
            EXPECT_EQ_INT(expect_num_of_dims, dims);

            //VX_TENSOR_DIMS
            src_check_size = sizeof(vx_size) * dims;
            vxQueryTensor(src_tensor, VX_TENSOR_DIMS, (void *)src_check_ptr, src_check_size);
            for (int tmpIdx = 0; tmpIdx < dims; tmpIdx++)
            {
                EXPECT_EQ_INT(src_check_ptr[tmpIdx], tensor_dims[tmpIdx]);
            }

            //VX_TENSOR_DATA_TYPE
            src_check_size = sizeof(vx_enum);
            vxQueryTensor(src_tensor, VX_TENSOR_DATA_TYPE, &expect_data_type, sizeof(vx_enum));
            EXPECT_EQ_INT(expect_data_type, data_type);

            //VX_TENSOR_FIXED_POINT_POSITION
            src_check_size = sizeof(vx_int8);
            vxQueryTensor(src_tensor, VX_TENSOR_FIXED_POINT_POSITION, &expect_fixed_point_position, src_check_size);
            EXPECT_EQ_INT(expect_fixed_point_position, fixed_point_position);

            ct_free_mem(src_check_ptr);

            VX_CALL(vxReleaseTensor(&src_tensor));
            VX_CALL(vxReleaseGraph(&graph));

            EXPECT_EQ_PTR(NULL, src_tensor);
            EXPECT_EQ_PTR(NULL, graph);
        }
    }

    ct_free_mem(tensor_dims);
}
#endif

#ifdef OPENVX_USE_ENHANCED_VISION
/* ***************************************************************************
 Test enhanced tensor interface:
         vxCreateImageObjectArrayFromTensor
*****************************************************************************/
TESTCASE(TensorEnhanced, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char * name;
    int width, height;
    enum TestTensorDF fmt;
} test_create_image_objectarray_tensor_arg;

#define TENSOR_FORMAT(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/fmt=Q78", __VA_ARGS__, TT_Q78)), \
    CT_EXPAND(nextmacro(testArgName "/fmt=U8", __VA_ARGS__, TT_U8))


#define CREATE_IMAGE_OBJECTARRAY_FROM_TENSOR_PARAMETERS \
    CT_GENERATE_PARAMETERS("Adjacent2D", ADD_SIZE_SMALL_SET, TENSOR_FORMAT, ARG)

TEST_WITH_ARG(TensorEnhanced, testvxCreateImageObjectArrayFromTensor,
              test_create_image_objectarray_tensor_arg,
              CREATE_IMAGE_OBJECTARRAY_FROM_TENSOR_PARAMETERS
    )
{
    const vx_context context = context_->vx_context_;
    const enum TestTensorDF fmt = arg_->fmt;
    assert(fmt == TT_Q78 || fmt == TT_U8 );
    vx_rectangle_t rect;
    vx_size array_size;
    vx_size stride;
    vx_df_image image_format = VX_DF_IMAGE_U8;
    vx_object_array objImgs = 0;
    vx_size view_start[3] = {0};

    vx_size max_dims = 3;
    vx_size dim1 = arg_->height;
    vx_size dim0 = arg_->width;

    rect.start_x = 0;
    rect.end_x = arg_->width;
    rect.start_y = 0;
    rect.end_y = arg_->height;

    uint64_t rng;
    {
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    size_t * const in0_dims = ct_alloc_mem(sizeof(*in0_dims) * max_dims);
    ASSERT(in0_dims);

    size_t * const in0_strides = ct_alloc_mem(sizeof(*in0_strides) * max_dims);
    ASSERT(in0_strides);

    in0_dims[2] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ + 1);
    in0_dims[0] = dim0;
    in0_dims[1] = dim1;
    for (vx_size i = 0; i < max_dims; ++i)
    {
        in0_strides[i] = i ? in0_strides[i - 1] * in0_dims[i - 1] : sizeof_data_type;
    }

    vx_tensor in0_tensor = vxCreateTensor(context, max_dims, in0_dims, data_type, fixed_point_position);
    ASSERT_VX_OBJECT(in0_tensor, VX_TYPE_TENSOR);

    size_t in0_bytes = 1;
    for (vx_size i = 0; i < max_dims; ++i)
    {
        in0_bytes *= in0_dims[i];
    }
    size_t malloc_bytes = in0_bytes * sizeof_data_type;

    void * in0_data = ct_alloc_mem(malloc_bytes);
    ASSERT(in0_data);
    //init in0_data
    ownFillRandData(fmt, &rng, in0_bytes, in0_data);
    VX_CALL(vxCopyTensorPatch(in0_tensor, max_dims, view_start, in0_dims, in0_strides, in0_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    array_size = in0_dims[2];
    stride = in0_strides[2];
    if (fmt == TT_Q78)
    {
        image_format = VX_DF_IMAGE_S16;
    }
    ASSERT_VX_OBJECT(objImgs = vxCreateImageObjectArrayFromTensor(in0_tensor, &rect, array_size, stride, image_format),
        VX_TYPE_OBJECT_ARRAY);

    //check result
    vx_size expect_itemnums = 0;
    VX_CALL(vxQueryObjectArray(objImgs, VX_OBJECT_ARRAY_NUMITEMS, (void *)&expect_itemnums, sizeof(expect_itemnums)));
    EXPECT_EQ_INT(expect_itemnums, array_size);

    VX_CALL(vxReleaseObjectArray(&objImgs));
    EXPECT_EQ_PTR(NULL, objImgs);

    VX_CALL(vxReleaseTensor(&in0_tensor));
    EXPECT_EQ_PTR(NULL, in0_tensor);

    ct_free_mem(in0_data);
    ct_free_mem(in0_dims);
    ct_free_mem(in0_strides);
}

#endif

#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_NEURAL_NETWORKS || OPENVX_CONFORMANCE_NNEF_IMPORT

TESTCASE_TESTS(Tensor,
    testvxCreateTensorFromHandle,
    testvxSwapTensorHandle,
    testMapandUnMapTensorPatch,
    testvxCreateVirtualTensor)

#endif

#ifdef OPENVX_USE_ENHANCED_VISION

TESTCASE_TESTS(TensorEnhanced, testvxCreateImageObjectArrayFromTensor)

#endif
