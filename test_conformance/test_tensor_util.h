//#define TT_ENABLE_MINIGRAPH_TEST



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


// Temporary defines for debug: Set to 0 when not used.
#define DEBUG_TEST_TENSOR_ENABLE_PRINTF 0
#define DEBUG_TEST_TENSOR_CONTINUE_AFTER_ERROR 0

// The conformance tests must only check the first 4 dimes, the rest being
// left to internal validation. However these tests already include support
// for upto the max reported dim by the context. Enabling this option will
// check dims > 4 upto 6, if supported by the impl.
#define DEBUG_TEST_TENSOR_BEYOND_FOUR_DIMS 1

//NOTE: TEST_TENSOR_MAX_DIM_SZ may be overriden in vxConvolutionLayer tests
#define TEST_TENSOR_NUM_ITERATIONS              1
#define TEST_TENSOR_MIN_DIM_SZ                  1
#define TEST_TENSOR_MAX_DIM_SZ                  20
#define TEST_TENSOR_INVERSE_MASK_PROBABILITY    4
#define TEST_TENSOR_INVERSE_SHRINK_PROBABILITY  8


/****************************************************************************
 *                                                                          *
 *                            Common Format Utils                           *
 *                                                                          *
 ***************************************************************************/

#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define MAX(a, b)   ((a) < (b) ? (b) : (a))
#define CLAMP(v, lower, upper) MAX((lower), MIN((v), (upper)))

#define Q78_FIXED_POINT_POSITION 8
#define Q78_SCALE   (1 << Q78_FIXED_POINT_POSITION)
#define Q78_HALF    (1 << (Q78_FIXED_POINT_POSITION - 1))

enum TestTensorDF
{
    TT_Q78,
    TT_U8,
    TT_S8,
};

static CT_INLINE int_fast32_t ownLoadValueAsRawInt(enum TestTensorDF fmt, const void * p)
{
    switch(fmt)
    {
        case TT_Q78: return *(vx_int16*)p;
        case TT_U8: return *(vx_uint8*)p;
        case TT_S8: return *(vx_int8*)p;
        default: assert(0); return 0;
    }
}

static CT_INLINE void ownStoreRawIntValue(enum TestTensorDF fmt, int_fast32_t val, void * p)
{
    switch(fmt)
    {
        case TT_Q78: *(vx_int16*)p = val; break;
        case TT_U8: *(vx_uint8*)p = val; break;
        case TT_S8: *(vx_int8*)p = val; break;
        default: assert(0);
    }
}

// Avoid impl defined behaviour when casting non representable values to signed
//TODO: is trunc indeed the type of cast the OpenVX spec demands??
static CT_INLINE int8_t trunc_to_int8(int_fast32_t val)
{
    union { int8_t i; uint8_t u; } tmp;
    tmp.u = val;
    return tmp.i;
}

// Avoid impl defined behaviour when casting non representable values to signed
//TODO: is trunc indeed the type of cast the OpenVX spec demands??
static CT_INLINE int16_t trunc_to_int16(int_fast32_t val)
{
    union { int16_t i; uint16_t u; } tmp;
    tmp.u = val;
    return tmp.i;
}

static CT_INLINE int_fast32_t ownWrapOrSat(enum TestTensorDF fmt, int_fast32_t val, bool wrap)
{
    switch(fmt)
    {
        case TT_Q78: return wrap? trunc_to_int16(val) : CLAMP(val, INT16_MIN, INT16_MAX);
        case TT_U8: return wrap? (uint8_t)val : CLAMP(val, 0, UINT8_MAX);
        case TT_S8: return wrap? trunc_to_int16(val) : CLAMP(val, INT8_MIN, INT8_MAX);
        default: assert(0); return 0;
    }
}

// Finalize the accum (sum of products) by applying rounding, norm and OF
//
// Rounding and scaling only apply to Q78, where the product results in 16
// "fractional" bits, rather than the normal 8.
static CT_INLINE int_fast32_t ownApplyWrapRoundingToAccum(
        enum TestTensorDF fmt, int_fast32_t val,
        bool wrap,  // true for WRAP, else SATURATE
        bool to_ne) // true for ROUND_TO_NE, else ROUND_TO_ZERO
{
    if (fmt == TT_Q78)
    {
       if (to_ne)
       {
           val += Q78_HALF;
       }

       val /= Q78_SCALE;
    }

    return ownWrapOrSat(fmt, val, wrap);
}

static CT_INLINE float ownUnquantize(enum TestTensorDF fmt, int_fast32_t val)
{
    return fmt == TT_Q78 ? ((float)val / Q78_SCALE) : val;
}

static CT_INLINE int_fast32_t ownQuantize(enum TestTensorDF fmt, float val)
{
    if (fmt == TT_Q78) val *= Q78_SCALE;

    return ownWrapOrSat(fmt, val, false);
}

static CT_INLINE int_fast32_t ownGetMinValue(enum TestTensorDF fmt)
{
    switch(fmt)
    {
        case TT_Q78: return INT16_MIN;
        case TT_U8: return 0;
        case TT_S8: return INT8_MIN;
        default: assert(0); return 0;
    }
}

static CT_INLINE int_fast32_t ownGetMaxValue(enum TestTensorDF fmt)
{
    switch(fmt)
    {
        case TT_Q78: return INT16_MAX;
        case TT_U8: return UINT8_MAX;
        case TT_S8: return INT8_MAX;
        default: assert(0); return 0;
    }
}

static int_fast32_t ownGetSizeofType(enum TestTensorDF fmt)
{
    switch(fmt)
    {
        case TT_Q78: return sizeof(vx_int16);
        case TT_U8: return sizeof(vx_uint8);
        case TT_S8: return sizeof(vx_int8);
        default: assert(0); return 1;
    }
}

static CT_INLINE void ownPrettyPrintVal(
        enum TestTensorDF fmt,
        void * v)
{
    switch(fmt)
    {
        case TT_Q78: printf("Q78{ .val: %f, .raw: %d }", *(vx_int16*)v / 256.f, *(vx_int16*)v); break;
        case TT_U8: printf("U8{ .val: %d }", *(vx_uint8*)v); break;
        case TT_S8: printf("S8{ .val: %d }", *(vx_int8*)v); break;
        default: assert(0);
    }
}


/****************************************************************************
 *                                                                          *
 *                            Common Tensor Utils                           *
 *                                                                          *
 ***************************************************************************/

// TODO: get rid of this, the test shouldn't have a hardcoded dim num!!!
// We assume that the OVX context supports no more than MAX_TENSOR_DIMS
// dimensions. This is used for the explicit for iterator as well array
// sizes. In practice only min(MAX_TENSOR_DIMS, OVX supported max dims)
// are used by the test.
// We should avoid it by looping up-to the item count and % by the dims
// as well as using dynamic arrays for the views and strides, if this
// won't suffice...
#define MAX_TENSOR_DIMS 6

typedef struct {
    size_t dim_num;
    const size_t * dims;
    const size_t * strides;
} tensor_desc_t;

static CT_INLINE size_t ownGetFlatByteOffset(
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

static CT_INLINE size_t ownGetFlatByteOffsetWithBroadcast(
        size_t index,
        vx_size dim_num,
        const vx_size * in_dims,
        const vx_size * in_strides,
        const vx_size * out_dims)
{
    size_t res = 0;

    for (vx_size d = 0; d < dim_num; ++d)
    {
        if (in_dims[d] == out_dims[d])
            res += in_strides[d] * (index % out_dims[d]);

        index /= out_dims[d];
    }

    return res;
}

static size_t ownGetItemCount(vx_size dim_num, const vx_size * dims)
{
    if (!dim_num) return 0;

    size_t res = dims[0];
    for (vx_size i = 1; i < dim_num; ++i)
        res *= dims[i];

    return res;
}

static CT_INLINE void ownGetFlatByteStrides(
        enum TestTensorDF fmt,
        const size_t * dims,
        size_t dim_num,
        /*OUT*/ size_t * strides)
{
    const size_t sizeof_type = ownGetSizeofType(fmt);

    for (size_t i = 0; i < dim_num; ++i)
    {
        strides[i] = i ? strides[i-1] * dims[i-1] : sizeof_type;
    }
}

// Since we calc offsets manually and cast to ptr type, we expect the strides
// to have the correct alignment
static void ownAssertStridesModSizeof(enum TestTensorDF fmt, tensor_desc_t td)
{
    const size_t sizeof_type = ownGetSizeofType(fmt);

    for (size_t i = 0; i < td.dim_num; ++i)
    {
        assert(td.strides[i] % sizeof_type == 0);
    }
}


/****************************************************************************
 *                                                                          *
 *                              Generic Test Code                           *
 *                                                                          *
 ***************************************************************************/

#define I64_ABS_DIFF(a, b) ((a) < (b) ? (int64_t)(b) - (a) : (int64_t)(a) - (b))

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
        case TT_S8:
            *data_type = VX_TYPE_INT8;
            *fixed_point_position = 0;
            *sizeof_data_type = sizeof(vx_int8);
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
        case TT_S8:
            for(size_t i = 0; i < count; ++i)
                ((vx_int8*)data)[i] = (vx_int8)CT_RNG_NEXT_INT(*rng, INT8_MIN, INT8_MAX+1);
            break;
        default:
            assert(0);
    }
}

// Some test for things like MatrixMultiply and Convolution perform a sum of
// products. The accumulator for these formats is supposed to have 32 bits
// and the behaviour for overflowing the accumulator is impl. defined.
// We therefore need to use sufficiently small values to avoid this issue in
// the tests.
static void ownFillSmallRandData(
        enum TestTensorDF fmt,
        uint64_t * rng,
        size_t count,
        int estimated_item_summation_count,
        /*OUT*/ void * data)
{
    switch(fmt)
    {
        case TT_Q78:
            {
                int16_t lower = INT16_MIN / sqrt(estimated_item_summation_count);
                int16_t upper = INT16_MAX / sqrt(estimated_item_summation_count);

                for(size_t i = 0; i < count; ++i)
                    ((vx_int16*)data)[i] = (vx_int16)CT_RNG_NEXT_INT(*rng, lower, upper + 1);
            }
            break;
        case TT_U8:
            {
                uint8_t upper = UINT8_MAX / sqrt(estimated_item_summation_count);

                for(size_t i = 0; i < count; ++i)
                    ((vx_uint8*)data)[i] = (vx_uint8)CT_RNG_NEXT_INT(*rng, 0, upper + 1);
            }
            break;
        case TT_S8:
            {
                int8_t lower = INT8_MIN / sqrt(estimated_item_summation_count);
                int8_t upper = INT8_MAX / sqrt(estimated_item_summation_count);

                for(size_t i = 0; i < count; ++i)
                    ((vx_int8*)data)[i] = (vx_int8)CT_RNG_NEXT_INT(*rng, lower, upper + 1);
            }
            break;
        default:
            assert(0);
    }
}

// Expecting identical item count, check if the contents are identical
// upto layout.
static bool ownExpectIdenticalData(
        enum TestTensorDF fmt,
        const void * data0, const vx_size * dims0, vx_size dim_num0, const vx_size * strides0,
        const void * data1, const vx_size * dims1, vx_size dim_num1, const vx_size * strides1,
        const int max_raw_int_diff,
        /*OUT*/ size_t * first_diff_index,          // only updated if res is false
        /*OUT*/ size_t * first_diff_byte_offset0,   // only updated if res is false
        /*OUT*/ size_t * first_diff_byte_offset1)   // only updated if res is false
{
    size_t count = ownGetItemCount(dim_num0, dims0);
    assert(count == ownGetItemCount(dim_num1, dims1));

    for (size_t i = 0; i < count; ++i)
    {
        const size_t byte_offset0 = ownGetFlatByteOffset(i, dim_num0, dims0, strides0);
        const size_t byte_offset1 = ownGetFlatByteOffset(i, dim_num1, dims1, strides1);

        int32_t a, b;

        switch(fmt)
        {
            case TT_Q78:
                a = *(vx_int16*)((char*)data0 + byte_offset0);
                b = *(vx_int16*)((char*)data1 + byte_offset1);
                break;
            case TT_U8:
                a = *(vx_uint8*)((char*)data0 + byte_offset0);
                b = *(vx_uint8*)((char*)data1 + byte_offset1);
                break;
            case TT_S8:
                a = *(vx_int8*)((char*)data0 + byte_offset0);
                b = *(vx_int8*)((char*)data1 + byte_offset1);
                break;
            default:
                assert(0);
        }

        if (I64_ABS_DIFF(a, b) > max_raw_int_diff) {
            if (first_diff_index) *first_diff_index = i;
            if (first_diff_byte_offset0) *first_diff_byte_offset0 = byte_offset0;
            if (first_diff_byte_offset1) *first_diff_byte_offset1 = byte_offset1;
            
            if (max_raw_int_diff)
            {
                EXPECT_EQ_INT(I64_ABS_DIFF(a, b) > max_raw_int_diff, 0);
            }
            else
            {
                EXPECT_EQ_INT(a, b);
            }

            return false;
        }
    }

    return true;
}

