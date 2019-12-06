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

#include "test_engine/test.h"

#include <VX/vx_types.h>
#include <VX/vx_khr_nn.h>
#include <VX/vxu.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <float.h>
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

#define  COLOR_WEIGHT_SIZE_PER_CHANNEL      256

static void getMinMax(const void* src, const vx_size* src_strides, const vx_size* dims, vx_size num_of_dims,
                      vx_int16 *max_value, vx_int16 *min_value)
{
    vx_int16 maxVal = INT16_MIN;
    vx_int16 minVal = INT16_MAX;
    if (num_of_dims == 2)
    {
        for (vx_uint32 y = 0; y < dims[1]; y++)
        {
            for (vx_uint32 x = 0; x < dims[0]; x++)
            {
                vx_uint32 offset = y * src_strides[1] + x * src_strides[0];
                vx_int16 val = *(vx_int16 *)((vx_int8 *)src + offset);
                if (val > maxVal)
                {
                    maxVal = val;
                }
                if (val < minVal)
                {
                    minVal = val;
                }
            }
        }
        *max_value = maxVal;
        *min_value = minVal;
    }
    else if (num_of_dims == 3)
    {
        for (vx_uint32 y = 0; y < dims[2]; y++)
        {
            for (vx_uint32 x = 0; x < dims[1]; x++)
            {
                for (vx_uint32 z = 0; z < dims[0]; z++)
                {
                    vx_uint32 offset = y * src_strides[2] + x * src_strides[1] + z * src_strides[0];
                    vx_int16 val = *(vx_int16 *)((vx_int8 *)src + offset);
                    if (val > maxVal)
                    {
                        maxVal = val;
                    }
                    if (val < minVal)
                    {
                        minVal = val;
                    }
                }
            }
        }
        *max_value = maxVal;
        *min_value = minVal;
    }
}

static void releaseRes(void *pData)
{
    if (NULL != pData)
    {
        ct_free_mem(pData);
    }
    return;
}

static vx_status calcColorWeight(vx_uint8 cn, vx_float64 gauss_color_coeff, vx_float32 **color_weight)
{
    vx_float32 *tmp_weight = (vx_float32 *)ct_alloc_mem(cn * COLOR_WEIGHT_SIZE_PER_CHANNEL * sizeof(vx_float32));
    if (NULL == tmp_weight)
    {
        return VX_ERROR_NO_MEMORY;
    }

    for (vx_int32 i = 0; i < (cn * COLOR_WEIGHT_SIZE_PER_CHANNEL); i++)
    {
        tmp_weight[i] = (vx_float32)exp(i * i * gauss_color_coeff);
    }

    *color_weight = tmp_weight;

    return VX_SUCCESS;
}

static vx_status calcSpaceWeight(vx_int32 diameter, vx_float64 gauss_space_coeff, vx_float32 **space_weight)
{
    vx_int32 radius = diameter / 2;
    vx_float32 *tmp_weight = (vx_float32 *)ct_alloc_mem(diameter * diameter * sizeof(vx_float32));
    if (NULL == tmp_weight)
    {
        return VX_ERROR_NO_MEMORY;
    }

    for (vx_int32 i = -radius; i <= radius; i++)
    {
        vx_int32 j = -radius;
        for (; j <= radius; j++)
        {
            vx_float64 r = sqrt((vx_float64)i * i + (vx_float64)j * j);
            if (r > radius)
            {
                continue;
            }
            tmp_weight[(i + radius) * diameter + (j + radius)] = (vx_float32)exp(r * r * gauss_space_coeff);
        }
    }

    *space_weight = tmp_weight;

    return VX_SUCCESS;
}


static void ownCheckBilateralFilterResult(
        const void * in_ptr, const vx_size * in_dims, const vx_size * in_strides,
        enum TestTensorDF fmt,
        vx_size dim_num,
        int   diameter,
        float sigmaSpace,
        float sigmaColor,
        void * out_ptr, const vx_size * out_dims, const vx_size * out_strides,
        vx_border_t border)
{
    vx_status status = VX_SUCCESS;
    vx_float32 tolerance = 0.0;
    vx_float32 total_num = 0.0;
    vx_float32 equal_num = 0.0;
    vx_int32 y = 0, x = 0;
    vx_int32 low_x, low_y, high_x, high_y;
    vx_int32 radius_y, radius_x;
    vx_float32 scale_index = 0;
    vx_int32 radius = diameter / 2;
    vx_enum border_mode = border.mode;
    vx_int16 out = 0, ref = 0;

    vx_float32 *color_weight = NULL;
    vx_float32 *space_weight = NULL;
    vx_uint8 cn = dim_num == 2 ? 1 : 3;

    vx_float64 gauss_color_coeff = -0.5/(sigmaColor*sigmaColor);
    vx_float64 gauss_space_coeff = -0.5/(sigmaSpace*sigmaSpace);

    if (border.mode == VX_BORDER_UNDEFINED)
    {
        low_x = radius;
        high_x = (in_dims[dim_num - 2] >= radius) ? in_dims[dim_num - 2] - radius : 0;
        low_y = radius;
        high_y = (in_dims[dim_num - 1] >= radius) ? in_dims[dim_num - 1] - radius : 0;
    }
    else
    {
        low_x = 0;
        high_x = in_dims[dim_num - 2];
        low_y = 0;
        high_y = in_dims[dim_num - 1];
    }

    if (fmt == TT_Q78)
    {
        vx_int16 minVal = -1;
        vx_int16 maxVal = 1;
        getMinMax(in_ptr, in_strides, in_dims, dim_num, &maxVal, &minVal);
        if ((vx_float32)(abs(maxVal - minVal)) < FLT_EPSILON)
        {
            if (dim_num == 2)
            {
                for (y = low_y; y < high_y; y++)
                {
                    for (x = low_x; x < high_x; x++)
                    {
                        out = *((vx_int16 *)((vx_uint8 *)out_ptr + y * in_strides[1] + x * in_strides[0]));
                        ref = *((vx_int16 *)((vx_uint8 *)in_ptr + y * in_strides[1] + x * in_strides[0]));
                        if (out == ref)
                        {
                            equal_num += 1;
                        }
                        total_num += 1;
                    }
                }
                tolerance = (equal_num / total_num);
                ASSERT(tolerance >= MIN_TOLERANCE);
                return;
            }
            else if (dim_num == 3)
            {
                for (y = low_y; y < high_y; y++)
                {
                    for (x = low_x; x < high_x; x++)
                    {
                        out = *(vx_int16 *)((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 0 * out_strides[0]);
                        ref = *(vx_int16 *)((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 0 * in_strides[0]);
                        if (out == ref)
                        {
                            equal_num += 1;
                        }
                        out = *(vx_int16 *)((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 1 * out_strides[0]);
                        ref = *(vx_int16 *)((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 1 * in_strides[0]);
                        if (out == ref)
                        {
                            equal_num += 1;
                        }
                        out = *(vx_int16 *)((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 2 * out_strides[0]);
                        ref = *(vx_int16 *)((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 2 * in_strides[0]);
                        if (out == ref)
                        {
                            equal_num += 1;
                        }
                        total_num += 3;
                    }
                }
                tolerance = (equal_num / total_num);
                ASSERT(tolerance >= MIN_TOLERANCE);
                return;
            }

            ASSERT(tolerance >= MIN_TOLERANCE);
            return;
        }

        //calculation color weight
        vx_int32 kExpNumBinsPerChannel = 1 << 12;
        vx_float32 lastExpVal = 1.f;
        vx_float32 len;
        vx_int32 kExpNumBins;
        len = (vx_float32)(maxVal - minVal) * cn;
        kExpNumBins = kExpNumBinsPerChannel * cn;
        color_weight = (vx_float32 *)ct_alloc_mem((kExpNumBins + 2) * sizeof(vx_float32));
        if (NULL == color_weight)
        {
            ASSERT(tolerance >= MIN_TOLERANCE);
            return;
        }
        scale_index = kExpNumBins / len;
        for (vx_uint32 i = 0; i < (kExpNumBins + 2); i++)
        {
            if (lastExpVal > 0.f)
            {
                vx_float64 val = i / scale_index;
                color_weight[i] = (vx_float32)exp(val * val * gauss_color_coeff);
                lastExpVal = color_weight[i];
            }
            else
            {
                color_weight[i] = 0.f;
            }
        }
    }
    else if (fmt == TT_U8)
    {
        (void)calcColorWeight(cn, gauss_color_coeff, &color_weight);
    }
    status = calcSpaceWeight(diameter, gauss_space_coeff, &space_weight);
    if (status != VX_SUCCESS)
    {
        releaseRes(color_weight);
        releaseRes(space_weight);
    }

    if (dim_num == 2)
    {
        for (y = low_y; y < high_y; y++)
        {
            for (x = low_x; x < high_x; x++)
            {
                vx_int16 value = 0;
                if (fmt == TT_U8)
                {
                    out = *((vx_uint8 *)out_ptr + y * out_strides[1] + x * out_strides[0]);
                    value = *((vx_uint8 *)in_ptr + y * in_strides[1] + x * in_strides[0]);
                }
                else if (fmt == TT_Q78)
                {
                    out = *((vx_int16 *)((vx_uint8 *)out_ptr + y * out_strides[1] + x * out_strides[0]));
                    value = *((vx_int16 *)((vx_uint8 *)in_ptr + y * in_strides[1] + x * in_strides[0]));
                }

                vx_float32 sum = 0, wsum = 0;
                //kernel filter
                for (radius_y = -radius; radius_y <= radius; radius_y++)
                {
                    for (radius_x = -radius; radius_x <= radius; radius_x++)
                    {
                        vx_float64 r = sqrt((vx_float64)radius_y * radius_y + (vx_float64)radius_x * radius_x);
                        if (r > radius)
                        {
                            continue;
                        }
                        vx_int32 neighbor_x = x + radius_x;
                        vx_int32 neighbor_y = y + radius_y;
                        vx_int16 neighborVal = 0;
                        if (border_mode == VX_BORDER_REPLICATE)
                        {
                            vx_int32 tmpx = neighbor_x < 0 ? 0 : (neighbor_x >((vx_int32)in_dims[0] - 1) ? ((vx_int32)in_dims[0] - 1) : neighbor_x);
                            vx_int32 tmpy = neighbor_y < 0 ? 0 : (neighbor_y >((vx_int32)in_dims[1] - 1) ? ((vx_int32)in_dims[1] - 1) : neighbor_y);
                            if (fmt == TT_U8)
                            {
                                neighborVal = *((vx_uint8 *)in_ptr + tmpy * in_strides[1] + tmpx * in_strides[0]);
                            }
                            else if (fmt == TT_Q78)
                            {
                                neighborVal = *((vx_int16 *)((vx_uint8 *)in_ptr + tmpy * in_strides[1] + tmpx * in_strides[0]));
                            }
                        }
                        else if (border_mode == VX_BORDER_CONSTANT)
                        {
                            vx_int32 tmpx = neighbor_x < 0 ? 0 : (neighbor_x >((vx_int32)in_dims[0] - 1) ? ((vx_int32)in_dims[0] - 1) : neighbor_x);
                            vx_int32 tmpy = neighbor_y < 0 ? 0 : (neighbor_y >((vx_int32)in_dims[1] - 1) ? ((vx_int32)in_dims[1] - 1) : neighbor_y);
                            if (neighbor_x < 0 || neighbor_y < 0)
                            {
                                if (fmt == TT_U8)
                                {
                                    neighborVal = border.constant_value.U8;
                                }
                                else if (fmt == TT_Q78)
                                {
                                    neighborVal = border.constant_value.S16;
                                }
                            }
                            else
                            {
                                if (fmt == TT_U8)
                                {
                                    neighborVal = *((vx_uint8 *)in_ptr + tmpy * in_strides[1] + tmpx * in_strides[0]);
                                }
                                else if (fmt == TT_Q78)
                                {
                                    neighborVal = *((vx_int16 *)((vx_uint8 *)in_ptr + tmpy * in_strides[1] + tmpx * in_strides[0]));
                                }
                            }
                        }

                        vx_float32 w = 0;
                        if (fmt == TT_U8)
                        {
                            w = space_weight[(radius_y + radius) * diameter + (radius_x + radius)] *
                                       color_weight[abs(neighborVal - value)];
                        }
                        else if (fmt == TT_Q78)
                        {
                            vx_float32 alpha = abs(neighborVal - value) * scale_index;
                            vx_int32 idx = (vx_int32)floorf(alpha);
                            alpha -= idx;
                            w = space_weight[(radius_y + radius) * diameter + (radius_x + radius)] *
                                (color_weight[idx] + alpha * (color_weight[idx + 1] - color_weight[idx]));
                        }
                        sum += neighborVal * w;
                        wsum += w;
                    }
                }

                if (fmt == TT_U8)
                {
                    ref = (vx_uint8)roundf(sum / wsum);
                }
                else if (fmt == TT_Q78)
                {
                    ref = (vx_int16)roundf(sum / wsum);
                }

                total_num += 1;

                if (ref == out)
                {
                    equal_num += 1;
                }
            }
        }
    }
    else if (dim_num == 3)
    {
        for (y = low_y; y < high_y; y++)
        {
            for (x = low_x; x < high_x; x++)
            {
                vx_int16 b0 = 0, g0 = 0, r0 = 0;
                vx_int16 outb = 0, outg = 0, outr = 0;
                if (fmt == TT_U8)
                {
                    outb = *((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 0 * out_strides[0]);
                    outg = *((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 1 * out_strides[0]);
                    outr = *((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 2 * out_strides[0]);
                    b0 = *((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 0 * in_strides[0]);
                    g0 = *((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 1 * in_strides[0]);
                    r0 = *((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 2 * in_strides[0]);
                }
                else if (fmt == TT_Q78)
                {
                    outb = *((vx_int16 *)((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 0 * out_strides[0]));
                    outg = *((vx_int16 *)((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 1 * out_strides[0]));
                    outr = *((vx_int16 *)((vx_uint8 *)out_ptr + y * out_strides[2] + x * out_strides[1] + 2 * out_strides[0]));
                    b0 = *((vx_int16 *)((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 0 * in_strides[0]));
                    g0 = *((vx_int16 *)((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 1 * in_strides[0]));
                    r0 = *((vx_int16 *)((vx_uint8 *)in_ptr + y * in_strides[2] + x * in_strides[1] + 2 * in_strides[0]));
                }
                vx_float32 sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                //kernel filter
                for (radius_y = -radius; radius_y <= radius; radius_y++)
                {
                    for (radius_x = -radius; radius_x <= radius; radius_x++)
                    {
                        vx_float64 dist = sqrt((vx_float64)radius_y * radius_y + (vx_float64)radius_x * radius_x);
                        if (dist > radius)
                        {
                            continue;
                        }
                        vx_int32 neighbor_x = x + radius_x;
                        vx_int32 neighbor_y = y + radius_y;
                        vx_int16 b = 0, g = 0, r = 0;
                        if (border_mode == VX_BORDER_REPLICATE)
                        {
                            vx_int32 tmpx = neighbor_x < 0 ? 0 : (neighbor_x >((vx_int32)in_dims[1] - 1) ? ((vx_int32)in_dims[1] - 1) : neighbor_x);
                            vx_int32 tmpy = neighbor_y < 0 ? 0 : (neighbor_y >((vx_int32)in_dims[2] - 1) ? ((vx_int32)in_dims[2] - 1) : neighbor_y);
                            if (fmt == TT_U8)
                            {
                                b = *((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 0 * in_strides[0]);
                                g = *((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 1 * in_strides[0]);
                                r = *((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 2 * in_strides[0]);
                            }
                            else if (fmt == TT_Q78)
                            {
                                b = *((vx_int16 *)((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 0 * in_strides[0]));
                                g = *((vx_int16 *)((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 1 * in_strides[0]));
                                r = *((vx_int16 *)((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 2 * in_strides[0]));
                            }
                        }
                        else if (border_mode == VX_BORDER_CONSTANT)
                        {
                            vx_int32 tmpx = neighbor_x < 0 ? 0 : (neighbor_x >((vx_int32)in_dims[1] - 1) ? ((vx_int32)in_dims[1] - 1) : neighbor_x);
                            vx_int32 tmpy = neighbor_y < 0 ? 0 : (neighbor_y >((vx_int32)in_dims[2] - 1) ? ((vx_int32)in_dims[2] - 1) : neighbor_y);
                            if (neighbor_x < 0 || neighbor_y < 0)
                            {
                                if (fmt == TT_U8)
                                {
                                    b = g = r = border.constant_value.U8;
                                }
                                else if (fmt == TT_Q78)
                                {
                                    b = g = r  = border.constant_value.S16;
                                }
                            }
                            else
                            {
                                if (fmt == TT_U8)
                                {
                                    b = *((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 0 * in_strides[0]);
                                    g = *((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 1 * in_strides[0]);
                                    r = *((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 2 * in_strides[0]);
                                }
                                else if (fmt == TT_Q78)
                                {
                                    b = *((vx_int16 *)((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 0 * in_strides[0]));
                                    g = *((vx_int16 *)((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 1 * in_strides[0]));
                                    r = *((vx_int16 *)((vx_uint8 *)in_ptr + tmpy * in_strides[2] + tmpx * in_strides[1] + 2 * in_strides[0]));
                                }
                            }
                        }

                        vx_float32 w = 0;
                        if (fmt == TT_U8)
                        {
                            w = space_weight[(radius_y + radius) * diameter + (radius_x + radius)] *
                                       color_weight[abs(b - b0) + abs(g - g0) + abs(r - r0)];
                        }
                        else if (fmt == TT_Q78)
                        {
                            vx_float32 alpha = (abs(b- b0) + abs(g - g0) + abs(r - r0)) * scale_index;
                            vx_int32 idx = (vx_int32)floorf(alpha);
                            alpha -= idx;
                            w = space_weight[(radius_y + radius) * diameter + (radius_x + radius)] *
                                (color_weight[idx] + alpha * (color_weight[idx + 1] - color_weight[idx]));
                        }
                        sum_b += b * w;
                        sum_g += g * w;
                        sum_r += r * w;
                        wsum += w;
                    }
                }

                vx_int16 refb = 0, refg = 0, refr = 0;
                if (fmt == TT_U8)
                {
                    refb = (vx_uint8)roundf(sum_b / wsum);
                    refg = (vx_uint8)roundf(sum_g / wsum);
                    refr = (vx_uint8)roundf(sum_r / wsum);
                }
                else if (fmt == TT_Q78)
                {
                    refb = (vx_int16)roundf(sum_b / wsum);
                    refg = (vx_int16)roundf(sum_g / wsum);
                    refr = (vx_int16)roundf(sum_r / wsum);
                }

                total_num += 3;

                if (refb == outb)
                {
                    equal_num += 1;
                }
                if (refg == outg)
                {
                    equal_num += 1;
                }
                if (refr == outr)
                {
                    equal_num += 1;
                }
            }
        }
    }

    tolerance = (equal_num / total_num);

    ASSERT(tolerance >= MIN_TOLERANCE);

    releaseRes(color_weight);
    releaseRes(space_weight);
}

/****************************************************************************
 *                                                                          *
 *                          Test vxBilateralFilterNode                      *
 *                                                                          *
 ***************************************************************************/

TESTCASE(BilateralFilter, CT_VXContext, ct_setup_vx_context, 0)

static void* bilateral_generate_random(int width, int height, int cn, enum TestTensorDF tensor_fmt)
{
    vx_enum data_type = VX_TYPE_UINT8;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = sizeof(vx_uint8);
    size_t count = 0;
    uint64_t rng;
    {
        uint64_t * seed = &CT()->seed_;
        //ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    ownUnpackFormat(tensor_fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    count = width * height * cn;
    void *data = ct_alloc_mem(count * sizeof_data_type);
    if (data != NULL)
    {
        ownFillRandData(tensor_fmt, &rng, count, data);
    }

    return data;
}

typedef struct {
    const char* testName;
    void* (*generator)(int width, int height, int cn, enum TestTensorDF tensor_fmt);
    const char* fileName;
    vx_border_t border;
    int width, height;
    int cn;
    int diameter;
    float sigmaSpace;
    float sigmaColor;
    enum TestTensorDF tensor_fmt;
} bilateral_arg;

#define BILATERAL_FILTER_BORDERS(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_REPLICATE", __VA_ARGS__, { VX_BORDER_REPLICATE, {{ 0 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=0", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 0 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=1", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 1 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=127", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 127 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=255", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 255 }} }))

#define BILATERAL_CHANNEL(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/channel=1", __VA_ARGS__, 1)), \
    CT_EXPAND(nextmacro(testArgName "/channel=3", __VA_ARGS__, 3))

#define BILATERAL_DIAMETER(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/diameter=5", __VA_ARGS__, 5)), \
    CT_EXPAND(nextmacro(testArgName "/diameter=7", __VA_ARGS__, 7)), \
    CT_EXPAND(nextmacro(testArgName "/diameter=9", __VA_ARGS__, 9)) \

#define BILATERAL_SPACE_WEIGHT(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sigmaSpace=10", __VA_ARGS__, 10))

#define BILATERAL_COLOR_WEIGHT(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sigmaColor=5", __VA_ARGS__, 5))

#define BILATERAL_FORMAT(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/TT_U8", __VA_ARGS__, TT_U8)), \
    CT_EXPAND(nextmacro(testArgName "/TT_Q78", __VA_ARGS__, TT_Q78))

#define BILATERAL_PARAMETERS \
    CT_GENERATE_PARAMETERS("randam", BILATERAL_FILTER_BORDERS, ADD_SIZE_SMALL_SET, BILATERAL_CHANNEL, BILATERAL_DIAMETER, BILATERAL_SPACE_WEIGHT, BILATERAL_COLOR_WEIGHT, BILATERAL_FORMAT, ARG, bilateral_generate_random, NULL)

TEST_WITH_ARG(BilateralFilter, testGraphProcessing, bilateral_arg,
        BILATERAL_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    const enum TestTensorDF src_fmt = arg_->tensor_fmt;
    const enum TestTensorDF dst_fmt = arg_->tensor_fmt;
    assert(src_fmt == TT_Q78 || src_fmt == TT_U8);
    assert(dst_fmt == TT_Q78 || dst_fmt == TT_U8);
    const vx_border_t border = arg_->border;
    const int diameter = arg_->diameter;
    const float sigmaSpace = arg_->sigmaSpace;
    const float sigmaColor = arg_->sigmaColor;
    const int cn = arg_->cn;
    const int width = arg_->width;
    const int height = arg_->height;
    vx_size num_of_dims = 2;

    vx_enum src_data_type = 0;
    vx_enum dst_data_type = 0;
    vx_uint8 src_fixed_point_position = 0;
    vx_uint8 dst_fixed_point_position= 0;
    vx_size src_sizeof_data_type = 1;
    vx_size dst_sizeof_data_type = 1;
    ownUnpackFormat(src_fmt, &src_data_type, &src_fixed_point_position, &src_sizeof_data_type);
    ownUnpackFormat(dst_fmt, &dst_data_type, &dst_fixed_point_position, &dst_sizeof_data_type);

    if (cn == 3)
    {
        num_of_dims = 3;
    }

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * num_of_dims);
    size_t * const src_tensor_strides = ct_alloc_mem(sizeof(*src_tensor_strides) * num_of_dims);
    size_t * const dst_tensor_strides = ct_alloc_mem(sizeof(*dst_tensor_strides) * num_of_dims);
    ASSERT(tensor_dims && src_tensor_strides && dst_tensor_strides);

    if (num_of_dims == 3)
    {
        tensor_dims[0] = 3;
        tensor_dims[1] = width;
        tensor_dims[2] = height;

        src_tensor_strides[0] = src_sizeof_data_type;
        src_tensor_strides[1] = tensor_dims[0] * src_tensor_strides[0];
        src_tensor_strides[2] = tensor_dims[1] * src_tensor_strides[1];

        dst_tensor_strides[0] = src_tensor_strides[0];
        dst_tensor_strides[1] = src_tensor_strides[1];
        dst_tensor_strides[2] = src_tensor_strides[2];
    }
    else
    {
        tensor_dims[0] = width;
        tensor_dims[1] = height;

        src_tensor_strides[0] = src_sizeof_data_type;
        src_tensor_strides[1] = tensor_dims[0] * src_tensor_strides[0];

        dst_tensor_strides[0] = src_tensor_strides[0];
        dst_tensor_strides[1] = src_tensor_strides[1];
    }

    const size_t dst_tensor_bytes = tensor_dims[num_of_dims-1] * dst_tensor_strides[num_of_dims-1];

    vx_tensor src_tensor = vxCreateTensor(context, num_of_dims, tensor_dims, src_data_type, src_fixed_point_position);
    vx_tensor dst_tensor = vxCreateTensor(context, num_of_dims, tensor_dims, dst_data_type, dst_fixed_point_position);

    void * const dst_data = ct_alloc_mem(dst_tensor_bytes);
    vx_size *view_start = (vx_size *)ct_alloc_mem(num_of_dims * sizeof(vx_size));
    memset(view_start, 0, num_of_dims * sizeof(vx_size));

    void *src_data = NULL;
    src_data = arg_->generator(arg_->width, arg_->height, arg_->cn, arg_->tensor_fmt);
    VX_CALL(vxCopyTensorPatch(src_tensor, num_of_dims, view_start, tensor_dims, src_tensor_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    vx_graph graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    vx_node node = vxBilateralFilterNode(graph, src_tensor, diameter,  sigmaSpace, sigmaColor, dst_tensor);

    ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
    VX_CALL(vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border)));
    VX_CALL(vxReleaseNode(&node));
    EXPECT_EQ_PTR(NULL, node);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseGraph(&graph));
    EXPECT_EQ_PTR(NULL, graph);

    VX_CALL(vxCopyTensorPatch(dst_tensor, num_of_dims, view_start, tensor_dims, dst_tensor_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    ownCheckBilateralFilterResult(
                    src_data, tensor_dims, src_tensor_strides,
                    dst_fmt,
                    num_of_dims,
                    diameter,
                    sigmaSpace,
                    sigmaColor,
                    dst_data, tensor_dims, dst_tensor_strides,
                    border);

    VX_CALL(vxReleaseTensor(&src_tensor));
    VX_CALL(vxReleaseTensor(&dst_tensor));
    EXPECT_EQ_PTR(NULL, src_tensor);
    EXPECT_EQ_PTR(NULL, dst_tensor);

    ct_free_mem(src_data);
    ct_free_mem(dst_data);
    ct_free_mem(view_start);
    ct_free_mem(tensor_dims);
    ct_free_mem(src_tensor_strides);
    ct_free_mem(dst_tensor_strides);
}

TEST_WITH_ARG(BilateralFilter, testImmediateProcessing, bilateral_arg,
        BILATERAL_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    const enum TestTensorDF src_fmt = arg_->tensor_fmt;
    const enum TestTensorDF dst_fmt = arg_->tensor_fmt;
    assert(src_fmt == TT_Q78 || src_fmt == TT_U8);
    assert(dst_fmt == TT_Q78 || dst_fmt == TT_U8);
    const vx_border_t border = arg_->border;
    const int diameter = arg_->diameter;
    const float sigmaSpace = arg_->sigmaSpace;
    const float sigmaColor = arg_->sigmaColor;
    const int cn = arg_->cn;
    const int width = arg_->width;
    const int height = arg_->height;
    vx_size num_of_dims = 2;

    vx_enum src_data_type = 0;
    vx_enum dst_data_type = 0;
    vx_uint8 src_fixed_point_position = 0;
    vx_uint8 dst_fixed_point_position = 0;
    vx_size src_sizeof_data_type = 1;
    vx_size dst_sizeof_data_type = 1;
    ownUnpackFormat(src_fmt, &src_data_type, &src_fixed_point_position, &src_sizeof_data_type);
    ownUnpackFormat(dst_fmt, &dst_data_type, &dst_fixed_point_position, &dst_sizeof_data_type);

    if (cn == 3)
    {
        num_of_dims = 3;
    }

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * num_of_dims);
    size_t * const src_tensor_strides = ct_alloc_mem(sizeof(*src_tensor_strides) * num_of_dims);
    size_t * const dst_tensor_strides = ct_alloc_mem(sizeof(*dst_tensor_strides) * num_of_dims);
    ASSERT(tensor_dims && src_tensor_strides && dst_tensor_strides);

    if (num_of_dims == 3)
    {
        tensor_dims[0] = 3;
        tensor_dims[1] = width;
        tensor_dims[2] = height;

        src_tensor_strides[0] = src_sizeof_data_type;
        src_tensor_strides[1] = tensor_dims[0] * src_tensor_strides[0];
        src_tensor_strides[2] = tensor_dims[1] * src_tensor_strides[1];

        dst_tensor_strides[0] = src_tensor_strides[0];
        dst_tensor_strides[1] = src_tensor_strides[1];
        dst_tensor_strides[2] = src_tensor_strides[2];
    }
    else
    {
        tensor_dims[0] = width;
        tensor_dims[1] = height;

        src_tensor_strides[0] = src_sizeof_data_type;
        src_tensor_strides[1] = tensor_dims[0] * src_tensor_strides[0];

        dst_tensor_strides[0] = src_tensor_strides[0];
        dst_tensor_strides[1] = src_tensor_strides[1];
    }

    const size_t dst_tensor_bytes = tensor_dims[num_of_dims-1] * dst_tensor_strides[num_of_dims-1];

    vx_tensor src_tensor = vxCreateTensor(context, num_of_dims, tensor_dims, src_data_type, src_fixed_point_position);
    vx_tensor dst_tensor = vxCreateTensor(context, num_of_dims, tensor_dims, dst_data_type, dst_fixed_point_position);

    void * const dst_data = ct_alloc_mem(dst_tensor_bytes);
    vx_size *view_start = (vx_size *)ct_alloc_mem(num_of_dims * sizeof(vx_size));
    memset(view_start, 0, num_of_dims * sizeof(vx_size));

    void *src_data = NULL;
    src_data = arg_->generator(arg_->width, arg_->height, arg_->cn, arg_->tensor_fmt);
    VX_CALL(vxCopyTensorPatch(src_tensor, num_of_dims, view_start, tensor_dims, src_tensor_strides, src_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));
    VX_CALL(vxuBilateralFilter(context, src_tensor, diameter, sigmaSpace, sigmaColor, dst_tensor));

    VX_CALL(vxCopyTensorPatch(dst_tensor, num_of_dims, view_start, tensor_dims, dst_tensor_strides, dst_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    ownCheckBilateralFilterResult(
                    src_data, tensor_dims, src_tensor_strides,
                    dst_fmt,
                    num_of_dims,
                    diameter,
                    sigmaSpace,
                    sigmaColor,
                    dst_data, tensor_dims, dst_tensor_strides,
                    border);

    VX_CALL(vxReleaseTensor(&src_tensor));
    VX_CALL(vxReleaseTensor(&dst_tensor));
    EXPECT_EQ_PTR(NULL, src_tensor);
    EXPECT_EQ_PTR(NULL, dst_tensor);

    ct_free_mem(src_data);
    ct_free_mem(dst_data);
    ct_free_mem(view_start);
    ct_free_mem(tensor_dims);
    ct_free_mem(src_tensor_strides);
    ct_free_mem(dst_tensor_strides);
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

    size_t * const tensor_dims = ct_alloc_mem(sizeof(*tensor_dims) * max_dims);
    size_t * const src_tensor_strides = ct_alloc_mem(sizeof(*src_tensor_strides) * max_dims);
    size_t * const dst_tensor_strides = ct_alloc_mem(sizeof(*dst_tensor_strides) * max_dims);
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

        void * const src_data = ct_alloc_mem(src_tensor_bytes);
        void * const dst_data = ct_alloc_mem(dst_tensor_bytes);
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

        ct_free_mem(src_data);
        ct_free_mem(dst_data);
    }

    ct_free_mem(tensor_dims);
    ct_free_mem(src_tensor_strides);
    ct_free_mem(dst_tensor_strides);
}

TESTCASE_TESTS(BilateralFilter,
    testNodeCreation,
    testGraphProcessing,
    testImmediateProcessing
);

#endif //OPENVX_USE_ENHANCED_VISION
