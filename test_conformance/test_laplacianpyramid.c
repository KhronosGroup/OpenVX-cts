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

#include <math.h>
#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"

#define VX_GAUSSIAN_PYRAMID_TOLERANCE 1

TESTCASE(LaplacianPyramid, CT_VXContext, ct_setup_vx_context, 0)

TEST(LaplacianPyramid, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image   input = 0;
    vx_pyramid laplacian = 0;
    vx_image   output = 0;
    vx_graph   graph = 0;
    vx_node    node = 0;
    const vx_size levels = 4;
    const vx_float32 scale = VX_SCALE_PYRAMID_HALF;
    const vx_uint32 width = 640;
    const vx_uint32 height = 480;
    const vx_df_image format = VX_DF_IMAGE_S16;
    vx_uint32 w = width;
    vx_uint32 h = height;
    vx_size L = levels - 1;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, width, height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(laplacian = vxCreatePyramid(context, levels, scale, width, height, format), VX_TYPE_PYRAMID);

    while (L--)
    {
        w = (vx_uint32)(w * scale);
        h = (vx_uint32)(h * scale);
    }

    ASSERT_VX_OBJECT(output = vxCreateImage(context, w * scale, h * scale, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxLaplacianPyramidNode(graph, input, laplacian, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    VX_CALL(vxReleaseImage(&input));
    VX_CALL(vxReleasePyramid(&laplacian));
    VX_CALL(vxReleaseImage(&output));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(input == 0);
    ASSERT(laplacian == 0);
    ASSERT(output == 0);
    ASSERT(node == 0);
    ASSERT(graph == 0);
}

#define LEVELS_COUNT_MAX    7

static CT_Image own_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
        image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}

static CT_Image own_read_image(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}

static vx_size own_pyramid_calc_max_levels_count(int width, int height, vx_float32 scale)
{
    vx_size level = 1;

    while ((16 <= width) && (16 <= height) && level < LEVELS_COUNT_MAX)
    {
        level++;
        width = (int)ceil((vx_float64)width * scale);
        height = (int)ceil((vx_float64)height * scale);
    }

    return level;
}

static vx_status ownCopyImage(vx_image input, vx_image output)
{
    vx_status status = VX_SUCCESS; // assume success until an error occurs.
    vx_uint32 p = 0;
    vx_uint32 y = 0, x = 0;
    vx_size planes = 0;

    void* src;
    void* dst;
    vx_imagepatch_addressing_t src_addr;
    vx_imagepatch_addressing_t dst_addr;
    vx_rectangle_t src_rect, dst_rect;
    vx_map_id map_id1;
    vx_map_id map_id2;
    vx_df_image src_format = 0;
    vx_df_image out_format = 0;

    status |= vxQueryImage(input, VX_IMAGE_PLANES, &planes, sizeof(planes));
    vxQueryImage(output, VX_IMAGE_FORMAT, &out_format, sizeof(out_format));
    vxQueryImage(input, VX_IMAGE_FORMAT, &src_format, sizeof(src_format));
    status |= vxGetValidRegionImage(input, &src_rect);
    status |= vxGetValidRegionImage(output, &dst_rect);
    for (p = 0; p < planes && status == VX_SUCCESS; p++)
    {
        status = VX_SUCCESS;
        src = NULL;
        dst = NULL;

        status |= vxMapImagePatch(input, &src_rect, p, &map_id1, &src_addr, &src, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        status |= vxMapImagePatch(output, &dst_rect, p, &map_id2, &dst_addr, &dst, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

        for (y = 0; y < src_addr.dim_y && status == VX_SUCCESS; y += src_addr.step_y)
        {
            for (x = 0; x < src_addr.dim_x && status == VX_SUCCESS; x += src_addr.step_x)
            {
                void* srcp = vxFormatImagePatchAddress2d(src, x, y, &src_addr);
                void* dstp = vxFormatImagePatchAddress2d(dst, x, y, &dst_addr);
                vx_int32 out0 = (src_format == VX_DF_IMAGE_U8) ? (*(vx_uint8 *)srcp) : (*(vx_int16 *)srcp);

                if (out_format == VX_DF_IMAGE_U8)
                {
                    if (out0 > UINT8_MAX)
                        out0 = UINT8_MAX;
                    else if (out0 < 0)
                        out0 = 0;
                    *(vx_uint8 *)dstp = (vx_uint8)out0;
                }
                else
                {
                    if (out0 > INT16_MAX)
                        out0 = INT16_MAX;
                    else if (out0 < INT16_MIN)
                        out0 = INT16_MIN;
                    *(vx_int16 *)dstp = (vx_int16)out0;
                }
            }
        }

        if (status == VX_SUCCESS)
        {
            status |= vxUnmapImagePatch(input, map_id1);
            status |= vxUnmapImagePatch(output, map_id2);
        }
    }

    return status;
}

static vx_bool own_read_pixel_16s(void *base, vx_imagepatch_addressing_t *addr,
    vx_int32 x, vx_int32 y, const vx_border_t *borders, vx_int16 *pixel)
{
    vx_uint32 bx;
    vx_uint32 by;
    vx_int16* bpixel;

    vx_bool out_of_bounds = (vx_bool)(x < 0 || y < 0 || x >= (vx_int32)addr->dim_x || y >= (vx_int32)addr->dim_y);

    if (out_of_bounds)
    {
        if (borders->mode == VX_BORDER_UNDEFINED)
            return vx_false_e;
        if (borders->mode == VX_BORDER_CONSTANT)
        {
            *pixel = (vx_int16)borders->constant_value.S16;
            return vx_true_e;
        }
    }

    // bounded x/y
    bx = x < 0 ? 0 : x >= (vx_int32)addr->dim_x ? addr->dim_x - 1 : (vx_uint32)x;
    by = y < 0 ? 0 : y >= (vx_int32)addr->dim_y ? addr->dim_y - 1 : (vx_uint32)y;

    bpixel = (vx_int16*)vxFormatImagePatchAddress2d(base, bx, by, addr);
    *pixel = *bpixel;

    return vx_true_e;
}

static vx_status ownScaleImageNearestS16(vx_image src_image, vx_image dst_image, const vx_border_t *borders)
{
    vx_status status = VX_SUCCESS;
    vx_int32 x1, y1, x2, y2;
    void* src_base = NULL;
    void* dst_base = NULL;
    vx_rectangle_t src_rect;
    vx_rectangle_t dst_rect;
    vx_imagepatch_addressing_t src_addr;
    vx_imagepatch_addressing_t dst_addr;
    vx_uint32 w1 = 0, h1 = 0, w2 = 0, h2 = 0;
    vx_float32 wr, hr;
    vx_map_id map_id1;
    vx_map_id map_id2;

    vxQueryImage(src_image, VX_IMAGE_WIDTH, &w1, sizeof(w1));
    vxQueryImage(src_image, VX_IMAGE_HEIGHT, &h1, sizeof(h1));

    vxQueryImage(dst_image, VX_IMAGE_WIDTH, &w2, sizeof(w2));
    vxQueryImage(dst_image, VX_IMAGE_HEIGHT, &h2, sizeof(h2));

    src_rect.start_x = src_rect.start_y = 0;
    src_rect.end_x = w1;
    src_rect.end_y = h1;

    dst_rect.start_x = dst_rect.start_y = 0;
    dst_rect.end_x = w2;
    dst_rect.end_y = h2;

    status = VX_SUCCESS;
    status |= vxMapImagePatch(src_image, &src_rect, 0, &map_id1, &src_addr, &src_base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    status |= vxMapImagePatch(dst_image, &dst_rect, 0, &map_id2, &dst_addr, &dst_base, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

    wr = (vx_float32)w1 / (vx_float32)w2;
    hr = (vx_float32)h1 / (vx_float32)h2;

    for (y2 = 0; y2 < (vx_int32)dst_addr.dim_y; y2 += dst_addr.step_y)
    {
        for (x2 = 0; x2 < (vx_int32)dst_addr.dim_x; x2 += dst_addr.step_x)
        {
            vx_int16 v = 0;
            vx_int16* dst = vxFormatImagePatchAddress2d(dst_base, x2, y2, &dst_addr);
            vx_float32 x_src = ((vx_float32)x2 + 0.5f)*wr - 0.5f;
            vx_float32 y_src = ((vx_float32)y2 + 0.5f)*hr - 0.5f;
            vx_float32 x_min = floorf(x_src);
            vx_float32 y_min = floorf(y_src);
            x1 = (vx_int32)x_min;
            y1 = (vx_int32)y_min;

            if (x_src - x_min >= 0.5f)
                x1++;
            if (y_src - y_min >= 0.5f)
                y1++;

            if (dst && vx_true_e == own_read_pixel_16s(src_base, &src_addr, x1, y1, borders, &v))
                *dst = v;
        }
    }

    status |= vxUnmapImagePatch(src_image, map_id1);
    status |= vxUnmapImagePatch(dst_image, map_id2);

    return VX_SUCCESS;
}

static const vx_uint32 gaussian5x5scale = 256;
static const vx_int16 gaussian5x5[5][5] =
{
    { 1, 4, 6, 4, 1 },
    { 4, 16, 24, 16, 4 },
    { 6, 24, 36, 24, 6 },
    { 4, 16, 24, 16, 4 },
    { 1, 4, 6, 4, 1 }
};

static vx_convolution vxCreateGaussian5x5Convolution(vx_context context)
{
    vx_convolution conv = vxCreateConvolution(context, 5, 5);
    vx_status status = vxCopyConvolutionCoefficients(conv, (vx_int16 *)gaussian5x5, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != VX_SUCCESS)
    {
        vxReleaseConvolution(&conv);
        return NULL;
    }

    status = vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, (void *)&gaussian5x5scale, sizeof(vx_uint32));
    if (status != VX_SUCCESS)
    {
        vxReleaseConvolution(&conv);
        return NULL;
    }
    return conv;
}

void readRect(const void *base, const vx_imagepatch_addressing_t *addr, const vx_border_t *borders, vx_df_image type,
    vx_uint32 center_x, vx_uint32 center_y, vx_uint32 radius_x, vx_uint32 radius_y, void *destination)
{
    vx_int32 width = (vx_int32)addr->dim_x, height = (vx_int32)addr->dim_y;
    vx_int32 stride_y = addr->stride_y;
    vx_int32 stride_x = addr->stride_x;
    const vx_uint8 *ptr = (const vx_uint8 *)base;
    vx_int32 ky, kx;
    vx_uint32 dest_index = 0;
    // kx, kx - kernel x and y
    if (borders->mode == VX_BORDER_REPLICATE || borders->mode == VX_BORDER_UNDEFINED)
    {
        for (ky = -(int32_t)radius_y; ky <= (int32_t)radius_y; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            y = y < 0 ? 0 : (y >= height ? height - 1 : y);

            for (kx = -(int32_t)radius_x; kx <= (int32_t)radius_x; ++kx, ++dest_index)
            {
                vx_int32 x = (int32_t)(center_x + kx);
                x = x < 0 ? 0 : (x >= width ? width - 1 : x);

                switch (type)
                {
                case VX_DF_IMAGE_U8:
                    ((vx_uint8*)destination)[dest_index] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                    break;
                case VX_DF_IMAGE_S16:
                case VX_DF_IMAGE_U16:
                    ((vx_uint16*)destination)[dest_index] = *(vx_uint16*)(ptr + y*stride_y + x*stride_x);
                    break;
                case VX_DF_IMAGE_S32:
                case VX_DF_IMAGE_U32:
                    ((vx_uint32*)destination)[dest_index] = *(vx_uint32*)(ptr + y*stride_y + x*stride_x);
                    break;
                default:
                    abort();
                }
            }
        }
    }
    else if (borders->mode == VX_BORDER_CONSTANT)
    {
        vx_pixel_value_t cval = borders->constant_value;
        for (ky = -(int32_t)radius_y; ky <= (int32_t)radius_y; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            int ccase_y = y < 0 || y >= height;

            for (kx = -(int32_t)radius_x; kx <= (int32_t)radius_x; ++kx, ++dest_index)
            {
                vx_int32 x = (int32_t)(center_x + kx);
                int ccase = ccase_y || x < 0 || x >= width;

                switch (type)
                {
                case VX_DF_IMAGE_U8:
                    if (!ccase)
                        ((vx_uint8*)destination)[dest_index] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint8*)destination)[dest_index] = (vx_uint8)cval.U8;
                    break;
                case VX_DF_IMAGE_S16:
                case VX_DF_IMAGE_U16:
                    if (!ccase)
                        ((vx_uint16*)destination)[dest_index] = *(vx_uint16*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint16*)destination)[dest_index] = (vx_uint16)cval.U16;
                    break;
                case VX_DF_IMAGE_S32:
                case VX_DF_IMAGE_U32:
                    if (!ccase)
                        ((vx_uint32*)destination)[dest_index] = *(vx_uint32*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint32*)destination)[dest_index] = (vx_uint32)cval.U32;
                    break;
                default:
                    abort();
                }
            }
        }
    }
    else
        abort();
}

#define CONV_DIM 5
#define CONV_DIM_HALF CONV_DIM / 2

#define INSERT_ZERO_Y(slice, y) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*(1-y)+i] = 0;
#define INSERT_VALUES_Y(slice, y) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*(high_y-y)+i+CONV_DIM_HALF*CONV_DIM] = slice[CONV_DIM*(high_y-y)+i];
#define INSERT_ZERO_X(slice, x) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*i+1-x] = 0;
#define INSERT_VALUES_X(slice, x) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*i+(high_x-x)+CONV_DIM_HALF] = slice[CONV_DIM*i+(high_x-x)];

#define C_MAX_CONVOLUTION_DIM 15
vx_status convolve(vx_image src, vx_convolution conv, vx_image dst, vx_border_t *bordermode)
{
    vx_int32 y, x, i;
    void *src_base = NULL;
    void *dst_base = NULL;
    vx_imagepatch_addressing_t src_addr, dst_addr;
    vx_rectangle_t rect;

    vx_size conv_width, conv_height;
    vx_int32 conv_radius_x, conv_radius_y;
    vx_int16 conv_mat[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };
    vx_int32 sum = 0, value = 0;
    vx_uint32 scale = 1;
    vx_df_image src_format = 0;
    vx_df_image dst_format = 0;
    vx_status status = VX_SUCCESS;
    vx_int32 low_x, low_y, high_x, high_y;

    vx_map_id src_map_id;
    vx_map_id dst_map_id;

    status |= vxQueryImage(src, VX_IMAGE_FORMAT, &src_format, sizeof(src_format));
    status |= vxQueryImage(dst, VX_IMAGE_FORMAT, &dst_format, sizeof(dst_format));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_COLUMNS, &conv_width, sizeof(conv_width));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_ROWS, &conv_height, sizeof(conv_height));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_SCALE, &scale, sizeof(scale));
    conv_radius_x = (vx_int32)conv_width / 2;
    conv_radius_y = (vx_int32)conv_height / 2;
    status |= vxCopyConvolutionCoefficients(conv, conv_mat, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxGetValidRegionImage(src, &rect);
    status |= vxMapImagePatch(src, &rect, 0, &src_map_id, &src_addr, (void **)&src_base,
                              VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    status |= vxMapImagePatch(dst, &rect, 0, &dst_map_id, &dst_addr, (void **)&dst_base,
                              VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

    low_x = 0;
    high_x = src_addr.dim_x;
    low_y = 0;
    high_y = src_addr.dim_y;

    for (y = low_y; y < high_y; ++y)
    {
        for (x = low_x; x < high_x; ++x)
        {
            sum = 0;

            if (src_format == VX_DF_IMAGE_U8)
            {
                vx_uint8 slice[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };

                readRect(src_base, &src_addr, bordermode, src_format, x, y, conv_radius_x, conv_radius_y, slice);

                // purpose of this section is to compensate extra terms caused by replicate border mode (it is the only one allowed)

                if (y < CONV_DIM_HALF)
                {
                    INSERT_ZERO_Y(slice, y)
                }
                else if (y >= high_y - CONV_DIM_HALF)
                {
                    INSERT_VALUES_Y(slice, y)
                }

                if (x < CONV_DIM_HALF)
                {
                    INSERT_ZERO_X(slice, x)
                }
                else if (x >= high_x - CONV_DIM_HALF)
                {
                    INSERT_VALUES_X(slice, x)
                }

                for (i = 0; i < (vx_int32)(conv_width * conv_height); ++i)
                    sum += conv_mat[conv_width * conv_height - 1 - i] * slice[i];
            }
            else if (src_format == VX_DF_IMAGE_S16)
            {
                vx_int16 slice[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };

                readRect(src_base, &src_addr, bordermode, src_format, x, y, conv_radius_x, conv_radius_y, slice);

                if (y < CONV_DIM_HALF)
                {
                    INSERT_ZERO_Y(slice, y)
                }
                else if (y >= high_y - CONV_DIM_HALF)
                {
                    INSERT_VALUES_Y(slice, y)
                }

                if (x < CONV_DIM_HALF)
                {
                    INSERT_ZERO_X(slice, x)
                }
                else if (x >= high_x - CONV_DIM_HALF)
                {
                    INSERT_VALUES_X(slice, x)
                }

                for (i = 0; i < (vx_int32)(conv_width * conv_height); ++i)
                    sum += conv_mat[conv_width * conv_height - 1 - i] * slice[i];
            }

            value = sum / (vx_int32)scale;

            if (dst_format == VX_DF_IMAGE_U8)
            {
                vx_uint8 *dstp = vxFormatImagePatchAddress2d(dst_base, x, y, &dst_addr);
                if (value < 0) *dstp = 0;
                else if (value > UINT8_MAX) *dstp = UINT8_MAX;
                else *dstp = value;
            }
            else if (dst_format == VX_DF_IMAGE_S16)
            {
                vx_int16 *dstp = vxFormatImagePatchAddress2d(dst_base, x, y, &dst_addr);
                if (value < INT16_MIN) *dstp = INT16_MIN;
                else if (value > INT16_MAX) *dstp = INT16_MAX;
                else *dstp = value;
            }
        }
    }

    status |= vxUnmapImagePatch(src, src_map_id);
    status |= vxUnmapImagePatch(dst, dst_map_id);

    return status;
}

static vx_status upsampleImage(vx_context context, vx_uint32 width, vx_uint32 height, vx_image filling, vx_convolution conv, vx_image upsample, vx_border_t *border)
{
    vx_status status = VX_SUCCESS;
    vx_df_image format, filling_format;

    format = VX_DF_IMAGE_U8;
    vx_image tmp = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    status |= vxQueryImage(filling, VX_IMAGE_FORMAT, &filling_format, sizeof(filling_format));

    vx_rectangle_t tmp_rect, filling_rect;
    vx_imagepatch_addressing_t tmp_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_imagepatch_addressing_t filling_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_map_id tmp_map_id, filling_map_id;
    void *tmp_base = NULL;
    void *filling_base = NULL;

    status = vxGetValidRegionImage(tmp, &tmp_rect);
    status |= vxMapImagePatch(tmp, &tmp_rect, 0, &tmp_map_id, &tmp_addr, (void **)&tmp_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
    status = vxGetValidRegionImage(filling, &filling_rect);
    status |= vxMapImagePatch(filling, &filling_rect, 0, &filling_map_id, &filling_addr, (void **)&filling_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);

    for (vx_uint32 ix = 0; ix < width; ix++)
    {
        for (vx_uint32 iy = 0; iy < height; iy++)
        {

            void* tmp_datap = vxFormatImagePatchAddress2d(tmp_base, ix, iy, &tmp_addr);

            if (iy % 2 != 0 || ix % 2 != 0)
            {
                if (format == VX_DF_IMAGE_U8)
                    *(vx_uint8 *)tmp_datap = (vx_uint8)0;
                else
                    *(vx_int16 *)tmp_datap = (vx_int16)0;
            }
            else
            {
                void* filling_tmp = vxFormatImagePatchAddress2d(filling_base, ix / 2, iy / 2, &filling_addr);
                vx_int32 filling_data = filling_format == VX_DF_IMAGE_U8 ? *(vx_uint8 *)filling_tmp : *(vx_int16 *)filling_tmp;
                if (format == VX_DF_IMAGE_U8)
                {
                    if (filling_data > UINT8_MAX)
                        filling_data = UINT8_MAX;
                    else if (filling_data < 0)
                        filling_data = 0;
                    *(vx_uint8 *)tmp_datap = (vx_uint8)filling_data;
                }
                else
                {
                    if (filling_data > INT16_MAX)
                        filling_data = INT16_MAX;
                    else if (filling_data < INT16_MIN)
                        filling_data = INT16_MIN;
                    *(vx_int16 *)tmp_datap = (vx_int16)filling_data;
                }
            }
        }
    }

    status |= vxUnmapImagePatch(tmp, tmp_map_id);
    status |= vxUnmapImagePatch(filling, filling_map_id);

    status |= convolve(tmp, conv, upsample, border);

    vx_rectangle_t upsample_rect;
    vx_imagepatch_addressing_t upsample_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_map_id upsample_map_id;
    void * upsample_base = NULL;
    vx_df_image upsample_format;

    status |= vxQueryImage(upsample, VX_IMAGE_FORMAT, &upsample_format, sizeof(upsample_format));
    status = vxGetValidRegionImage(upsample, &upsample_rect);
    status |= vxMapImagePatch(upsample, &upsample_rect, 0, &upsample_map_id, &upsample_addr, (void **)&upsample_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);

    for (vx_uint32 ix = 0; ix < width; ix++)
    {
        for (vx_uint32 iy = 0; iy < height; iy++)
        {
            void* upsample_p = vxFormatImagePatchAddress2d(upsample_base, ix, iy, &upsample_addr);
            vx_int32 upsample_data = upsample_format == VX_DF_IMAGE_U8 ? *(vx_uint8 *)upsample_p : *(vx_int16 *)upsample_p;
            upsample_data *= 4;
            if (upsample_format == VX_DF_IMAGE_U8)
            {
                if (upsample_data > UINT8_MAX)
                    upsample_data = UINT8_MAX;
                else if (upsample_data < 0)
                    upsample_data = 0;
                *(vx_uint8 *)upsample_p = (vx_uint8)upsample_data;
            }
            else
            {
                if (upsample_data > INT16_MAX)
                    upsample_data = INT16_MAX;
                else if (upsample_data < INT16_MIN)
                    upsample_data = INT16_MIN;
                *(vx_int16 *)upsample_p = (vx_int16)upsample_data;
            }
        }
    }
    status |= vxUnmapImagePatch(upsample, upsample_map_id);
    status |= vxReleaseImage(&tmp);
    return status;
}

static void own_laplacian_pyramid_reference(vx_context context, vx_border_t border, vx_image input, vx_pyramid laplacian, vx_image output)
{
    vx_uint32 i;
    vx_size levels = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_df_image format = 0;
    vx_pyramid gaussian = 0;
    vx_convolution conv = 0;

    border.mode = VX_BORDER_REPLICATE;

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));

    VX_CALL(vxQueryPyramid(laplacian, VX_PYRAMID_LEVELS, &levels, sizeof(levels)));

    VX_CALL(vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width)));
    VX_CALL(vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height)));
    VX_CALL(vxQueryImage(input, VX_IMAGE_FORMAT, &format, sizeof(format)));

    ASSERT_VX_OBJECT(gaussian = vxCreatePyramid(context, levels + 1, VX_SCALE_PYRAMID_HALF, width, height, VX_DF_IMAGE_U8), VX_TYPE_PYRAMID);

    VX_CALL(vxuGaussianPyramid(context, input, gaussian));

    ASSERT_VX_OBJECT(conv = vxCreateConvolution(context, 5, 5), VX_TYPE_CONVOLUTION);

    VX_CALL(vxCopyConvolutionCoefficients(conv, (vx_int16*)gaussian5x5, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    VX_CALL(vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, (void*)&gaussian5x5scale, sizeof(gaussian5x5scale)));

    for (i = 0; i < levels; i++)
    {
        vx_uint32 w = 0;
        vx_uint32 h = 0;
        vx_image in1 = 0;
        vx_image in2 = 0;
        vx_image filter = 0;
        vx_image out = 0;

        ASSERT_VX_OBJECT(in1 = vxGetPyramidLevel(gaussian, i), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(in2 = vxGetPyramidLevel(gaussian, i + 1), VX_TYPE_IMAGE);

        VX_CALL(vxQueryImage(in1, VX_IMAGE_WIDTH, &w, sizeof(vx_uint32)));
        VX_CALL(vxQueryImage(in1, VX_IMAGE_HEIGHT, &h, sizeof(vx_uint32)));

        ASSERT_VX_OBJECT(filter = vxCreateImage(context, w, h, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
        border.mode = VX_BORDER_REPLICATE;
        upsampleImage(context, w, h, in2, conv, filter, &border);
        /* laplacian is S16 format */
        ASSERT_VX_OBJECT(out = vxGetPyramidLevel(laplacian, i), VX_TYPE_IMAGE);
        VX_CALL(vxuSubtract(context, in1, filter, VX_CONVERT_POLICY_SATURATE, out));

        if (i == levels - 1)
        {
            vx_image tmp = vxGetPyramidLevel(gaussian, levels);
            ownCopyImage(tmp, output);
            VX_CALL(vxReleaseImage(&tmp));
        }

        VX_CALL(vxReleaseImage(&filter));
        VX_CALL(vxReleaseImage(&in1));
        VX_CALL(vxReleaseImage(&in2));
        VX_CALL(vxReleaseImage(&out));

    }

    VX_CALL(vxReleasePyramid(&gaussian));
    VX_CALL(vxReleaseConvolution(&conv));

    ASSERT(conv == 0);
    ASSERT(gaussian == 0);

    return;
}

static void own_laplacian_pyramid_openvx(vx_context context, vx_border_t border, vx_image input, vx_pyramid laplacian, vx_image output)
{
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxLaplacianPyramidNode(graph, input, laplacian, output), VX_TYPE_NODE);

    VX_CALL(vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(node == 0);
    ASSERT(graph == 0);

    return;
}

typedef struct
{
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height);
    const char* fileName;
    vx_border_t border;
    int width;
    int height;
} Arg;


#define ADD_SIZE_OWN_SET(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sz=128x128", __VA_ARGS__, 128, 128)), \
    CT_EXPAND(nextmacro(testArgName "/sz=256x256", __VA_ARGS__, 256, 256)), \
    CT_EXPAND(nextmacro(testArgName "/sz=640x480", __VA_ARGS__, 640, 480))


#define LAPLACIAN_PYRAMID_PARAMETERS \
    CT_GENERATE_PARAMETERS("randomInput", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_OWN_SET, ARG, own_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ARG, own_read_image, "lena.bmp")

TEST_WITH_ARG(LaplacianPyramid, testGraphProcessing, Arg, LAPLACIAN_PYRAMID_PARAMETERS)
{
    vx_context context = context_->vx_context_;
    vx_size levels = 0;
    vx_uint32 i;
    vx_image src = 0;
    vx_image ref_dst = 0;
    vx_image tst_dst = 0;
    vx_pyramid ref_pyr = 0;
    vx_pyramid tst_pyr = 0;
    int undefined_border = 0;

    CT_Image input = NULL;

    vx_border_t border = arg_->border;

    if(border.mode == VX_BORDER_UNDEFINED){
        undefined_border = 2; // 5x5 kernel border
    }

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, arg_->width, arg_->height));
    ASSERT_VX_OBJECT(src = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    levels = own_pyramid_calc_max_levels_count(input->width, input->height, VX_SCALE_PYRAMID_HALF) - 1;

    {
        vx_uint32 next_lev_width = input->width;
        vx_uint32 next_lev_height = input->height;

        ASSERT_VX_OBJECT(ref_pyr = vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF, input->width, input->height, VX_DF_IMAGE_S16), VX_TYPE_PYRAMID);
        ASSERT_VX_OBJECT(tst_pyr = vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF, input->width, input->height, VX_DF_IMAGE_S16), VX_TYPE_PYRAMID);

        for (i = 1; i < levels + 1; i++)
        {
            next_lev_width = (vx_uint32)ceilf(next_lev_width * VX_SCALE_PYRAMID_HALF);
            next_lev_height = (vx_uint32)ceilf(next_lev_height * VX_SCALE_PYRAMID_HALF);
        }

        ASSERT_VX_OBJECT(ref_dst = vxCreateImage(context, next_lev_width, next_lev_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(tst_dst = vxCreateImage(context, next_lev_width, next_lev_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    }

    own_laplacian_pyramid_reference(context, border, src, ref_pyr, ref_dst);
    own_laplacian_pyramid_openvx(context, border, src, tst_pyr, tst_dst);

    {
        CT_Image ct_ref_dst = 0;
        CT_Image ct_tst_dst = 0;

        ASSERT_NO_FAILURE(ct_ref_dst = ct_image_from_vx_image(ref_dst));
        ASSERT_NO_FAILURE(ct_tst_dst = ct_image_from_vx_image(tst_dst));
        ASSERT_NO_FAILURE(ct_adjust_roi(ct_ref_dst, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border));
        ASSERT_NO_FAILURE(ct_adjust_roi(ct_tst_dst, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border));
        EXPECT_CTIMAGE_NEAR(ct_ref_dst, ct_tst_dst, 1);

        for (i = 0; i < levels; i++)
        {
            CT_Image ct_ref_lev = 0;
            CT_Image ct_tst_lev = 0;
            vx_image ref_lev = 0;
            vx_image tst_lev = 0;

            ASSERT_VX_OBJECT(ref_lev = vxGetPyramidLevel(ref_pyr, i), VX_TYPE_IMAGE);
            ASSERT_VX_OBJECT(tst_lev = vxGetPyramidLevel(tst_pyr, i), VX_TYPE_IMAGE);

            ASSERT_NO_FAILURE(ct_ref_lev = ct_image_from_vx_image(ref_lev));
            ASSERT_NO_FAILURE(ct_tst_lev = ct_image_from_vx_image(tst_lev));
            ASSERT_NO_FAILURE(ct_adjust_roi(ct_ref_lev, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border));
            ASSERT_NO_FAILURE(ct_adjust_roi(ct_tst_lev, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border));
            EXPECT_CTIMAGE_NEAR(ct_ref_lev, ct_tst_lev, 1);

            VX_CALL(vxReleaseImage(&ref_lev));
            VX_CALL(vxReleaseImage(&tst_lev));
        }
    }

    VX_CALL(vxReleaseImage(&src));
    VX_CALL(vxReleasePyramid(&ref_pyr));
    VX_CALL(vxReleasePyramid(&tst_pyr));
    VX_CALL(vxReleaseImage(&ref_dst));
    VX_CALL(vxReleaseImage(&tst_dst));

    ASSERT(src == 0);
    ASSERT(ref_pyr == 0);
    ASSERT(tst_pyr == 0);
    ASSERT(ref_dst == 0);
    ASSERT(tst_dst == 0);
}

TEST_WITH_ARG(LaplacianPyramid, testImmediateProcessing, Arg, LAPLACIAN_PYRAMID_PARAMETERS)
{
    vx_context context = context_->vx_context_;
    vx_size levels = 0;
    vx_uint32 i;
    vx_image src = 0;
    vx_image ref_dst = 0;
    vx_image tst_dst = 0;
    vx_pyramid ref_pyr = 0;
    vx_pyramid tst_pyr = 0;
    int undefined_border = 2; // 5x5 kernel border

    CT_Image input = NULL;

    vx_border_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, arg_->width, arg_->height));
    ASSERT_VX_OBJECT(src = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    levels = own_pyramid_calc_max_levels_count(input->width, input->height, VX_SCALE_PYRAMID_HALF) - 1;

    {
        vx_uint32 next_lev_width = input->width;
        vx_uint32 next_lev_height = input->height;

        ASSERT_VX_OBJECT(ref_pyr = vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF, input->width, input->height, VX_DF_IMAGE_S16), VX_TYPE_PYRAMID);
        ASSERT_VX_OBJECT(tst_pyr = vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF, input->width, input->height, VX_DF_IMAGE_S16), VX_TYPE_PYRAMID);

        for (i = 1; i < levels + 1; i++)
        {
            next_lev_width = (vx_uint32)ceilf(next_lev_width * VX_SCALE_PYRAMID_HALF);
            next_lev_height = (vx_uint32)ceilf(next_lev_height * VX_SCALE_PYRAMID_HALF);
        }

        ASSERT_VX_OBJECT(ref_dst = vxCreateImage(context, next_lev_width, next_lev_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(tst_dst = vxCreateImage(context, next_lev_width, next_lev_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    }

    own_laplacian_pyramid_reference(context, border, src, ref_pyr, ref_dst);

    border.mode = VX_BORDER_REPLICATE;

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));
    VX_CALL(vxuLaplacianPyramid(context, src, tst_pyr, tst_dst));

    {
        CT_Image ct_ref_dst = 0;
        CT_Image ct_tst_dst = 0;

        ASSERT_NO_FAILURE(ct_ref_dst = ct_image_from_vx_image(ref_dst));
        ASSERT_NO_FAILURE(ct_tst_dst = ct_image_from_vx_image(tst_dst));
        ASSERT_NO_FAILURE(ct_adjust_roi(ct_ref_dst, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border));
        ASSERT_NO_FAILURE(ct_adjust_roi(ct_tst_dst, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border));
        EXPECT_CTIMAGE_NEAR(ct_ref_dst, ct_tst_dst, 1);

        for (i = 0; i < levels; i++)
        {
            CT_Image ct_ref_lev = 0;
            CT_Image ct_tst_lev = 0;
            vx_image ref_lev = 0;
            vx_image tst_lev = 0;

            ASSERT_VX_OBJECT(ref_lev = vxGetPyramidLevel(ref_pyr, i), VX_TYPE_IMAGE);
            ASSERT_VX_OBJECT(tst_lev = vxGetPyramidLevel(tst_pyr, i), VX_TYPE_IMAGE);

            ASSERT_NO_FAILURE(ct_ref_lev = ct_image_from_vx_image(ref_lev));
            ASSERT_NO_FAILURE(ct_tst_lev = ct_image_from_vx_image(tst_lev));
            ASSERT_NO_FAILURE(ct_adjust_roi(ct_ref_lev, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border));
            ASSERT_NO_FAILURE(ct_adjust_roi(ct_tst_lev, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border, 2 * undefined_border));
            EXPECT_CTIMAGE_NEAR(ct_ref_lev, ct_tst_lev, 1);

            VX_CALL(vxReleaseImage(&ref_lev));
            VX_CALL(vxReleaseImage(&tst_lev));
        }
    }

    VX_CALL(vxReleaseImage(&src));
    VX_CALL(vxReleasePyramid(&ref_pyr));
    VX_CALL(vxReleasePyramid(&tst_pyr));
    VX_CALL(vxReleaseImage(&ref_dst));
    VX_CALL(vxReleaseImage(&tst_dst));

    ASSERT(src == 0);
    ASSERT(ref_pyr == 0);
    ASSERT(tst_pyr == 0);
    ASSERT(ref_dst == 0);
    ASSERT(tst_dst == 0);
}

TESTCASE_TESTS(LaplacianPyramid,
    testNodeCreation,
    testGraphProcessing,
    testImmediateProcessing
)

/* reconstruct image from laplacian pyramid */

#define VX_SCALE_PYRAMID_DOUBLE (2.0f)

TESTCASE(LaplacianReconstruct, CT_VXContext, ct_setup_vx_context, 0)

TEST(LaplacianReconstruct, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_pyramid laplacian = 0;
    vx_image   input = 0;
    vx_image   output = 0;
    vx_graph   graph = 0;
    vx_node    node = 0;
    const vx_size levels = 4;
    const vx_float32 scale = VX_SCALE_PYRAMID_HALF;
    const vx_uint32 width = 640;
    const vx_uint32 height = 480;
    vx_size num_levels = levels;
    vx_uint32 w = width;
    vx_uint32 h = height;

    while (num_levels--)
    {
        w = (vx_uint32)ceilf(w * scale);
        h = (vx_uint32)ceilf(h * scale);
    }

    ASSERT_VX_OBJECT(input = vxCreateImage(context, w, h, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, width, height, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(laplacian = vxCreatePyramid(context, levels, scale, width, height, VX_DF_IMAGE_S16), VX_TYPE_PYRAMID);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxLaplacianReconstructNode(graph, laplacian, input, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    VX_CALL(vxReleaseImage(&input));
    VX_CALL(vxReleasePyramid(&laplacian));
    VX_CALL(vxReleaseImage(&output));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(laplacian == 0);
    ASSERT(input == 0);
    ASSERT(output == 0);
    ASSERT(node == 0);
    ASSERT(graph == 0);
}

static void own_laplacian_reconstruct_reference(vx_context context, vx_border_t border, vx_pyramid laplacian, vx_image input, vx_image output)
{
    vx_size i;
    vx_size levels = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_uint32 prev_lev_width = 0;
    vx_uint32 prev_lev_height = 0;
    vx_df_image format = 0;
    vx_image pyr_level = 0;
    vx_image filling = 0;
    vx_image filter = 0;
    vx_image out = 0;
    vx_convolution conv;

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));

    VX_CALL(vxQueryPyramid(laplacian, VX_PYRAMID_LEVELS, &levels, sizeof(vx_size)));
    VX_CALL(vxQueryPyramid(laplacian, VX_PYRAMID_FORMAT, &format, sizeof(vx_df_image)));

    VX_CALL(vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(vx_uint32)));
    VX_CALL(vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(vx_uint32)));

    conv = vxCreateGaussian5x5Convolution(context);

    prev_lev_width = (vx_uint32)ceilf(width  * VX_SCALE_PYRAMID_DOUBLE);
    prev_lev_height = (vx_uint32)ceilf(height * VX_SCALE_PYRAMID_DOUBLE);

    ASSERT_VX_OBJECT(filling = vxCreateImage(context, width, height, format), VX_TYPE_IMAGE);
    VX_CALL(ownCopyImage(input, filling));

    for (i = 0; i < levels; i++)
    {
        ASSERT_VX_OBJECT(pyr_level = vxGetPyramidLevel(laplacian, (vx_uint32)((levels - 1) - i)), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(out = vxCreateImage(context, prev_lev_width, prev_lev_height, format), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(filter = vxCreateImage(context, prev_lev_width, prev_lev_height, format), VX_TYPE_IMAGE);

        upsampleImage(context, prev_lev_width, prev_lev_height, filling, conv, filter, &border);
        VX_CALL(vxuAdd(context, filter, pyr_level, VX_CONVERT_POLICY_SATURATE, out));

        VX_CALL(vxReleaseImage(&pyr_level));

        if ((levels - 1) - i == 0)
        {
            VX_CALL(ownCopyImage(out, output));
            VX_CALL(vxReleaseImage(&filling));
        }
        else
        {
            VX_CALL(vxReleaseImage(&filling));
            ASSERT_VX_OBJECT(filling = vxCreateImage(context, prev_lev_width, prev_lev_height, format), VX_TYPE_IMAGE);
            VX_CALL(ownCopyImage(out, filling));
            /* compute dimensions for the prev level */
            prev_lev_width = (vx_uint32)ceilf(prev_lev_width * VX_SCALE_PYRAMID_DOUBLE);
            prev_lev_height = (vx_uint32)ceilf(prev_lev_height * VX_SCALE_PYRAMID_DOUBLE);
        }

        VX_CALL(vxReleaseImage(&out));
        VX_CALL(vxReleaseImage(&filter));
    }

    VX_CALL(vxReleaseConvolution(&conv));

    return;
}

static void own_laplacian_reconstruct_openvx(vx_context context, vx_border_t border, vx_pyramid laplacian, vx_image input, vx_image output)
{
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxLaplacianReconstructNode(graph, laplacian, input, output), VX_TYPE_NODE);

    VX_CALL(vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(node == 0);
    ASSERT(graph == 0);

    return;
}

#define LAPLACIAN_RECONSTRUCT_PARAMETERS \
    CT_GENERATE_PARAMETERS("randomInput", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_OWN_SET, ARG, own_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ARG, own_read_image, "lena.bmp")

TEST_WITH_ARG(LaplacianReconstruct, testGraphProcessing, Arg, LAPLACIAN_RECONSTRUCT_PARAMETERS)
{
    vx_uint32 i;
    vx_context context = context_->vx_context_;
    vx_size levels = 0;
    vx_image src = 0;
    vx_image ref_lowest_res = 0;
    vx_image ref_dst = 0;
    vx_image tst_dst = 0;
    vx_pyramid ref_pyr = 0;

    CT_Image input = NULL;

    //vx_border_t border = arg_->border;
    vx_border_t build_border = { VX_BORDER_REPLICATE };

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, arg_->width, arg_->height));
    ASSERT_VX_OBJECT(src = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    levels = own_pyramid_calc_max_levels_count(input->width, input->height, VX_SCALE_PYRAMID_HALF) - 1;

    {
        vx_uint32 lowest_res_width = input->width;
        vx_uint32 lowest_res_height = input->height;

        for (i = 0; i < levels; i++)
        {
            lowest_res_width = (vx_uint32)ceilf(lowest_res_width * VX_SCALE_PYRAMID_HALF);
            lowest_res_height = (vx_uint32)ceilf(lowest_res_height * VX_SCALE_PYRAMID_HALF);
        }

        ASSERT_VX_OBJECT(ref_pyr = vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF, input->width, input->height, VX_DF_IMAGE_S16), VX_TYPE_PYRAMID);

        ASSERT_VX_OBJECT(ref_lowest_res = vxCreateImage(context, lowest_res_width, lowest_res_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(ref_dst = vxCreateImage(context, input->width, input->height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(tst_dst = vxCreateImage(context, input->width, input->height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    }

    own_laplacian_pyramid_reference(context, build_border, src, ref_pyr, ref_lowest_res);
    own_laplacian_reconstruct_reference(context, build_border, ref_pyr, ref_lowest_res, ref_dst);
    own_laplacian_reconstruct_openvx(context, build_border, ref_pyr, ref_lowest_res, tst_dst);

    {
        CT_Image ct_ref_dst = 0;
        CT_Image ct_tst_dst = 0;

        ASSERT_NO_FAILURE(ct_ref_dst = ct_image_from_vx_image(ref_dst));
        ASSERT_NO_FAILURE(ct_tst_dst = ct_image_from_vx_image(tst_dst));
        EXPECT_CTIMAGE_NEAR(ct_ref_dst, ct_tst_dst, 1);
    }

    VX_CALL(vxReleaseImage(&src));
    VX_CALL(vxReleasePyramid(&ref_pyr));
    VX_CALL(vxReleaseImage(&ref_lowest_res));
    VX_CALL(vxReleaseImage(&ref_dst));
    VX_CALL(vxReleaseImage(&tst_dst));

    ASSERT(src == 0);
    ASSERT(ref_pyr == 0);
    ASSERT(ref_lowest_res == 0);
    ASSERT(ref_dst == 0);
    ASSERT(tst_dst == 0);
}

TEST_WITH_ARG(LaplacianReconstruct, testImmediateProcessing, Arg, LAPLACIAN_RECONSTRUCT_PARAMETERS)
{
    vx_uint32 i;
    vx_context context = context_->vx_context_;
    vx_size levels = 0;
    vx_image src = 0;
    vx_image ref_lovest_res = 0;
    vx_image ref_dst = 0;
    vx_image tst_dst = 0;
    vx_pyramid ref_pyr = 0;

    CT_Image input = NULL;

    //vx_border_t border = arg_->border;
    vx_border_t build_border = { VX_BORDER_REPLICATE };

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, arg_->width, arg_->height));
    ASSERT_VX_OBJECT(src = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    levels = own_pyramid_calc_max_levels_count(input->width, input->height, VX_SCALE_PYRAMID_HALF) - 1;

    {
        vx_uint32 lowest_res_width = input->width;
        vx_uint32 lowest_res_height = input->height;

        for (i = 0; i < levels; i++)
        {
            lowest_res_width = (vx_uint32)ceilf(lowest_res_width * VX_SCALE_PYRAMID_HALF);
            lowest_res_height = (vx_uint32)ceilf(lowest_res_height * VX_SCALE_PYRAMID_HALF);
        }

        ASSERT_VX_OBJECT(ref_pyr = vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF, input->width, input->height, VX_DF_IMAGE_S16), VX_TYPE_PYRAMID);

        ASSERT_VX_OBJECT(ref_lovest_res = vxCreateImage(context, lowest_res_width, lowest_res_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(ref_dst = vxCreateImage(context, input->width, input->height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
        ASSERT_VX_OBJECT(tst_dst = vxCreateImage(context, input->width, input->height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    }

    own_laplacian_pyramid_reference(context, build_border, src, ref_pyr, ref_lovest_res);
    own_laplacian_reconstruct_reference(context, build_border, ref_pyr, ref_lovest_res, ref_dst);

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &build_border, sizeof(build_border)));
    VX_CALL(vxuLaplacianReconstruct(context, ref_pyr, ref_lovest_res, tst_dst));

    {
        CT_Image ct_ref_dst = 0;
        CT_Image ct_tst_dst = 0;

        ASSERT_NO_FAILURE(ct_ref_dst = ct_image_from_vx_image(ref_dst));
        ASSERT_NO_FAILURE(ct_tst_dst = ct_image_from_vx_image(tst_dst));
        EXPECT_CTIMAGE_NEAR(ct_ref_dst, ct_tst_dst, 1);
    }

    VX_CALL(vxReleaseImage(&src));
    VX_CALL(vxReleasePyramid(&ref_pyr));
    VX_CALL(vxReleaseImage(&ref_lovest_res));
    VX_CALL(vxReleaseImage(&ref_dst));
    VX_CALL(vxReleaseImage(&tst_dst));

    ASSERT(src == 0);
    ASSERT(ref_pyr == 0);
    ASSERT(ref_lovest_res == 0);
    ASSERT(ref_dst == 0);
    ASSERT(tst_dst == 0);
}

TESTCASE_TESTS(LaplacianReconstruct,
    testNodeCreation,
    testGraphProcessing,
    testImmediateProcessing
)

#endif //OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION
