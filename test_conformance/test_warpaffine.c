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
#include <float.h>
#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"

TESTCASE(WarpAffine, CT_VXContext, ct_setup_vx_context, 0)

TEST(WarpAffine, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0, output = 0;
    vx_matrix matrix = 0;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_enum type = VX_INTERPOLATION_NEAREST_NEIGHBOR;
    const vx_enum matrix_type = VX_TYPE_FLOAT32;
    const vx_size matrix_rows = 3;
    const vx_size matrix_cols = 2;
    const vx_size matrix_data_size = 4 * matrix_rows * matrix_cols;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(matrix = vxCreateMatrix(context, matrix_type, matrix_cols, matrix_rows), VX_TYPE_MATRIX);

    {
        vx_enum ch_matrix_type;
        vx_size ch_matrix_rows, ch_matrix_cols, ch_data_size;

        VX_CALL(vxQueryMatrix(matrix, VX_MATRIX_TYPE, &ch_matrix_type, sizeof(ch_matrix_type)));
        if (matrix_type != ch_matrix_type)
        {
            CT_FAIL("check for Matrix attribute VX_MATRIX_TYPE failed\n");
        }
        VX_CALL(vxQueryMatrix(matrix, VX_MATRIX_ROWS, &ch_matrix_rows, sizeof(ch_matrix_rows)));
        if (matrix_rows != ch_matrix_rows)
        {
            CT_FAIL("check for Matrix attribute VX_MATRIX_ROWS failed\n");
        }
        VX_CALL(vxQueryMatrix(matrix, VX_MATRIX_COLUMNS, &ch_matrix_cols, sizeof(ch_matrix_cols)));
        if (matrix_cols != ch_matrix_cols)
        {
            CT_FAIL("check for Matrix attribute VX_MATRIX_COLUMNS failed\n");
        }
        VX_CALL(vxQueryMatrix(matrix, VX_MATRIX_SIZE, &ch_data_size, sizeof(ch_data_size)));
        if (matrix_data_size > ch_data_size)
        {
            CT_FAIL("check for Matrix attribute VX_MATRIX_SIZE failed\n");
        }
    }

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxWarpAffineNode(graph, input, matrix, type, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseMatrix(&matrix));
    VX_CALL(vxReleaseImage(&output));
    VX_CALL(vxReleaseImage(&input));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(matrix == 0);
    ASSERT(output == 0);
    ASSERT(input == 0);
}

enum CT_AffineMatrixType {
    VX_MATRIX_IDENT = 0,
    VX_MATRIX_ROTATE_90,
    VX_MATRIX_SCALE,
    VX_MATRIX_SCALE_ROTATE,
    VX_MATRIX_RANDOM
};

#define VX_NN_AREA_SIZE         1.5
#define VX_BILINEAR_TOLERANCE   1

static CT_Image warp_affine_read_image(const char* fileName, int width, int height, vx_df_image format)
{
    CT_Image image_load = NULL, image_ret = NULL;
    ASSERT_(return 0, format == VX_DF_IMAGE_U1 || format == VX_DF_IMAGE_U8);

    image_load = ct_read_image(fileName, 1);
    ASSERT_(return 0, image_load);
    ASSERT_(return 0, image_load->format == VX_DF_IMAGE_U8);

    if (format == VX_DF_IMAGE_U1)
    {
        ASSERT_NO_FAILURE_(return 0, threshold_U8_ct_image(image_load, 127));   // Threshold to make the U1 image less trivial
        ASSERT_NO_FAILURE_(return 0, image_ret = ct_allocate_image(image_load->width, image_load->height, VX_DF_IMAGE_U1));
        ASSERT_NO_FAILURE_(return 0, U8_ct_image_to_U1_ct_image(image_load, image_ret));
    }
    else    // format == VX_DF_IMAGE_U8
    {
        image_ret = image_load;
    }

    ASSERT_(return 0, image_ret);
    ASSERT_(return 0, image_ret->format == format);

    return image_ret;
}

static CT_Image warp_affine_generate_random(const char* fileName, int width, int height, vx_df_image format)
{
    CT_Image image;
    ASSERT_(return 0, format == VX_DF_IMAGE_U1 || format == VX_DF_IMAGE_U8);

    if (format == VX_DF_IMAGE_U1)
        ASSERT_NO_FAILURE_(return 0, image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 2));
    else    // format == VX_DF_IMAGE_U8
        ASSERT_NO_FAILURE_(return 0, image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 256));

    return image;
}

#define RND_FLT(low, high)      (vx_float32)CT_RNG_NEXT_REAL(CT()->seed_, low, high);

static void warp_affine_generate_matrix(vx_float32* m, int src_width, int src_height, int dst_width, int dst_height, int type)
{
    vx_float32 mat[3][2];
    vx_float32 angle, scale_x, scale_y, cos_a, sin_a;
    if (VX_MATRIX_IDENT == type)
    {
        mat[0][0] = 1.f;
        mat[0][1] = 0.f;

        mat[1][0] = 0.f;
        mat[1][1] = 1.f;

        mat[2][0] = 0.f;
        mat[2][1] = 0.f;
    }
    else if (VX_MATRIX_ROTATE_90 == type)
    {
        mat[0][0] = 0.f;
        mat[0][1] = 1.f;

        mat[1][0] = -1.f;
        mat[1][1] = 0.f;

        mat[2][0] = (vx_float32)src_width;
        mat[2][1] = 0.f;
    }
    else if (VX_MATRIX_SCALE == type)
    {
        scale_x = src_width / (vx_float32)dst_width;
        scale_y = src_height / (vx_float32)dst_height;

        mat[0][0] = scale_x;
        mat[0][1] = 0.f;

        mat[1][0] = 0.f;
        mat[1][1] = scale_y;

        mat[2][0] = 0.f;
        mat[2][1] = 0.f;
    }
    else if (VX_MATRIX_SCALE_ROTATE == type)
    {
        angle = M_PIF / RND_FLT(3.f, 6.f);
        scale_x = src_width / (vx_float32)dst_width;
        scale_y = src_height / (vx_float32)dst_height;
        cos_a = cosf(angle);
        sin_a = sinf(angle);

        mat[0][0] = cos_a * scale_x;
        mat[0][1] = sin_a * scale_y;

        mat[1][0] = -sin_a * scale_x;
        mat[1][1] = cos_a  * scale_y;

        mat[2][0] = 0.f;
        mat[2][1] = 0.f;
    }
    else// if (VX_MATRIX_RANDOM == type)
    {
        angle = M_PIF / RND_FLT(3.f, 6.f);
        scale_x = src_width / (vx_float32)dst_width;
        scale_y = src_height / (vx_float32)dst_height;
        cos_a = cosf(angle);
        sin_a = sinf(angle);

        mat[0][0] = cos_a * RND_FLT(scale_x / 2.f, scale_x);
        mat[0][1] = sin_a * RND_FLT(scale_y / 2.f, scale_y);

        mat[1][0] = -sin_a * RND_FLT(scale_y / 2.f, scale_y);
        mat[1][1] = cos_a  * RND_FLT(scale_x / 2.f, scale_x);

        mat[2][0] = src_width  / 5.f * RND_FLT(-1.f, 1.f);
        mat[2][1] = src_height / 5.f * RND_FLT(-1.f, 1.f);
    }
    memcpy(m, mat, sizeof(mat));
}

static vx_matrix warp_affine_create_matrix(vx_context context, vx_float32 *m)
{
    vx_matrix matrix;
    matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 2, 3);
    if (vxGetStatus((vx_reference)matrix) == VX_SUCCESS)
    {
        if (VX_SUCCESS != vxCopyMatrix(matrix, m, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST))
        {
            VX_CALL_(return 0, vxReleaseMatrix(&matrix));
        }
    }
    return matrix;
}

static int warp_affine_check_pixel(CT_Image input, CT_Image output, int x, int y, vx_enum interp_type, vx_border_t border, vx_float32 *m)
{
    vx_float64 x0, y0, xlower, ylower, s, t;
    vx_int32 xo, yo, xi, yi, roi_xi, roi_yi, xiShft;
    int candidate;
    vx_df_image format = input->format;

    xo = x + output->roi.x - (format == VX_DF_IMAGE_U1 ? output->roi.x % 8 : 0);    // ROI independent cordinates
    yo = y + output->roi.y;
    roi_xi = input->roi.x;
    roi_yi = input->roi.y;
    xiShft = (format == VX_DF_IMAGE_U1) ? input->roi.x % 8 : 0; // Bit-shift used for U1 input image

    vx_uint8 res;
    if (format == VX_DF_IMAGE_U1)
        res = (*CT_IMAGE_DATA_PTR_1U(output, x, y) & (1 << (x % 8))) >> (x % 8);
    else
        res =  *CT_IMAGE_DATA_PTR_8U(output, x, y);

    x0 = (vx_float64)m[2 * 0 + 0] * (vx_float64)xo + (vx_float64)m[2 * 1 + 0] * (vx_float64)yo + (vx_float64)m[2 * 2 + 0];
    y0 = (vx_float64)m[2 * 0 + 1] * (vx_float64)xo + (vx_float64)m[2 * 1 + 1] * (vx_float64)yo + (vx_float64)m[2 * 2 + 1];
    x0 = x0 - (vx_float64)roi_xi + xiShft;       // Switch to ROI-respecting coordinates
    y0 = y0 - (vx_float64)roi_yi;

    if (VX_INTERPOLATION_NEAREST_NEIGHBOR == interp_type)
    {
        for (yi = (vx_int32)ceil(y0 - VX_NN_AREA_SIZE); (vx_float64)yi <= y0 + VX_NN_AREA_SIZE; yi++)
        {
            for (xi = (vx_int32)ceil(x0 - VX_NN_AREA_SIZE); (vx_float64)xi <= x0 + VX_NN_AREA_SIZE; xi++)
            {
                if (xi >= xiShft                          && yi >= 0 &&
                    xi <  (vx_int32)input->width + xiShft && yi <  (vx_int32)input->height)
                {
                    if (format == VX_DF_IMAGE_U1)
                        candidate = (*CT_IMAGE_DATA_PTR_1U(input, xi, yi) & (1 << (xi % 8))) >> (xi % 8);
                    else
                        candidate =  *CT_IMAGE_DATA_PTR_8U(input, xi, yi);
                }
                else if (VX_BORDER_CONSTANT == border.mode)
                {
                    if (format == VX_DF_IMAGE_U1)
                        candidate = border.constant_value.U1 ? 1 : 0;
                    else
                        candidate = border.constant_value.U8;
                }
                else
                {
                    candidate = -1;
                }
                if (candidate == -1 || candidate == res)
                    return 0;
            }
        }
        CT_FAIL_(return 1, "Check failed for pixel (%d, %d): %d", xo, yo, (int)res);
    }
    else if (VX_INTERPOLATION_BILINEAR == interp_type)
    {
        xlower = floor(x0);
        ylower = floor(y0);

        s = x0 - xlower;
        t = y0 - ylower;

        xi = (vx_int32)xlower;
        yi = (vx_int32)ylower;

        candidate = -1;
        if (VX_BORDER_UNDEFINED == border.mode)
        {
            if (xi >= xiShft                             && yi >= 0 &&
                xi < (vx_int32)input->width - 1 + xiShft && yi < (vx_int32)input->height - 1)
            {
                if (format == VX_DF_IMAGE_U1)
                {
                    vx_uint8 p00 = (*CT_IMAGE_DATA_PTR_1U(input, xi    , yi    ) & (1 <<  xi      % 8)) >> ( xi      % 8);
                    vx_uint8 p10 = (*CT_IMAGE_DATA_PTR_1U(input, xi + 1, yi    ) & (1 << (xi + 1) % 8)) >> ((xi + 1) % 8);
                    vx_uint8 p01 = (*CT_IMAGE_DATA_PTR_1U(input, xi    , yi + 1) & (1 <<  xi      % 8)) >> ( xi      % 8);
                    vx_uint8 p11 = (*CT_IMAGE_DATA_PTR_1U(input, xi + 1, yi + 1) & (1 << (xi + 1) % 8)) >> ((xi + 1) % 8);
                    candidate = (int)((1. - s) * (1. - t) * (vx_float64) p00 +
                                            s  * (1. - t) * (vx_float64) p10 +
                                      (1. - s) *       t  * (vx_float64) p01 +
                                            s  *       t  * (vx_float64) p11 + 0.5); // Arithmetic rounding instead of truncation
                    candidate = (candidate > 1) ? 1 : (candidate < 0) ? 0 : candidate;
                }
                else
                {
                    candidate = (int)((1. - s) * (1. - t) * (vx_float64) *CT_IMAGE_DATA_PTR_8U(input, xi    , yi    ) +
                                            s  * (1. - t) * (vx_float64) *CT_IMAGE_DATA_PTR_8U(input, xi + 1, yi    ) +
                                      (1. - s) *       t  * (vx_float64) *CT_IMAGE_DATA_PTR_8U(input, xi    , yi + 1) +
                                            s  *       t  * (vx_float64) *CT_IMAGE_DATA_PTR_8U(input, xi + 1, yi + 1));
                }
            }
        }
        else if (VX_BORDER_CONSTANT == border.mode)
        {
            if (format == VX_DF_IMAGE_U1)
            {
                vx_uint8 p00 = CT_IMAGE_DATA_CONSTANT_1U(input, xi    , yi    , border.constant_value.U1);
                vx_uint8 p10 = CT_IMAGE_DATA_CONSTANT_1U(input, xi + 1, yi    , border.constant_value.U1);
                vx_uint8 p01 = CT_IMAGE_DATA_CONSTANT_1U(input, xi    , yi + 1, border.constant_value.U1);
                vx_uint8 p11 = CT_IMAGE_DATA_CONSTANT_1U(input, xi + 1, yi + 1, border.constant_value.U1);
                candidate = (int)((1. - s) * (1. - t) * (vx_float32)p00 +
                                        s  * (1. - t) * (vx_float32)p10 +
                                  (1. - s) *       t  * (vx_float32)p01 +
                                        s  *       t  * (vx_float32)p11 + 0.5);
                candidate = (candidate > 1) ? 1 : (candidate < 0) ? 0 : candidate;
            }
            else
            {
                vx_uint8 p00 = CT_IMAGE_DATA_CONSTANT_8U(input, xi    , yi    , border.constant_value.U8);
                vx_uint8 p10 = CT_IMAGE_DATA_CONSTANT_8U(input, xi + 1, yi    , border.constant_value.U8);
                vx_uint8 p01 = CT_IMAGE_DATA_CONSTANT_8U(input, xi    , yi + 1, border.constant_value.U8);
                vx_uint8 p11 = CT_IMAGE_DATA_CONSTANT_8U(input, xi + 1, yi + 1, border.constant_value.U8);
                candidate = (int)((1. - s) * (1. - t) * (vx_float32)p00 +
                                        s  * (1. - t) * (vx_float32)p10 +
                                  (1. - s) *       t  * (vx_float32)p01 +
                                        s  *       t  * (vx_float32)p11);
            }
        }
        // A tolerance of 1 would make tests on U1 images trivial
        if ( candidate == -1 || (abs(candidate - res) <= ((format == VX_DF_IMAGE_U1) ? 0 : VX_BILINEAR_TOLERANCE)) )
            return 0;
        else
            return 1;
    }
    CT_FAIL_(return 1, "Interpolation type undefined");
}

static void warp_affine_validate(CT_Image input, CT_Image output, vx_enum interp_type, vx_border_t border, vx_float32* m)
{
    vx_uint32 err_count = 0;

    if (input->format == VX_DF_IMAGE_U1)
    {
        CT_FILL_IMAGE_1U(, output,
                {
                    ASSERT_NO_FAILURE(err_count += warp_affine_check_pixel(input, output, xShftd, y, interp_type, border, m));
                });
    }
    else
    {
        CT_FILL_IMAGE_8U(, output,
                {
                    ASSERT_NO_FAILURE(err_count += warp_affine_check_pixel(input, output, x, y, interp_type, border, m));
                });
    }
    if (10 * err_count > output->width * output->height)
        CT_FAIL_(return, "Check failed for %d pixels", err_count);
}

static void warp_affine_check(CT_Image input, CT_Image output, vx_enum interp_type, vx_border_t border, vx_float32* m)
{
    ASSERT(input && output);
    ASSERT( (interp_type == VX_INTERPOLATION_NEAREST_NEIGHBOR) ||
            (interp_type == VX_INTERPOLATION_BILINEAR));

    ASSERT( (border.mode == VX_BORDER_UNDEFINED) ||
            (border.mode == VX_BORDER_CONSTANT));

    ASSERT( (input->format == output->format) &&
            (input->format == VX_DF_IMAGE_U1 || input->format == VX_DF_IMAGE_U8));

    ASSERT( ((input->width  == output->width)  || (input->roi.width  == output->roi.width)) &&
            ((input->height == output->height) || (input->roi.height == output->roi.height)));

    warp_affine_validate(input, output, interp_type, border, m);
    if (CT_HasFailure())
    {
        printf("=== INPUT ===\n");
        ct_dump_image_info(input);
        printf("=== OUTPUT ===\n");
        ct_dump_image_info(output);
        printf("Matrix:\n%g %g %g\n%g %g %g\n",
                m[0], m[2], m[4],
                m[1], m[3], m[5]);
    }
}

typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    int width, height;
    vx_border_t border;
    vx_enum interp_type;
    int matrix_type;
    vx_df_image format;
} Arg;

#define ADD_VX_BORDERS_WARP_AFFINE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_UNDEFINED", __VA_ARGS__, { VX_BORDER_UNDEFINED, {{ 0 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=0", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 0 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=1", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 1 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=127", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 127 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=255", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 255 }} }))

#define ADD_VX_BORDERS_WARP_AFFINE_MINIMAL(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_UNDEFINED", __VA_ARGS__, { VX_BORDER_UNDEFINED, {{ 0 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=0", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 0 }} })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_CONSTANT=255", __VA_ARGS__, { VX_BORDER_CONSTANT, {{ 255 }} }))

#define ADD_VX_INTERP_TYPE_WARP_AFFINE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_NEAREST_NEIGHBOR", __VA_ARGS__, VX_INTERPOLATION_NEAREST_NEIGHBOR)), \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_BILINEAR", __VA_ARGS__, VX_INTERPOLATION_BILINEAR ))

#define ADD_VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_NEAREST_NEIGHBOR", __VA_ARGS__, VX_INTERPOLATION_NEAREST_NEIGHBOR))

#define ADD_VX_MATRIX_PARAM_WARP_AFFINE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_IDENT", __VA_ARGS__,        VX_MATRIX_IDENT)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_ROTATE_90", __VA_ARGS__,    VX_MATRIX_ROTATE_90)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_SCALE", __VA_ARGS__,        VX_MATRIX_SCALE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_SCALE_ROTATE", __VA_ARGS__, VX_MATRIX_SCALE_ROTATE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_RANDOM", __VA_ARGS__,       VX_MATRIX_RANDOM))

#define ADD_VX_MATRIX_PARAM_WARP_AFFINE_MINIMAL(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_IDENT", __VA_ARGS__,        VX_MATRIX_IDENT)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_SCALE_ROTATE", __VA_ARGS__, VX_MATRIX_SCALE_ROTATE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_RANDOM", __VA_ARGS__,       VX_MATRIX_RANDOM))

#define PARAMETERS \
    CT_GENERATE_PARAMETERS("random", ADD_SIZE_SMALL_SET, ADD_VX_BORDERS_WARP_AFFINE, ADD_VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, ADD_VX_MATRIX_PARAM_WARP_AFFINE, ADD_TYPE_U8, ARG, warp_affine_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_SIZE_NONE, ADD_VX_BORDERS_WARP_AFFINE, ADD_VX_INTERP_TYPE_WARP_AFFINE, ADD_VX_MATRIX_PARAM_WARP_AFFINE, ADD_TYPE_U8, ARG, warp_affine_read_image, "lena.bmp"), \
    CT_GENERATE_PARAMETERS("_U1_/random", ADD_SIZE_SMALL_SET, ADD_VX_BORDERS_WARP_AFFINE_MINIMAL, ADD_VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, ADD_VX_MATRIX_PARAM_WARP_AFFINE, ADD_TYPE_U1, ARG, warp_affine_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("_U1_/lena", ADD_SIZE_NONE, ADD_VX_BORDERS_WARP_AFFINE_MINIMAL, ADD_VX_INTERP_TYPE_WARP_AFFINE, ADD_VX_MATRIX_PARAM_WARP_AFFINE, ADD_TYPE_U1, ARG, warp_affine_read_image, "lena.bmp")

TEST_WITH_ARG(WarpAffine, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_image input_image = 0, output_image = 0;
    vx_matrix matrix = 0;
    vx_float32 m[6];

    CT_Image input = NULL, output = NULL;

    vx_border_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));
    ASSERT_NO_FAILURE(output = ct_allocate_image(input->width, input->height, input->format));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(warp_affine_generate_matrix(m, input->width, input->height, input->width/2, input->height/2, arg_->matrix_type));
    ASSERT_VX_OBJECT(matrix = warp_affine_create_matrix(context, m), VX_TYPE_MATRIX);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxWarpAffineNode(graph, input_image, matrix, arg_->interp_type, output_image), VX_TYPE_NODE);

    VX_CALL(vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));
    ASSERT_NO_FAILURE(warp_affine_check(input, output, arg_->interp_type, arg_->border, m));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseMatrix(&matrix));
    VX_CALL(vxReleaseImage(&output_image));
    VX_CALL(vxReleaseImage(&input_image));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(matrix == 0);
    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

TEST_WITH_ARG(WarpAffine, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, output_image = 0;
    vx_matrix matrix = 0;
    vx_float32 m[6];

    CT_Image input = NULL, output = NULL;

    vx_border_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));
    ASSERT_NO_FAILURE(output = ct_allocate_image(input->width, input->height, input->format));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(warp_affine_generate_matrix(m, input->width, input->height, input->width/2, input->height/2, arg_->matrix_type));
    ASSERT_VX_OBJECT(matrix = warp_affine_create_matrix(context, m), VX_TYPE_MATRIX);

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));

    VX_CALL(vxuWarpAffine(context, input_image, matrix, arg_->interp_type, output_image));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));

    ASSERT_NO_FAILURE(warp_affine_check(input, output, arg_->interp_type, arg_->border, m));

    VX_CALL(vxReleaseMatrix(&matrix));
    VX_CALL(vxReleaseImage(&output_image));
    VX_CALL(vxReleaseImage(&input_image));

    ASSERT(matrix == 0);
    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

typedef struct {
    const char* testName;
    CT_Image (*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    int width, height;
    vx_border_t border;
    vx_enum interp_type;
    int matrix_type;
    vx_df_image format;
    vx_rectangle_t region_shift;
} ValidRegionTest_Arg;

#define REGION_PARAMETERS \
    CT_GENERATE_PARAMETERS("lena", ADD_SIZE_256x256, ADD_VX_BORDERS_WARP_AFFINE_MINIMAL, ADD_VX_INTERP_TYPE_WARP_AFFINE, ADD_VX_MATRIX_PARAM_WARP_AFFINE_MINIMAL, ADD_TYPE_U8, ADD_VALID_REGION_SHRINKS, ARG, warp_affine_read_image, "lena.bmp"), \
    CT_GENERATE_PARAMETERS("_U1_/lena", ADD_SIZE_256x256, ADD_VX_BORDERS_WARP_AFFINE_MINIMAL, ADD_VX_INTERP_TYPE_WARP_AFFINE, ADD_VX_MATRIX_PARAM_WARP_AFFINE_MINIMAL, ADD_TYPE_U1, ADD_VALID_REGION_SHRINKS, ARG, warp_affine_read_image, "lena.bmp")

TEST_WITH_ARG(WarpAffine, testWithValidRegion, ValidRegionTest_Arg,
    REGION_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, output_image = 0;
    vx_matrix matrix = 0;
    vx_float32 m[6];

    CT_Image input = NULL, output = NULL;

    vx_border_t border = arg_->border;
    vx_rectangle_t rect = {0, 0, 0, 0}, rect_shft = arg_->region_shift;

    ASSERT_NO_FAILURE(input  = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));
    ASSERT_NO_FAILURE(output = ct_allocate_image(input->width, input->height, input->format));

    ASSERT_VX_OBJECT(input_image  = ct_image_to_vx_image(input,  context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(warp_affine_generate_matrix(m, input->width, input->height, input->width/2, input->height/2, arg_->matrix_type));
    ASSERT_VX_OBJECT(matrix = warp_affine_create_matrix(context, m), VX_TYPE_MATRIX);

    ASSERT_NO_FAILURE(vxGetValidRegionImage(input_image, &rect));
    ALTERRECTANGLE(rect, rect_shft.start_x, rect_shft.start_y, rect_shft.end_x, rect_shft.end_y);
    ASSERT_NO_FAILURE(vxSetImageValidRectangle(input_image, &rect));

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));

    VX_CALL(vxuWarpAffine(context, input_image, matrix, arg_->interp_type, output_image));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));

    ASSERT_NO_FAILURE(ct_adjust_roi(input, rect_shft.start_x, rect_shft.start_y, -rect_shft.end_x, -rect_shft.end_y));
    ASSERT_NO_FAILURE(ct_adjust_roi(output, rect_shft.start_x, rect_shft.start_y, -rect_shft.end_x, -rect_shft.end_y));
    ASSERT_NO_FAILURE(warp_affine_check(input, output, arg_->interp_type, border, m));

    VX_CALL(vxReleaseMatrix(&matrix));
    VX_CALL(vxReleaseImage(&output_image));
    VX_CALL(vxReleaseImage(&input_image));

    ASSERT(matrix == 0);
    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

TESTCASE_TESTS(WarpAffine,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing,
        testWithValidRegion
)

#endif //OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION
