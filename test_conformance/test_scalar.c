/*

 * Copyright (c) 2016-2017 The Khronos Group Inc.
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
#include <float.h>
#include <VX/vx.h>

#include "test_engine/test.h"

TESTCASE(Scalar, CT_VXContext, ct_setup_vx_context, 0)

typedef union
{
    vx_char     chr;
    vx_int8     s08;
    vx_uint8    u08;
    vx_int16    s16;
    vx_uint16   u16;
    vx_int32    s32;
    vx_uint32   u32;
    vx_int64    s64;
    vx_uint64   u64;
    vx_float32  f32;
    vx_float64  f64;
    vx_enum     enm;
    vx_size     size;
    vx_df_image fcc;
    vx_bool     boolean;
    vx_uint8    data[8];

    /* support type of scalar with size */
    vx_rectangle_t rect;
    vx_keypoint_t  key_point;
    vx_coordinates2d_t coord2d;
    vx_coordinates3d_t coord3d;
#ifdef OPENVX_USE_ENHANCED_VISION
    vx_coordinates2df_t coord2df;
    vx_hog_t  hog;
    vx_hough_lines_p_t houghlines;
    vx_line2d_t line2d;
    vx_tensor_matrix_multiply_params_t matrix;
#endif
} scalar_val;

typedef struct
{
    const char* name;
    vx_enum data_type;
} format_arg;

static void own_init_scalar_value(vx_enum type, scalar_val* val, vx_bool variant)
{
    switch (type)
    {
    case VX_TYPE_CHAR:     val->chr = variant == vx_true_e ? 1 : 2; break;
    case VX_TYPE_INT8:     val->s08 = variant == vx_true_e ? 2 : 3; break;
    case VX_TYPE_UINT8:    val->u08 = variant == vx_true_e ? 3 : 4; break;
    case VX_TYPE_INT16:    val->s16 = variant == vx_true_e ? 4 : 5; break;
    case VX_TYPE_UINT16:   val->u16 = variant == vx_true_e ? 5 : 6; break;
    case VX_TYPE_INT32:    val->s32 = variant == vx_true_e ? 6 : 7; break;
    case VX_TYPE_UINT32:   val->u32 = variant == vx_true_e ? 7 : 8; break;
    case VX_TYPE_INT64:    val->s64 = variant == vx_true_e ? 8 : 9; break;
    case VX_TYPE_UINT64:   val->u64 = variant == vx_true_e ? 9 : 10; break;
    case VX_TYPE_FLOAT32:  val->f32 = variant == vx_true_e ? 1.5f : 9.9f; break;
    case VX_TYPE_FLOAT64:  val->f64 = variant == vx_true_e ? 1.5 : 9.9; break;
    case VX_TYPE_ENUM:     val->enm = variant == vx_true_e ? VX_BORDER_CONSTANT : VX_BORDER_REPLICATE; break;
    case VX_TYPE_SIZE:     val->size = variant == vx_true_e ? 10 : 999; break;
    case VX_TYPE_DF_IMAGE: val->fcc = variant == vx_true_e ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_U8; break;
    case VX_TYPE_BOOL:     val->boolean = variant == vx_true_e ? vx_true_e : vx_false_e; break;

    /* support type of scalar with size */
    case VX_TYPE_RECTANGLE:
        if (variant == vx_true_e)
        {
            val->rect.start_x = 0;
            val->rect.start_y = 0;
            val->rect.end_x = 1280;
            val->rect.end_y = 720;
        }
        break;
    case VX_TYPE_KEYPOINT:
        if (variant == vx_true_e)
        {
            val->key_point.x = 128;
            val->key_point.y = 256;
            val->key_point.strength = 20.0;
            val->key_point.scale = 0.8;
            val->key_point.orientation = 0.3;
            val->key_point.tracking_status = 1;
            val->key_point.error = 0;
        }
        break;
    case VX_TYPE_COORDINATES2D:
        if (variant == vx_true_e)
        {
            val->coord2d.x = 10;
            val->coord2d.y = 9;
        }
        break;
    case VX_TYPE_COORDINATES3D:
        if (variant == vx_true_e)
        {
            val->coord3d.x = 16;
            val->coord3d.y = 31;
            val->coord3d.z = 22;
        }
        break;
#ifdef OPENVX_USE_ENHANCED_VISION
    case VX_TYPE_COORDINATES2DF:
        if (variant == vx_true_e)
        {
            val->coord2df.x = 2.7f;
            val->coord2df.y = 3.5f;
        }
        break;
    case VX_TYPE_HOG_PARAMS:
        if (variant == vx_true_e)
        {
            val->hog.cell_width = 16;
            val->hog.cell_height = 16;
            val->hog.block_width = 128;
            val->hog.block_height = 128;
            val->hog.block_stride = 1280;
            val->hog.num_bins = 5;
            val->hog.window_width = 1280;
            val->hog.window_height = 720;
            val->hog.window_stride = 1280;
            val->hog.threshold = 0.2f;
        }
        break;
    case VX_TYPE_HOUGH_LINES_PARAMS:
        if (variant == vx_true_e)
        {
            val->houghlines.rho = 0.8f;
            val->houghlines.theta = 0.5f;
            val->houghlines.threshold = 235;
            val->houghlines.line_length = 8;
            val->houghlines.line_gap = 3;
            val->houghlines.theta_max = 1.2f;
            val->houghlines.theta_min = 0.1f;
        }
        break;
    case VX_TYPE_LINE_2D:
        if (variant == vx_true_e)
        {
            val->line2d.start_x = 2.3f;
            val->line2d.start_y = 1.5f;
            val->line2d.end_x = 1279.9f;
            val->line2d.end_y = 718.8f;
        }
        break;
    case VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS:
        if (variant == vx_true_e)
        {
            val->matrix.transpose_input1 = vx_true_e;
            val->matrix.transpose_input2 = vx_false_e;
            val->matrix.transpose_input3 = vx_true_e;
        }
        break;
#endif
    default:
        FAIL("Unsupported type: (%.4s)", &type);
    }
    return;
}

static vx_size ownGetSizeByType(vx_enum type)
{
    vx_size size = 0;
    switch (type)
    {
    case VX_TYPE_CHAR:     size = sizeof(vx_char); break;
    case VX_TYPE_INT8:     size = sizeof(vx_int8); break;
    case VX_TYPE_UINT8:    size = sizeof(vx_uint8); break;
    case VX_TYPE_INT16:    size = sizeof(vx_int16); break;
    case VX_TYPE_UINT16:   size = sizeof(vx_uint16); break;
    case VX_TYPE_INT32:    size = sizeof(vx_int32); break;
    case VX_TYPE_UINT32:   size = sizeof(vx_uint32); break;
    case VX_TYPE_INT64:    size = sizeof(vx_int64); break;
    case VX_TYPE_UINT64:   size = sizeof(vx_uint64); break;
    case VX_TYPE_FLOAT32:  size = sizeof(vx_float32); break;
    case VX_TYPE_FLOAT64:  size = sizeof(vx_float64); break;
    case VX_TYPE_ENUM:     size = sizeof(vx_int32); break;
    case VX_TYPE_SIZE:     size = sizeof(vx_size); break;
    case VX_TYPE_DF_IMAGE: size = sizeof(vx_df_image); break;
    case VX_TYPE_BOOL:     size = sizeof(vx_bool); break;

    /* support type of scalar with size */
    case VX_TYPE_RECTANGLE:
        size = sizeof(vx_rectangle_t);
        break;
    case VX_TYPE_KEYPOINT:
        size = sizeof(vx_keypoint_t);
        break;
    case VX_TYPE_COORDINATES2D:
        size = sizeof(vx_coordinates2d_t);
        break;
    case VX_TYPE_COORDINATES3D:
        size = sizeof(vx_coordinates3d_t);
        break;
#ifdef OPENVX_USE_ENHANCED_VISION
    case VX_TYPE_COORDINATES2DF:
        size = sizeof(vx_coordinates2df_t);
        break;
    case VX_TYPE_HOG_PARAMS:
        size = sizeof(vx_hog_t);
        break;
    case VX_TYPE_HOUGH_LINES_PARAMS:
        size = sizeof(vx_hough_lines_p_t);
        break;
    case VX_TYPE_LINE_2D:
        size = sizeof(vx_line2d_t);
        break;
    case VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS:
        size = sizeof(vx_tensor_matrix_multiply_params_t);
        break;
#endif
    default:
        CT_RecordFailureAtFormat("Unsupported type: (%.4s)", __FUNCTION__, __FILE__, __LINE__, &type);
        break;
    }
    return size;
}

static void ownCheckScalarVal(vx_enum type, scalar_val *actual_val, scalar_val *expect_val)
{
    switch (type)
    {
    case VX_TYPE_CHAR:
        ASSERT_EQ_INT(actual_val->chr, expect_val->chr);
        break;

    case VX_TYPE_INT8:
        ASSERT_EQ_INT(actual_val->s08, expect_val->s08);
        break;

    case VX_TYPE_UINT8:
        ASSERT_EQ_INT(actual_val->u08, expect_val->u08);
        break;

    case VX_TYPE_INT16:
        ASSERT_EQ_INT(actual_val->s16, expect_val->s16);
        break;

    case VX_TYPE_UINT16:
        ASSERT_EQ_INT(actual_val->u16, expect_val->u16);
        break;

    case VX_TYPE_INT32:
        ASSERT_EQ_INT(actual_val->s32, expect_val->s32);
        break;

    case VX_TYPE_UINT32:
        ASSERT_EQ_INT(actual_val->u32, expect_val->u32);
        break;

    case VX_TYPE_INT64:
        ASSERT_EQ_INT(actual_val->s64, expect_val->s64);
        break;

    case VX_TYPE_UINT64:
        ASSERT_EQ_INT(actual_val->u64, expect_val->u64);
        break;

    case VX_TYPE_FLOAT32:
        ASSERT(fabs(actual_val->f32 - expect_val->f32) < 0.000001f);
        break;

    case VX_TYPE_FLOAT64:
        ASSERT(fabs(actual_val->f64 - expect_val->f64) < 0.000001f);
        break;

    case VX_TYPE_DF_IMAGE:
        ASSERT_EQ_INT(actual_val->fcc, expect_val->fcc);
        break;

    case VX_TYPE_ENUM:
        ASSERT_EQ_INT(actual_val->enm, expect_val->enm);
        break;

    case VX_TYPE_SIZE:
        ASSERT_EQ_INT(actual_val->size, expect_val->size);
        break;

    case VX_TYPE_BOOL:
        ASSERT_EQ_INT(actual_val->boolean, expect_val->boolean);
        break;

    case VX_TYPE_RECTANGLE:
        ASSERT_EQ_INT(memcmp(&actual_val->rect, &expect_val->rect, sizeof(VX_TYPE_RECTANGLE)), 0);
        break;

    case VX_TYPE_KEYPOINT:
        ASSERT_EQ_INT(memcmp(&actual_val->key_point, &expect_val->key_point, sizeof(VX_TYPE_KEYPOINT)), 0);
        break;

    case VX_TYPE_COORDINATES2D:
        ASSERT_EQ_INT(memcmp(&actual_val->coord2d, &expect_val->coord2d, sizeof(VX_TYPE_COORDINATES2D)), 0);
        break;

    case VX_TYPE_COORDINATES3D:
        ASSERT_EQ_INT(memcmp(&actual_val->coord3d, &expect_val->coord3d, sizeof(VX_TYPE_COORDINATES3D)), 0);
        break;

#ifdef OPENVX_USE_ENHANCED_VISION
    case VX_TYPE_COORDINATES2DF:
        ASSERT_EQ_INT(memcmp(&actual_val->coord2df, &expect_val->coord2df, sizeof(VX_TYPE_COORDINATES2DF)), 0);
        break;

    case VX_TYPE_HOG_PARAMS:
        ASSERT_EQ_INT(memcmp(&actual_val->hog, &expect_val->hog, sizeof(VX_TYPE_HOG_PARAMS)), 0);
        break;

    case VX_TYPE_HOUGH_LINES_PARAMS:
        ASSERT_EQ_INT(memcmp(&actual_val->houghlines, &expect_val->houghlines, sizeof(VX_TYPE_HOUGH_LINES_PARAMS)), 0);
        break;

    case VX_TYPE_LINE_2D:
        ASSERT_EQ_INT(memcmp(&actual_val->line2d, &expect_val->line2d, sizeof(VX_TYPE_LINE_2D)), 0);
        break;

    case VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS:
        ASSERT_EQ_INT(memcmp(&actual_val->matrix, &expect_val->matrix, sizeof(VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS)), 0);
        break;
#endif
    default:
        FAIL("Unsupported type: (%.4s)", &type);
        break;
    }
    return;
}

TEST_WITH_ARG(Scalar, testCreateScalar, format_arg,
    ARG_ENUM(VX_TYPE_CHAR),
    ARG_ENUM(VX_TYPE_INT8),
    ARG_ENUM(VX_TYPE_UINT8),
    ARG_ENUM(VX_TYPE_INT16),
    ARG_ENUM(VX_TYPE_UINT16),
    ARG_ENUM(VX_TYPE_INT32),
    ARG_ENUM(VX_TYPE_UINT32),
    ARG_ENUM(VX_TYPE_INT64),
    ARG_ENUM(VX_TYPE_UINT64),
    ARG_ENUM(VX_TYPE_FLOAT32),
    ARG_ENUM(VX_TYPE_FLOAT64),
    ARG_ENUM(VX_TYPE_ENUM),
    ARG_ENUM(VX_TYPE_SIZE),
    ARG_ENUM(VX_TYPE_DF_IMAGE),
    ARG_ENUM(VX_TYPE_BOOL)
    )
{
    vx_context context = context_->vx_context_;
    vx_scalar  scalar = 0;
    vx_enum    ref_type = arg_->data_type;
    scalar_val ref;

    own_init_scalar_value(ref_type, &ref, vx_true_e);

    ASSERT_VX_OBJECT(scalar = vxCreateScalar(context, ref_type, &ref), VX_TYPE_SCALAR);

    VX_CALL(vxReleaseScalar(&scalar));

    ASSERT(scalar == 0);

    return;
} /* testCreateScalar() */

TEST_WITH_ARG(Scalar, testQueryScalar, format_arg,
    ARG_ENUM(VX_TYPE_CHAR),
    ARG_ENUM(VX_TYPE_INT8),
    ARG_ENUM(VX_TYPE_UINT8),
    ARG_ENUM(VX_TYPE_INT16),
    ARG_ENUM(VX_TYPE_UINT16),
    ARG_ENUM(VX_TYPE_INT32),
    ARG_ENUM(VX_TYPE_UINT32),
    ARG_ENUM(VX_TYPE_INT64),
    ARG_ENUM(VX_TYPE_UINT64),
    ARG_ENUM(VX_TYPE_FLOAT32),
    ARG_ENUM(VX_TYPE_FLOAT64),
    ARG_ENUM(VX_TYPE_ENUM),
    ARG_ENUM(VX_TYPE_SIZE),
    ARG_ENUM(VX_TYPE_DF_IMAGE),
    ARG_ENUM(VX_TYPE_BOOL)
    )
{
    vx_context context   = context_->vx_context_;
    vx_scalar  scalar    = 0;
    vx_enum    ref_type = arg_->data_type;
    vx_enum    tst_type = 0;
    scalar_val ref;

    own_init_scalar_value(ref_type, &ref, vx_true_e);

    ASSERT_VX_OBJECT(scalar = vxCreateScalar(context, ref_type, &ref), VX_TYPE_SCALAR);

    VX_CALL(vxQueryScalar(scalar, VX_SCALAR_TYPE, &tst_type, sizeof(tst_type)));
    ASSERT_EQ_INT(ref_type, tst_type);

    VX_CALL(vxReleaseScalar(&scalar));

    ASSERT(scalar == 0);

    return;
} /* testQueryScalar() */

TEST_WITH_ARG(Scalar, testCopyScalar, format_arg,
    ARG_ENUM(VX_TYPE_CHAR),
    ARG_ENUM(VX_TYPE_INT8),
    ARG_ENUM(VX_TYPE_UINT8),
    ARG_ENUM(VX_TYPE_INT16),
    ARG_ENUM(VX_TYPE_UINT16),
    ARG_ENUM(VX_TYPE_INT32),
    ARG_ENUM(VX_TYPE_UINT32),
    ARG_ENUM(VX_TYPE_INT64),
    ARG_ENUM(VX_TYPE_UINT64),
    ARG_ENUM(VX_TYPE_FLOAT32),
    ARG_ENUM(VX_TYPE_FLOAT64),
    ARG_ENUM(VX_TYPE_ENUM),
    ARG_ENUM(VX_TYPE_SIZE),
    ARG_ENUM(VX_TYPE_DF_IMAGE),
    ARG_ENUM(VX_TYPE_BOOL)
    )
{
    vx_context context   = context_->vx_context_;
    vx_scalar  scalar    = 0;
    vx_enum    type = arg_->data_type;
    scalar_val val1;
    scalar_val val2;
    scalar_val tst;

    own_init_scalar_value(type, &val1, vx_true_e);
    own_init_scalar_value(type, &val2, vx_false_e);

    ASSERT_VX_OBJECT(scalar = vxCreateScalar(context, type, &val1), VX_TYPE_SCALAR);

    VX_CALL(vxCopyScalar(scalar, &tst, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    switch (type)
    {
    case VX_TYPE_CHAR:
        ASSERT_EQ_INT(val1.chr, tst.chr);
        break;

    case VX_TYPE_INT8:
        ASSERT_EQ_INT(val1.s08, tst.s08);
        break;

    case VX_TYPE_UINT8:
        ASSERT_EQ_INT(val1.u08, tst.u08);
        break;

    case VX_TYPE_INT16:
        ASSERT_EQ_INT(val1.s16, tst.s16);
        break;

    case VX_TYPE_UINT16:
        ASSERT_EQ_INT(val1.u16, tst.u16);
        break;

    case VX_TYPE_INT32:
        ASSERT_EQ_INT(val1.s32, tst.s32);
        break;

    case VX_TYPE_UINT32:
        ASSERT_EQ_INT(val1.u32, tst.u32);
        break;

    case VX_TYPE_INT64:
        ASSERT_EQ_INT(val1.s64, tst.s64);
        break;

    case VX_TYPE_UINT64:
        ASSERT_EQ_INT(val1.u64, tst.u64);
        break;

    case VX_TYPE_FLOAT32:
        ASSERT(fabs(val1.f32 - tst.f32) < 0.000001f);
        break;

    case VX_TYPE_FLOAT64:
        ASSERT(fabs(val1.f64 - tst.f64) < 0.000001f);
        break;

    case VX_TYPE_DF_IMAGE:
        ASSERT_EQ_INT(val1.fcc, tst.fcc);
        break;

    case VX_TYPE_ENUM:
        ASSERT_EQ_INT(val1.enm, tst.enm);
        break;

    case VX_TYPE_SIZE:
        ASSERT_EQ_INT(val1.size, tst.size);
        break;

    case VX_TYPE_BOOL:
        ASSERT_EQ_INT(val1.boolean, tst.boolean);
        break;

    default:
        FAIL("Unsupported type: (%.4s)", &type);
        break;
    }

    /* change scalar value */
    VX_CALL(vxCopyScalar(scalar, &val2, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    /* read value back */
    VX_CALL(vxCopyScalar(scalar, &tst, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    switch (type)
    {
    case VX_TYPE_CHAR:
        ASSERT_EQ_INT(val2.chr, tst.chr);
        break;

    case VX_TYPE_INT8:
        ASSERT_EQ_INT(val2.s08, tst.s08);
        break;

    case VX_TYPE_UINT8:
        ASSERT_EQ_INT(val2.u08, tst.u08);
        break;

    case VX_TYPE_INT16:
        ASSERT_EQ_INT(val2.s16, tst.s16);
        break;

    case VX_TYPE_UINT16:
        ASSERT_EQ_INT(val2.u16, tst.u16);
        break;

    case VX_TYPE_INT32:
        ASSERT_EQ_INT(val2.s32, tst.s32);
        break;

    case VX_TYPE_UINT32:
        ASSERT_EQ_INT(val2.u32, tst.u32);
        break;

    case VX_TYPE_INT64:
        ASSERT_EQ_INT(val2.s64, tst.s64);
        break;

    case VX_TYPE_UINT64:
        ASSERT_EQ_INT(val2.u64, tst.u64);
        break;

    case VX_TYPE_FLOAT32:
        ASSERT(fabs(val2.f32 - tst.f32) < 0.000001f);
        break;

    case VX_TYPE_FLOAT64:
        ASSERT(fabs(val2.f64 - tst.f64) < 0.000001f);
        break;

    case VX_TYPE_DF_IMAGE:
        ASSERT_EQ_INT(val2.fcc, tst.fcc);
        break;

    case VX_TYPE_ENUM:
        ASSERT_EQ_INT(val2.enm, tst.enm);
        break;

    case VX_TYPE_SIZE:
        ASSERT_EQ_INT(val2.size, tst.size);
        break;

    case VX_TYPE_BOOL:
        ASSERT_EQ_INT(val2.boolean, tst.boolean);
        break;

    default:
        FAIL("Unsupported type: (%.4s)", &type);
        break;
    }

    VX_CALL(vxReleaseScalar(&scalar));

    ASSERT(scalar == 0);

    return;
} /* testCopyScalar() */

TEST_WITH_ARG(Scalar, testCreateScalarWithSize, format_arg,
    ARG_ENUM(VX_TYPE_CHAR),
    ARG_ENUM(VX_TYPE_INT8),
    ARG_ENUM(VX_TYPE_UINT8),
    ARG_ENUM(VX_TYPE_INT16),
    ARG_ENUM(VX_TYPE_UINT16),
    ARG_ENUM(VX_TYPE_INT32),
    ARG_ENUM(VX_TYPE_UINT32),
    ARG_ENUM(VX_TYPE_INT64),
    ARG_ENUM(VX_TYPE_UINT64),
    ARG_ENUM(VX_TYPE_FLOAT32),
    ARG_ENUM(VX_TYPE_FLOAT64),
    ARG_ENUM(VX_TYPE_ENUM),
    ARG_ENUM(VX_TYPE_SIZE),
    ARG_ENUM(VX_TYPE_DF_IMAGE),
    ARG_ENUM(VX_TYPE_BOOL),
    ARG_ENUM(VX_TYPE_RECTANGLE),
    ARG_ENUM(VX_TYPE_KEYPOINT),
    ARG_ENUM(VX_TYPE_COORDINATES2D),
    ARG_ENUM(VX_TYPE_COORDINATES3D),
#ifdef OPENVX_USE_ENHANCED_VISION
    ARG_ENUM(VX_TYPE_COORDINATES2DF),
    ARG_ENUM(VX_TYPE_HOG_PARAMS),
    ARG_ENUM(VX_TYPE_HOUGH_LINES_PARAMS),
    ARG_ENUM(VX_TYPE_LINE_2D),
    ARG_ENUM(VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS)
#endif
    )
{
    vx_context context = context_->vx_context_;
    vx_scalar  scalar = 0;
    vx_enum    ref_type = arg_->data_type;
    scalar_val ref;
    vx_size    ref_size = ownGetSizeByType(ref_type);

    own_init_scalar_value(ref_type, &ref, vx_true_e);

    ASSERT_VX_OBJECT(scalar = vxCreateScalarWithSize(context, ref_type, &ref, ref_size), VX_TYPE_SCALAR);

    VX_CALL(vxReleaseScalar(&scalar));

    ASSERT(scalar == 0);

    return;
}

TEST_WITH_ARG(Scalar, testCopyScalarWithSize, format_arg,
    ARG_ENUM(VX_TYPE_CHAR),
    ARG_ENUM(VX_TYPE_INT8),
    ARG_ENUM(VX_TYPE_UINT8),
    ARG_ENUM(VX_TYPE_INT16),
    ARG_ENUM(VX_TYPE_UINT16),
    ARG_ENUM(VX_TYPE_INT32),
    ARG_ENUM(VX_TYPE_UINT32),
    ARG_ENUM(VX_TYPE_INT64),
    ARG_ENUM(VX_TYPE_UINT64),
    ARG_ENUM(VX_TYPE_FLOAT32),
    ARG_ENUM(VX_TYPE_FLOAT64),
    ARG_ENUM(VX_TYPE_ENUM),
    ARG_ENUM(VX_TYPE_SIZE),
    ARG_ENUM(VX_TYPE_DF_IMAGE),
    ARG_ENUM(VX_TYPE_BOOL),
    ARG_ENUM(VX_TYPE_RECTANGLE),
    ARG_ENUM(VX_TYPE_KEYPOINT),
    ARG_ENUM(VX_TYPE_COORDINATES2D),
    ARG_ENUM(VX_TYPE_COORDINATES3D),
#ifdef OPENVX_USE_ENHANCED_VISION
    ARG_ENUM(VX_TYPE_COORDINATES2DF),
    ARG_ENUM(VX_TYPE_HOG_PARAMS),
    ARG_ENUM(VX_TYPE_HOUGH_LINES_PARAMS),
    ARG_ENUM(VX_TYPE_LINE_2D),
    ARG_ENUM(VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS)
#endif
    )
{
    vx_context context = context_->vx_context_;
    vx_scalar  scalar = 0;
    vx_enum    ref_type = arg_->data_type;
    scalar_val ref;
    vx_size    ref_size = ownGetSizeByType(ref_type);
    scalar_val expect_ref;
    scalar_val actual_ref;

    own_init_scalar_value(ref_type, &ref, vx_true_e);

    ASSERT_VX_OBJECT(scalar = vxCreateScalarWithSize(context, ref_type, &ref, ref_size), VX_TYPE_SCALAR);

    //check result
    VX_CALL(vxCopyScalarWithSize(scalar, ref_size, &expect_ref, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ownCheckScalarVal(ref_type, &ref, &expect_ref);

    own_init_scalar_value(ref_type, &actual_ref, vx_false_e);
    VX_CALL(vxCopyScalarWithSize(scalar, ref_size, &actual_ref, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    VX_CALL(vxCopyScalarWithSize(scalar, ref_size, &expect_ref, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ownCheckScalarVal(ref_type, &actual_ref, &expect_ref);

    VX_CALL(vxReleaseScalar(&scalar));

    ASSERT(scalar == 0);

    return;
}

TEST_WITH_ARG(Scalar, testCreateVirtualScalar, format_arg,
    ARG_ENUM(VX_TYPE_CHAR),
    ARG_ENUM(VX_TYPE_INT8),
    ARG_ENUM(VX_TYPE_UINT8),
    ARG_ENUM(VX_TYPE_INT16),
    ARG_ENUM(VX_TYPE_UINT16),
    ARG_ENUM(VX_TYPE_INT32),
    ARG_ENUM(VX_TYPE_UINT32),
    ARG_ENUM(VX_TYPE_INT64),
    ARG_ENUM(VX_TYPE_UINT64),
    ARG_ENUM(VX_TYPE_FLOAT32),
    ARG_ENUM(VX_TYPE_FLOAT64),
    ARG_ENUM(VX_TYPE_ENUM),
    ARG_ENUM(VX_TYPE_SIZE),
    ARG_ENUM(VX_TYPE_DF_IMAGE),
    ARG_ENUM(VX_TYPE_BOOL),
    ARG_ENUM(VX_TYPE_RECTANGLE),
    ARG_ENUM(VX_TYPE_KEYPOINT),
    ARG_ENUM(VX_TYPE_COORDINATES2D),
    ARG_ENUM(VX_TYPE_COORDINATES3D),
#ifdef OPENVX_USE_ENHANCED_VISION
    ARG_ENUM(VX_TYPE_COORDINATES2DF),
    ARG_ENUM(VX_TYPE_HOG_PARAMS),
    ARG_ENUM(VX_TYPE_HOUGH_LINES_PARAMS),
    ARG_ENUM(VX_TYPE_LINE_2D),
    ARG_ENUM(VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS)
#endif
    )
{
    vx_context context = context_->vx_context_;
    vx_scalar  scalar = 0;
    vx_enum    ref_type = arg_->data_type;
    vx_graph graph = 0;
    vx_enum    expect_type = VX_TYPE_INVALID;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(scalar = vxCreateVirtualScalar(graph, ref_type), VX_TYPE_SCALAR);

    VX_CALL(vxQueryScalar(scalar, VX_SCALAR_TYPE, &expect_type, sizeof(vx_enum)));
    EXPECT_EQ_INT(expect_type, ref_type);

    VX_CALL(vxReleaseScalar(&scalar));
    ASSERT(scalar == 0);

    VX_CALL(vxReleaseGraph(&graph));
    ASSERT(graph == 0);

    return;
}

TESTCASE_TESTS(Scalar,
    testCreateScalar,
    testQueryScalar,
    testCopyScalar,
    testCreateScalarWithSize,
    testCopyScalarWithSize,
    testCreateVirtualScalar
    )

#endif //OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION
