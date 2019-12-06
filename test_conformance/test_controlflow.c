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

#include <math.h>
#include <float.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"

TESTCASE(ControlFlow, CT_VXContext, ct_setup_vx_context, 0)

static vx_reference create_reference(vx_context context, vx_enum type)
{
    vx_reference ref = 0;
    vx_uint8 value = 0;
    vx_enum format = VX_DF_IMAGE_U8;
    vx_uint32 src_width = 128, src_height = 128;
    vx_uint32 img_width = 10, img_height = 10;
    vx_enum item_type = VX_TYPE_UINT8;
    vx_size capacity = 20;
    vx_size levels = 8;
    vx_float32 scale = 0.5f;
    vx_size bins = 36;
    vx_int32 offset = 0;
    vx_uint32 range = 360;
    vx_enum thresh_type = VX_THRESHOLD_TYPE_BINARY;
    vx_size num_items = 100;
    vx_size m = 5, n = 5;
    vx_size object_array_count = 10;
    vx_size convolution_x = 3;
    vx_size convolution_y = 3;
    vx_scalar scalar = 0;
    vx_size tensor_dims_num = 2;
    vx_size tensor_dims_length = 8;
    vx_size *dims;

    switch (type)
    {
    case VX_TYPE_IMAGE:
        ref = (vx_reference)vxCreateImage(context, src_width, src_height, format);
        break;
    case VX_TYPE_ARRAY:
        ref = (vx_reference)vxCreateArray(context, item_type, capacity);
        break;
    case VX_TYPE_PYRAMID:
        ref = (vx_reference)vxCreatePyramid(context, levels, scale, src_width, src_height, format);
        break;
    case VX_TYPE_SCALAR:
        ref = (vx_reference)vxCreateScalar(context, item_type, &value);
        break;
    case VX_TYPE_MATRIX:
        ref = (vx_reference)vxCreateMatrix(context, item_type, m, n);
        break;
    case VX_TYPE_CONVOLUTION:
        ref = (vx_reference)vxCreateConvolution(context, convolution_x, convolution_y);
        break;
    case VX_TYPE_DISTRIBUTION:
        ref = (vx_reference)vxCreateDistribution(context, bins, offset, range);
        break;
    case VX_TYPE_REMAP:
    {
        vx_remap remap = vxCreateRemap(context, img_width, img_height, img_width*2, img_height*2);

        vx_rectangle_t rect = { 0, 0,  img_width*2, img_height*2};
        vx_size stride = img_width*2;
        vx_size stride_y = sizeof(vx_coordinates2df_t) * (stride);
        vx_size size = stride * img_height*2;
        vx_coordinates2df_t* ptr_w = ct_alloc_mem(sizeof(vx_coordinates2df_t) * size);
        for (vx_size i = 0; i < img_height*2; i++)
        {
            for (vx_size j = 0; j < img_width*2; j++)
            {
                vx_coordinates2df_t *coord_ptr = &(ptr_w[i * stride + j]);
                coord_ptr->x = (vx_float32)j;
                coord_ptr->y = (vx_float32)i;
            }
        }
        vxCopyRemapPatch(remap, &rect, stride_y, ptr_w, VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

        ref = (vx_reference)remap;
        ct_free_mem(ptr_w);
        break;
    }
    case VX_TYPE_LUT:
        ref = (vx_reference)vxCreateLUT(context, item_type, num_items);
        break;
    case VX_TYPE_THRESHOLD:
        ref = (vx_reference)vxCreateThresholdForImage(context, thresh_type, format, format);
        break;
    case VX_TYPE_OBJECT_ARRAY:
        scalar = vxCreateScalar(context, item_type, &value);
        ref = (vx_reference)vxCreateObjectArray(context, (vx_reference)scalar, object_array_count);
        vxReleaseScalar(&scalar);
        break;
    case VX_TYPE_TENSOR:
        dims = ct_alloc_mem(tensor_dims_num * sizeof(vx_size));
        for(vx_size i = 0; i < tensor_dims_num; i++)
        {
            dims[i] = tensor_dims_length;
        }
        ref = (vx_reference)vxCreateTensor(context, tensor_dims_num, dims, VX_TYPE_UINT8, 0);
        ct_free_mem(dims);
        break;
    default:
        break;
    }
    return ref;
}

typedef struct
{
    const char* testName;
    const char* p;
    vx_enum item_type;
} select_arg;

#define  ADD_VX_CONTROLFLOW_TYPES(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_IMAGE", __VA_ARGS__, VX_TYPE_IMAGE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_SCALAR", __VA_ARGS__, VX_TYPE_SCALAR)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_MATRIX", __VA_ARGS__, VX_TYPE_MATRIX)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_CONVOLUTION", __VA_ARGS__, VX_TYPE_CONVOLUTION)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_DISTRIBUTION", __VA_ARGS__, VX_TYPE_DISTRIBUTION)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_LUT", __VA_ARGS__, VX_TYPE_LUT)),\
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_OBJECT_ARRAY", __VA_ARGS__, VX_TYPE_OBJECT_ARRAY)),\
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_PYRAMID", __VA_ARGS__, VX_TYPE_PYRAMID)),\
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_THRESHOLD", __VA_ARGS__, VX_TYPE_THRESHOLD)),\
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_TENSOR", __VA_ARGS__, VX_TYPE_TENSOR )),\
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_REMAP", __VA_ARGS__, VX_TYPE_REMAP))

#define PARAMETERS \
    CT_GENERATE_PARAMETERS("controlflow", ADD_VX_CONTROLFLOW_TYPES, ARG, NULL)


TEST_WITH_ARG(ControlFlow, testSelectNode, select_arg, PARAMETERS)
{
    vx_reference true_value = NULL;
    vx_reference false_value = NULL;
    vx_reference output = NULL;
    vx_enum input_type = arg_->item_type;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_scalar scalar_true = 0;
    vx_bool bool_true = vx_true_e;

    vx_context context = context_->vx_context_;
    ASSERT_VX_OBJECT(true_value = create_reference(context, input_type), (enum vx_type_e)input_type);
    ASSERT_VX_OBJECT(false_value = create_reference(context, input_type), (enum vx_type_e)input_type);
    ASSERT_VX_OBJECT(output = create_reference(context, input_type), (enum vx_type_e)input_type);
    ASSERT_VX_OBJECT(scalar_true = vxCreateScalar(context, VX_TYPE_BOOL, &bool_true), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxSelectNode(graph, scalar_true, true_value, false_value, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseReference(&true_value));
    VX_CALL(vxReleaseReference(&false_value));
    VX_CALL(vxReleaseReference(&output));
    VX_CALL(vxReleaseScalar(&scalar_true));
}

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

} scalar_val;

typedef struct
{
    const char* testName;
    vx_enum operation;
    vx_enum param_a_type;
    vx_enum param_b_type;
    vx_enum result_type;
    const char * a_value;
    const char * b_value;
    const char * o_check;
} scalar_op_arg;

#define PARAMETERS_SCALAR_OP \
    ARG("ControlFlow/AND/BOOL",  VX_SCALAR_OP_AND,  VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "0|vx_false_e", "0|vx_false_e", "0|vx_false_e"),\
    ARG("ControlFlow/AND/BOOL",  VX_SCALAR_OP_AND,  VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "0|vx_false_e", "1|vx_true_e",  "0|vx_false_e"),\
    ARG("ControlFlow/AND/BOOL",  VX_SCALAR_OP_AND,  VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "1|vx_true_e",  "0|vx_false_e", "0|vx_false_e"),\
    ARG("ControlFlow/AND/BOOL",  VX_SCALAR_OP_AND,  VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "1|vx_true_e",  "1|vx_true_e",  "1|vx_true_e") ,\
    ARG("ControlFlow/OR/BOOL",   VX_SCALAR_OP_OR,   VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "0|vx_false_e", "0|vx_false_e", "0|vx_false_e"),\
    ARG("ControlFlow/OR/BOOL",   VX_SCALAR_OP_OR,   VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "0|vx_false_e", "1|vx_true_e",  "1|vx_true_e"),\
    ARG("ControlFlow/OR/BOOL",   VX_SCALAR_OP_OR,   VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "1|vx_true_e",  "0|vx_false_e", "1|vx_true_e"),\
    ARG("ControlFlow/OR/BOOL",   VX_SCALAR_OP_OR,   VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "1|vx_true_e",  "1|vx_true_e",  "1|vx_true_e"),\
    ARG("ControlFlow/XOR/BOOL",  VX_SCALAR_OP_XOR,  VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "0|vx_false_e", "0|vx_false_e", "0|vx_false_e"),\
    ARG("ControlFlow/XOR/BOOL",  VX_SCALAR_OP_XOR,  VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "0|vx_false_e", "1|vx_true_e",  "1|vx_true_e"),\
    ARG("ControlFlow/XOR/BOOL",  VX_SCALAR_OP_XOR,  VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "1|vx_true_e",  "0|vx_false_e", "1|vx_true_e"),\
    ARG("ControlFlow/XOR/BOOL",  VX_SCALAR_OP_XOR,  VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "1|vx_true_e",  "1|vx_true_e",  "0|vx_false_e"),\
    ARG("ControlFlow/NAND/BOOL", VX_SCALAR_OP_NAND, VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "0|vx_false_e", "0|vx_false_e", "1|vx_true_e"),\
    ARG("ControlFlow/NAND/BOOL", VX_SCALAR_OP_NAND, VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "0|vx_false_e", "1|vx_true_e",  "1|vx_true_e"),\
    ARG("ControlFlow/NAND/BOOL", VX_SCALAR_OP_NAND, VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "1|vx_true_e",  "0|vx_false_e", "1|vx_true_e"),\
    ARG("ControlFlow/NAND/BOOL", VX_SCALAR_OP_NAND, VX_TYPE_BOOL, VX_TYPE_BOOL, VX_TYPE_BOOL, "1|vx_true_e",  "1|vx_true_e",  "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/INT8",   VX_SCALAR_OP_EQUAL, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "-1", "1", "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/INT8",   VX_SCALAR_OP_EQUAL, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "1", "1", "1|vx_true_e"),\
    ARG("ControlFlow/EQUAL/UINT8",  VX_SCALAR_OP_EQUAL, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "0", "2", "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/UINT8",  VX_SCALAR_OP_EQUAL, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "2", "2", "1|vx_true_e"),\
    ARG("ControlFlow/EQUAL/INT16",  VX_SCALAR_OP_EQUAL, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "-3", "3", "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/INT16",  VX_SCALAR_OP_EQUAL, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "3", "3", "1|vx_true_e"),\
    ARG("ControlFlow/EQUAL/UINT16", VX_SCALAR_OP_EQUAL, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "0", "4", "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/UINT16", VX_SCALAR_OP_EQUAL, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "4", "4", "1|vx_true_e"),\
    ARG("ControlFlow/EQUAL/INT32",  VX_SCALAR_OP_EQUAL, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "-5", "5", "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/INT32",  VX_SCALAR_OP_EQUAL, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "5", "5", "1|vx_true_e"),\
    ARG("ControlFlow/EQUAL/UINT32", VX_SCALAR_OP_EQUAL, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "0", "6", "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/UINT32", VX_SCALAR_OP_EQUAL, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "6", "6", "1|vx_true_e"),\
    ARG("ControlFlow/EQUAL/SIZE",   VX_SCALAR_OP_EQUAL, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "-7", "7", "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/SIZE",   VX_SCALAR_OP_EQUAL, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "7", "7", "1|vx_true_e"),\
    ARG("ControlFlow/EQUAL/FLOAT32",VX_SCALAR_OP_EQUAL, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "-1.5f", "1.5f", "0|vx_false_e"),\
    ARG("ControlFlow/EQUAL/FLOAT32",VX_SCALAR_OP_EQUAL, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "1.5f", "1.5f", "1|vx_true_e"),\
    ARG("ControlFlow/NOTEQUAL/INT8",   VX_SCALAR_OP_NOTEQUAL, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "1", "1", "0|vx_false_e"),\
    ARG("ControlFlow/NOTEQUAL/UINT8",  VX_SCALAR_OP_NOTEQUAL, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "2", "2", "0|vx_false_e"),\
    ARG("ControlFlow/NOTEQUAL/INT16",  VX_SCALAR_OP_NOTEQUAL, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "3", "3", "0|vx_false_e"),\
    ARG("ControlFlow/NOTEQUAL/UINT16", VX_SCALAR_OP_NOTEQUAL, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "4", "4", "0|vx_false_e"),\
    ARG("ControlFlow/NOTEQUAL/INT32",  VX_SCALAR_OP_NOTEQUAL, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "5", "5", "0|vx_false_e"),\
    ARG("ControlFlow/NOTEQUAL/UINT32", VX_SCALAR_OP_NOTEQUAL, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "6", "6", "0|vx_false_e"),\
    ARG("ControlFlow/NOTEQUAL/SIZE",   VX_SCALAR_OP_NOTEQUAL, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "7", "7", "0|vx_false_e"),\
    ARG("ControlFlow/NOTEQUAL/FLOAT32",VX_SCALAR_OP_NOTEQUAL, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "1.5f", "1.5f", "0|vx_false_e"),\
    ARG("ControlFlow/NOTEQUAL/INT8",   VX_SCALAR_OP_NOTEQUAL, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "1", "2", "1|vx_true_e"),\
    ARG("ControlFlow/NOTEQUAL/UINT8",  VX_SCALAR_OP_NOTEQUAL, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "2", "4", "1|vx_true_e"),\
    ARG("ControlFlow/NOTEQUAL/INT16",  VX_SCALAR_OP_NOTEQUAL, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "3", "5", "1|vx_true_e"),\
    ARG("ControlFlow/NOTEQUAL/UINT16", VX_SCALAR_OP_NOTEQUAL, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "4", "6", "1|vx_true_e"),\
    ARG("ControlFlow/NOTEQUAL/INT32",  VX_SCALAR_OP_NOTEQUAL, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "5", "7", "1|vx_true_e"),\
    ARG("ControlFlow/NOTEQUAL/UINT32", VX_SCALAR_OP_NOTEQUAL, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "6", "7", "1|vx_true_e"),\
    ARG("ControlFlow/NOTEQUAL/SIZE",   VX_SCALAR_OP_NOTEQUAL, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "7", "8", "1|vx_true_e"),\
    ARG("ControlFlow/NOTEQUAL/FLOAT32",VX_SCALAR_OP_NOTEQUAL, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "1.5f", "2.5f", "1|vx_true_e"),\
    ARG("ControlFlow/LESS/INT8",   VX_SCALAR_OP_LESS, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "1", "-2", "0|vx_false_e"),\
    ARG("ControlFlow/LESS/UINT8",  VX_SCALAR_OP_LESS, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "2", "1", "0|vx_false_e"),\
    ARG("ControlFlow/LESS/INT16",  VX_SCALAR_OP_LESS, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "3", "-5", "0|vx_false_e"),\
    ARG("ControlFlow/LESS/UINT16", VX_SCALAR_OP_LESS, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "4", "1", "0|vx_false_e"),\
    ARG("ControlFlow/LESS/INT32",  VX_SCALAR_OP_LESS, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "5", "-7", "0|vx_false_e"),\
    ARG("ControlFlow/LESS/UINT32", VX_SCALAR_OP_LESS, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "6", "1", "0|vx_false_e"),\
    ARG("ControlFlow/LESS/SIZE",   VX_SCALAR_OP_LESS, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "7", "1", "0|vx_false_e"),\
    ARG("ControlFlow/LESS/FLOAT32",VX_SCALAR_OP_LESS, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "1.5f", "-2.5f", "0|vx_false_e"),\
    ARG("ControlFlow/LESS/INT8",   VX_SCALAR_OP_LESS, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "1", "2", "1|vx_true_e"),\
    ARG("ControlFlow/LESS/UINT8",  VX_SCALAR_OP_LESS, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "2", "4", "1|vx_true_e"),\
    ARG("ControlFlow/LESS/INT16",  VX_SCALAR_OP_LESS, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "3", "5", "1|vx_true_e"),\
    ARG("ControlFlow/LESS/UINT16", VX_SCALAR_OP_LESS, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "4", "6", "1|vx_true_e"),\
    ARG("ControlFlow/LESS/INT32",  VX_SCALAR_OP_LESS, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "5", "7", "1|vx_true_e"),\
    ARG("ControlFlow/LESS/UINT32", VX_SCALAR_OP_LESS, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "6", "7", "1|vx_true_e"),\
    ARG("ControlFlow/LESS/SIZE",   VX_SCALAR_OP_LESS, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "7", "8", "1|vx_true_e"),\
    ARG("ControlFlow/LESS/FLOAT32",VX_SCALAR_OP_LESS, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "1.5f", "2.5f", "1|vx_true_e"),\
    ARG("ControlFlow/LESSEQ/INT8",   VX_SCALAR_OP_LESSEQ, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "1", "-2", "0|vx_false_e"),\
    ARG("ControlFlow/LESSEQ/UINT8",  VX_SCALAR_OP_LESSEQ, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "2", "0", "0|vx_false_e"),\
    ARG("ControlFlow/LESSEQ/INT16",  VX_SCALAR_OP_LESSEQ, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "4", "-5", "0|vx_false_e"),\
    ARG("ControlFlow/LESSEQ/UINT16", VX_SCALAR_OP_LESSEQ, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "5", "0", "0|vx_false_e"),\
    ARG("ControlFlow/LESSEQ/INT32",  VX_SCALAR_OP_LESSEQ, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "6", "-7", "0|vx_false_e"),\
    ARG("ControlFlow/LESSEQ/UINT32", VX_SCALAR_OP_LESSEQ, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "7", "0", "0|vx_false_e"),\
    ARG("ControlFlow/LESSEQ/SIZE",   VX_SCALAR_OP_LESSEQ, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "8", "7", "0|vx_false_e"),\
    ARG("ControlFlow/LESSEQ/FLOAT32",VX_SCALAR_OP_LESSEQ, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "1.5f", "-1.5f", "0|vx_false_e"),\
    ARG("ControlFlow/LESSEQ/INT8",   VX_SCALAR_OP_LESSEQ, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "1", "2", "1|vx_true_e"),\
    ARG("ControlFlow/LESSEQ/UINT8",  VX_SCALAR_OP_LESSEQ, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "2", "3", "1|vx_true_e"),\
    ARG("ControlFlow/LESSEQ/INT16",  VX_SCALAR_OP_LESSEQ, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "4", "5", "1|vx_true_e"),\
    ARG("ControlFlow/LESSEQ/UINT16", VX_SCALAR_OP_LESSEQ, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "5", "5", "1|vx_true_e"),\
    ARG("ControlFlow/LESSEQ/INT32",  VX_SCALAR_OP_LESSEQ, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "6", "7", "1|vx_true_e"),\
    ARG("ControlFlow/LESSEQ/UINT32", VX_SCALAR_OP_LESSEQ, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "7", "8", "1|vx_true_e"),\
    ARG("ControlFlow/LESSEQ/SIZE",   VX_SCALAR_OP_LESSEQ, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "8", "9", "1|vx_true_e"),\
    ARG("ControlFlow/LESSEQ/FLOAT32",VX_SCALAR_OP_LESSEQ, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "1.5f", "2.0f", "1|vx_true_e"),\
    ARG("ControlFlow/GREATER/INT8",   VX_SCALAR_OP_GREATER, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "-2", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATER/UINT8",  VX_SCALAR_OP_GREATER, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "0", "2", "0|vx_false_e"),\
    ARG("ControlFlow/GREATER/INT16",  VX_SCALAR_OP_GREATER, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "-4", "3", "0|vx_false_e"),\
    ARG("ControlFlow/GREATER/UINT16", VX_SCALAR_OP_GREATER, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "0", "4", "0|vx_false_e"),\
    ARG("ControlFlow/GREATER/INT32",  VX_SCALAR_OP_GREATER, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "-6", "5", "0|vx_false_e"),\
    ARG("ControlFlow/GREATER/UINT32", VX_SCALAR_OP_GREATER, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "1", "6", "0|vx_false_e"),\
    ARG("ControlFlow/GREATER/SIZE",   VX_SCALAR_OP_GREATER, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "0", "7", "0|vx_false_e"),\
    ARG("ControlFlow/GREATER/FLOAT32",VX_SCALAR_OP_GREATER, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "-2.0f", "1.5f", "0|vx_false_e"),\
    ARG("ControlFlow/GREATER/INT8",   VX_SCALAR_OP_GREATER, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "2", "1", "1|vx_true_e"),\
    ARG("ControlFlow/GREATER/UINT8",  VX_SCALAR_OP_GREATER, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "3", "2", "1|vx_true_e"),\
    ARG("ControlFlow/GREATER/INT16",  VX_SCALAR_OP_GREATER, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "4", "3", "1|vx_true_e"),\
    ARG("ControlFlow/GREATER/UINT16", VX_SCALAR_OP_GREATER, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "5", "4", "1|vx_true_e"),\
    ARG("ControlFlow/GREATER/INT32",  VX_SCALAR_OP_GREATER, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "6", "5", "1|vx_true_e"),\
    ARG("ControlFlow/GREATER/UINT32", VX_SCALAR_OP_GREATER, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "7", "6", "1|vx_true_e"),\
    ARG("ControlFlow/GREATER/SIZE",   VX_SCALAR_OP_GREATER, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "8", "7", "1|vx_true_e"),\
    ARG("ControlFlow/GREATER/FLOAT32",VX_SCALAR_OP_GREATER, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "2.0f", "1.5f", "1|vx_true_e"),\
    ARG("ControlFlow/GREATEREQ/INT8",    VX_SCALAR_OP_GREATEREQ, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "-2", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATEREQ/UINT8",   VX_SCALAR_OP_GREATEREQ, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "0", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATEREQ/INT16",   VX_SCALAR_OP_GREATEREQ, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "-2", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATEREQ/UINT16",  VX_SCALAR_OP_GREATEREQ, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "0", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATEREQ/INT32",   VX_SCALAR_OP_GREATEREQ, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "-2", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATEREQ/UINT32",  VX_SCALAR_OP_GREATEREQ, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "0", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATEREQ/SIZE",    VX_SCALAR_OP_GREATEREQ, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "0", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATEREQ/FLOAT32", VX_SCALAR_OP_GREATEREQ, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "0", "1", "0|vx_false_e"),\
    ARG("ControlFlow/GREATEREQ/INT8",    VX_SCALAR_OP_GREATEREQ, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_BOOL, "-1", "-1", "1|vx_true_e"),\
    ARG("ControlFlow/GREATEREQ/UINT8",   VX_SCALAR_OP_GREATEREQ, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_BOOL, "2", "1", "1|vx_true_e"),\
    ARG("ControlFlow/GREATEREQ/INT16",   VX_SCALAR_OP_GREATEREQ, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_BOOL, "2", "1", "1|vx_true_e"),\
    ARG("ControlFlow/GREATEREQ/UINT16",  VX_SCALAR_OP_GREATEREQ, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_BOOL, "2", "1", "1|vx_true_e"),\
    ARG("ControlFlow/GREATEREQ/INT32",   VX_SCALAR_OP_GREATEREQ, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_BOOL, "2", "1", "1|vx_true_e"),\
    ARG("ControlFlow/GREATEREQ/UINT32",  VX_SCALAR_OP_GREATEREQ, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_BOOL, "2", "1", "1|vx_true_e"),\
    ARG("ControlFlow/GREATEREQ/SIZE",    VX_SCALAR_OP_GREATEREQ, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_BOOL, "2", "1", "1|vx_true_e"),\
    ARG("ControlFlow/GREATEREQ/FLOAT32", VX_SCALAR_OP_GREATEREQ, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_BOOL, "2.0f", "1.0f", "1|vx_true_e"),\
    ARG("ControlFlow/ADD/INT8",   VX_SCALAR_OP_ADD, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_INT8,   "2",   "1",    "3"),\
    ARG("ControlFlow/ADD/INT8",   VX_SCALAR_OP_ADD, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_INT8,   "2",   "-1",    "1"),\
    ARG("ControlFlow/ADD/UINT8",  VX_SCALAR_OP_ADD, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_UINT8,  "3",   "2",    "5"),\
    ARG("ControlFlow/ADD/INT16",  VX_SCALAR_OP_ADD, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_INT16,  "4",   "3",    "7"),\
    ARG("ControlFlow/ADD/INT16",  VX_SCALAR_OP_ADD, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_INT16,  "4",   "-3",    "1"),\
    ARG("ControlFlow/ADD/UINT16", VX_SCALAR_OP_ADD, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_UINT16, "5",   "4",    "9"),\
    ARG("ControlFlow/ADD/INT32",  VX_SCALAR_OP_ADD, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_INT32,  "6",   "5",    "11"),\
    ARG("ControlFlow/ADD/INT32",  VX_SCALAR_OP_ADD, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_INT32,  "-6",   "5",    "-1"),\
    ARG("ControlFlow/ADD/UINT32", VX_SCALAR_OP_ADD, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_UINT32, "7",   "6",    "13"),\
    ARG("ControlFlow/ADD/SIZE",   VX_SCALAR_OP_ADD, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_SIZE,   "8",   "7",    "15"),\
    ARG("ControlFlow/ADD/FLOAT32",VX_SCALAR_OP_ADD, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, "-1.5f", "1.0f", "-0.5f"),\
    ARG("ControlFlow/SUBTRACT/INT8",   VX_SCALAR_OP_SUBTRACT, VX_TYPE_INT8,    VX_TYPE_INT8,   VX_TYPE_INT8,    "2",    "1",     "1"),\
    ARG("ControlFlow/SUBTRACT/UINT8",  VX_SCALAR_OP_SUBTRACT, VX_TYPE_UINT8,   VX_TYPE_UINT8,  VX_TYPE_UINT8,   "3",    "2",     "1"),\
    ARG("ControlFlow/SUBTRACT/INT16",  VX_SCALAR_OP_SUBTRACT, VX_TYPE_INT16,   VX_TYPE_INT16,  VX_TYPE_INT16,   "4",    "3",     "1"),\
    ARG("ControlFlow/SUBTRACT/UINT16", VX_SCALAR_OP_SUBTRACT, VX_TYPE_UINT16,  VX_TYPE_UINT16, VX_TYPE_UINT16,  "5",    "4",     "1"),\
    ARG("ControlFlow/SUBTRACT/INT32",  VX_SCALAR_OP_SUBTRACT, VX_TYPE_INT32,   VX_TYPE_INT32,  VX_TYPE_INT32,   "6",    "5",     "1"),\
    ARG("ControlFlow/SUBTRACT/UINT32", VX_SCALAR_OP_SUBTRACT, VX_TYPE_UINT32,  VX_TYPE_UINT32, VX_TYPE_UINT32,  "7",    "6",     "1"),\
    ARG("ControlFlow/SUBTRACT/SIZE",   VX_SCALAR_OP_SUBTRACT, VX_TYPE_SIZE,    VX_TYPE_SIZE,   VX_TYPE_SIZE,    "8",    "7",     "1"),\
    ARG("ControlFlow/SUBTRACT/FLOAT32",VX_SCALAR_OP_SUBTRACT, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, "1.0f", "1.0f", "0.0f"),\
    ARG("ControlFlow/MULTIPLY/INT8",   VX_SCALAR_OP_MULTIPLY, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_INT8,    "2",    "2",     "4"),\
    ARG("ControlFlow/MULTIPLY/UINT8",  VX_SCALAR_OP_MULTIPLY, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_UINT8,   "3",    "2",     "6"),\
    ARG("ControlFlow/MULTIPLY/INT16",  VX_SCALAR_OP_MULTIPLY, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_INT16,   "4",    "2",     "8"),\
    ARG("ControlFlow/MULTIPLY/UINT16", VX_SCALAR_OP_MULTIPLY, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_UINT16,  "5",    "2",     "10"),\
    ARG("ControlFlow/MULTIPLY/INT32",  VX_SCALAR_OP_MULTIPLY, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_INT32,   "6",    "2",     "12"),\
    ARG("ControlFlow/MULTIPLY/UINT32", VX_SCALAR_OP_MULTIPLY, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_UINT32,  "7",    "2",     "14"),\
    ARG("ControlFlow/MULTIPLY/SIZE",   VX_SCALAR_OP_MULTIPLY, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_SIZE,    "8",    "2",     "16"),\
    ARG("ControlFlow/MULTIPLY/FLOAT32",VX_SCALAR_OP_MULTIPLY, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, "1.0f", "2.0f", "2.0f"),\
    ARG("ControlFlow/MULTIPLY/FLOAT32",VX_SCALAR_OP_MULTIPLY, VX_TYPE_SIZE, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, "4", "0.5f", "2.0f"),\
    ARG("ControlFlow/DIVIDE/INT8",   VX_SCALAR_OP_DIVIDE, VX_TYPE_INT8,    VX_TYPE_INT8,   VX_TYPE_INT8,    "4",     "2",     "2"),\
    ARG("ControlFlow/DIVIDE/UINT8",  VX_SCALAR_OP_DIVIDE, VX_TYPE_UINT8,   VX_TYPE_UINT8,  VX_TYPE_UINT8,   "6",     "3",     "2"),\
    ARG("ControlFlow/DIVIDE/INT16",  VX_SCALAR_OP_DIVIDE, VX_TYPE_INT16,   VX_TYPE_INT16,  VX_TYPE_INT16,    "8",     "4",     "2"),\
    ARG("ControlFlow/DIVIDE/UINT16", VX_SCALAR_OP_DIVIDE, VX_TYPE_UINT16,  VX_TYPE_UINT16, VX_TYPE_UINT16,  "10",    "5",     "2"),\
    ARG("ControlFlow/DIVIDE/INT32",  VX_SCALAR_OP_DIVIDE, VX_TYPE_INT32,   VX_TYPE_INT32,  VX_TYPE_INT32,   "12",    "6",     "2"),\
    ARG("ControlFlow/DIVIDE/UINT32", VX_SCALAR_OP_DIVIDE, VX_TYPE_UINT32,  VX_TYPE_UINT32, VX_TYPE_UINT32,  "14",    "7",     "2"),\
    ARG("ControlFlow/DIVIDE/SIZE",   VX_SCALAR_OP_DIVIDE, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_SIZE,    "16",    "8",     "2"),\
    ARG("ControlFlow/DIVIDE/FLOAT32",VX_SCALAR_OP_DIVIDE, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, "1.0f", "2.0f", "0.5f"),\
    ARG("ControlFlow/DIVIDE/FLOAT32",VX_SCALAR_OP_DIVIDE, VX_TYPE_SIZE, VX_TYPE_SIZE, VX_TYPE_FLOAT32, "7", "2", "3"),\
    ARG("ControlFlow/MODULUS/INT8",   VX_SCALAR_OP_MODULUS, VX_TYPE_INT8,   VX_TYPE_INT8,   VX_TYPE_INT8,    "5",     "2",     "1"),\
    ARG("ControlFlow/MODULUS/UINT8",  VX_SCALAR_OP_MODULUS, VX_TYPE_UINT8,  VX_TYPE_UINT8,  VX_TYPE_UINT8,   "8",     "3",     "2"),\
    ARG("ControlFlow/MODULUS/INT16",  VX_SCALAR_OP_MODULUS, VX_TYPE_INT16,  VX_TYPE_INT16,  VX_TYPE_INT16,    "8",     "4",     "0"),\
    ARG("ControlFlow/MODULUS/UINT16", VX_SCALAR_OP_MODULUS, VX_TYPE_UINT16, VX_TYPE_UINT16, VX_TYPE_UINT16,  "10",    "6",     "4"),\
    ARG("ControlFlow/MODULUS/INT32",  VX_SCALAR_OP_MODULUS, VX_TYPE_INT32,  VX_TYPE_INT32,  VX_TYPE_INT32,   "12",    "7",     "5"),\
    ARG("ControlFlow/MODULUS/UINT32", VX_SCALAR_OP_MODULUS, VX_TYPE_UINT32, VX_TYPE_UINT32, VX_TYPE_UINT32,  "14",    "13",     "1"),\
    ARG("ControlFlow/MIN/INT8",   VX_SCALAR_OP_MIN, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_INT8,    "4",     "-2",     "-2"),\
    ARG("ControlFlow/MIN/UINT8",  VX_SCALAR_OP_MIN, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_UINT8,   "6",     "3",     "3"),\
    ARG("ControlFlow/MIN/INT16",  VX_SCALAR_OP_MIN, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_INT16,    "8",     "4",     "4"),\
    ARG("ControlFlow/MIN/UINT16", VX_SCALAR_OP_MIN, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_UINT16,  "10",    "5",     "5"),\
    ARG("ControlFlow/MIN/INT32",  VX_SCALAR_OP_MIN, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_INT32,   "12",    "6",     "6"),\
    ARG("ControlFlow/MIN/UINT32", VX_SCALAR_OP_MIN, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_UINT32,  "14",    "7",     "7"),\
    ARG("ControlFlow/MIN/SIZE",   VX_SCALAR_OP_MIN, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_SIZE,    "16",    "8",     "8"),\
    ARG("ControlFlow/MIN/FLOAT32",VX_SCALAR_OP_MIN, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, "2.0f", "-1.0f", "-1.0f"),\
    ARG("ControlFlow/MIN/FLOAT32",VX_SCALAR_OP_MIN, VX_TYPE_FLOAT32, VX_TYPE_INT32,   VX_TYPE_INT32,   "7.1f", "7",     "7"),\
    ARG("ControlFlow/MIN/FLOAT32",VX_SCALAR_OP_MIN, VX_TYPE_FLOAT32, VX_TYPE_UINT32,  VX_TYPE_UINT32,  "0.5f",  "1",    "0"),\
    ARG("ControlFlow/MAX_INT8",   VX_SCALAR_OP_MAX, VX_TYPE_INT8,    VX_TYPE_INT8,    VX_TYPE_INT8,    "4",     "2",     "4"),\
    ARG("ControlFlow/MAX_UINT8",  VX_SCALAR_OP_MAX, VX_TYPE_UINT8,   VX_TYPE_UINT8,   VX_TYPE_UINT8,   "6",     "3",     "6"),\
    ARG("ControlFlow/MAX_INT16",  VX_SCALAR_OP_MAX, VX_TYPE_INT16,   VX_TYPE_INT16,   VX_TYPE_INT16,    "8",     "4",     "8"),\
    ARG("ControlFlow/MAX_UINT16", VX_SCALAR_OP_MAX, VX_TYPE_UINT16,  VX_TYPE_UINT16,  VX_TYPE_UINT16,  "10",    "5",     "10"),\
    ARG("ControlFlow/MAX_INT32",  VX_SCALAR_OP_MAX, VX_TYPE_INT32,   VX_TYPE_INT32,   VX_TYPE_INT32,   "12",    "-6",     "12"),\
    ARG("ControlFlow/MAX_UINT32", VX_SCALAR_OP_MAX, VX_TYPE_UINT32,  VX_TYPE_UINT32,  VX_TYPE_UINT32,  "14",    "7",     "14"),\
    ARG("ControlFlow/MAX_SIZE",   VX_SCALAR_OP_MAX, VX_TYPE_SIZE,    VX_TYPE_SIZE,    VX_TYPE_SIZE,    "16",    "8",     "16"),\
    ARG("ControlFlow/MAX/FLOAT32",VX_SCALAR_OP_MAX, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, "2.0f", "1.0f", "2.0f"),\
    ARG("ControlFlow/MAX/FLOAT32",VX_SCALAR_OP_MAX, VX_TYPE_FLOAT32, VX_TYPE_UINT32,  VX_TYPE_UINT32,  "2.99f", "3",     "3"),\
    ARG("ControlFlow/MAX/FLOAT32",VX_SCALAR_OP_MAX, VX_TYPE_FLOAT32, VX_TYPE_INT32,   VX_TYPE_FLOAT32, "2.99f","-3",     "2.99f"),\

static vx_status ReadValueFromString(const char * str, vx_enum type, scalar_val * val)
{
    vx_status status = VX_SUCCESS;
    switch (type)
    {
    case VX_TYPE_CHAR:
        val->chr = atoi(str);
        break;

    case VX_TYPE_INT8:
        val->s08 = atoi(str);
        break;

    case VX_TYPE_UINT8:
        val->u08 = atoi(str);
        break;

    case VX_TYPE_INT16:
        val->s16 = atoi(str);
        break;

    case VX_TYPE_UINT16:
        val->u16 = atoi(str);
        break;

    case VX_TYPE_INT32:
        val->s32 = atoi(str);
        break;

    case VX_TYPE_UINT32:
        val->u32 = atoi(str);
        break;

    case VX_TYPE_INT64:
        val->s64 = atoi(str);
        break;

    case VX_TYPE_UINT64:
        val->u64 = atoi(str);
        break;

    case VX_TYPE_FLOAT32:
        val->f32 = atof(str);
        break;

    case VX_TYPE_SIZE:
        val->size = atoi(str);
        break;

    case VX_TYPE_BOOL:
        val->boolean = atoi(str) ? vx_true_e : vx_false_e;
        break;

    default:
        status = VX_ERROR_INVALID_TYPE;
        break;
    }
    return status;
}


TEST_WITH_ARG(ControlFlow, testScalarOperationNode, scalar_op_arg, PARAMETERS_SCALAR_OP)
{
    vx_node node = 0;
    vx_graph graph = 0;
    vx_scalar a = 0, b = 0, o = 0;
    vx_context context = context_->vx_context_;

    vx_enum param_a_type = arg_->param_a_type;
    vx_enum param_b_type = arg_->param_b_type;
    vx_enum operation = arg_->operation;
    vx_enum result_type = arg_->result_type;
    scalar_val a_value;
    scalar_val b_value;
    scalar_val o_check;
    scalar_val o_value;

    VX_CALL(ReadValueFromString(arg_->a_value, arg_->param_a_type, &a_value));
    VX_CALL(ReadValueFromString(arg_->b_value, arg_->param_b_type, &b_value));
    VX_CALL(ReadValueFromString(arg_->o_check, arg_->result_type,  &o_check));

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(a = vxCreateScalar(context, param_a_type, &a_value), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(b = vxCreateScalar(context, param_b_type, &b_value), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(o = vxCreateScalar(context, result_type, &o_value), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(node = vxScalarOperationNode(graph, operation , a, b, o), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    switch (result_type)
    {
    case VX_TYPE_CHAR:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.chr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.chr, o_value.chr);
        break;

    case VX_TYPE_INT8:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.s08, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.s08, o_value.s08);
        break;

    case VX_TYPE_UINT8:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.u08, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.u08, o_value.u08);
        break;

    case VX_TYPE_INT16:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.s16, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.s16, o_value.s16);
        break;

    case VX_TYPE_UINT16:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.u16, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.u16, o_value.u16);
        break;

    case VX_TYPE_INT32:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.s32, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.s32, o_value.s32);
        break;

    case VX_TYPE_UINT32:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.u32, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.u32, o_value.u32);
        break;

    case VX_TYPE_INT64:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.s64, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.s64, o_value.s64);
        break;

    case VX_TYPE_UINT64:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.u64, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.u64, o_value.u64);
        break;

    case VX_TYPE_FLOAT32:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.f32, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT(fabs(o_check.f32 - o_value.f32) < 0.000001f);
        break;

    case VX_TYPE_SIZE:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.size, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.size, o_value.size);
        break;

    case VX_TYPE_BOOL:
        VX_CALL(vxCopyScalar(o, (void *)&o_value.boolean, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT_EQ_INT(o_check.boolean, o_value.boolean);
        break;

    default:
        FAIL("Unsupported type: (%.4s)", &result_type);
        break;
    }

    VX_CALL(vxReleaseScalar(&a));
    VX_CALL(vxReleaseScalar(&b));
    VX_CALL(vxReleaseScalar(&o));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(b == 0);
    ASSERT(a == 0);
    ASSERT(o == 0);
    ASSERT(node == 0);
    ASSERT(graph == 0);
}

TESTCASE_TESTS(ControlFlow, testSelectNode, testScalarOperationNode)

#endif //OPENVX_USE_ENHANCED_VISION
