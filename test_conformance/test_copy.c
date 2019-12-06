/*

 * Copyright (c) 2017-2017 The Khronos Group Inc.
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

#include "test_engine/test.h"

#define IMAGE_SIZE_X 10
#define IMAGE_SIZE_Y 10
#define MATRIX_SIZE_X 5
#define MATRIX_SIZE_Y 5
#define CONVOLUTION_X 3
#define CONVOLUTION_Y 3
#define N 100
#define OBJECT_ARRAY_COUNT 10
#define PYRAMID_LEVELS 2
#define TENSOR_DIMS_NUM 2
#define TENSOR_DIMS_LENGTH 8
TESTCASE(Copy, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char* testName;
    const char* p;
    vx_enum item_type;
} copy_arg;


static vx_reference own_create_exemplar(vx_context context, vx_enum item_type, vx_uint8 value)
{
    vx_reference exemplar = NULL;
    vx_enum format = VX_DF_IMAGE_U8;
    vx_enum obj_item_type = VX_TYPE_UINT8;
    vx_size levels = PYRAMID_LEVELS;
    vx_size bins = 36;
    vx_int32 offset = 0;
    vx_uint32 range = 360;
    vx_enum thresh_type = VX_THRESHOLD_TYPE_BINARY;
    vx_scalar scalar_exemplar;
    vx_uint8 scalar_value = 0;
    vx_size * dims;
    switch (item_type)
    {
        case VX_TYPE_IMAGE:
            exemplar = (vx_reference)vxCreateImage(context, IMAGE_SIZE_X, IMAGE_SIZE_Y, format);
            break;
        case VX_TYPE_ARRAY:
            exemplar = (vx_reference)vxCreateArray(context, VX_TYPE_COORDINATES2D, N);
            break;
        case VX_TYPE_SCALAR:
            exemplar = (vx_reference)vxCreateScalar(context, obj_item_type, &value);
            break;
        case VX_TYPE_MATRIX:
            exemplar = (vx_reference)vxCreateMatrix(context, obj_item_type, MATRIX_SIZE_X, MATRIX_SIZE_Y);
            break;
        case VX_TYPE_CONVOLUTION:
            exemplar = (vx_reference)vxCreateConvolution(context, CONVOLUTION_X, CONVOLUTION_Y);
            break;
        case VX_TYPE_DISTRIBUTION:
            exemplar = (vx_reference)vxCreateDistribution(context, bins, offset, range);
            break;
        case VX_TYPE_LUT:
            exemplar = (vx_reference)vxCreateLUT(context, obj_item_type, N);
            break;
        case VX_TYPE_PYRAMID:
            exemplar = (vx_reference)vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF, IMAGE_SIZE_X, IMAGE_SIZE_Y, format);
            break;
        case VX_TYPE_REMAP:
            exemplar = (vx_reference)vxCreateRemap(context, IMAGE_SIZE_X, IMAGE_SIZE_Y, IMAGE_SIZE_X*2, IMAGE_SIZE_Y*2);
           break;
        case VX_TYPE_THRESHOLD:
            exemplar = (vx_reference)vxCreateThresholdForImage(context, thresh_type, format, format);
            break;
        case VX_TYPE_OBJECT_ARRAY:
            scalar_exemplar = vxCreateScalar(context, obj_item_type, &scalar_value);
            exemplar = (vx_reference)vxCreateObjectArray(context, (vx_reference)scalar_exemplar, OBJECT_ARRAY_COUNT);
            vxReleaseReference((vx_reference*)&scalar_exemplar);
            break;
        case VX_TYPE_TENSOR:
            dims = ct_alloc_mem(TENSOR_DIMS_NUM * sizeof(vx_size));
            for(vx_size i = 0; i < TENSOR_DIMS_NUM; i++)
            {
                dims[i] = TENSOR_DIMS_LENGTH;
            }
            exemplar = (vx_reference)vxCreateTensor(context, TENSOR_DIMS_NUM, dims, obj_item_type, 0);
            ct_free_mem(dims);
            break;
        default:
            break;
    }
    return exemplar;
}

#define  ADD_VX_COPY_TYPES(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_IMAGE", __VA_ARGS__, VX_TYPE_IMAGE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_ARRAY", __VA_ARGS__, VX_TYPE_ARRAY)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_SCALAR", __VA_ARGS__, VX_TYPE_SCALAR)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_MATRIX", __VA_ARGS__, VX_TYPE_MATRIX)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_CONVOLUTION", __VA_ARGS__, VX_TYPE_CONVOLUTION)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_DISTRIBUTION", __VA_ARGS__, VX_TYPE_DISTRIBUTION)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_LUT", __VA_ARGS__, VX_TYPE_LUT)),\
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_OBJECT_ARRAY", __VA_ARGS__, VX_TYPE_OBJECT_ARRAY)),\
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_PYRAMID", __VA_ARGS__, VX_TYPE_PYRAMID)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_TENSOR", __VA_ARGS__, VX_TYPE_TENSOR )), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_THRESHOLD", __VA_ARGS__, VX_TYPE_THRESHOLD)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_REMAP", __VA_ARGS__, VX_TYPE_REMAP))
#define PARAMETERS \
    CT_GENERATE_PARAMETERS("Copy", ADD_VX_COPY_TYPES, ARG, NULL)

TEST_WITH_ARG(Copy, testNodeCreation, copy_arg, PARAMETERS)
{
    vx_context context = context_->vx_context_;
    vx_reference input = NULL;
    vx_reference output = NULL;
    vx_enum input_type = arg_->item_type;

    ASSERT_VX_OBJECT(input = own_create_exemplar(context, input_type, 0), (enum vx_type_e)input_type);
    ASSERT_VX_OBJECT(output = own_create_exemplar(context, input_type, 1), (enum vx_type_e)input_type);

    vx_graph graph = 0;
    vx_node node = 0;
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxCopyNode(graph, input, output), VX_TYPE_NODE);

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseReference(&input));
    VX_CALL(vxReleaseReference(&output));
    ASSERT(input == 0);
    ASSERT(output == 0);
}

TEST_WITH_ARG(Copy, testGraphProcessing, copy_arg, PARAMETERS)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;

    vx_reference input = NULL;
    vx_reference output = NULL;
    vx_enum input_type = arg_->item_type;
    vx_enum output_type = VX_TYPE_INVALID;
    vx_int16 gx[CONVOLUTION_X][CONVOLUTION_Y] = {
        { 3, 0, -3 },
        { 10, 0, -10 },
        { 3, 0, -3 },
    };

    ASSERT_VX_OBJECT(input = own_create_exemplar(context, input_type, 0), (enum vx_type_e)input_type);
    ASSERT_VX_OBJECT(output = own_create_exemplar(context, input_type, 1), (enum vx_type_e)input_type);
    switch (input_type)
    {
        case VX_TYPE_IMAGE:
            {
            vx_image input_image = (vx_image)input;
            void *p = NULL;
            vx_map_id input_map_id;
            vx_rectangle_t rect;
            vx_imagepatch_addressing_t addr;
            VX_CALL(vxGetValidRegionImage(input_image, &rect));
            VX_CALL(vxMapImagePatch(input_image, &rect, 0, &input_map_id, &addr, &p, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
            for (vx_size i = 0; i < addr.dim_x*addr.dim_y; i++) {
                vx_uint8 *pPixel = vxFormatImagePatchAddress1d(p, i, &addr);
                *pPixel = i;
            }
            VX_CALL( vxUnmapImagePatch(input_image, input_map_id));
            break;
            }
        case VX_TYPE_ARRAY:
            {
                vx_coordinates2d_t localArrayInit[N];
                vx_array array = (vx_array)input;
                /* Initialization */
                for (int i = 0; i < N; i++) {
                    localArrayInit[i].x = i;
                    localArrayInit[i].y = i;
                }
                VX_CALL( vxAddArrayItems(array, N, &localArrayInit[0], sizeof(vx_coordinates2d_t)) );
                break;
            }
        case VX_TYPE_MATRIX:
            {
                vx_uint8* data = ct_alloc_mem(MATRIX_SIZE_X * MATRIX_SIZE_Y * sizeof(vx_uint8));
                vx_size i;
                for (i = 0; i < MATRIX_SIZE_X * MATRIX_SIZE_Y; i++)
                {
                    data[i] = 1;
                }
                VX_CALL(vxCopyMatrix((vx_matrix)input, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ct_free_mem(data);
                break;
            }
        case VX_TYPE_CONVOLUTION:
            {
                VX_CALL(vxCopyConvolutionCoefficients((vx_convolution)input, gx, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                break;
            }
        case VX_TYPE_OBJECT_ARRAY:
            {
                vx_scalar input_item = NULL;
                vx_uint8  scalar_value=1;
                for (vx_size i = 0; i < OBJECT_ARRAY_COUNT; i++)
                {
                    ASSERT_VX_OBJECT(input_item = (vx_scalar)vxGetObjectArrayItem((vx_object_array)input, i), VX_TYPE_SCALAR);
                    VX_CALL(vxCopyScalar(input_item, &scalar_value, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

                    VX_CALL(vxReleaseReference((vx_reference*)&input_item));
                    ASSERT(input_item == 0);
                }
                break;
            }
        case VX_TYPE_LUT:
            {
                vx_size size = N*sizeof(vx_uint8);
                void* data =  ct_alloc_mem(size);
                vx_uint8* data8 = (vx_uint8*)data;
                for (vx_size i = 0; i < N; ++i)
                    data8[i] = 1;
                VX_CALL(vxCopyLUT((vx_lut)input, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ct_free_mem(data);
                break;
            }
        case VX_TYPE_PYRAMID:
            {
                vx_pyramid input_pyramid = (vx_pyramid)input;
                vx_image input_image = NULL;
                ASSERT_VX_OBJECT(input_image = vxGetPyramidLevel(input_pyramid, 0), VX_TYPE_IMAGE);
                void *p = NULL;
                vx_map_id input_map_id;
                vx_rectangle_t rect;
                vx_imagepatch_addressing_t addr;
                VX_CALL(vxGetValidRegionImage(input_image, &rect));
                VX_CALL(vxMapImagePatch(input_image, &rect, 0, &input_map_id, &addr, &p, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
                for (vx_size i = 0; i < addr.dim_x*addr.dim_y; i++)
                {
                    vx_uint8 *pPixel = vxFormatImagePatchAddress1d(p, i, &addr);
                    *pPixel = 1;
                }
                VX_CALL( vxUnmapImagePatch(input_image, input_map_id));
                VX_CALL(vxReleaseImage(&input_image));
                ASSERT(input_image == 0);
                break;
            }
        case VX_TYPE_REMAP:
            {
                vx_remap input_remap = (vx_remap)input;
                vx_rectangle_t rect = { 0, 0,  IMAGE_SIZE_X*2, IMAGE_SIZE_Y*2};
                vx_size stride = IMAGE_SIZE_X*2;
                vx_size stride_y = sizeof(vx_coordinates2df_t) * (stride);
                vx_size size = stride * IMAGE_SIZE_Y*2;
                vx_coordinates2df_t* ptr_w = ct_alloc_mem(sizeof(vx_coordinates2df_t) * size);

                for (vx_size i = 0; i < IMAGE_SIZE_Y*2; i++)
                {
                    for (vx_size j = 0; j < IMAGE_SIZE_X*2; j++)
                    {
                        vx_coordinates2df_t *coord_ptr = &(ptr_w[i * stride + j]);
                        coord_ptr->x = (vx_float32)j;
                        coord_ptr->y = (vx_float32)i;
                    }
                }

                VX_CALL(vxCopyRemapPatch(input_remap, &rect, stride_y, ptr_w, VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ct_free_mem(ptr_w);
                break;
            }
        case VX_TYPE_THRESHOLD:
            {
                vx_threshold input_threshold = (vx_threshold)input;
                vx_pixel_value_t pa;
                pa.U8 = 8;
                ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCopyThresholdValue(input_threshold, &pa, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                break;
            }
        case VX_TYPE_TENSOR:
            {
                vx_tensor input_tensor = (vx_tensor)input;
                vx_size start[TENSOR_DIMS_NUM] = { 0 };
                vx_size strides[TENSOR_DIMS_NUM]= { 0 };
                vx_size * dims = ct_alloc_mem(TENSOR_DIMS_NUM * sizeof(vx_size));
                for(vx_size i = 0; i < TENSOR_DIMS_NUM; i++)
                {
                    dims[i] = TENSOR_DIMS_LENGTH;
                    start[i] = 0;
                    strides[i] = i ? strides[i - 1] * dims[i - 1] : sizeof(vx_uint8);
                }
                const vx_size bytes = dims[TENSOR_DIMS_NUM - 1] * strides[TENSOR_DIMS_NUM - 1];
                void * data = ct_alloc_mem(bytes);
                vx_uint8* u8_data = (vx_uint8*)data;
                for(vx_size i = 0; i < bytes; i++)
                {
                    u8_data[i] = 2;
                }

                VX_CALL(vxCopyTensorPatch(input_tensor, TENSOR_DIMS_NUM, start, dims, strides, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ct_free_mem(dims);
                ct_free_mem(data);
                break;
            }
        default:
            break;
    }
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxCopyNode(graph, input, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxQueryReference((vx_reference)output, VX_REFERENCE_TYPE, &output_type, sizeof(output_type)));
    ASSERT_EQ_INT(input_type, output_type);

    switch (output_type)
    {
        case VX_TYPE_IMAGE:
            {
            vx_image output_image = (vx_image)output;
            void *p = NULL;
            vx_map_id output_map_id;
            vx_rectangle_t rect;
            vx_imagepatch_addressing_t addr;
            VX_CALL(vxGetValidRegionImage(output_image, &rect));
            VX_CALL(vxMapImagePatch(output_image, &rect, 0, &output_map_id, &addr, &p,
                        VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
            for (vx_uint32 i = 0; i < (addr.dim_x * addr.dim_y); i++) {
                vx_uint8 *pPixel = vxFormatImagePatchAddress1d(p, i, &addr);
                ASSERT(*pPixel == i);
            }

            VX_CALL( vxUnmapImagePatch(output_image, output_map_id));
            break;
            }
        case VX_TYPE_ARRAY:
            {
                vx_array array = (vx_array)output;
                vx_uint8 *p = NULL;
                vx_size stride = 0;
                vx_map_id map_id;
                VX_CALL( vxMapArrayRange(array, N/2, N, &map_id, &stride, (void **)&p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
                ASSERT(stride >=  sizeof(vx_coordinates2d_t));
                ASSERT(p != NULL);

                for (int i = N/2; i<N; i++) {
                    ASSERT(((vx_coordinates2d_t *)(p+stride*(i-N/2)))->x == i);
                    ASSERT(((vx_coordinates2d_t *)(p+stride*(i-N/2)))->y == i);
                }
                VX_CALL( vxUnmapArrayRange (array, map_id));
                break;
            }
        case VX_TYPE_SCALAR:
            {
                vx_uint8  in=2, out=2;
                vx_scalar input_scalar = (vx_scalar)input;
                vx_scalar output_scalar = (vx_scalar)output;
                VX_CALL(vxCopyScalar(input_scalar, &in, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                VX_CALL(vxCopyScalar(output_scalar, &out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                ASSERT_EQ_INT(in, out);
                break;
            }
        case VX_TYPE_MATRIX:
            {
                vx_uint8* data = ct_alloc_mem(MATRIX_SIZE_X * MATRIX_SIZE_Y * sizeof(vx_uint8));
                VX_CALL(vxCopyMatrix((vx_matrix)output, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                vx_size i;
                for (i = 0; i < MATRIX_SIZE_X * MATRIX_SIZE_Y; i++)
                {
                    ASSERT_EQ_INT(data[i], 1);
                }
                ct_free_mem(data);
                break;
            }
        case VX_TYPE_CONVOLUTION:
            {
              vx_int16 *data = (vx_int16 *)ct_alloc_mem(CONVOLUTION_X*CONVOLUTION_Y*sizeof(vx_int16));
              VX_CALL(vxCopyConvolutionCoefficients((vx_convolution)output, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

              for (vx_size i = 0; i < CONVOLUTION_X; i++)
              {
                  for (vx_size j = 0; j < CONVOLUTION_Y; j++)
                  {
                      ASSERT(gx[i][j] == data[i * CONVOLUTION_X + j]);
                  }
              }

              ct_free_mem(data);
              break;
            }
        case VX_TYPE_OBJECT_ARRAY:
            {
                vx_scalar output_item = NULL;
                vx_uint8  input_value=1, output_value = 0;
                for (vx_size i = 0; i < OBJECT_ARRAY_COUNT; i++)
                {
                    ASSERT_VX_OBJECT(output_item = (vx_scalar)vxGetObjectArrayItem((vx_object_array)output, i), VX_TYPE_SCALAR);
                    VX_CALL(vxCopyScalar(output_item, &output_value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                    ASSERT_EQ_INT(output_value, input_value);
                    VX_CALL(vxReleaseReference((vx_reference*)&output_item));
                    ASSERT(output_item == 0);
                }
                break;
            }
        case VX_TYPE_LUT:
            {
                vx_map_id map_id;
                void* lut_data = NULL;
                VX_CALL(vxMapLUT((vx_lut)output, &map_id, &lut_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                vx_uint8* data8 = (vx_uint8*)lut_data;
                for (vx_size i = 0; i < N; i++)
                {
                    ASSERT_EQ_INT(data8[i], 1);
                }
                VX_CALL(vxUnmapLUT((vx_lut)output, map_id));
                break;
            }
        case VX_TYPE_PYRAMID:
            {
                vx_pyramid output_pyramid = (vx_pyramid)output;
                vx_image output_image = NULL;
                ASSERT_VX_OBJECT(output_image = vxGetPyramidLevel(output_pyramid, 0), VX_TYPE_IMAGE);
                void *p = NULL;
                vx_map_id map_id;
                vx_rectangle_t rect;
                vx_imagepatch_addressing_t addr;
                VX_CALL(vxGetValidRegionImage(output_image, &rect));
                VX_CALL(vxMapImagePatch(output_image, &rect, 0, &map_id, &addr, &p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                for (vx_size i = 0; i < addr.dim_x*addr.dim_y; i++)
                {
                    vx_uint8 *pPixel = vxFormatImagePatchAddress1d(p, i, &addr);
                    ASSERT_EQ_INT(*pPixel, 1);
                }
                VX_CALL( vxUnmapImagePatch(output_image, map_id));
                VX_CALL(vxReleaseImage(&output_image));
                ASSERT(output_image == 0);
                break;
            }
        case VX_TYPE_THRESHOLD:
            {
                vx_threshold output_threshold = (vx_threshold)output;
                vx_pixel_value_t pa;
                ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCopyThresholdValue(output_threshold, &pa, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                ASSERT(pa.U8 == 8);
                break;
            }
        case VX_TYPE_REMAP:
            {
                vx_remap output_remap = (vx_remap)output;
                vx_rectangle_t rect = { 0, 0,  IMAGE_SIZE_X*2, IMAGE_SIZE_Y*2};
                vx_size stride = IMAGE_SIZE_X*2;
                vx_size stride_y = 0;
                vx_coordinates2df_t *ptr_r = 0;
                vx_map_id map_id;

                VX_CALL(vxMapRemapPatch(output_remap, &rect, &map_id, &stride_y, (void **)&ptr_r, VX_TYPE_COORDINATES2DF, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

                for (vx_size i = 0; i < IMAGE_SIZE_Y*2; i++)
                {
                    for (vx_size j = 0; j < IMAGE_SIZE_X*2; j++)
                    {
                        vx_coordinates2df_t *coord_ptr = &(ptr_r[i * stride + j]);
                        ASSERT_EQ_INT((vx_uint32)coord_ptr->x, j);
                        ASSERT_EQ_INT((vx_uint32)coord_ptr->y, i);
                    }
                }

                VX_CALL(vxUnmapRemapPatch(output_remap, map_id));
                break;
            }
        case VX_TYPE_TENSOR:
            {
                vx_tensor output_tensor = (vx_tensor)output;
                vx_size start[TENSOR_DIMS_NUM] = { 0 };
                vx_size strides[TENSOR_DIMS_NUM]= { 0 };
                vx_size * dims = ct_alloc_mem(TENSOR_DIMS_NUM * sizeof(vx_size));
                for(vx_size i = 0; i < TENSOR_DIMS_NUM; i++)
                {
                    dims[i] = TENSOR_DIMS_LENGTH;
                    start[i] = 0;
                    strides[i] = i ? strides[i - 1] * dims[i - 1] : sizeof(vx_uint8);
                }
                const vx_size bytes = dims[TENSOR_DIMS_NUM - 1] * strides[TENSOR_DIMS_NUM - 1];
                void * data = ct_alloc_mem(bytes);
                VX_CALL(vxCopyTensorPatch(output_tensor, TENSOR_DIMS_NUM, start, dims, strides, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                vx_uint8* u8_data = (vx_uint8*)data;
                for(vx_size i = 0; i < bytes; i++)
                {
                    ASSERT(u8_data[i] == 2);
                }
                ct_free_mem(dims);
                ct_free_mem(data);
                break;
            }
        default:
            break;
    }
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT(node == 0);
    ASSERT(graph == 0);
    VX_CALL(vxReleaseReference(&input));
    VX_CALL(vxReleaseReference(&output));
    ASSERT(input == 0);
    ASSERT(output == 0);
}

TEST_WITH_ARG(Copy, testImmediateProcessing, copy_arg, PARAMETERS)
{
    vx_context context = context_->vx_context_;

    vx_reference input = NULL;
    vx_reference output = NULL;
    vx_enum input_type = arg_->item_type;
    vx_enum output_type = VX_TYPE_INVALID;
    vx_int16 gx[CONVOLUTION_X][CONVOLUTION_Y] = {
        { 3, 0, -3 },
        { 10, 0, -10 },
        { 3, 0, -3 },
    };

    ASSERT_VX_OBJECT(input = own_create_exemplar(context, input_type, 0), (enum vx_type_e)input_type);
    ASSERT_VX_OBJECT(output = own_create_exemplar(context, input_type, 1), (enum vx_type_e)input_type);
    switch (input_type)
    {
        case VX_TYPE_IMAGE:
            {
            vx_image input_image = (vx_image)input;
            void *p = NULL;
            vx_map_id input_map_id;
            vx_rectangle_t rect;
            vx_imagepatch_addressing_t addr;
            VX_CALL(vxGetValidRegionImage(input_image, &rect));
            VX_CALL(vxMapImagePatch(input_image, &rect, 0, &input_map_id, &addr, &p, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
            for (vx_size i = 0; i < addr.dim_x*addr.dim_y; i++) {
                vx_uint8 *pPixel = vxFormatImagePatchAddress1d(p, i, &addr);
                *pPixel = i;
            }
            VX_CALL( vxUnmapImagePatch(input_image, input_map_id));
            break;
            }
        case VX_TYPE_ARRAY:
            {
                vx_coordinates2d_t localArrayInit[N];
                vx_array array = (vx_array)input;
                /* Initialization */
                for (int i = 0; i < N; i++) {
                    localArrayInit[i].x = i;
                    localArrayInit[i].y = i;
                }
                VX_CALL( vxAddArrayItems(array, N, &localArrayInit[0], sizeof(vx_coordinates2d_t)) );
                break;
            }
        case VX_TYPE_MATRIX:
            {
                vx_uint8* data = ct_alloc_mem(MATRIX_SIZE_X * MATRIX_SIZE_Y * sizeof(vx_uint8));
                vx_size i;
                for (i = 0; i < MATRIX_SIZE_X * MATRIX_SIZE_Y; i++)
                {
                    data[i] = 1;
                }
                VX_CALL(vxCopyMatrix((vx_matrix)input, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ct_free_mem(data);
                break;
            }
        case VX_TYPE_CONVOLUTION:
            {
                VX_CALL(vxCopyConvolutionCoefficients((vx_convolution)input, gx, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                break;
            }
        case VX_TYPE_OBJECT_ARRAY:
            {
                vx_scalar input_item = NULL;
                vx_uint8  scalar_value=1;
                for (vx_size i = 0; i < OBJECT_ARRAY_COUNT; i++)
                {
                    ASSERT_VX_OBJECT(input_item = (vx_scalar)vxGetObjectArrayItem((vx_object_array)input, i), VX_TYPE_SCALAR);
                    VX_CALL(vxCopyScalar(input_item, &scalar_value, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

                    VX_CALL(vxReleaseReference((vx_reference*)&input_item));
                    ASSERT(input_item == 0);
                }
                break;
            }
        case VX_TYPE_LUT:
            {
                vx_size size = N*sizeof(vx_uint8);
                void* data =  ct_alloc_mem(size);
                vx_uint8* data8 = (vx_uint8*)data;
                for (vx_size i = 0; i < N; ++i)
                    data8[i] = 1;
                VX_CALL(vxCopyLUT((vx_lut)input, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ct_free_mem(data);
                break;
            }
        case VX_TYPE_PYRAMID:
            {
                vx_pyramid input_pyramid = (vx_pyramid)input;
                vx_image input_image = NULL;
                ASSERT_VX_OBJECT(input_image = vxGetPyramidLevel(input_pyramid, 0), VX_TYPE_IMAGE);
                void *p = NULL;
                vx_map_id input_map_id;
                vx_rectangle_t rect;
                vx_imagepatch_addressing_t addr;
                VX_CALL(vxGetValidRegionImage(input_image, &rect));
                VX_CALL(vxMapImagePatch(input_image, &rect, 0, &input_map_id, &addr, &p, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
                for (vx_size i = 0; i < addr.dim_x*addr.dim_y; i++)
                {
                    vx_uint8 *pPixel = vxFormatImagePatchAddress1d(p, i, &addr);
                    *pPixel = 1;
                }
                VX_CALL( vxUnmapImagePatch(input_image, input_map_id));
                VX_CALL(vxReleaseImage(&input_image));
                ASSERT(input_image == 0);
                break;
            }
        case VX_TYPE_THRESHOLD:
            {
                vx_threshold input_threshold = (vx_threshold)input;
                vx_pixel_value_t pa;
                pa.U8 = 8;
                ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCopyThresholdValue(input_threshold, &pa, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                break;
            }
        case VX_TYPE_REMAP:
            {
                vx_remap input_remap = (vx_remap)input;
                vx_rectangle_t rect = { 0, 0,  IMAGE_SIZE_X*2, IMAGE_SIZE_Y*2};
                vx_size stride = IMAGE_SIZE_X*2;
                vx_size stride_y = sizeof(vx_coordinates2df_t) * (stride);
                vx_size size = stride * IMAGE_SIZE_Y*2;
                vx_coordinates2df_t* ptr_w = ct_alloc_mem(sizeof(vx_coordinates2df_t) * size);

                for (vx_size i = 0; i < IMAGE_SIZE_Y*2; i++)
                {
                    for (vx_size j = 0; j < IMAGE_SIZE_X*2; j++)
                    {
                        vx_coordinates2df_t *coord_ptr = &(ptr_w[i * stride + j]);
                        coord_ptr->x = (vx_float32)j;
                        coord_ptr->y = (vx_float32)i;
                    }
                }

                VX_CALL(vxCopyRemapPatch(input_remap, &rect, stride_y, ptr_w, VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ct_free_mem(ptr_w);
                break;
            }
        case VX_TYPE_TENSOR:
            {
                vx_tensor input_tensor = (vx_tensor)input;
                vx_size start[TENSOR_DIMS_NUM] = { 0 };
                vx_size strides[TENSOR_DIMS_NUM]= { 0 };
                vx_size * dims = ct_alloc_mem(TENSOR_DIMS_NUM * sizeof(vx_size));
                for(vx_size i = 0; i < TENSOR_DIMS_NUM; i++)
                {
                    dims[i] = TENSOR_DIMS_LENGTH;
                    start[i] = 0;
                    strides[i] = i ? strides[i - 1] * dims[i - 1] : sizeof(vx_uint8);
                }
                const vx_size bytes = dims[TENSOR_DIMS_NUM - 1] * strides[TENSOR_DIMS_NUM - 1];
                void * data = ct_alloc_mem(bytes);
                vx_uint8* u8_data = (vx_uint8*)data;
                for(vx_size i = 0; i < bytes; i++)
                {
                    u8_data[i] = 2;
                }

                VX_CALL(vxCopyTensorPatch(input_tensor, TENSOR_DIMS_NUM, start, dims, strides, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                ct_free_mem(dims);
                ct_free_mem(data);
                break;
            }
        default:
            break;
    }
    VX_CALL(vxuCopy(context, input, output));
    VX_CALL(vxQueryReference((vx_reference)output, VX_REFERENCE_TYPE, &output_type, sizeof(output_type)));
    ASSERT_EQ_INT(input_type, output_type);

    switch (output_type)
    {
        case VX_TYPE_IMAGE:
            {
            vx_image output_image = (vx_image)output;
            void *p = NULL;
            vx_map_id output_map_id;
            vx_rectangle_t rect;
            vx_imagepatch_addressing_t addr;
            VX_CALL(vxGetValidRegionImage(output_image, &rect));
            VX_CALL(vxMapImagePatch(output_image, &rect, 0, &output_map_id, &addr, &p,
                        VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
            for (vx_uint32 i = 0; i < (addr.dim_x * addr.dim_y); i++) {
                vx_uint8 *pPixel = vxFormatImagePatchAddress1d(p, i, &addr);
                ASSERT(*pPixel == i);
            }

            VX_CALL( vxUnmapImagePatch(output_image, output_map_id));
            break;
            }
        case VX_TYPE_ARRAY:
            {
                vx_array array = (vx_array)output;
                vx_uint8 *p = NULL;
                vx_size stride = 0;
                vx_map_id map_id;
                VX_CALL( vxMapArrayRange(array, N/2, N, &map_id, &stride, (void **)&p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
                ASSERT(stride >=  sizeof(vx_coordinates2d_t));
                ASSERT(p != NULL);

                for (int i = N/2; i<N; i++) {
                    ASSERT(((vx_coordinates2d_t *)(p+stride*(i-N/2)))->x == i);
                    ASSERT(((vx_coordinates2d_t *)(p+stride*(i-N/2)))->y == i);
                }
                VX_CALL( vxUnmapArrayRange (array, map_id));
                break;
            }
        case VX_TYPE_SCALAR:
            {
                vx_uint8  in=2, out=2;
                vx_scalar input_scalar = (vx_scalar)input;
                vx_scalar output_scalar = (vx_scalar)output;
                VX_CALL(vxCopyScalar(input_scalar, &in, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                VX_CALL(vxCopyScalar(output_scalar, &out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                ASSERT_EQ_INT(in, out);
                break;
            }
        case VX_TYPE_MATRIX:
            {
                vx_uint8* data = ct_alloc_mem(MATRIX_SIZE_X * MATRIX_SIZE_Y * sizeof(vx_uint8));
                VX_CALL(vxCopyMatrix((vx_matrix)output, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                vx_size i;
                for (i = 0; i < MATRIX_SIZE_X * MATRIX_SIZE_Y; i++)
                {
                    ASSERT_EQ_INT(data[i], 1);
                }
                ct_free_mem(data);
                break;
            }
        case VX_TYPE_CONVOLUTION:
            {
              vx_int16 *data = (vx_int16 *)ct_alloc_mem(CONVOLUTION_X*CONVOLUTION_Y*sizeof(vx_int16));
              VX_CALL(vxCopyConvolutionCoefficients((vx_convolution)output, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

              for (vx_size i = 0; i < CONVOLUTION_X; i++)
              {
                  for (vx_size j = 0; j < CONVOLUTION_Y; j++)
                  {
                      ASSERT(gx[i][j] == data[i * CONVOLUTION_X + j]);
                  }
              }

              ct_free_mem(data);
              break;
            }
        case VX_TYPE_OBJECT_ARRAY:
            {
                vx_scalar output_item = NULL;
                vx_uint8  input_value=1, output_value = 0;
                for (vx_size i = 0; i < OBJECT_ARRAY_COUNT; i++)
                {
                    ASSERT_VX_OBJECT(output_item = (vx_scalar)vxGetObjectArrayItem((vx_object_array)output, i), VX_TYPE_SCALAR);
                    VX_CALL(vxCopyScalar(output_item, &output_value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                    ASSERT_EQ_INT(output_value, input_value);
                    VX_CALL(vxReleaseReference((vx_reference*)&output_item));
                    ASSERT(output_item == 0);
                }
                break;
            }
        case VX_TYPE_LUT:
            {
                vx_map_id map_id;
                void* lut_data = NULL;
                VX_CALL(vxMapLUT((vx_lut)output, &map_id, &lut_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                vx_uint8* data8 = (vx_uint8*)lut_data;
                for (vx_size i = 0; i < N; i++)
                {
                    ASSERT_EQ_INT(data8[i], 1);
                }
                VX_CALL(vxUnmapLUT((vx_lut)output, map_id));
                break;
            }
        case VX_TYPE_PYRAMID:
            {
                vx_pyramid output_pyramid = (vx_pyramid)output;
                vx_image output_image = NULL;
                ASSERT_VX_OBJECT(output_image = vxGetPyramidLevel(output_pyramid, 0), VX_TYPE_IMAGE);
                void *p = NULL;
                vx_map_id map_id;
                vx_rectangle_t rect;
                vx_imagepatch_addressing_t addr;
                VX_CALL(vxGetValidRegionImage(output_image, &rect));
                VX_CALL(vxMapImagePatch(output_image, &rect, 0, &map_id, &addr, &p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                for (vx_size i = 0; i < addr.dim_x*addr.dim_y; i++)
                {
                    vx_uint8 *pPixel = vxFormatImagePatchAddress1d(p, i, &addr);
                    ASSERT_EQ_INT(*pPixel, 1);
                }
                VX_CALL( vxUnmapImagePatch(output_image, map_id));
                VX_CALL(vxReleaseImage(&output_image));
                ASSERT(output_image == 0);
                break;
            }
        case VX_TYPE_THRESHOLD:
            {
                vx_threshold output_threshold = (vx_threshold)output;
                vx_pixel_value_t pa;
                ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCopyThresholdValue(output_threshold, &pa, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                ASSERT(pa.U8 == 8);
                break;
            }
        case VX_TYPE_REMAP:
            {
                vx_remap output_remap = (vx_remap)output;
                vx_rectangle_t rect = { 0, 0,  IMAGE_SIZE_X*2, IMAGE_SIZE_Y*2};
                vx_size stride = IMAGE_SIZE_X*2;
                vx_size stride_y = 0;
                vx_coordinates2df_t *ptr_r = 0;
                vx_map_id map_id;

                VX_CALL(vxMapRemapPatch(output_remap, &rect, &map_id, &stride_y, (void **)&ptr_r, VX_TYPE_COORDINATES2DF, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

                for (vx_size i = 0; i < IMAGE_SIZE_Y*2; i++)
                {
                    for (vx_size j = 0; j < IMAGE_SIZE_X*2; j++)
                    {
                        vx_coordinates2df_t *coord_ptr = &(ptr_r[i * stride + j]);
                        ASSERT_EQ_INT((vx_uint32)coord_ptr->x, j);
                        ASSERT_EQ_INT((vx_uint32)coord_ptr->y, i);
                    }
                }

                VX_CALL(vxUnmapRemapPatch(output_remap, map_id));
                break;
            }
        case VX_TYPE_TENSOR:
            {
                vx_tensor output_tensor = (vx_tensor)output;
                vx_size start[TENSOR_DIMS_NUM] = { 0 };
                vx_size strides[TENSOR_DIMS_NUM]= { 0 };
                vx_size * dims = ct_alloc_mem(TENSOR_DIMS_NUM * sizeof(vx_size));
                for(vx_size i = 0; i < TENSOR_DIMS_NUM; i++)
                {
                    dims[i] = TENSOR_DIMS_LENGTH;
                    start[i] = 0;
                    strides[i] = i ? strides[i - 1] * dims[i - 1] : sizeof(vx_uint8);
                }
                const vx_size bytes = dims[TENSOR_DIMS_NUM - 1] * strides[TENSOR_DIMS_NUM - 1];
                void * data = ct_alloc_mem(bytes);
                VX_CALL(vxCopyTensorPatch(output_tensor, TENSOR_DIMS_NUM, start, dims, strides, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
                vx_uint8* u8_data = (vx_uint8*)data;
                for(vx_size i = 0; i < bytes; i++)
                {
                    ASSERT(u8_data[i] == 2);
                }
                ct_free_mem(dims);
                ct_free_mem(data);
                break;
            }
        default:
            break;
    }
    VX_CALL(vxReleaseReference(&input));
    VX_CALL(vxReleaseReference(&output));
    ASSERT(input == 0);
    ASSERT(output == 0);
}
TESTCASE_TESTS(Copy,
    testNodeCreation,
    testGraphProcessing,
    testImmediateProcessing
)

#endif //OPENVX_USE_ENHANCED_VISION
