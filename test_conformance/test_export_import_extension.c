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

#ifdef OPENVX_USE_IX

#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_khr_ix.h>

#include "test_engine/test.h"

#define TEST_TENSOR_MIN_DIM_SZ                  1
#define TEST_TENSOR_MAX_DIM_SZ                  20

TESTCASE(ExtensionObject, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char* name;
    vx_enum     type;
} format_arg;

#define EXPORT_IMPORT_TEST_CASE(tp) \
    {#tp, tp}

static vx_size lut_count(vx_enum data_type)
{
    vx_size count = 0;

    switch (data_type)
    {
    case VX_TYPE_UINT8:
        count = 256;
        break;
    case VX_TYPE_INT16:
        count = 65536;
        break;
    }

    return count;
}

TEST_WITH_ARG(ExtensionObject, testExtensionObject, format_arg,
        EXPORT_IMPORT_TEST_CASE(VX_IX_USE_EXPORT_VALUES),
        EXPORT_IMPORT_TEST_CASE(VX_IX_USE_NO_EXPORT_VALUES),
        EXPORT_IMPORT_TEST_CASE(VX_IX_USE_APPLICATION_CREATE),
        )
{
    vx_context context = context_->vx_context_;
    vx_reference reference_list[13];
    vx_reference import_list[13]={NULL};
    vx_enum uses_list[] = {
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        VX_IX_USE_EXPORT_VALUES
    };

    vx_enum uses_import_list[] = {
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        arg_->type,
        VX_IX_USE_EXPORT_VALUES
    };

    const vx_uint8 *blob_ptr = NULL;
    vx_size blob_bytes = 0;

    vx_image image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8);
    reference_list[0] = (vx_reference ) image;

    vx_size count = lut_count(VX_TYPE_UINT8);
    vx_lut lut = vxCreateLUT(context, VX_TYPE_UINT8, count);
    const char* lut_name= "Lut";
    VX_CALL(vxSetReferenceName((vx_reference)lut, lut_name));
    reference_list[1] = (vx_reference ) lut;

    vx_distribution distribution = vxCreateDistribution(context, 32, 0, 255);
    reference_list[2] = (vx_reference ) distribution;

    vx_threshold threshold = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    reference_list[3] = (vx_reference ) threshold;

    vx_matrix matrix = vxCreateMatrix(context, VX_TYPE_UINT8, 3, 5);
    reference_list[4] = (vx_reference ) matrix;

    vx_convolution conv = vxCreateConvolution(context, 3, 3);
    reference_list[5] = (vx_reference ) conv;

    vx_char ref = 1;
    vx_scalar scalar = vxCreateScalar(context, VX_TYPE_CHAR, &ref);
    reference_list[6] = (vx_reference ) scalar;

    vx_array array = vxCreateArray(context, VX_TYPE_COORDINATES2D, 10);
    reference_list[7] = (vx_reference ) array;

    vx_remap map = vxCreateRemap(context, 16, 32, 128, 64);
    reference_list[8] = (vx_reference ) map;

    vx_pyramid pyramid = vxCreatePyramid(context, 2, VX_SCALE_PYRAMID_HALF,  320, 240, VX_DF_IMAGE_S16);
    reference_list[9] = (vx_reference ) pyramid;

    vx_image imagedelay = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8);
    vx_delay delay = vxCreateDelay(context, (vx_reference)imagedelay, 2);
    reference_list[10] = (vx_reference ) delay;

    uint64_t rng;
    {
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    size_t in_dims[4];
    in_dims[0] = 5;
    in_dims[1] = 3;
    in_dims[2] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);
    in_dims[3] = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ+1);

    vx_uint8 fixed_point_position = 0;
    vx_tensor tensor = vxCreateTensor(context, 4, in_dims, VX_TYPE_UINT8, fixed_point_position);
    ASSERT_VX_OBJECT(tensor, VX_TYPE_TENSOR);
    reference_list[11] = (vx_reference ) tensor;

    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);
    vx_graph graph = vxCreateGraph(context);
    vxGaussian3x3Node(graph, src_image, interm_image);
    vxIntegralImageNode(graph, interm_image, dst_image);
    VX_CALL(vxVerifyGraph(graph));
    reference_list[12] = (vx_reference ) graph;

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxExportObjectsToMemory( context, 13, &reference_list[0], &uses_list[0], &blob_ptr, &blob_bytes));
    void * export_blob = ct_alloc_mem(blob_bytes);
    memcpy(export_blob, blob_ptr, blob_bytes);

    VX_CALL(vxReleaseImage(&image));
    VX_CALL(vxReleaseLUT(&lut));
    VX_CALL(vxReleaseDistribution(&distribution));
    VX_CALL(vxReleaseThreshold(&threshold));
    VX_CALL(vxReleaseMatrix(&matrix));
    VX_CALL(vxReleaseConvolution(&conv));
    VX_CALL(vxReleaseScalar(&scalar));
    VX_CALL(vxReleaseArray(&array));
    VX_CALL(vxReleaseRemap(&map));
    VX_CALL(vxReleasePyramid(&pyramid));
    VX_CALL(vxReleaseDelay(&delay));
    VX_CALL(vxReleaseImage(&imagedelay));
    VX_CALL(vxReleaseTensor(&tensor));
    VX_CALL(vxReleaseImage(&src_image));
    VX_CALL(vxReleaseImage(&interm_image));
    VX_CALL(vxReleaseImage(&dst_image));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseExportedMemory(context, &blob_ptr));
    //VX_CALL(vxReleaseContext(&context));

    vx_context context1 = vxCreateContext();
    vx_image image1 = vxCreateImage(context1, 128, 128, VX_DF_IMAGE_U8);
    import_list[0] = (vx_reference ) image1;
    vx_lut lut1 = vxCreateLUT(context1, VX_TYPE_UINT8, count);
    VX_CALL(vxSetReferenceName((vx_reference)lut1, lut_name));
    import_list[1] = (vx_reference ) lut1;
    vx_distribution distribution1 = vxCreateDistribution(context1, 32, 0, 255);
    import_list[2] = (vx_reference ) distribution1;
    vx_threshold threshold1 = vxCreateThresholdForImage(context1, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    import_list[3] = (vx_reference ) threshold1;
    vx_matrix matrix1 = vxCreateMatrix(context1, VX_TYPE_UINT8, 3, 5);
    import_list[4] = (vx_reference ) matrix1;
    vx_convolution conv1 = vxCreateConvolution(context1, 3, 3);
    import_list[5] = (vx_reference ) conv1;
    vx_scalar scalar1 = vxCreateScalar(context1, VX_TYPE_CHAR, &ref);
    import_list[6] = (vx_reference ) scalar1;
    vx_array array1 = vxCreateArray(context1, VX_TYPE_COORDINATES2D, 10);
    import_list[7] = (vx_reference ) array1;
    vx_remap map1 = vxCreateRemap(context1, 16, 32, 128, 64);
    import_list[8] = (vx_reference ) map1;
    vx_pyramid pyramid1 = vxCreatePyramid(context1, 2, VX_SCALE_PYRAMID_HALF,  320, 240, VX_DF_IMAGE_S16);
    import_list[9] = (vx_reference ) pyramid1;
    vx_image imagedelay1 = vxCreateImage(context1, 128, 128, VX_DF_IMAGE_U8);
    vx_delay delay1 = vxCreateDelay(context1, (vx_reference)imagedelay1, 2);
    import_list[10] = (vx_reference ) delay1;
    vx_tensor tensor1 = vxCreateTensor(context1, 4, in_dims, VX_TYPE_UINT8, fixed_point_position);
    ASSERT_VX_OBJECT(tensor1, VX_TYPE_TENSOR);
    import_list[11] = (vx_reference ) tensor1;

    vx_import import = vxImportObjectsFromMemory ( context1, 13, &import_list[0], &uses_import_list[0], export_blob, blob_bytes );
    ASSERT_VX_OBJECT(import, VX_TYPE_IMPORT);
    ASSERT_VX_OBJECT((vx_lut)import_list[2], VX_TYPE_DISTRIBUTION);

    vx_reference name_reference = vxGetImportReferenceByName(import, lut_name);

    ASSERT_VX_OBJECT((vx_lut)name_reference, VX_TYPE_LUT);
    ASSERT(vxGetStatus(name_reference) == VX_SUCCESS);


    VX_CALL(vxReleaseImage(&image1));
    VX_CALL(vxReleaseLUT(&lut1));
    VX_CALL(vxReleaseDistribution(&distribution1));
    VX_CALL(vxReleaseThreshold(&threshold1));
    VX_CALL(vxReleaseMatrix(&matrix1));
    VX_CALL(vxReleaseConvolution(&conv1));
    VX_CALL(vxReleaseScalar(&scalar1));
    VX_CALL(vxReleaseArray(&array1));
    VX_CALL(vxReleaseRemap(&map1));
    VX_CALL(vxReleasePyramid(&pyramid1));
    VX_CALL(vxReleaseImage(&imagedelay1));
    VX_CALL(vxReleaseDelay(&delay1));
    VX_CALL(vxReleaseTensor(&tensor1));


    if(arg_->type != VX_IX_USE_APPLICATION_CREATE)
    {
        vxReleaseReference(&import_list[0]);
        vxReleaseReference(&import_list[1]);
        vxReleaseReference(&import_list[2]);
        vxReleaseReference(&import_list[3]);
        vxReleaseReference(&import_list[4]);
        vxReleaseReference(&import_list[5]);
        vxReleaseReference(&import_list[6]);
        vxReleaseReference(&import_list[7]);
        vxReleaseReference(&import_list[8]);
        vxReleaseReference(&import_list[9]);
        vxReleaseReference(&import_list[10]);
        vxReleaseReference(&import_list[11]);
    }
    vxReleaseReference(&import_list[12]);

    vxReleaseReference(&name_reference);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS,vxReleaseImport(&import));
    ct_free_mem(export_blob);
}

TESTCASE_TESTS(ExtensionObject, testExtensionObject)

#endif
