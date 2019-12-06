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

#include "test_engine/test.h"
#include <VX/vx.h>
#include <VX/vxu.h>

typedef struct
{
    const char* name;
    vx_enum type;
} method_type_arg;
TESTCASE(MatchTemplate, CT_VXContext, ct_setup_vx_context, 0)

TEST(MatchTemplate, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_image vx_template_image, vx_source_image, vx_result_image;
    CT_Image ct_template_image, ct_source_image;
    ASSERT_NO_FAILURE(ct_template_image = ct_read_image("lena_gray_template.bmp", 1));
    ASSERT_NO_FAILURE(vx_template_image = ct_image_to_vx_image(ct_template_image, context));
    ASSERT_NO_FAILURE(ct_source_image = ct_read_image("lena_gray_source.bmp", 1));
    ASSERT_NO_FAILURE(vx_source_image = ct_image_to_vx_image(ct_source_image, context));

    vx_uint32 source_width;
    vx_uint32 source_height;
    vxQueryImage(vx_source_image, VX_IMAGE_WIDTH, &source_width, sizeof(vx_uint32));
    vxQueryImage(vx_source_image, VX_IMAGE_HEIGHT, &source_height, sizeof(vx_uint32));

    ASSERT_VX_OBJECT(vx_result_image = vxCreateImage(context, source_width, source_height, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxMatchTemplateNode(graph, vx_source_image, vx_template_image, VX_COMPARE_HAMMING, vx_result_image), VX_TYPE_NODE);

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&vx_template_image));
    VX_CALL(vxReleaseImage(&vx_source_image));
    VX_CALL(vxReleaseImage(&vx_result_image));

    ASSERT(vx_template_image == 0);
    ASSERT(vx_source_image == 0);
    ASSERT(vx_result_image == 0);
}

TEST_WITH_ARG(MatchTemplate, testGraphProcessing, method_type_arg,
        ARG_ENUM(VX_COMPARE_HAMMING),
        ARG_ENUM(VX_COMPARE_L1),
        ARG_ENUM(VX_COMPARE_L2),
        //ARG_ENUM(VX_COMPARE_CCORR),
        ARG_ENUM(VX_COMPARE_L2_NORM),
        ARG_ENUM(VX_COMPARE_CCORR_NORM))
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_image vx_template_image, vx_source_image, vx_result_image;
    CT_Image ct_template_image, ct_source_image;
    ASSERT_NO_FAILURE(ct_template_image = ct_read_image("lena_gray_template.bmp", 1));
    ASSERT_NO_FAILURE(vx_template_image = ct_image_to_vx_image(ct_template_image, context));
    ASSERT_NO_FAILURE(ct_source_image = ct_read_image("lena_gray_source.bmp", 1));
    ASSERT_NO_FAILURE(vx_source_image = ct_image_to_vx_image(ct_source_image, context));

    vx_uint32 template_width, template_height, source_width, source_height, result_width, result_height;
    vxQueryImage(vx_source_image, VX_IMAGE_WIDTH, &source_width, sizeof(vx_uint32));
    vxQueryImage(vx_source_image, VX_IMAGE_HEIGHT, &source_height, sizeof(vx_uint32));
    vxQueryImage(vx_template_image, VX_IMAGE_WIDTH, &template_width, sizeof(vx_uint32));
    vxQueryImage(vx_template_image, VX_IMAGE_HEIGHT, &template_height, sizeof(vx_uint32));
    result_width = source_width - template_width + 1;
    result_height = source_height - template_height + 1;
    ASSERT_VX_OBJECT(vx_result_image = vxCreateImage(context, result_width, result_height, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxMatchTemplateNode(graph, vx_source_image, vx_template_image, arg_->type, vx_result_image), VX_TYPE_NODE);
    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    uint32_t mincount = 0, maxcount = 0;
    vx_scalar minval_, maxval_, mincount_, maxcount_;
    vx_array minloc_ = 0, maxloc_ = 0;
    vx_enum sctype = VX_TYPE_INT16;
    minval_ = ct_scalar_from_int(context, sctype, 0);
    maxval_ = ct_scalar_from_int(context, sctype, 0);
    mincount_ = ct_scalar_from_int(context, VX_TYPE_UINT32, 0);
    maxcount_ = ct_scalar_from_int(context, VX_TYPE_UINT32, 0);
    minloc_ = vxCreateArray(context, VX_TYPE_COORDINATES2D, 300);
    maxloc_ = vxCreateArray(context, VX_TYPE_COORDINATES2D, 300);
    node = vxMinMaxLocNode(graph, vx_result_image, minval_, maxval_,
                                   minloc_,
                                   maxloc_,
                                   mincount_,
                                   maxcount_);
    ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));
    vxCopyScalar(mincount_, &mincount, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(maxcount_, &maxcount, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vx_array resultloc = minloc_;
    if(arg_->type == VX_COMPARE_CCORR || arg_->type == VX_COMPARE_CCORR_NORM)
    {
        resultloc = maxloc_;
    }
    vx_size stride = 0;
    vx_coordinates2d_t *p = NULL;
    vx_map_id map_id;
    vx_coordinates2d_t result = {0,0};
    VX_CALL(vxMapArrayRange(resultloc, 0, 1, &map_id, &stride, (void **)&p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
    result = p[0];
    VX_CALL(vxUnmapArrayRange(resultloc, map_id));

    //The correct match point is (24, 24), and error range is +/- 1 pixel.
    ASSERT(result.x > 24 - 2 && result.x < 24 + 2);
    ASSERT(result.y > 24 - 2 && result.y < 24 + 2);

    VX_CALL(vxReleaseScalar(&minval_));
    VX_CALL(vxReleaseScalar(&maxval_));
    VX_CALL(vxReleaseScalar(&mincount_));
    VX_CALL(vxReleaseScalar(&maxcount_));
    VX_CALL(vxReleaseArray(&minloc_));
    VX_CALL(vxReleaseArray(&maxloc_));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&vx_template_image));
    VX_CALL(vxReleaseImage(&vx_source_image));
    VX_CALL(vxReleaseImage(&vx_result_image));

    ASSERT(vx_template_image == 0);
    ASSERT(vx_source_image == 0);
    ASSERT(vx_result_image == 0);
}
TEST_WITH_ARG(MatchTemplate,  testImmediateProcessing, method_type_arg,
        ARG_ENUM(VX_COMPARE_HAMMING),
        ARG_ENUM(VX_COMPARE_L1),
        ARG_ENUM(VX_COMPARE_L2),
        //ARG_ENUM(VX_COMPARE_CCORR),
        ARG_ENUM(VX_COMPARE_L2_NORM),
        ARG_ENUM(VX_COMPARE_CCORR_NORM))
{
    vx_context context = context_->vx_context_;
    vx_image vx_template_image, vx_source_image, vx_result_image;
    CT_Image ct_template_image, ct_source_image;
    ASSERT_NO_FAILURE(ct_template_image = ct_read_image("lena_gray_template.bmp", 1));
    ASSERT_NO_FAILURE(vx_template_image = ct_image_to_vx_image(ct_template_image, context));
    ASSERT_NO_FAILURE(ct_source_image = ct_read_image("lena_gray_source.bmp", 1));
    ASSERT_NO_FAILURE(vx_source_image = ct_image_to_vx_image(ct_source_image, context));

    vx_uint32 template_width, template_height, source_width, source_height, result_width, result_height;
    vxQueryImage(vx_source_image, VX_IMAGE_WIDTH, &source_width, sizeof(vx_uint32));
    vxQueryImage(vx_source_image, VX_IMAGE_HEIGHT, &source_height, sizeof(vx_uint32));
    vxQueryImage(vx_template_image, VX_IMAGE_WIDTH, &template_width, sizeof(vx_uint32));
    vxQueryImage(vx_template_image, VX_IMAGE_HEIGHT, &template_height, sizeof(vx_uint32));
    result_width = source_width - template_width + 1;
    result_height = source_height - template_height + 1;
    ASSERT_VX_OBJECT(vx_result_image = vxCreateImage(context, result_width, result_height, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);

    VX_CALL(vxuMatchTemplate(context, vx_source_image, vx_template_image, arg_->type, vx_result_image));

    uint32_t mincount = 0, maxcount = 0;
    vx_scalar minval_, maxval_, mincount_, maxcount_;
    vx_array minloc_ = 0, maxloc_ = 0;
    vx_enum sctype = VX_TYPE_INT16;
    minval_ = ct_scalar_from_int(context, sctype, 0);
    maxval_ = ct_scalar_from_int(context, sctype, 0);
    mincount_ = ct_scalar_from_int(context, VX_TYPE_UINT32, 0);
    maxcount_ = ct_scalar_from_int(context, VX_TYPE_UINT32, 0);
    minloc_ = vxCreateArray(context, VX_TYPE_COORDINATES2D, 300);
    maxloc_ = vxCreateArray(context, VX_TYPE_COORDINATES2D, 300);
    VX_CALL(vxuMinMaxLoc(context, vx_result_image, minval_, maxval_,
                                   minloc_,
                                   maxloc_,
                                   mincount_,
                                   maxcount_));
    vxCopyScalar(mincount_, &mincount, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(maxcount_, &maxcount, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vx_array resultloc = minloc_;
    if(arg_->type == VX_COMPARE_CCORR || arg_->type == VX_COMPARE_CCORR_NORM)
    {
        resultloc = maxloc_;
    }
    vx_size stride = 0;
    vx_coordinates2d_t *p = NULL;
    vx_map_id map_id;
    vx_coordinates2d_t result = {0,0};
    VX_CALL(vxMapArrayRange(resultloc, 0, 1, &map_id, &stride, (void **)&p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
    result = p[0];
    VX_CALL(vxUnmapArrayRange(resultloc, map_id));

    //The correct match point is (24, 24), and error range is +/- 1 pixel.
    ASSERT(result.x > 24 - 2 && result.x < 24 + 2);
    ASSERT(result.y > 24 - 2 && result.y < 24 + 2);

    VX_CALL(vxReleaseScalar(&minval_));
    VX_CALL(vxReleaseScalar(&maxval_));
    VX_CALL(vxReleaseScalar(&mincount_));
    VX_CALL(vxReleaseScalar(&maxcount_));
    VX_CALL(vxReleaseArray(&minloc_));
    VX_CALL(vxReleaseArray(&maxloc_));
    VX_CALL(vxReleaseImage(&vx_template_image));
    VX_CALL(vxReleaseImage(&vx_source_image));
    VX_CALL(vxReleaseImage(&vx_result_image));

    ASSERT(vx_template_image == 0);
    ASSERT(vx_source_image == 0);
    ASSERT(vx_result_image == 0);}

TESTCASE_TESTS(MatchTemplate, testNodeCreation, testGraphProcessing, testImmediateProcessing)

#endif //OPENVX_USE_ENHANCED_VISION
