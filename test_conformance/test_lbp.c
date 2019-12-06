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
#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"

TESTCASE(LBP, CT_VXContext, ct_setup_vx_context, 0)

TEST(LBP, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0, output = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxLBPNode(graph, input, VX_LBP, 3, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&output));
    VX_CALL(vxReleaseImage(&input));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(output == 0);
    ASSERT(input == 0);
}


static CT_Image lbp_read_image(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}

typedef struct {
    const char* filename;
    CT_Image(*generator)(const char* fileName, int width, int height);
    vx_enum format;
    vx_int8 kernel_size;
    const char* golden_image;
} Arg;

#define PARAMETERS \
    ARG("lbp_3x3_standard.bmp", lbp_read_image, (vx_enum)VX_LBP, (vx_int8)3, "lbp_3x3_standard_golden.bmp"),\
    ARG("lbp_5x5_standard.bmp", lbp_read_image, (vx_enum)VX_LBP, (vx_int8)5, "lbp_5x5_standard_golden.bmp"),\
    ARG("lbp_5x5_modified.bmp", lbp_read_image, (vx_enum)VX_MLBP, (vx_int8)5, "lbp_5x5_modified_golden.bmp"),\
    ARG("lbp_3x3_uniform.bmp", lbp_read_image, (vx_enum)VX_ULBP, (vx_int8)3, "lbp_3x3_uniform_golden.bmp"),\
    ARG("lbp_5x5_uniform.bmp", lbp_read_image, (vx_enum)VX_ULBP, (vx_int8)5, "lbp_5x5_uniform_golden.bmp"),\


TEST_WITH_ARG(LBP, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_image input_image = 0, output_image = 0;
    CT_Image input = NULL, output = NULL;
    vx_uint32 border = arg_->kernel_size / 2;


    ASSERT_NO_FAILURE(input = arg_->generator(arg_->filename, 0, 0));
    ASSERT_NO_FAILURE(output = ct_allocate_image(input->width, input->height, VX_DF_IMAGE_U8));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxLBPNode(graph, input_image, arg_->format, arg_->kernel_size, output_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));

    //bit match begin
    CT_Image golden_image = lbp_read_image(arg_->golden_image, 0, 0);
    ct_adjust_roi(output, border, border, border, border);
    ct_adjust_roi(golden_image, border, border, border, border);
    EXPECT_EQ_CTIMAGE(golden_image, output);
    //bit match end

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseImage(&output_image));
    VX_CALL(vxReleaseImage(&input_image));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

TEST_WITH_ARG(LBP, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, output_image = 0;
    CT_Image input = NULL, output = NULL;
    vx_uint32 border = arg_->kernel_size / 2;

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->filename, 0, 0));
    ASSERT_NO_FAILURE(output = ct_allocate_image(input->width, input->height, VX_DF_IMAGE_U8));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);

    VX_CALL(vxuLBP(context, input_image, arg_->format, arg_->kernel_size, output_image));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));

    //bit match begin
    CT_Image golden_image = lbp_read_image(arg_->golden_image, 0, 0);
    ct_adjust_roi(output, border, border, border, border);
    ct_adjust_roi(golden_image, border, border, border, border);
    EXPECT_EQ_CTIMAGE(golden_image, output);
    //bit match end

    VX_CALL(vxReleaseImage(&output_image));
    VX_CALL(vxReleaseImage(&input_image));

    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

TESTCASE_TESTS(LBP, testNodeCreation, testGraphProcessing, testImmediateProcessing)

#endif //OPENVX_USE_ENHANCED_VISION
