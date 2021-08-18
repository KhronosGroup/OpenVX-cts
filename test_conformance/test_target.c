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


#include "test_engine/test.h"
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_kernels.h>

typedef struct
{
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height);
    vx_enum target_enum;
    const char* target_string;

} SetTarget_Arg;


#define ADD_SET_TARGET_PARAMETERS(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_TARGET_ANY", __VA_ARGS__, VX_TARGET_ANY, NULL)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TARGET_STRING=any", __VA_ARGS__, VX_TARGET_STRING, "any")), \
    CT_EXPAND(nextmacro(testArgName "/VX_TARGET_STRING=aNy", __VA_ARGS__, VX_TARGET_STRING, "aNy")), \
    CT_EXPAND(nextmacro(testArgName "/VX_TARGET_STRING=ANY", __VA_ARGS__, VX_TARGET_STRING, "ANY"))

#define SET_NODE_TARGET_PARAMETERS \
    CT_GENERATE_PARAMETERS("target", ADD_SET_TARGET_PARAMETERS, ARG, NULL)

#define SET_IMM_MODE_TARGET_PARAMETERS \
    CT_GENERATE_PARAMETERS("target", ADD_SET_TARGET_PARAMETERS, ARG, NULL)



#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION

TESTCASE(Target, CT_VXContext, ct_setup_vx_context, 0)

TEST_WITH_ARG(Target, testvxSetNodeTarget, SetTarget_Arg, SET_NODE_TARGET_PARAMETERS)
{
    vx_context context = context_->vx_context_;

    vx_image src1 = 0;
    vx_image src2 = 0;
    vx_image dst = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(src1 = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2 = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst  = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxAddNode(graph, src1, src2, VX_CONVERT_POLICY_WRAP, dst), VX_TYPE_NODE);

    VX_CALL(vxSetNodeTarget(node, arg_->target_enum, arg_->target_string));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));

    VX_CALL(vxReleaseImage(&src1));
    VX_CALL(vxReleaseImage(&src2));
    VX_CALL(vxReleaseImage(&dst));

    return;
}


TEST_WITH_ARG(Target, testvxSetImmediateModeTarget, SetTarget_Arg, SET_IMM_MODE_TARGET_PARAMETERS)
{
    vx_context context = context_->vx_context_;

    vx_image src1 = 0;
    vx_image src2 = 0;
    vx_image dst = 0;

    ASSERT_VX_OBJECT(src1 = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2 = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst  = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    VX_CALL(vxSetImmediateModeTarget(context, arg_->target_enum, arg_->target_string));

    VX_CALL(vxuAdd(context, src1, src2, VX_CONVERT_POLICY_WRAP, dst));

    VX_CALL(vxReleaseImage(&src1));
    VX_CALL(vxReleaseImage(&src2));
    VX_CALL(vxReleaseImage(&dst));

    return;
}

#endif

TESTCASE(TargetBase, CT_VXContext, ct_setup_vx_context, 0)

TEST(TargetBase, testvxCreateContext)
{
    vx_context context = vxCreateContext();
    ASSERT_VX_OBJECT(context, VX_TYPE_CONTEXT);
    vxReleaseContext(&context);
}

TEST(TargetBase, testvxQueryContext)
{
    vx_context context = context_->vx_context_;
    vx_status status = VX_SUCCESS;
    vx_uint32 num_refs1 = 0;
    char * test = (char*)ct_alloc_mem(VX_MAX_IMPLEMENTATION_NAME);

    vx_context context_test = NULL;
    status = vxQueryContext(context_test, VX_CONTEXT_REFERENCES, (void*)&num_refs1, sizeof(num_refs1));
    ASSERT_EQ_INT(VX_ERROR_INVALID_REFERENCE, status);

    status = vxQueryContext(context, VX_CONTEXT_VENDOR_ID, test, sizeof(vx_uint16));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_VENDOR_ID, test, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_VERSION, test, sizeof(vx_uint16));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_VERSION, test, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_MODULES, (void*)&num_refs1, sizeof(num_refs1));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_MODULES, (void*)&num_refs1, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_REFERENCES, (void*)&num_refs1, sizeof(num_refs1));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_REFERENCES, (void*)&num_refs1, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_IMPLEMENTATION, (void*)test, VX_MAX_IMPLEMENTATION_NAME);
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_IMPLEMENTATION, (void*)&test, VX_MAX_IMPLEMENTATION_NAME + 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_EXTENSIONS_SIZE, test, sizeof(vx_size));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_EXTENSIONS_SIZE, test, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_EXTENSIONS, test, 2);
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_EXTENSIONS, NULL, 2);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_CONVOLUTION_MAX_DIMENSION, test, sizeof(vx_size));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_CONVOLUTION_MAX_DIMENSION, test, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_NONLINEAR_MAX_DIMENSION, test, sizeof(vx_size));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_NONLINEAR_MAX_DIMENSION, test, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_OPTICAL_FLOW_MAX_WINDOW_DIMENSION, test, sizeof(vx_size));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_OPTICAL_FLOW_MAX_WINDOW_DIMENSION, test, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_IMMEDIATE_BORDER, test, sizeof(vx_border_t));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_IMMEDIATE_BORDER, NULL, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_IMMEDIATE_BORDER_POLICY, test, sizeof(vx_enum));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_IMMEDIATE_BORDER_POLICY, NULL, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_UNIQUE_KERNELS, (void*)&num_refs1, sizeof(num_refs1));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_UNIQUE_KERNELS, NULL, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, test, sizeof(vx_size));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, NULL, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxQueryContext(context, VX_CONTEXT_UNIQUE_KERNEL_TABLE, NULL, 1);
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);
    status = vxQueryContext(context, VX_ERROR_INVALID_TYPE, (void*)&num_refs1, sizeof(num_refs1));
    ASSERT_EQ_INT(VX_ERROR_NOT_SUPPORTED, status);

    ct_free_mem(test);
}

TEST(TargetBase, testvxReleaseContext)
{
    vx_status status = VX_SUCCESS;
    vx_context context_test = 0;

    status = vxReleaseContext(&context_test);
    ASSERT_EQ_INT(VX_ERROR_INVALID_REFERENCE, status);

    context_test = vxCreateContext();
    ASSERT_VX_OBJECT(context_test, VX_TYPE_CONTEXT);
    vx_border_t ptr;
    ptr.mode = VX_BORDER_CONSTANT;
    status = vxSetContextAttribute(context_test, VX_CONTEXT_IMMEDIATE_BORDER, &ptr, sizeof(vx_border_t));
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxReleaseContext(&context_test);
    ASSERT_EQ_INT(VX_SUCCESS, status);
}

TEST(TargetBase, testvxSetContextAttribute)
{
    vx_context context = context_->vx_context_;
    vx_status status = VX_SUCCESS;

    vx_context context_test = 0;
    status = vxSetContextAttribute(context_test, VX_CONTEXT_IMMEDIATE_BORDER, NULL, 0);
    ASSERT_EQ_INT(VX_ERROR_INVALID_REFERENCE, status);

    status = vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, NULL, sizeof(int));
    ASSERT_EQ_INT(VX_ERROR_INVALID_PARAMETERS, status);

    status = vxSetContextAttribute(context, VX_CONTEXT_EXTENSIONS, NULL, sizeof(vx_border_t));
    ASSERT_EQ_INT(VX_ERROR_NOT_SUPPORTED, status);

    vx_border_t ptr = { 0 };
    ptr.mode = VX_BORDER_POLICY_DEFAULT_TO_UNDEFINED;
    status = vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &ptr, sizeof(vx_border_t));
    ASSERT_EQ_INT(VX_ERROR_INVALID_VALUE, status);

    ptr.mode = VX_BORDER_CONSTANT;
    status = vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &ptr, sizeof(vx_border_t));
    ASSERT_EQ_INT(VX_SUCCESS, status);
}

TEST_WITH_ARG(TargetBase, testvxSetImmediateModeTargetBase, SetTarget_Arg, SET_IMM_MODE_TARGET_PARAMETERS)
{
    vx_context context = context_->vx_context_;
    vx_status status = VX_SUCCESS;
    char * string = "test";

    vx_context context_test = 0;
    status = vxSetImmediateModeTarget(context_test, arg_->target_enum, arg_->target_string);
    ASSERT_EQ_INT(VX_ERROR_INVALID_REFERENCE, status);

    status = vxSetImmediateModeTarget(context, arg_->target_enum, arg_->target_string);
    ASSERT_EQ_INT(VX_SUCCESS, status);
    status = vxSetImmediateModeTarget(context, VX_TARGET_STRING, string);
    ASSERT_EQ_INT(VX_ERROR_NOT_SUPPORTED, status);
}

TEST(TargetBase, testvxSetNodeTargetBase)
{
    vx_node node = 0;
    vx_status status = VX_SUCCESS;

    status = vxSetNodeTarget(node, VX_TARGET_ANY, "any");
    ASSERT_EQ_INT(VX_ERROR_INVALID_REFERENCE, status);
}

#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION

TESTCASE_TESTS(Target,
        testvxSetNodeTarget,
        testvxSetImmediateModeTarget
        )

#endif

TESTCASE_TESTS(TargetBase,
        testvxCreateContext,
        //testvxQueryContext, - negative test turn off
        testvxReleaseContext,
        //testvxSetContextAttribute, - negative test turn off
        //testvxSetImmediateModeTargetBase, - negative test turn off
        testvxSetNodeTargetBase
        )
