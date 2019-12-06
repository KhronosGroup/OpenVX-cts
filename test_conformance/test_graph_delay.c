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

#include "test_engine/test.h"
#include "test_tensor_util.h"
#include <VX/vx.h>
#include <VX/vxu.h>

TESTCASE(GraphDelay, CT_VXContext, ct_setup_vx_context, 0)

TEST(GraphDelay, testSimple)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    int w = 128, h = 128;
    vx_df_image f = VX_DF_IMAGE_U8;
    vx_image images[3];
    vx_node nodes[3];
    vx_delay delay = 0;
    int i;
    vx_size delay_count = 0;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(images[0] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(images[1] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(images[2] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(delay = vxCreateDelay(context, (vx_reference)images[0], 2), VX_TYPE_DELAY);

    VX_CALL(vxQueryDelay(delay, VX_DELAY_SLOTS, &delay_count, sizeof(delay_count)));
    ASSERT(delay_count == 2);

    ASSERT_VX_OBJECT(nodes[0] = vxBox3x3Node(graph, images[0], (vx_image)vxGetReferenceFromDelay(delay, 0)), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(nodes[1] = vxMedian3x3Node(graph, (vx_image)vxGetReferenceFromDelay(delay, -1), images[1]), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(nodes[2] = vxGaussian3x3Node(graph, (vx_image)vxGetReferenceFromDelay(delay, -1), images[2]), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxAgeDelay(delay));

    VX_CALL(vxProcessGraph(graph));

    for (i = 0; i < 3; i++)
    {
        VX_CALL(vxReleaseNode(&nodes[i]));
    }
    for (i = 0; i < 3; i++)
    {
        VX_CALL(vxReleaseImage(&images[i]));
    }
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseDelay(&delay));

    ASSERT(graph == 0);
    ASSERT(delay == 0);
}

TEST(GraphDelay, testPyramid)
{
    int w = 128, h = 128;
    vx_df_image f = VX_DF_IMAGE_U8;
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_image input = 0;
    vx_image output = 0;
    vx_image image_pyr = 0;
    vx_image image_node = 0;
    vx_pyramid pyr = 0;
    vx_delay delay = 0;
    vx_node node_0 = 0;
    vx_node node_1 = 0;
    vx_parameter param = 0;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(input = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, w / 4, h / 4, f), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(pyr = vxCreatePyramid(context, 3, VX_SCALE_PYRAMID_HALF, w, h, f), VX_TYPE_PYRAMID);
    ASSERT_VX_OBJECT(delay = vxCreateDelay(context, (vx_reference)pyr, 2), VX_TYPE_DELAY);
    VX_CALL(vxReleasePyramid(&pyr));
    ASSERT(pyr == 0);

    ASSERT_VX_OBJECT(pyr = (vx_pyramid)vxGetReferenceFromDelay(delay, 0), VX_TYPE_PYRAMID);
    ASSERT_VX_OBJECT(node_0 = vxGaussianPyramidNode(graph, input, pyr), VX_TYPE_NODE);

    ASSERT_VX_OBJECT(image_pyr = vxGetPyramidLevel(pyr, 2), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(node_1 = vxMedian3x3Node(graph, image_pyr, output), VX_TYPE_NODE);
    VX_CALL(vxReleaseImage(&image_pyr));
    ASSERT(image_pyr == 0);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));
    VX_CALL(vxAgeDelay(delay));

    ASSERT_VX_OBJECT(pyr = (vx_pyramid)vxGetReferenceFromDelay(delay, 0), VX_TYPE_PYRAMID);
    ASSERT_VX_OBJECT(image_pyr = vxGetPyramidLevel(pyr, 2), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(node_1, 0), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &image_node, sizeof(image_node)));
    VX_CALL(vxReleaseParameter(&param));

    EXPECT_EQ_PTR(image_pyr, image_node);

    VX_CALL(vxReleaseImage(&image_node));
    VX_CALL(vxReleaseImage(&image_pyr));
    ASSERT(image_node == 0);
    ASSERT(image_pyr == 0);

    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&node_0));
    VX_CALL(vxReleaseNode(&node_1));
    ASSERT(node_0 == 0);
    ASSERT(node_1 == 0);

    VX_CALL(vxReleaseDelay(&delay));
    VX_CALL(vxReleaseImage(&input));
    VX_CALL(vxReleaseImage(&output));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(graph == 0);
    ASSERT(delay == 0);
    ASSERT(input == 0);
    ASSERT(output == 0);
}

TEST(GraphDelay, testRegisterAutoAging)
{
    int i, w = 128, h = 128;
    vx_df_image f = VX_DF_IMAGE_U8;
    vx_context context = context_->vx_context_;
    vx_graph graph_0 = 0;
    vx_graph graph_1 = 0;
    vx_image images[3];
    vx_node nodes[3];
    vx_delay delay = 0;
    vx_image delay_image_0 = 0;
    vx_image delay_image_1 = 0;
    vx_image node_image = 0;
    vx_parameter param = 0;
    vx_imagepatch_addressing_t addr;
    vx_uint8 *pdata = 0;
    vx_rectangle_t rect = {0, 0, 1, 1};
    vx_map_id map_id;

    ASSERT_VX_OBJECT(graph_0 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(graph_1 = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(images[0] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(images[1] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(images[2] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(delay = vxCreateDelay(context, (vx_reference)images[0], 2), VX_TYPE_DELAY);

    ASSERT_VX_OBJECT(delay_image_0 = (vx_image)vxGetReferenceFromDelay(delay, 0), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(delay_image_1 = (vx_image)vxGetReferenceFromDelay(delay,-1), VX_TYPE_IMAGE);


    /* image[0] gets 1 */
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(images[0], &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
    *pdata = 1;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(images[0], map_id));

    /* Initialize the each delay slots with different values */
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(delay_image_0, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
    /* Slot 0 gets 10 */
    *pdata = 10;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(delay_image_0, map_id));

    /* Slot -1 gets 2 */
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(delay_image_1, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
    *pdata = 2;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(delay_image_1, map_id));

    ASSERT_VX_OBJECT(nodes[0] = vxAddNode(graph_0, images[0], (vx_image)vxGetReferenceFromDelay(delay, -1), VX_CONVERT_POLICY_WRAP, (vx_image)vxGetReferenceFromDelay(delay, 0)), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(nodes[1] = vxGaussian3x3Node(graph_0, (vx_image)vxGetReferenceFromDelay(delay, -1), images[1]), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(nodes[2] = vxGaussian3x3Node(graph_1, (vx_image)vxGetReferenceFromDelay(delay, 0), images[2]), VX_TYPE_NODE);

    VX_CALL(vxRegisterAutoAging(graph_0, delay));
    VX_CALL(vxRegisterAutoAging(graph_1, delay));
    VX_CALL(vxVerifyGraph(graph_0));
    VX_CALL(vxVerifyGraph(graph_1));

    /* 1 + 2 (slot -1) -> 3 (slot 0) */
    /* Ageing shifts slots: slot -1 = 3 ; slot 0 = 2 */
    VX_CALL(vxProcessGraph(graph_0));

    /* check if delay was really aged */

    /* Slot 0 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[0], 3), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 2);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* Slot -1 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[0], 1), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 3);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* Slot -1 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[1], 0), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 3);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* Slot 0 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[2], 0), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 2);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* check auto-aging multiple registration */
    VX_CALL(vxRegisterAutoAging(graph_0, delay)); /* Register auto-ageing a second time */
    VX_CALL(vxVerifyGraph(graph_0));
    VX_CALL(vxProcessGraph(graph_0));

    /* the delay must be aged once */
    /* 1 + 3 (slot -1) -> 4 (slot 0) */
    /* Ageing shifts slots: slot -1 = 4 ; slot 0 = 3 */

    /* Slot 0 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[0], 3), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    VX_CALL(vxReleaseParameter(&param));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 3);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    ASSERT(node_image == 0);

    /* Slot -1 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[0], 1), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 4);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* Slot -1 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[1], 0), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 4);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* Slot 0 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[2], 0), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 3);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* check second graph */
    VX_CALL(vxProcessGraph(graph_1));

    /* the delay must be aged once more */
    /* Ageing shifts slots: slot -1 = 3 ; slot 0 = 4 */

    /* Slot 0 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[0], 3), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    VX_CALL(vxReleaseParameter(&param));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 4);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    ASSERT(node_image == 0);

    /* Slot -1 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[0], 1), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 3);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* Slot -1 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[1], 0), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 3);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    /* Slot 0 */
    ASSERT_VX_OBJECT(param = vxGetParameterByIndex(nodes[2], 0), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(param, VX_PARAMETER_REF, &node_image, sizeof(node_image)));
    pdata = NULL;
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxMapImagePatch(node_image, &rect, 0, &map_id, &addr, (void **)&pdata,
                                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
    ASSERT(*pdata == 4);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxUnmapImagePatch(node_image, map_id));
    VX_CALL(vxReleaseImage(&node_image));
    VX_CALL(vxReleaseParameter(&param));
    ASSERT(node_image == 0);
    ASSERT(param == 0);

    for (i = 0; i < (sizeof(nodes)/sizeof(nodes[0])); i++)
    {
        VX_CALL(vxReleaseNode(&nodes[i]));
    }

    for (i = 0; i < (sizeof(images)/sizeof(images[0])); i++)
    {
        VX_CALL(vxReleaseImage(&images[i]));
    }

    VX_CALL(vxReleaseGraph(&graph_0));
    VX_CALL(vxReleaseGraph(&graph_1));
    VX_CALL(vxReleaseDelay(&delay));

    ASSERT(graph_0 == 0);
    ASSERT(graph_1 == 0);
    ASSERT(delay == 0);
}

typedef struct
{
    const char* testName;
    const char* p;
    vx_enum item_type;
} Obj_Array_Arg;

static vx_reference own_create_exemplar(vx_context context, vx_enum item_type)
{
    vx_reference exemplar = NULL;

    vx_uint8 value = 0;
    vx_enum format = VX_DF_IMAGE_U8;
    vx_uint32 obj_width = 128, obj_height = 128;
    vx_enum obj_item_type = VX_TYPE_UINT8;
    vx_size capacity = 100;
    vx_size levels = 8;
    vx_float32 scale = 0.5f;
    vx_size bins = 36;
    vx_int32 offset = 0;
    vx_uint32 range = 360;
    vx_enum thresh_type = VX_THRESHOLD_TYPE_BINARY;
    vx_size lut_num_items = 100;
    vx_size m = 5, n = 5;

    switch (item_type)
    {
    case VX_TYPE_IMAGE:
        exemplar = (vx_reference)vxCreateImage(context, obj_width, obj_height, format);
        break;
    case VX_TYPE_ARRAY:
        exemplar = (vx_reference)vxCreateArray(context, obj_item_type, capacity);
        break;
    case VX_TYPE_PYRAMID:
        exemplar = (vx_reference)vxCreatePyramid(context, levels, scale, obj_width, obj_height, format);
        break;
    case VX_TYPE_SCALAR:
        exemplar = (vx_reference)vxCreateScalar(context, obj_item_type, &value);
        break;
    case VX_TYPE_MATRIX:
        exemplar = (vx_reference)vxCreateMatrix(context, obj_item_type, m, n);
        break;
    case VX_TYPE_DISTRIBUTION:
        exemplar = (vx_reference)vxCreateDistribution(context, bins, offset, range);
        break;
    case VX_TYPE_REMAP:
        exemplar = (vx_reference)vxCreateRemap(context, obj_width, obj_height, obj_width, obj_height);
        break;
    case VX_TYPE_LUT:
        exemplar = (vx_reference)vxCreateLUT(context, obj_item_type, lut_num_items);
        break;
    case VX_TYPE_THRESHOLD:
        exemplar = (vx_reference)vxCreateThresholdForImage(context, thresh_type, format, format);
        break;
    default:
        break;
    }

    return exemplar;
}

static void own_check_meta(vx_reference item, vx_reference ref)
{
    vx_enum ref_type, item_type;

    VX_CALL(vxQueryReference(ref, VX_REFERENCE_TYPE, &ref_type, sizeof(ref_type)));

    VX_CALL(vxQueryReference(item, VX_REFERENCE_TYPE, &item_type, sizeof(item_type)));

    ASSERT(item_type == ref_type);

    switch (item_type)
    {
    case VX_TYPE_IMAGE:
    {
        vx_uint32 ref_width, item_width;
        vx_uint32 ref_height, item_height;
        vx_df_image ref_format, item_format;

        VX_CALL(vxQueryImage((vx_image)ref, VX_IMAGE_WIDTH, &ref_width, sizeof(ref_width)));
        VX_CALL(vxQueryImage((vx_image)ref, VX_IMAGE_HEIGHT, &ref_height, sizeof(ref_height)));
        VX_CALL(vxQueryImage((vx_image)ref, VX_IMAGE_FORMAT, &ref_format, sizeof(ref_format)));

        VX_CALL(vxQueryImage((vx_image)item, VX_IMAGE_WIDTH, &item_width, sizeof(item_width)));
        VX_CALL(vxQueryImage((vx_image)item, VX_IMAGE_HEIGHT, &item_height, sizeof(item_height)));
        VX_CALL(vxQueryImage((vx_image)item, VX_IMAGE_FORMAT, &item_format, sizeof(item_format)));

        ASSERT(ref_width == item_width);
        ASSERT(ref_height == item_height);
        ASSERT(ref_format == item_format);
    }   break;
    case VX_TYPE_ARRAY:
    {
        vx_size ref_capacity, item_capacity;
        vx_enum ref_itemtype, item_itemtype;

        VX_CALL(vxQueryArray((vx_array)ref, VX_ARRAY_CAPACITY, &ref_capacity, sizeof(ref_capacity)));
        VX_CALL(vxQueryArray((vx_array)ref, VX_ARRAY_ITEMTYPE, &ref_itemtype, sizeof(ref_itemtype)));

        VX_CALL(vxQueryArray((vx_array)item, VX_ARRAY_CAPACITY, &item_capacity, sizeof(item_capacity)));
        VX_CALL(vxQueryArray((vx_array)item, VX_ARRAY_ITEMTYPE, &item_itemtype, sizeof(item_itemtype)));

        ASSERT(ref_capacity == item_capacity);
        ASSERT(ref_itemtype == item_itemtype);
    }   break;
    case VX_TYPE_PYRAMID:
    {
        vx_uint32 ref_width, item_width;
        vx_uint32 ref_height, item_height;
        vx_df_image ref_format, item_format;
        vx_size ref_levels, item_levels;
        vx_float32 ref_scale, item_scale;

        VX_CALL(vxQueryPyramid((vx_pyramid)ref, VX_PYRAMID_WIDTH, &ref_width, sizeof(ref_width)));
        VX_CALL(vxQueryPyramid((vx_pyramid)ref, VX_PYRAMID_HEIGHT, &ref_height, sizeof(ref_height)));
        VX_CALL(vxQueryPyramid((vx_pyramid)ref, VX_PYRAMID_FORMAT, &ref_format, sizeof(ref_format)));
        VX_CALL(vxQueryPyramid((vx_pyramid)ref, VX_PYRAMID_LEVELS, &ref_levels, sizeof(ref_levels)));
        VX_CALL(vxQueryPyramid((vx_pyramid)ref, VX_PYRAMID_SCALE, &ref_scale, sizeof(ref_scale)));

        VX_CALL(vxQueryPyramid((vx_pyramid)item, VX_PYRAMID_WIDTH, &item_width, sizeof(item_width)));
        VX_CALL(vxQueryPyramid((vx_pyramid)item, VX_PYRAMID_HEIGHT, &item_height, sizeof(item_height)));
        VX_CALL(vxQueryPyramid((vx_pyramid)item, VX_PYRAMID_FORMAT, &item_format, sizeof(item_format)));
        VX_CALL(vxQueryPyramid((vx_pyramid)item, VX_PYRAMID_LEVELS, &item_levels, sizeof(item_levels)));
        VX_CALL(vxQueryPyramid((vx_pyramid)item, VX_PYRAMID_SCALE, &item_scale, sizeof(item_scale)));

        ASSERT(ref_width == item_width);
        ASSERT(ref_height == item_height);
        ASSERT(ref_format == item_format);
        ASSERT(ref_levels == item_levels);
        ASSERT(ref_scale == item_scale);
    }   break;
    case VX_TYPE_SCALAR:
    {
        vx_enum ref_type, item_type;

        VX_CALL(vxQueryScalar((vx_scalar)ref, VX_SCALAR_TYPE, &ref_type, sizeof(ref_type)));

        VX_CALL(vxQueryScalar((vx_scalar)item, VX_SCALAR_TYPE, &item_type, sizeof(item_type)));

        ASSERT(ref_type == item_type);
    }   break;
    case VX_TYPE_MATRIX:
    {
        vx_enum ref_type, item_type;
        vx_size ref_rows, item_rows;
        vx_size ref_cols, item_cols;

        VX_CALL(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_TYPE, &ref_type, sizeof(ref_type)));
        VX_CALL(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ROWS, &ref_rows, sizeof(ref_rows)));
        VX_CALL(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_COLUMNS, &ref_cols, sizeof(ref_cols)));

        VX_CALL(vxQueryMatrix((vx_matrix)item, VX_MATRIX_TYPE, &item_type, sizeof(item_type)));
        VX_CALL(vxQueryMatrix((vx_matrix)item, VX_MATRIX_ROWS, &item_rows, sizeof(item_rows)));
        VX_CALL(vxQueryMatrix((vx_matrix)item, VX_MATRIX_COLUMNS, &item_cols, sizeof(item_cols)));

        ASSERT(ref_type == item_type);
        ASSERT(ref_rows == item_rows);
        ASSERT(ref_cols == item_cols);
    }   break;
    case VX_TYPE_DISTRIBUTION:
    {
        vx_size ref_bins, item_bins;
        vx_int32 ref_offset, item_offset;
        vx_uint32 ref_range, item_range;

        VX_CALL(vxQueryDistribution((vx_distribution)ref, VX_DISTRIBUTION_BINS, &ref_bins, sizeof(ref_bins)));
        VX_CALL(vxQueryDistribution((vx_distribution)ref, VX_DISTRIBUTION_OFFSET, &ref_offset, sizeof(ref_offset)));
        VX_CALL(vxQueryDistribution((vx_distribution)ref, VX_DISTRIBUTION_RANGE, &ref_range, sizeof(ref_range)));

        VX_CALL(vxQueryDistribution((vx_distribution)item, VX_DISTRIBUTION_BINS, &item_bins, sizeof(item_bins)));
        VX_CALL(vxQueryDistribution((vx_distribution)item, VX_DISTRIBUTION_OFFSET, &item_offset, sizeof(item_offset)));
        VX_CALL(vxQueryDistribution((vx_distribution)item, VX_DISTRIBUTION_RANGE, &item_range, sizeof(item_range)));

        ASSERT(ref_bins == item_bins);
        ASSERT(ref_offset == item_offset);
        ASSERT(ref_range == item_range);
    }   break;
    case VX_TYPE_REMAP:
    {
        vx_uint32 ref_srcwidth, item_srcwidth;
        vx_uint32 ref_srcheight, item_srcheight;
        vx_uint32 ref_dstwidth, item_dstwidth;
        vx_uint32 ref_dstheight, item_dstheight;

        VX_CALL(vxQueryRemap((vx_remap)ref, VX_REMAP_SOURCE_WIDTH, &ref_srcwidth, sizeof(ref_srcwidth)));
        VX_CALL(vxQueryRemap((vx_remap)ref, VX_REMAP_SOURCE_HEIGHT, &ref_srcheight, sizeof(ref_srcheight)));
        VX_CALL(vxQueryRemap((vx_remap)ref, VX_REMAP_DESTINATION_WIDTH, &ref_dstwidth, sizeof(ref_dstwidth)));
        VX_CALL(vxQueryRemap((vx_remap)ref, VX_REMAP_DESTINATION_HEIGHT, &ref_dstheight, sizeof(ref_dstheight)));

        VX_CALL(vxQueryRemap((vx_remap)item, VX_REMAP_SOURCE_WIDTH, &item_srcwidth, sizeof(item_srcwidth)));
        VX_CALL(vxQueryRemap((vx_remap)item, VX_REMAP_SOURCE_HEIGHT, &item_srcheight, sizeof(item_srcheight)));
        VX_CALL(vxQueryRemap((vx_remap)item, VX_REMAP_DESTINATION_WIDTH, &item_dstwidth, sizeof(item_dstwidth)));
        VX_CALL(vxQueryRemap((vx_remap)item, VX_REMAP_DESTINATION_HEIGHT, &item_dstheight, sizeof(item_dstheight)));

        ASSERT(ref_srcwidth == item_srcwidth);
        ASSERT(ref_srcheight == item_srcheight);
        ASSERT(ref_dstwidth == item_dstwidth);
        ASSERT(ref_dstheight == item_dstheight);
    }   break;
    case VX_TYPE_LUT:
    {
        vx_enum ref_type, item_type;
        vx_size ref_count, item_count;

        VX_CALL(vxQueryLUT((vx_lut)ref, VX_LUT_TYPE, &ref_type, sizeof(ref_type)));
        VX_CALL(vxQueryLUT((vx_lut)ref, VX_LUT_COUNT, &ref_count, sizeof(ref_count)));

        VX_CALL(vxQueryLUT((vx_lut)item, VX_LUT_TYPE, &item_type, sizeof(item_type)));
        VX_CALL(vxQueryLUT((vx_lut)item, VX_LUT_COUNT, &item_count, sizeof(item_count)));

        ASSERT(ref_type == item_type);
        ASSERT(ref_count == item_count);
    }   break;
    case VX_TYPE_THRESHOLD:
    {
        vx_enum ref_type, item_type;

        VX_CALL(vxQueryThreshold((vx_threshold)ref, VX_THRESHOLD_TYPE, &ref_type, sizeof(ref_type)));

        VX_CALL(vxQueryThreshold((vx_threshold)item, VX_THRESHOLD_TYPE, &item_type, sizeof(item_type)));

        ASSERT(ref_type == item_type);
    }   break;
    default:
        ASSERT(0 == 1);
    }
}

#define ADD_VX_OBJECT_ARRAY_TYPES(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_IMAGE", __VA_ARGS__, VX_TYPE_IMAGE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_ARRAY", __VA_ARGS__, VX_TYPE_ARRAY)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_PYRAMID", __VA_ARGS__, VX_TYPE_PYRAMID)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_SCALAR", __VA_ARGS__, VX_TYPE_SCALAR)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_MATRIX", __VA_ARGS__, VX_TYPE_MATRIX)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_DISTRIBUTION", __VA_ARGS__, VX_TYPE_DISTRIBUTION)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_REMAP", __VA_ARGS__, VX_TYPE_REMAP)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_LUT", __VA_ARGS__, VX_TYPE_LUT)), \
    CT_EXPAND(nextmacro(testArgName "/VX_TYPE_THRESHOLD", __VA_ARGS__, VX_TYPE_THRESHOLD ))

#define PARAMETERS \
    CT_GENERATE_PARAMETERS("object_array", ADD_VX_OBJECT_ARRAY_TYPES, ARG, NULL)

TEST_WITH_ARG(GraphDelay, testObjectArray, Obj_Array_Arg, PARAMETERS)
{
    vx_context context = context_->vx_context_;
    vx_delay delay;
    vx_reference exemplar = NULL;
    vx_size num_items = 1;
    vx_enum item_type = arg_->item_type;

    vx_object_array object_array = 0;
    vx_object_array delayobjarray;

    vx_reference expect_item = NULL;
    vx_enum expect_type = VX_TYPE_INVALID;
    vx_size expect_num_items = 0;

    vx_uint32 i;

    ASSERT_VX_OBJECT(exemplar = own_create_exemplar(context, item_type), (enum vx_type_e)item_type);

    /* 1. check if object array can be created with allowed types*/
    ASSERT_VX_OBJECT(object_array = vxCreateObjectArray(context, exemplar, num_items), VX_TYPE_OBJECT_ARRAY);

    /* 2. create delay with object array*/
    ASSERT_VX_OBJECT(delay = vxCreateDelay(context, (vx_reference)object_array, 1), VX_TYPE_DELAY);

    ASSERT_VX_OBJECT(delayobjarray = (vx_object_array)vxGetReferenceFromDelay(delay, 0), VX_TYPE_OBJECT_ARRAY);

    VX_CALL(vxQueryObjectArray(delayobjarray, VX_OBJECT_ARRAY_ITEMTYPE, &expect_type, sizeof(expect_type)));
    ASSERT_EQ_INT(item_type, expect_type);

    VX_CALL(vxQueryObjectArray(delayobjarray, VX_OBJECT_ARRAY_NUMITEMS, &expect_num_items, sizeof(expect_num_items)));
    ASSERT_EQ_INT(num_items, expect_num_items);

    for (i = 0u; i < num_items; i++)
    {
        ASSERT_VX_OBJECT(expect_item = vxGetObjectArrayItem(delayobjarray, i), (enum vx_type_e)item_type);

        ASSERT_NO_FAILURE(own_check_meta(expect_item, exemplar));

        VX_CALL(vxReleaseReference(&expect_item));
        ASSERT(expect_item == 0);
    }

    expect_item = vxGetObjectArrayItem(delayobjarray, (vx_uint32)num_items);
    ASSERT_NE_VX_STATUS(VX_SUCCESS, vxGetStatus((vx_reference)expect_item));

    VX_CALL(vxReleaseReference(&exemplar));
    ASSERT(exemplar == 0);

    VX_CALL(vxReleaseObjectArray(&object_array));
    ASSERT(object_array == 0);

    VX_CALL(vxReleaseDelay(&delay));
    ASSERT(delay == 0);
}

#endif //OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION

#ifdef OPENVX_USE_ENHANCED_VISION

/* *****************testTensor tests*******************************/
TESTCASE(GraphDelayTensor, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char *name;
    enum TestTensorDF fmt;
} test_tensor_op_arg;

TEST_WITH_ARG(GraphDelayTensor, testTensor, test_tensor_op_arg,
    ARG("Q78_DELAYTENSOR", TT_Q78),
    ARG("U8_DELAYTENSOR", TT_U8),
    ARG("S8_DELAYTENSOR", TT_S8))
{
    vx_context context = context_->vx_context_;
    const enum TestTensorDF fmt = arg_->fmt;
    vx_size max_dims = 0;
    vx_enum data_type = 0;
    vx_uint8 fixed_point_position = 0;
    vx_size sizeof_data_type = 0;
    vx_delay delay;
    vx_status status;
    vx_size delaytensor_dims;
    vx_enum delaytensor_datatype;
    vx_uint8 delaytensor_fppos;
    vx_size delaytensor_in0dims[MAX_TENSOR_DIMS];

    ownUnpackFormat(fmt, &data_type, &fixed_point_position, &sizeof_data_type);

    VX_CALL(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &max_dims, sizeof(max_dims)));
    ASSERT(max_dims > 3);

    size_t * const in0_dims = ct_alloc_mem(sizeof(*in0_dims) * max_dims);
    ASSERT(in0_dims);

    uint64_t rng;
    {   // TODO: ownTestGetRNG() ?
        uint64_t * seed = &CT()->seed_;
        ASSERT(!!seed);
        CT_RNG_INIT(rng, *seed);
    }

    for (vx_size i = 0; i < max_dims; ++i)
    {
        const size_t new_dim = (size_t)CT_RNG_NEXT_INT(rng, TEST_TENSOR_MIN_DIM_SZ, TEST_TENSOR_MAX_DIM_SZ + 1);

        const int mask0 = !!CT_RNG_NEXT_INT(rng, 0, TEST_TENSOR_INVERSE_MASK_PROBABILITY);

        // Note: Broadcasting is described as for each dim, either in0 and in1 have the same
        // size or "1" for a broadcasted value. And the output is strictly determined by them
        // so that the implementation is required to support
        // { in0, in1, out } = { 1, 5, 5 } but not { in0, in1, out } = { 1, 1, 5 }
        // even though the KHR sample implementation currently supports both.
        in0_dims[i] = mask0 ? new_dim : 1;
    }

    vx_tensor tensor = vxCreateTensor(context, max_dims, in0_dims, data_type, fixed_point_position);
    ASSERT_VX_OBJECT(tensor, VX_TYPE_TENSOR);

    ASSERT_VX_OBJECT(delay = vxCreateDelay(context, (vx_reference)tensor, 1), VX_TYPE_DELAY);

    vx_tensor delaytensor;
    ASSERT_VX_OBJECT(delaytensor = (vx_tensor)vxGetReferenceFromDelay(delay, 0), VX_TYPE_TENSOR);

    status = vxQueryTensor(delaytensor, VX_TENSOR_NUMBER_OF_DIMS, (void *)&delaytensor_dims, sizeof(delaytensor_dims));
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, status);
    EXPECT_EQ_INT(max_dims, delaytensor_dims);

    status = vxQueryTensor(delaytensor, VX_TENSOR_DATA_TYPE, (void *)&delaytensor_datatype, sizeof(delaytensor_datatype));
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, status);
    EXPECT_EQ_INT(data_type, delaytensor_datatype);

    status = vxQueryTensor(delaytensor, VX_TENSOR_FIXED_POINT_POSITION, (void *)&delaytensor_fppos, sizeof(delaytensor_fppos));
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, status);
    EXPECT_EQ_INT(max_dims, delaytensor_dims);

    status = vxQueryTensor(delaytensor, VX_TENSOR_DIMS, (void *)&delaytensor_in0dims, sizeof(vx_size)*max_dims);
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, status);

    for (vx_size i = 0; i < max_dims; i++)
    {
        EXPECT_EQ_INT(in0_dims[i], delaytensor_in0dims[i]);
    }

    VX_CALL(vxReleaseDelay(&delay));
    VX_CALL(vxReleaseTensor(&tensor));

    ASSERT(delay == 0);
    ASSERT(tensor == 0);

    ct_free_mem(in0_dims);
}

#endif //OPENVX_USE_ENHANCED_VISION

#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION

TESTCASE_TESTS(
    GraphDelay,
    testSimple,
    testPyramid,
    testRegisterAutoAging,
    testObjectArray
    )

#endif

#ifdef OPENVX_USE_ENHANCED_VISION

TESTCASE_TESTS(
    GraphDelayTensor,
    testTensor
    )
#endif
