/** @file graph.h
 *  @brief
 *  This file contains the implementation of the generated graph factory function
 */

#ifdef OPENVX_CONFORMANCE_NEURAL_NETWORKS

#include <stdio.h>
#include <stdlib.h>
#include <VX/vx_khr_nn.h>
#include <VX/vx_compatibility.h>  // for vxNormalizeLayer
#include "graph.h"



/** @brief Constructs OpenVX graph and connects to the input/output references
 *
 *  @param context The OpenVX context
 *  @param graph The OpenVX graph
 *  @param pObjectContainer The pointer to object container.
 *  @param filteredNodesList The list of filtered nodes to create in the graph (can be empty)
 *  @param filteredNodesCount The number of filtered nodes to create in the graph
 *  @return vx_status code.
 */
static vx_status Graph(vx_context context, vx_graph graph, ObjectRefContainerType* pObjectContainer, char* filteredNodeList[], size_t filteredNodeCount, vx_tensor org_khronos_nn_extension_convolution_layer_0_p0, vx_tensor org_khronos_nn_extension_convolution_layer_0_p1, vx_tensor org_khronos_nn_extension_convolution_layer_0_p2, vx_scalar org_khronos_nn_extension_convolution_layer_0_p3, vx_scalar org_khronos_nn_extension_convolution_layer_0_p4, vx_scalar org_khronos_nn_extension_convolution_layer_0_p5, vx_scalar org_khronos_nn_extension_convolution_layer_0_p6, vx_scalar org_khronos_nn_extension_convolution_layer_0_p7, vx_tensor org_khronos_nn_extension_convolution_layer_0_p8);

/** @brief Implements the OpenVX graph factory
 *
 *  @param context The OpenVX context
 *  @param graph The OpenVX graph
 *  @param pObjectContainer The pointer to object container.
 *  @param filteredNodesList The list of filtered nodes to create in the graph (can be empty)
 *  @param filteredNodesCount The number of filtered nodes to create in the graph
 *  @return vx_status code.
 */
vx_status _GraphFactoryAlexnet(vx_context context, vx_graph graph, ObjectRefContainerType* pObjectContainer, char* filteredNodeList[], size_t filteredNodeCount)
{
    vx_status status = VX_SUCCESS;

    //
    // Primitive Declarations
    //

    vx_tensor org_khronos_nn_extension_convolution_layer_0_p0;
    vx_tensor org_khronos_nn_extension_convolution_layer_0_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_0_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_0_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_0_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_0_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_0_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_0_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_0_p8;

    //
    // Other Declarations
    //

    vx_size org_khronos_nn_extension_convolution_layer_0_p0Dimensions[4] = {227,227,3,1};
    vx_size org_khronos_nn_extension_convolution_layer_0_p1Dimensions[4] = {11,11,3,96};
    vx_size org_khronos_nn_extension_convolution_layer_0_p2Dimensions[1] = {96};
    vx_size org_khronos_nn_extension_convolution_layer_0_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_0_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_0_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_0_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_0_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_0_p8Dimensions[4] = {55,55,96,1};

    //
    // Source Primitives Assignments
    // ( source primitives are created here. These are used as inputs to Graph(), which will query the primitives for their respective attribute values )
    //

    org_khronos_nn_extension_convolution_layer_0_p0 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_0_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p0);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p0, VX_TYPE_TENSOR, "cnn_input");

    org_khronos_nn_extension_convolution_layer_0_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_0_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p1);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p1, VX_TYPE_TENSOR, "conv1_weights");

    org_khronos_nn_extension_convolution_layer_0_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_0_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p2);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p2, VX_TYPE_TENSOR, "conv1_bias");

    org_khronos_nn_extension_convolution_layer_0_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p3);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p3, VX_TYPE_SCALAR, "conv1_3");

    org_khronos_nn_extension_convolution_layer_0_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_0_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p4);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p4, VX_TYPE_SCALAR, "conv1_4");

    org_khronos_nn_extension_convolution_layer_0_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_0_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p5);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p5, VX_TYPE_SCALAR, "conv1_5");

    org_khronos_nn_extension_convolution_layer_0_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_0_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p6);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p6, VX_TYPE_SCALAR, "conv1_6");

    org_khronos_nn_extension_convolution_layer_0_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_0_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p7);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p7, VX_TYPE_SCALAR, "conv1_7");

    org_khronos_nn_extension_convolution_layer_0_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_0_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p8);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p8, VX_TYPE_TENSOR, "conv1_8");


    //
    // All nodes and primitives (except primitives associated with source nodes) of the graph are setup in Graph()
    //

    status = Graph(context, graph, pObjectContainer, filteredNodeList, filteredNodeCount, org_khronos_nn_extension_convolution_layer_0_p0, org_khronos_nn_extension_convolution_layer_0_p1, org_khronos_nn_extension_convolution_layer_0_p2, org_khronos_nn_extension_convolution_layer_0_p3, org_khronos_nn_extension_convolution_layer_0_p4, org_khronos_nn_extension_convolution_layer_0_p5, org_khronos_nn_extension_convolution_layer_0_p6, org_khronos_nn_extension_convolution_layer_0_p7, org_khronos_nn_extension_convolution_layer_0_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create graph workload [Graph()]\n");
        return VX_FAILURE;
    }

    return status;
}

static vx_status Graph(vx_context context, vx_graph graph, ObjectRefContainerType* pObjectContainer, char* filteredNodeList[], size_t filteredNodeCount, vx_tensor org_khronos_nn_extension_convolution_layer_0_p0, vx_tensor org_khronos_nn_extension_convolution_layer_0_p1, vx_tensor org_khronos_nn_extension_convolution_layer_0_p2, vx_scalar org_khronos_nn_extension_convolution_layer_0_p3, vx_scalar org_khronos_nn_extension_convolution_layer_0_p4, vx_scalar org_khronos_nn_extension_convolution_layer_0_p5, vx_scalar org_khronos_nn_extension_convolution_layer_0_p6, vx_scalar org_khronos_nn_extension_convolution_layer_0_p7, vx_tensor org_khronos_nn_extension_convolution_layer_0_p8)
{
    vx_status status = VX_SUCCESS;

    //
    // Kernel Declarations
    //


    //
    // Node Declarations
    //

    vx_node org_khronos_nn_extension_convolution_layer_0;
    vx_node org_khronos_nn_extension_activation_layer_0;
    vx_node org_khronos_nn_extension_normalization_layer_0;
    vx_node org_khronos_nn_extension_pooling_layer_0;
    vx_node org_khronos_nn_extension_convolution_layer_2;
    vx_node org_khronos_nn_extension_convolution_layer_1;
    vx_node org_khronos_nn_extension_activation_layer_1;
    vx_node org_khronos_nn_extension_normalization_layer_1;
    vx_node org_khronos_nn_extension_pooling_layer_1;
    vx_node org_khronos_nn_extension_convolution_layer_3;
    vx_node org_khronos_nn_extension_activation_layer_2;
    vx_node org_khronos_nn_extension_convolution_layer_5;
    vx_node org_khronos_nn_extension_convolution_layer_4;
    vx_node org_khronos_nn_extension_activation_layer_3;
    vx_node org_khronos_nn_extension_convolution_layer_7;
    vx_node org_khronos_nn_extension_convolution_layer_6;
    vx_node org_khronos_nn_extension_activation_layer_4;
    vx_node org_khronos_nn_extension_pooling_layer_2;
    vx_node org_khronos_nn_extension_fully_connected_layer_0;
    vx_node org_khronos_nn_extension_activation_layer_5;
    vx_node org_khronos_nn_extension_fully_connected_layer_1;
    vx_node org_khronos_nn_extension_activation_layer_6;
    vx_node org_khronos_nn_extension_fully_connected_layer_2;
    vx_node org_khronos_nn_extension_softmax_layer_0;

    //
    // Primitive Declarations
    //

    vx_tensor outputAllocators_MergeTensor_2_p0;
    vx_tensor outputAllocators_MergeTensor_1_p0;
    vx_tensor outputAllocators_MergeTensor_0_p0;
    vx_scalar org_khronos_nn_extension_activation_layer_0_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_0_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_0_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_0_p4;
    vx_scalar org_khronos_nn_extension_normalization_layer_0_p1;
    vx_scalar org_khronos_nn_extension_normalization_layer_0_p2;
    vx_scalar org_khronos_nn_extension_normalization_layer_0_p3;
    vx_scalar org_khronos_nn_extension_normalization_layer_0_p4;
    vx_tensor org_khronos_nn_extension_normalization_layer_0_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_0_p7;
    vx_tensor outputAllocators_SplitTensor_0_p1;
    vx_tensor outputAllocators_SplitTensor_0_p2;
    vx_tensor org_khronos_nn_extension_convolution_layer_2_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_2_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_2_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_1_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_1_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_1_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_1_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_1_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_1_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_1_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_1_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_1_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_1_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_1_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_1_p4;
    vx_scalar org_khronos_nn_extension_normalization_layer_1_p1;
    vx_scalar org_khronos_nn_extension_normalization_layer_1_p2;
    vx_scalar org_khronos_nn_extension_normalization_layer_1_p3;
    vx_scalar org_khronos_nn_extension_normalization_layer_1_p4;
    vx_tensor org_khronos_nn_extension_normalization_layer_1_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_1_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_1_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_1_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_1_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_1_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_1_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_1_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_3_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_3_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_3_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_2_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_2_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_2_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_2_p4;
    vx_tensor outputAllocators_SplitTensor_1_p1;
    vx_tensor outputAllocators_SplitTensor_1_p2;
    vx_tensor org_khronos_nn_extension_convolution_layer_5_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_5_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_5_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_4_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_4_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_4_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_3_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_3_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_3_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_3_p4;
    vx_tensor outputAllocators_SplitTensor_2_p1;
    vx_tensor outputAllocators_SplitTensor_2_p2;
    vx_tensor org_khronos_nn_extension_convolution_layer_7_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_7_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_7_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_6_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_6_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_6_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_4_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_4_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_4_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_4_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_2_p7;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_0_p1;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_0_p2;
    vx_scalar org_khronos_nn_extension_fully_connected_layer_0_p3;
    vx_scalar org_khronos_nn_extension_fully_connected_layer_0_p4;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_0_p5;
    vx_scalar org_khronos_nn_extension_activation_layer_5_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_5_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_5_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_5_p4;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_1_p1;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_1_p2;
    vx_scalar org_khronos_nn_extension_fully_connected_layer_1_p3;
    vx_scalar org_khronos_nn_extension_fully_connected_layer_1_p4;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_1_p5;
    vx_scalar org_khronos_nn_extension_activation_layer_6_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_6_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_6_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_6_p4;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_2_p1;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_2_p2;
    vx_scalar org_khronos_nn_extension_fully_connected_layer_2_p3;
    vx_scalar org_khronos_nn_extension_fully_connected_layer_2_p4;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_2_p5;
    vx_scalar com_cnn_helpers_scalemddata_0_p1;
    vx_tensor com_cnn_helpers_scalemddata_0_p2;
    vx_tensor org_khronos_nn_extension_softmax_layer_0_p1;

    //
    // Other Declarations
    //

    vx_size outputAllocators_MergeTensor_2_p0Dimensions[4] = {13,13,256,1};
    vx_size outputAllocators_MergeTensor_1_p0Dimensions[4] = {13,13,384,1};
    vx_size outputAllocators_MergeTensor_0_p0Dimensions[4] = {27,27,256,1};
    vx_enum org_khronos_nn_extension_activation_layer_0_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_0_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_0_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_0_p4Dimensions[4] = {55,55,96,1};
    vx_enum org_khronos_nn_extension_normalization_layer_0_scalar_p1 = VX_NN_NORMALIZATION_ACROSS_MAPS;
    vx_size org_khronos_nn_extension_normalization_layer_0_scalar_p2 = 5;
    vx_float32 org_khronos_nn_extension_normalization_layer_0_scalar_p3 = 0.0063999998;
    vx_float32 org_khronos_nn_extension_normalization_layer_0_scalar_p4 = 0.750000;
    vx_size org_khronos_nn_extension_normalization_layer_0_p5Dimensions[4] = {55,55,96,1};
    vx_enum org_khronos_nn_extension_pooling_layer_0_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_0_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_0_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_0_scalar_p4 = 0;
    vx_size org_khronos_nn_extension_pooling_layer_0_scalar_p5 = 0;
    vx_enum org_khronos_nn_extension_pooling_layer_0_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_0_p7Dimensions[4] = {27,27,96,1};
    vx_size org_khronos_nn_extension_pooling_layer_0_p7_view1_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_pooling_layer_0_p7_view1_view_end[4] = {27,27,48,1};
    vx_size org_khronos_nn_extension_pooling_layer_0_p7_view2_view_start[4] = {0,0,48,0};
    vx_size org_khronos_nn_extension_pooling_layer_0_p7_view2_view_end[4] = {27,27,96,1};
    vx_size org_khronos_nn_extension_convolution_layer_2_p1Dimensions[4] = {5,5,48,128};
    vx_size org_khronos_nn_extension_convolution_layer_2_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_2_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_2_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_2_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_2_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_2_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_2_p8_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_convolution_layer_2_p8_view_view_end[4] = {27,27,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_1_p1Dimensions[4] = {5,5,48,128};
    vx_size org_khronos_nn_extension_convolution_layer_1_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_1_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_1_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_1_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_1_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_1_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_1_p8_view_view_start[4] = {0,0,128,0};
    vx_size org_khronos_nn_extension_convolution_layer_1_p8_view_view_end[4] = {27,27,256,1};
    vx_enum org_khronos_nn_extension_activation_layer_1_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_1_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_1_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_1_p4Dimensions[4] = {27,27,256,1};
    vx_enum org_khronos_nn_extension_normalization_layer_1_scalar_p1 = VX_NN_NORMALIZATION_ACROSS_MAPS;
    vx_size org_khronos_nn_extension_normalization_layer_1_scalar_p2 = 5;
    vx_float32 org_khronos_nn_extension_normalization_layer_1_scalar_p3 = 0.0063999998;
    vx_float32 org_khronos_nn_extension_normalization_layer_1_scalar_p4 = 0.750000;
    vx_size org_khronos_nn_extension_normalization_layer_1_p5Dimensions[4] = {27,27,256,1};
    vx_enum org_khronos_nn_extension_pooling_layer_1_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_1_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_1_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_1_scalar_p4 = 0;
    vx_size org_khronos_nn_extension_pooling_layer_1_scalar_p5 = 0;
    vx_enum org_khronos_nn_extension_pooling_layer_1_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_1_p7Dimensions[4] = {13,13,256,1};
    vx_size org_khronos_nn_extension_convolution_layer_3_p1Dimensions[4] = {3,3,256,384};
    vx_size org_khronos_nn_extension_convolution_layer_3_p2Dimensions[1] = {384};
    vx_size org_khronos_nn_extension_convolution_layer_3_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_3_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_3_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_3_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_3_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_3_p8Dimensions[4] = {13,13,384,1};
    vx_enum org_khronos_nn_extension_activation_layer_2_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_2_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_2_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_2_p4Dimensions[4] = {13,13,384,1};
    vx_size org_khronos_nn_extension_activation_layer_2_p4_view1_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_2_p4_view1_view_end[4] = {13,13,192,1};
    vx_size org_khronos_nn_extension_activation_layer_2_p4_view2_view_start[4] = {0,0,192,0};
    vx_size org_khronos_nn_extension_activation_layer_2_p4_view2_view_end[4] = {13,13,384,1};
    vx_size org_khronos_nn_extension_convolution_layer_5_p1Dimensions[4] = {3,3,192,192};
    vx_size org_khronos_nn_extension_convolution_layer_5_p2Dimensions[1] = {192};
    vx_size org_khronos_nn_extension_convolution_layer_5_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_5_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_5_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_5_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_5_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_5_p8_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_convolution_layer_5_p8_view_view_end[4] = {13,13,192,1};
    vx_size org_khronos_nn_extension_convolution_layer_4_p1Dimensions[4] = {3,3,192,192};
    vx_size org_khronos_nn_extension_convolution_layer_4_p2Dimensions[1] = {192};
    vx_size org_khronos_nn_extension_convolution_layer_4_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_4_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_4_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_4_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_4_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_4_p8_view_view_start[4] = {0,0,192,0};
    vx_size org_khronos_nn_extension_convolution_layer_4_p8_view_view_end[4] = {13,13,384,1};
    vx_enum org_khronos_nn_extension_activation_layer_3_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_3_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_3_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_3_p4Dimensions[4] = {13,13,384,1};
    vx_size org_khronos_nn_extension_activation_layer_3_p4_view1_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_3_p4_view1_view_end[4] = {13,13,192,1};
    vx_size org_khronos_nn_extension_activation_layer_3_p4_view2_view_start[4] = {0,0,192,0};
    vx_size org_khronos_nn_extension_activation_layer_3_p4_view2_view_end[4] = {13,13,384,1};
    vx_size org_khronos_nn_extension_convolution_layer_7_p1Dimensions[4] = {3,3,192,128};
    vx_size org_khronos_nn_extension_convolution_layer_7_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_7_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_7_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_7_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_7_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_7_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_7_p8_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_convolution_layer_7_p8_view_view_end[4] = {13,13,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_6_p1Dimensions[4] = {3,3,192,128};
    vx_size org_khronos_nn_extension_convolution_layer_6_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_6_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_6_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_6_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_6_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_6_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_6_p8_view_view_start[4] = {0,0,128,0};
    vx_size org_khronos_nn_extension_convolution_layer_6_p8_view_view_end[4] = {13,13,256,1};
    vx_enum org_khronos_nn_extension_activation_layer_4_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_4_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_4_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_4_p4Dimensions[4] = {13,13,256,1};
    vx_enum org_khronos_nn_extension_pooling_layer_2_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_2_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_2_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_2_scalar_p4 = 0;
    vx_size org_khronos_nn_extension_pooling_layer_2_scalar_p5 = 0;
    vx_enum org_khronos_nn_extension_pooling_layer_2_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_2_p7Dimensions[4] = {6,6,256,1};
    vx_size org_khronos_nn_extension_fully_connected_layer_0_p1Dimensions[4] = {6,6,256,4096};
    vx_size org_khronos_nn_extension_fully_connected_layer_0_p2Dimensions[1] = {4096};
    vx_enum org_khronos_nn_extension_fully_connected_layer_0_scalar_p3 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_fully_connected_layer_0_scalar_p4 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_size org_khronos_nn_extension_fully_connected_layer_0_p5Dimensions[2] = {4096,1};
    vx_enum org_khronos_nn_extension_activation_layer_5_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_5_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_5_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_5_p4Dimensions[2] = {4096,1};
    vx_size org_khronos_nn_extension_fully_connected_layer_1_p1Dimensions[2] = {4096,4096};
    vx_size org_khronos_nn_extension_fully_connected_layer_1_p2Dimensions[1] = {4096};
    vx_enum org_khronos_nn_extension_fully_connected_layer_1_scalar_p3 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_fully_connected_layer_1_scalar_p4 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_size org_khronos_nn_extension_fully_connected_layer_1_p5Dimensions[2] = {4096,1};
    vx_enum org_khronos_nn_extension_activation_layer_6_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_6_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_6_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_6_p4Dimensions[2] = {4096,1};
    vx_size org_khronos_nn_extension_fully_connected_layer_2_p1Dimensions[2] = {4096,1000};
    vx_size org_khronos_nn_extension_fully_connected_layer_2_p2Dimensions[1] = {1000};
    vx_enum org_khronos_nn_extension_fully_connected_layer_2_scalar_p3 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_fully_connected_layer_2_scalar_p4 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_size org_khronos_nn_extension_fully_connected_layer_2_p5Dimensions[2] = {1000,1};
    vx_float32 com_cnn_helpers_scalemddata_0_scalar_p1 = 8;
    vx_size com_cnn_helpers_scalemddata_0_p2Dimensions[2] = {1000,1};
    vx_size org_khronos_nn_extension_softmax_layer_0_p1Dimensions[2] = {1000,1};


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Manual types
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	pooling_params pool_params = { 3, 3, 0, 0, VX_NN_DS_SIZE_ROUNDING_FLOOR };
	activation_params relu_params = { VX_NN_ACTIVATION_RELU, 0, 0 };
	vx_nn_convolution_params_t conv_params[3] = { { 0, 0, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0 },
	{ 2, 2, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0 },
	{ 1, 1, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0 } };
	normalization_params norm_params = { VX_NN_NORMALIZATION_ACROSS_MAPS, 5, 0.0001f * 64, 0.75f };

	vx_enum overflowPolicy = VX_CONVERT_POLICY_SATURATE;
	vx_scalar overflowPolicy_scalar = vxCreateScalar(context, VX_TYPE_ENUM, &overflowPolicy);
    status = vxGetStatus((vx_reference)overflowPolicy_scalar);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter overflowPolicy_scalar (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)overflowPolicy_scalar, VX_TYPE_SCALAR, "overflowPolicy_scalar");
	vx_enum roundingPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
	vx_scalar roundingPolicy_scalar = vxCreateScalar(context, VX_TYPE_ENUM, &roundingPolicy);
    status = vxGetStatus((vx_reference)roundingPolicy_scalar);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter roundingPolicy_scalar (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)roundingPolicy_scalar, VX_TYPE_SCALAR, "roundingPolicy_scalar");



    //
    // Primitive Assignments
    //
    outputAllocators_MergeTensor_2_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_2_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_2_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_2_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_2_p0, VX_TYPE_TENSOR, "relu5_0");

    outputAllocators_MergeTensor_1_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_1_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_1_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_1_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_1_p0, VX_TYPE_TENSOR, "relu4_0");

    outputAllocators_MergeTensor_0_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_0_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_0_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_0_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_0_p0, VX_TYPE_TENSOR, "relu2_0");

    org_khronos_nn_extension_activation_layer_0_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_0_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0_p1, VX_TYPE_SCALAR, "relu1_1");

    org_khronos_nn_extension_activation_layer_0_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_0_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0_p2, VX_TYPE_SCALAR, "relu1_2");

    org_khronos_nn_extension_activation_layer_0_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0_p3, VX_TYPE_SCALAR, "relu1_2");

    org_khronos_nn_extension_activation_layer_0_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_0_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0_p4, VX_TYPE_TENSOR, "relu1_4");

    org_khronos_nn_extension_normalization_layer_0_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_normalization_layer_0_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p1, VX_TYPE_SCALAR, "norm1_1");

    org_khronos_nn_extension_normalization_layer_0_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_normalization_layer_0_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p2, VX_TYPE_SCALAR, "norm1_2");

    org_khronos_nn_extension_normalization_layer_0_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_normalization_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p3, VX_TYPE_SCALAR, "norm1_3");

    org_khronos_nn_extension_normalization_layer_0_p4 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_normalization_layer_0_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p4, VX_TYPE_SCALAR, "norm1_4");

    org_khronos_nn_extension_normalization_layer_0_p5 = vxCreateTensor(context, 4, org_khronos_nn_extension_normalization_layer_0_p5Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p5, VX_TYPE_TENSOR, "norm1_5");

    org_khronos_nn_extension_pooling_layer_0_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p1, VX_TYPE_SCALAR, "pool1_1");

    org_khronos_nn_extension_pooling_layer_0_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p2, VX_TYPE_SCALAR, "pool1_2");

    org_khronos_nn_extension_pooling_layer_0_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p3, VX_TYPE_SCALAR, "pool1_3");

    org_khronos_nn_extension_pooling_layer_0_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p4, VX_TYPE_SCALAR, "pool1_4");

    org_khronos_nn_extension_pooling_layer_0_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p5, VX_TYPE_SCALAR, "pool1_5");

    org_khronos_nn_extension_pooling_layer_0_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p6, VX_TYPE_SCALAR, "pool1_6");

    org_khronos_nn_extension_pooling_layer_0_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_0_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p7, VX_TYPE_TENSOR, "pool1_7");

    outputAllocators_SplitTensor_0_p1 = vxCreateTensorFromView(org_khronos_nn_extension_pooling_layer_0_p7, 4, org_khronos_nn_extension_pooling_layer_0_p7_view1_view_start, org_khronos_nn_extension_pooling_layer_0_p7_view1_view_end);
    status = vxGetStatus((vx_reference)outputAllocators_SplitTensor_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_SplitTensor_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_SplitTensor_0_p1, VX_TYPE_TENSOR, "conv2_0_0");

    outputAllocators_SplitTensor_0_p2 = vxCreateTensorFromView(org_khronos_nn_extension_pooling_layer_0_p7, 4, org_khronos_nn_extension_pooling_layer_0_p7_view2_view_start, org_khronos_nn_extension_pooling_layer_0_p7_view2_view_end);
    status = vxGetStatus((vx_reference)outputAllocators_SplitTensor_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_SplitTensor_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_SplitTensor_0_p2, VX_TYPE_TENSOR, "conv2_1_0");

    org_khronos_nn_extension_convolution_layer_2_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_2_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p1, VX_TYPE_TENSOR, "conv2_0_weights");

    org_khronos_nn_extension_convolution_layer_2_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_2_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p2, VX_TYPE_TENSOR, "conv2_0_bias");

    org_khronos_nn_extension_convolution_layer_2_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p3, VX_TYPE_SCALAR, "conv2_0_3");

    org_khronos_nn_extension_convolution_layer_2_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p4, VX_TYPE_SCALAR, "conv2_0_4");

    org_khronos_nn_extension_convolution_layer_2_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p5, VX_TYPE_SCALAR, "conv2_0_5");

    org_khronos_nn_extension_convolution_layer_2_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p6, VX_TYPE_SCALAR, "conv2_0_6");

    org_khronos_nn_extension_convolution_layer_2_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p7, VX_TYPE_SCALAR, "conv2_0_7");

    org_khronos_nn_extension_convolution_layer_2_p8 = vxCreateTensorFromView(outputAllocators_MergeTensor_0_p0, 4, org_khronos_nn_extension_convolution_layer_2_p8_view_view_start, org_khronos_nn_extension_convolution_layer_2_p8_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p8, VX_TYPE_TENSOR, "conv2_0_8");

    org_khronos_nn_extension_convolution_layer_1_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_1_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p1, VX_TYPE_TENSOR, "conv2_1_weights");

    org_khronos_nn_extension_convolution_layer_1_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_1_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p2, VX_TYPE_TENSOR, "conv2_1_bias");

    org_khronos_nn_extension_convolution_layer_1_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p3, VX_TYPE_SCALAR, "conv2_1_3");

    org_khronos_nn_extension_convolution_layer_1_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p4, VX_TYPE_SCALAR, "conv2_1_4");

    org_khronos_nn_extension_convolution_layer_1_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p5, VX_TYPE_SCALAR, "conv2_1_5");

    org_khronos_nn_extension_convolution_layer_1_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p6, VX_TYPE_SCALAR, "conv2_1_6");

    org_khronos_nn_extension_convolution_layer_1_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p7, VX_TYPE_SCALAR, "conv2_1_7");

    org_khronos_nn_extension_convolution_layer_1_p8 = vxCreateTensorFromView(outputAllocators_MergeTensor_0_p0, 4, org_khronos_nn_extension_convolution_layer_1_p8_view_view_start, org_khronos_nn_extension_convolution_layer_1_p8_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p8, VX_TYPE_TENSOR, "conv2_1_8");

    org_khronos_nn_extension_activation_layer_1_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_1_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1_p1, VX_TYPE_SCALAR, "relu2_1");

    org_khronos_nn_extension_activation_layer_1_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_1_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1_p2, VX_TYPE_SCALAR, "relu2_2");

    org_khronos_nn_extension_activation_layer_1_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1_p3, VX_TYPE_SCALAR, "relu2_2");

    org_khronos_nn_extension_activation_layer_1_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_1_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1_p4, VX_TYPE_TENSOR, "relu2_4");

    org_khronos_nn_extension_normalization_layer_1_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_normalization_layer_1_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p1, VX_TYPE_SCALAR, "norm2_1");

    org_khronos_nn_extension_normalization_layer_1_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_normalization_layer_1_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p2, VX_TYPE_SCALAR, "norm2_2");

    org_khronos_nn_extension_normalization_layer_1_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_normalization_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p3, VX_TYPE_SCALAR, "norm2_3");

    org_khronos_nn_extension_normalization_layer_1_p4 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_normalization_layer_1_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p4, VX_TYPE_SCALAR, "norm2_4");

    org_khronos_nn_extension_normalization_layer_1_p5 = vxCreateTensor(context, 4, org_khronos_nn_extension_normalization_layer_1_p5Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p5, VX_TYPE_TENSOR, "norm2_5");

    org_khronos_nn_extension_pooling_layer_1_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p1, VX_TYPE_SCALAR, "pool2_1");

    org_khronos_nn_extension_pooling_layer_1_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p2, VX_TYPE_SCALAR, "pool2_2");

    org_khronos_nn_extension_pooling_layer_1_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p3, VX_TYPE_SCALAR, "pool2_3");

    org_khronos_nn_extension_pooling_layer_1_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p4, VX_TYPE_SCALAR, "pool2_4");

    org_khronos_nn_extension_pooling_layer_1_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p5, VX_TYPE_SCALAR, "pool2_5");

    org_khronos_nn_extension_pooling_layer_1_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p6, VX_TYPE_SCALAR, "pool2_6");

    org_khronos_nn_extension_pooling_layer_1_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_1_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p7, VX_TYPE_TENSOR, "pool2_7");

    org_khronos_nn_extension_convolution_layer_3_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_3_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p1, VX_TYPE_TENSOR, "conv3_weights");

    org_khronos_nn_extension_convolution_layer_3_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_3_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p2, VX_TYPE_TENSOR, "conv3_bias");

    org_khronos_nn_extension_convolution_layer_3_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p3, VX_TYPE_SCALAR, "conv3_3");

    org_khronos_nn_extension_convolution_layer_3_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p4, VX_TYPE_SCALAR, "conv3_4");

    org_khronos_nn_extension_convolution_layer_3_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p5, VX_TYPE_SCALAR, "conv3_5");

    org_khronos_nn_extension_convolution_layer_3_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p6, VX_TYPE_SCALAR, "conv3_6");

    org_khronos_nn_extension_convolution_layer_3_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p7, VX_TYPE_SCALAR, "conv3_7");

    org_khronos_nn_extension_convolution_layer_3_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_3_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p8, VX_TYPE_TENSOR, "conv3_8");

    org_khronos_nn_extension_activation_layer_2_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_2_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_2_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_2_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2_p1, VX_TYPE_SCALAR, "relu3_1");

    org_khronos_nn_extension_activation_layer_2_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_2_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_2_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_2_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2_p2, VX_TYPE_SCALAR, "relu3_2");

    org_khronos_nn_extension_activation_layer_2_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_2_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_2_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_2_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2_p3, VX_TYPE_SCALAR, "relu3_2");

    org_khronos_nn_extension_activation_layer_2_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_2_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_2_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_2_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2_p4, VX_TYPE_TENSOR, "relu3_4");

    outputAllocators_SplitTensor_1_p1 = vxCreateTensorFromView(org_khronos_nn_extension_activation_layer_2_p4, 4, org_khronos_nn_extension_activation_layer_2_p4_view1_view_start, org_khronos_nn_extension_activation_layer_2_p4_view1_view_end);
    status = vxGetStatus((vx_reference)outputAllocators_SplitTensor_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_SplitTensor_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_SplitTensor_1_p1, VX_TYPE_TENSOR, "conv4_0_0");

    outputAllocators_SplitTensor_1_p2 = vxCreateTensorFromView(org_khronos_nn_extension_activation_layer_2_p4, 4, org_khronos_nn_extension_activation_layer_2_p4_view2_view_start, org_khronos_nn_extension_activation_layer_2_p4_view2_view_end);
    status = vxGetStatus((vx_reference)outputAllocators_SplitTensor_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_SplitTensor_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_SplitTensor_1_p2, VX_TYPE_TENSOR, "conv4_1_0");

    org_khronos_nn_extension_convolution_layer_5_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_5_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p1, VX_TYPE_TENSOR, "conv4_0_weights");

    org_khronos_nn_extension_convolution_layer_5_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_5_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p2, VX_TYPE_TENSOR, "conv4_0_bias");

    org_khronos_nn_extension_convolution_layer_5_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p3, VX_TYPE_SCALAR, "conv4_0_3");

    org_khronos_nn_extension_convolution_layer_5_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p4, VX_TYPE_SCALAR, "conv4_0_4");

    org_khronos_nn_extension_convolution_layer_5_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p5, VX_TYPE_SCALAR, "conv4_0_5");

    org_khronos_nn_extension_convolution_layer_5_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p6, VX_TYPE_SCALAR, "conv4_0_6");

    org_khronos_nn_extension_convolution_layer_5_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p7, VX_TYPE_SCALAR, "conv4_0_7");

    org_khronos_nn_extension_convolution_layer_5_p8 = vxCreateTensorFromView(outputAllocators_MergeTensor_1_p0, 4, org_khronos_nn_extension_convolution_layer_5_p8_view_view_start, org_khronos_nn_extension_convolution_layer_5_p8_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p8, VX_TYPE_TENSOR, "conv4_0_8");

    org_khronos_nn_extension_convolution_layer_4_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_4_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p1, VX_TYPE_TENSOR, "conv4_1_weights");

    org_khronos_nn_extension_convolution_layer_4_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_4_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p2, VX_TYPE_TENSOR, "conv4_1_bias");

    org_khronos_nn_extension_convolution_layer_4_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p3, VX_TYPE_SCALAR, "conv4_1_3");

    org_khronos_nn_extension_convolution_layer_4_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p4, VX_TYPE_SCALAR, "conv4_1_4");

    org_khronos_nn_extension_convolution_layer_4_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p5, VX_TYPE_SCALAR, "conv4_1_5");

    org_khronos_nn_extension_convolution_layer_4_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p6, VX_TYPE_SCALAR, "conv4_1_6");

    org_khronos_nn_extension_convolution_layer_4_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p7, VX_TYPE_SCALAR, "conv4_1_7");

    org_khronos_nn_extension_convolution_layer_4_p8 = vxCreateTensorFromView(outputAllocators_MergeTensor_1_p0, 4, org_khronos_nn_extension_convolution_layer_4_p8_view_view_start, org_khronos_nn_extension_convolution_layer_4_p8_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p8, VX_TYPE_TENSOR, "conv4_1_8");

    org_khronos_nn_extension_activation_layer_3_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_3_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_3_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_3_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3_p1, VX_TYPE_SCALAR, "relu4_1");

    org_khronos_nn_extension_activation_layer_3_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_3_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_3_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_3_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3_p2, VX_TYPE_SCALAR, "relu4_2");

    org_khronos_nn_extension_activation_layer_3_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_3_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_3_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_3_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3_p3, VX_TYPE_SCALAR, "relu4_2");

    org_khronos_nn_extension_activation_layer_3_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_3_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_3_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_3_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3_p4, VX_TYPE_TENSOR, "relu4_4");

    outputAllocators_SplitTensor_2_p1 = vxCreateTensorFromView(org_khronos_nn_extension_activation_layer_3_p4, 4, org_khronos_nn_extension_activation_layer_3_p4_view1_view_start, org_khronos_nn_extension_activation_layer_3_p4_view1_view_end);
    status = vxGetStatus((vx_reference)outputAllocators_SplitTensor_2_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_SplitTensor_2_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_SplitTensor_2_p1, VX_TYPE_TENSOR, "conv5_0_0");

    outputAllocators_SplitTensor_2_p2 = vxCreateTensorFromView(org_khronos_nn_extension_activation_layer_3_p4, 4, org_khronos_nn_extension_activation_layer_3_p4_view2_view_start, org_khronos_nn_extension_activation_layer_3_p4_view2_view_end);
    status = vxGetStatus((vx_reference)outputAllocators_SplitTensor_2_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_SplitTensor_2_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_SplitTensor_2_p2, VX_TYPE_TENSOR, "conv5_1_0");

    org_khronos_nn_extension_convolution_layer_7_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_7_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p1, VX_TYPE_TENSOR, "conv5_0_weights");

    org_khronos_nn_extension_convolution_layer_7_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_7_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p2, VX_TYPE_TENSOR, "conv5_0_bias");

    org_khronos_nn_extension_convolution_layer_7_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p3, VX_TYPE_SCALAR, "conv5_0_3");

    org_khronos_nn_extension_convolution_layer_7_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p4, VX_TYPE_SCALAR, "conv5_0_4");

    org_khronos_nn_extension_convolution_layer_7_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p5, VX_TYPE_SCALAR, "conv5_0_5");

    org_khronos_nn_extension_convolution_layer_7_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p6, VX_TYPE_SCALAR, "conv5_0_6");

    org_khronos_nn_extension_convolution_layer_7_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p7, VX_TYPE_SCALAR, "conv5_0_7");

    org_khronos_nn_extension_convolution_layer_7_p8 = vxCreateTensorFromView(outputAllocators_MergeTensor_2_p0, 4, org_khronos_nn_extension_convolution_layer_7_p8_view_view_start, org_khronos_nn_extension_convolution_layer_7_p8_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p8, VX_TYPE_TENSOR, "conv5_0_8");

    org_khronos_nn_extension_convolution_layer_6_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_6_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p1, VX_TYPE_TENSOR, "conv5_1_weights");

    org_khronos_nn_extension_convolution_layer_6_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_6_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p2, VX_TYPE_TENSOR, "conv5_1_bias");

    org_khronos_nn_extension_convolution_layer_6_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p3, VX_TYPE_SCALAR, "conv5_1_3");

    org_khronos_nn_extension_convolution_layer_6_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p4, VX_TYPE_SCALAR, "conv5_1_4");

    org_khronos_nn_extension_convolution_layer_6_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p5, VX_TYPE_SCALAR, "conv5_1_5");

    org_khronos_nn_extension_convolution_layer_6_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p6, VX_TYPE_SCALAR, "conv5_1_6");

    org_khronos_nn_extension_convolution_layer_6_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p7, VX_TYPE_SCALAR, "conv5_1_7");

    org_khronos_nn_extension_convolution_layer_6_p8 = vxCreateTensorFromView(outputAllocators_MergeTensor_2_p0, 4, org_khronos_nn_extension_convolution_layer_6_p8_view_view_start, org_khronos_nn_extension_convolution_layer_6_p8_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p8, VX_TYPE_TENSOR, "conv5_1_8");

    org_khronos_nn_extension_activation_layer_4_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_4_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_4_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_4_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4_p1, VX_TYPE_SCALAR, "relu5_1");

    org_khronos_nn_extension_activation_layer_4_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_4_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_4_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_4_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4_p2, VX_TYPE_SCALAR, "relu5_2");

    org_khronos_nn_extension_activation_layer_4_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_4_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_4_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_4_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4_p3, VX_TYPE_SCALAR, "relu5_2");

    org_khronos_nn_extension_activation_layer_4_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_4_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_4_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_4_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4_p4, VX_TYPE_TENSOR, "relu5_4");

    org_khronos_nn_extension_pooling_layer_2_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p1, VX_TYPE_SCALAR, "pool5_1");

    org_khronos_nn_extension_pooling_layer_2_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p2, VX_TYPE_SCALAR, "pool5_2");

    org_khronos_nn_extension_pooling_layer_2_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p3, VX_TYPE_SCALAR, "pool5_3");

    org_khronos_nn_extension_pooling_layer_2_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p4, VX_TYPE_SCALAR, "pool5_4");

    org_khronos_nn_extension_pooling_layer_2_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p5, VX_TYPE_SCALAR, "pool5_5");

    org_khronos_nn_extension_pooling_layer_2_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p6, VX_TYPE_SCALAR, "pool5_6");

    org_khronos_nn_extension_pooling_layer_2_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_2_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p7, VX_TYPE_TENSOR, "pool5_7");

    org_khronos_nn_extension_fully_connected_layer_0_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_fully_connected_layer_0_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p1, VX_TYPE_TENSOR, "fc6_weights");

    org_khronos_nn_extension_fully_connected_layer_0_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_fully_connected_layer_0_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p2, VX_TYPE_TENSOR, "fc6_bias");

    org_khronos_nn_extension_fully_connected_layer_0_p3 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_fully_connected_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p3, VX_TYPE_SCALAR, "fc6_3");

    org_khronos_nn_extension_fully_connected_layer_0_p4 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_fully_connected_layer_0_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p4, VX_TYPE_SCALAR, "fc6_4");

    org_khronos_nn_extension_fully_connected_layer_0_p5 = vxCreateTensor(context, 2, org_khronos_nn_extension_fully_connected_layer_0_p5Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p5, VX_TYPE_TENSOR, "fc6_5");

    org_khronos_nn_extension_activation_layer_5_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_5_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_5_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_5_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5_p1, VX_TYPE_SCALAR, "relu6_1");

    org_khronos_nn_extension_activation_layer_5_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_5_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_5_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_5_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5_p2, VX_TYPE_SCALAR, "relu6_2");

    org_khronos_nn_extension_activation_layer_5_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_5_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_5_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_5_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5_p3, VX_TYPE_SCALAR, "relu6_2");

    org_khronos_nn_extension_activation_layer_5_p4 = vxCreateTensor(context, 2, org_khronos_nn_extension_activation_layer_5_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_5_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_5_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5_p4, VX_TYPE_TENSOR, "relu6_4");

    org_khronos_nn_extension_fully_connected_layer_1_p1 = vxCreateTensor(context, 2, org_khronos_nn_extension_fully_connected_layer_1_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p1, VX_TYPE_TENSOR, "fc7_weights");

    org_khronos_nn_extension_fully_connected_layer_1_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_fully_connected_layer_1_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p2, VX_TYPE_TENSOR, "fc7_bias");

    org_khronos_nn_extension_fully_connected_layer_1_p3 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_fully_connected_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p3, VX_TYPE_SCALAR, "fc7_3");

    org_khronos_nn_extension_fully_connected_layer_1_p4 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_fully_connected_layer_1_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p4, VX_TYPE_SCALAR, "fc7_4");

    org_khronos_nn_extension_fully_connected_layer_1_p5 = vxCreateTensor(context, 2, org_khronos_nn_extension_fully_connected_layer_1_p5Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_1_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_1_p5, VX_TYPE_TENSOR, "fc7_5");

    org_khronos_nn_extension_activation_layer_6_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_6_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_6_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_6_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6_p1, VX_TYPE_SCALAR, "relu7_1");

    org_khronos_nn_extension_activation_layer_6_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_6_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_6_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_6_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6_p2, VX_TYPE_SCALAR, "relu7_2");

    org_khronos_nn_extension_activation_layer_6_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_6_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_6_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_6_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6_p3, VX_TYPE_SCALAR, "relu7_2");

    org_khronos_nn_extension_activation_layer_6_p4 = vxCreateTensor(context, 2, org_khronos_nn_extension_activation_layer_6_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_6_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_6_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6_p4, VX_TYPE_TENSOR, "relu7_4");

    org_khronos_nn_extension_fully_connected_layer_2_p1 = vxCreateTensor(context, 2, org_khronos_nn_extension_fully_connected_layer_2_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_2_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p1, VX_TYPE_TENSOR, "fc8_weights");

    org_khronos_nn_extension_fully_connected_layer_2_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_fully_connected_layer_2_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_2_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p2, VX_TYPE_TENSOR, "fc8_bias");

    org_khronos_nn_extension_fully_connected_layer_2_p3 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_fully_connected_layer_2_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_2_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p3, VX_TYPE_SCALAR, "fc8_3");

    org_khronos_nn_extension_fully_connected_layer_2_p4 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_fully_connected_layer_2_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_2_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p4, VX_TYPE_SCALAR, "fc8_4");

    org_khronos_nn_extension_fully_connected_layer_2_p5 = vxCreateTensor(context, 2, org_khronos_nn_extension_fully_connected_layer_2_p5Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_2_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_2_p5, VX_TYPE_TENSOR, "fc8_5");

    com_cnn_helpers_scalemddata_0_p1 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&com_cnn_helpers_scalemddata_0_scalar_p1);
    status = vxGetStatus((vx_reference)com_cnn_helpers_scalemddata_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter com_cnn_helpers_scalemddata_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)com_cnn_helpers_scalemddata_0_p1, VX_TYPE_SCALAR, "Power0_1");

    com_cnn_helpers_scalemddata_0_p2 = vxCreateTensor(context, 2, com_cnn_helpers_scalemddata_0_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)com_cnn_helpers_scalemddata_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter com_cnn_helpers_scalemddata_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)com_cnn_helpers_scalemddata_0_p2, VX_TYPE_TENSOR, "Power0_2");

    org_khronos_nn_extension_softmax_layer_0_p1 = vxCreateTensor(context, 2, org_khronos_nn_extension_softmax_layer_0_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_softmax_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_softmax_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_softmax_layer_0_p1, VX_TYPE_TENSOR, "cnn_output");


	//
	// Node Assignments
	//
	WriteLog("Adding graph nodes...\n");
	//status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_0);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_convolution_layer_0 = vxConvolutionLayer(graph, org_khronos_nn_extension_convolution_layer_0_p0, org_khronos_nn_extension_convolution_layer_0_p1,
		org_khronos_nn_extension_convolution_layer_0_p2, &conv_params[0], sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_0_p8);
	status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_0\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_0");

	//status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_0);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_activation_layer_0 = vxActivationLayer(graph, org_khronos_nn_extension_convolution_layer_0_p8, relu_params.function, relu_params.a, relu_params.b, org_khronos_nn_extension_activation_layer_0_p4);
	status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_activation_layer_0\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0, VX_TYPE_NODE, "org_khronos_nn_extension_activation_layer_0");


	//status = CreateNode(graph, org_khronos_nn_extension_normalization_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_normalization_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_normalization_layer_0);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_normalization_layer_0 = vxLocalResponseNormalizationLayer(graph, org_khronos_nn_extension_activation_layer_0_p4, norm_params.type, norm_params.normalization_size, norm_params.alpha, norm_params.beta, 1.0f,
		org_khronos_nn_extension_normalization_layer_0_p5);
	status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_normalization_layer_0\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0, VX_TYPE_NODE, "org_khronos_nn_extension_normalization_layer_0");


	//status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_0);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_pooling_layer_0 = vxPoolingLayer(graph, org_khronos_nn_extension_normalization_layer_0_p5, VX_NN_POOLING_MAX, pool_params.pooling_size_x, pool_params.pooling_size_y, pool_params.pooling_padding_x,
		pool_params.pooling_padding_y, pool_params.rounding, org_khronos_nn_extension_pooling_layer_0_p7);
	status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_pooling_layer_0\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0, VX_TYPE_NODE, "org_khronos_nn_extension_pooling_layer_0");

	//status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_2", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_2);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_convolution_layer_2 = vxConvolutionLayer(graph, outputAllocators_SplitTensor_0_p1, org_khronos_nn_extension_convolution_layer_2_p1, org_khronos_nn_extension_convolution_layer_2_p2,
		&conv_params[1], sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_2_p8);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_2\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_2");


	//status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_1);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_convolution_layer_1 = vxConvolutionLayer(graph, outputAllocators_SplitTensor_0_p2, org_khronos_nn_extension_convolution_layer_1_p1, org_khronos_nn_extension_convolution_layer_1_p2,
		&conv_params[1], sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_1_p8);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_1\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_1");


	//status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_1);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_activation_layer_1 = vxActivationLayer(graph, outputAllocators_MergeTensor_0_p0, relu_params.function, relu_params.a, relu_params.b, org_khronos_nn_extension_activation_layer_1_p4);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_activation_layer_1\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1, VX_TYPE_NODE, "org_khronos_nn_extension_activation_layer_1");


	//status = CreateNode(graph, org_khronos_nn_extension_normalization_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_normalization_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_normalization_layer_1);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_normalization_layer_1 = vxLocalResponseNormalizationLayer(graph, org_khronos_nn_extension_activation_layer_1_p4, norm_params.type, norm_params.normalization_size, norm_params.alpha,
		norm_params.beta, 1.0f, org_khronos_nn_extension_normalization_layer_1_p5);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_normalization_layer_1\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1, VX_TYPE_NODE, "org_khronos_nn_extension_normalization_layer_1");

	//status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_1);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_pooling_layer_1 = vxPoolingLayer(graph, org_khronos_nn_extension_normalization_layer_1_p5, VX_NN_POOLING_MAX, pool_params.pooling_size_x, pool_params.pooling_size_y, pool_params.pooling_padding_x,
		pool_params.pooling_padding_y, pool_params.rounding, org_khronos_nn_extension_pooling_layer_1_p7);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_pooling_layer_1\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1, VX_TYPE_NODE, "org_khronos_nn_extension_pooling_layer_1");


	//status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_3", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_3);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_convolution_layer_3 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_1_p7, org_khronos_nn_extension_convolution_layer_3_p1, org_khronos_nn_extension_convolution_layer_3_p2,
		&conv_params[2], sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_3_p8);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_3\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_3");


	//status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_2", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_2);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_activation_layer_2 = vxActivationLayer(graph, org_khronos_nn_extension_convolution_layer_3_p8, relu_params.function, relu_params.a, relu_params.b, org_khronos_nn_extension_activation_layer_2_p4);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_activation_layer_2\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2, VX_TYPE_NODE, "org_khronos_nn_extension_activation_layer_2");

	//status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_5", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_5);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_convolution_layer_5 = vxConvolutionLayer(graph, outputAllocators_SplitTensor_1_p1, org_khronos_nn_extension_convolution_layer_5_p1, org_khronos_nn_extension_convolution_layer_5_p2,
		&conv_params[2], sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_5_p8);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_5\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_5");


	//status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_4", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_4);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_convolution_layer_4 = vxConvolutionLayer(graph, outputAllocators_SplitTensor_1_p2, org_khronos_nn_extension_convolution_layer_4_p1, org_khronos_nn_extension_convolution_layer_4_p2,
		&conv_params[2], sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_4_p8);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_4\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_4");

	//status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_3", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_3);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_activation_layer_3 = vxActivationLayer(graph, outputAllocators_MergeTensor_1_p0, relu_params.function, relu_params.a, relu_params.b, org_khronos_nn_extension_activation_layer_3_p4);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_activation_layer_3\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3, VX_TYPE_NODE, "org_khronos_nn_extension_activation_layer_3");

	//status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_7", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_7);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_convolution_layer_7 = vxConvolutionLayer(graph, outputAllocators_SplitTensor_2_p1, org_khronos_nn_extension_convolution_layer_7_p1, org_khronos_nn_extension_convolution_layer_7_p2,
		&conv_params[2], sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_7_p8);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_7\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_7");


	//status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_6", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_6);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_convolution_layer_6 = vxConvolutionLayer(graph, outputAllocators_SplitTensor_2_p2, org_khronos_nn_extension_convolution_layer_6_p1, org_khronos_nn_extension_convolution_layer_6_p2,
		&conv_params[2], sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_6_p8);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_6\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_6");

	//status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_4", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_4);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_activation_layer_4 = vxActivationLayer(graph, outputAllocators_MergeTensor_2_p0, relu_params.function, relu_params.a, relu_params.b, org_khronos_nn_extension_activation_layer_4_p4);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_activation_layer_4\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4, VX_TYPE_NODE, "org_khronos_nn_extension_activation_layer_4");


	//status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_2", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_2);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_pooling_layer_2 = vxPoolingLayer(graph, org_khronos_nn_extension_activation_layer_4_p4, VX_NN_POOLING_MAX, pool_params.pooling_size_x, pool_params.pooling_size_y, pool_params.pooling_padding_x,
		pool_params.pooling_padding_y, pool_params.rounding, org_khronos_nn_extension_pooling_layer_2_p7);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_pooling_layer_2\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2, VX_TYPE_NODE, "org_khronos_nn_extension_pooling_layer_2");

	//status = CreateNode(graph, org_khronos_nn_extension_fully_connected_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_fully_connected_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_fully_connected_layer_0);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_fully_connected_layer_0 = vxFullyConnectedLayer(graph, org_khronos_nn_extension_pooling_layer_2_p7, org_khronos_nn_extension_fully_connected_layer_0_p1,
		org_khronos_nn_extension_fully_connected_layer_0_p2, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, org_khronos_nn_extension_fully_connected_layer_0_p5);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_fully_connected_layer_0\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0, VX_TYPE_NODE, "org_khronos_nn_extension_fully_connected_layer_0");

	//status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_5", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_5);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_activation_layer_5 = vxActivationLayer(graph, org_khronos_nn_extension_fully_connected_layer_0_p5, relu_params.function, relu_params.a, relu_params.b, org_khronos_nn_extension_activation_layer_5_p4);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_activation_layer_5\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5, VX_TYPE_NODE, "org_khronos_nn_extension_activation_layer_5");

	//status = CreateNode(graph, org_khronos_nn_extension_fully_connected_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_fully_connected_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_fully_connected_layer_1);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_fully_connected_layer_1 = vxFullyConnectedLayer(graph, org_khronos_nn_extension_activation_layer_5_p4, org_khronos_nn_extension_fully_connected_layer_1_p1,
		org_khronos_nn_extension_fully_connected_layer_1_p2, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, org_khronos_nn_extension_fully_connected_layer_1_p5);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_fully_connected_layer_1\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_1, VX_TYPE_NODE, "org_khronos_nn_extension_fully_connected_layer_1");

	//status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_6", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_6);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_activation_layer_6 = vxActivationLayer(graph, org_khronos_nn_extension_fully_connected_layer_1_p5, relu_params.function, relu_params.a, relu_params.b, org_khronos_nn_extension_activation_layer_6_p4);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_activation_layer_5\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6, VX_TYPE_NODE, "org_khronos_nn_extension_activation_layer_6");

	//status = CreateNode(graph, org_khronos_nn_extension_fully_connected_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_fully_connected_layer_2", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_fully_connected_layer_2);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_fully_connected_layer_2 = vxFullyConnectedLayer(graph, org_khronos_nn_extension_activation_layer_6_p4, org_khronos_nn_extension_fully_connected_layer_2_p1,
		org_khronos_nn_extension_fully_connected_layer_2_p2, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, org_khronos_nn_extension_fully_connected_layer_2_p5);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_fully_connected_layer_2\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_2, VX_TYPE_NODE, "org_khronos_nn_extension_fully_connected_layer_2");


	{

		const vx_size unit_tensor_view_start[2] = { 0, 0 };
		const vx_size unit_tensor_dims[2] = { 1000, 1 };
		const vx_size unit_tensor_strides[2] = { sizeof(vx_int16), sizeof(vx_int16) * 1000 };
		vx_int16 unit_tensor_data[1000];
		for (int i = 0; i < 1000; ++i) unit_tensor_data[i] = 1 << 8; // 1 in Q78

		vx_tensor unit_tensor = vxCreateTensor(context, 2, unit_tensor_dims, VX_TYPE_INT16, 8);
		status = vxGetStatus((vx_reference)unit_tensor);
		if (status != VX_SUCCESS) return status;

		status = vxCopyTensorPatch(unit_tensor, 2, unit_tensor_view_start, unit_tensor_dims, unit_tensor_strides, &unit_tensor_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
		if (status != VX_SUCCESS)
		{
			vxReleaseTensor(&unit_tensor);
			return status;
		}

		vx_node node = vxTensorMultiplyNode(
			graph,
			org_khronos_nn_extension_fully_connected_layer_2_p5,
			unit_tensor,
			com_cnn_helpers_scalemddata_0_p1,
			VX_CONVERT_POLICY_SATURATE,
			VX_ROUND_POLICY_TO_ZERO,
			com_cnn_helpers_scalemddata_0_p2);
		status = vxGetStatus((vx_reference)node);
		if (status != VX_SUCCESS) return status;

		status = vxReleaseNode(&node);
		if (status != VX_SUCCESS) return status;

		status = vxReleaseTensor(&unit_tensor);
		if (status != VX_SUCCESS) return status;
	}

	//status = CreateNode(graph, org_khronos_nn_extension_softmax_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_softmax_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_softmax_layer_0);
	//if(status != VX_SUCCESS)
	//    return status;
	org_khronos_nn_extension_softmax_layer_0 = vxSoftmaxLayer(graph, com_cnn_helpers_scalemddata_0_p2, org_khronos_nn_extension_softmax_layer_0_p1);
	if (status != VX_SUCCESS)
	{
		WriteLog("ERROR: failed to create node org_khronos_nn_extension_softmax_layer_0\n");
		return status;
	}
	AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_softmax_layer_0, VX_TYPE_NODE, "org_khronos_nn_extension_softmax_layer_0");



    return status;
}

#endif//OPENVX_CONFORMANCE_NEURAL_NETWORKS
