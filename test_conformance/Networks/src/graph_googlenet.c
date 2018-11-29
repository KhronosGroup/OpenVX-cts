/** @file graph.h
 *  @brief 
 *  This file contains the implementation of the generated graph factory function
 */

#ifdef OPENVX_USE_NN

#include <stdio.h>
#include <stdlib.h>
#include "graph.h"
#include <VX/vx_khr_nn.h>



/** @brief Constructs OpenVX graph and connects to the input/output references
 *
 *  @param context The OpenVX context
 *  @param graph The OpenVX graph
 *  @param pObjectContainer The pointer to object container.
 *  @param filteredNodesList The list of filtered nodes to create in the graph (can be empty)
 *  @param filteredNodesCount The number of filtered nodes to create in the graph
 *  @return vx_status code.
 */
static vx_status Graph(vx_context context, vx_graph graph, ObjectRefContainerType* pObjectContainer, char* filteredNodeList[], size_t filteredNodeCount, vx_tensor org_khronos_nn_extension_convolution_layer_0_p0, vx_tensor org_khronos_nn_extension_convolution_layer_0_p1, vx_tensor org_khronos_nn_extension_convolution_layer_0_p2, vx_nn_convolution_params_t org_khronos_nn_extension_convolution_layer_0_p3, vx_tensor org_khronos_nn_extension_convolution_layer_0_p8);
//static vx_status Graph(vx_context context, vx_graph graph, ObjectRefContainerType* pObjectContainer, char* filteredNodeList[], size_t filteredNodeCount, vx_tensor org_khronos_nn_extension_convolution_layer_0_p0, vx_tensor org_khronos_nn_extension_convolution_layer_0_p1, vx_tensor org_khronos_nn_extension_convolution_layer_0_p2, vx_scalar org_khronos_nn_extension_convolution_layer_0_p3, vx_scalar org_khronos_nn_extension_convolution_layer_0_p4, vx_scalar org_khronos_nn_extension_convolution_layer_0_p5, vx_scalar org_khronos_nn_extension_convolution_layer_0_p6, vx_scalar org_khronos_nn_extension_convolution_layer_0_p7, vx_tensor org_khronos_nn_extension_convolution_layer_0_p8);

/** @brief Implements the OpenVX graph factory
 *
 *  @param context The OpenVX context
 *  @param graph The OpenVX graph
 *  @param pObjectContainer The pointer to object container.
 *  @param filteredNodesList The list of filtered nodes to create in the graph (can be empty)
 *  @param filteredNodesCount The number of filtered nodes to create in the graph
 *  @return vx_status code.
 */
vx_status _GraphFactoryGooglenet(vx_context context, vx_graph graph, ObjectRefContainerType* pObjectContainer, char* filteredNodeList[], size_t filteredNodeCount)
{
    vx_status status = VX_SUCCESS;

    //
    // Primitive Declarations
    //

    vx_tensor org_khronos_nn_extension_convolution_layer_0_p0;
    vx_tensor org_khronos_nn_extension_convolution_layer_0_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_0_p2;
    vx_tensor org_khronos_nn_extension_convolution_layer_0_p8;

    //
    // Other Declarations
    //

    vx_size org_khronos_nn_extension_convolution_layer_0_p0Dimensions[4] = {224,224,3,1};
    vx_size org_khronos_nn_extension_convolution_layer_0_p1Dimensions[4] = {7,7,3,64};
    vx_size org_khronos_nn_extension_convolution_layer_0_p2Dimensions[1] = {64};
//    vx_size org_khronos_nn_extension_convolution_layer_0_scalar_p3 = 3;
//    vx_size org_khronos_nn_extension_convolution_layer_0_scalar_p4 = 3;
//    vx_enum org_khronos_nn_extension_convolution_layer_0_scalar_p5 = VX_CONVERT_POLICY_WRAP;
//    vx_enum org_khronos_nn_extension_convolution_layer_0_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
//    vx_enum org_khronos_nn_extension_convolution_layer_0_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_0_p8Dimensions[4] = {112,112,64,1};
    vx_nn_convolution_params_t org_khronos_nn_extension_convolution_layer_0_p3 = {3,3,VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0};

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
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p1, VX_TYPE_TENSOR, "conv1_7x7_s2_weights");

    org_khronos_nn_extension_convolution_layer_0_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_0_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p2);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p2, VX_TYPE_TENSOR, "conv1_7x7_s2_bias");

    org_khronos_nn_extension_convolution_layer_0_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_0_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0_p8);
    if(status != VX_SUCCESS)
    {
		WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_0_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p8, VX_TYPE_TENSOR, "conv1_7x7_s2_8");


    //
    // All nodes and primitives (except primitives associated with source nodes) of the graph are setup in Graph()
    //

    status = Graph(context, graph, pObjectContainer, filteredNodeList, filteredNodeCount, org_khronos_nn_extension_convolution_layer_0_p0, org_khronos_nn_extension_convolution_layer_0_p1, org_khronos_nn_extension_convolution_layer_0_p2, org_khronos_nn_extension_convolution_layer_0_p3, org_khronos_nn_extension_convolution_layer_0_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create graph workload [Graph()]\n");
        return VX_FAILURE;
    }

    return status;
}

static vx_status Graph(vx_context context, vx_graph graph, ObjectRefContainerType* pObjectContainer, char* filteredNodeList[], size_t filteredNodeCount, vx_tensor org_khronos_nn_extension_convolution_layer_0_p0, vx_tensor org_khronos_nn_extension_convolution_layer_0_p1, vx_tensor org_khronos_nn_extension_convolution_layer_0_p2, vx_nn_convolution_params_t org_khronos_nn_extension_convolution_layer_0_p3, vx_tensor org_khronos_nn_extension_convolution_layer_0_p8)
{
    vx_status status = VX_SUCCESS;    

    //
    // Kernel Declarations
    //

    vx_kernel org_khronos_nn_extension_convolution_layer_Kernel;
    vx_kernel org_khronos_nn_extension_activation_layer_Kernel;
    vx_kernel org_khronos_nn_extension_pooling_layer_Kernel;
    vx_kernel org_khronos_nn_extension_normalization_layer_Kernel;
    vx_kernel org_khronos_nn_extension_fully_connected_layer_Kernel;
    vx_kernel org_khronos_openvx_tensor_multiply_Kernel;
    vx_kernel org_khronos_nn_extension_softmax_layer_Kernel;

    //
    // Node Declarations
    //

    vx_node org_khronos_nn_extension_convolution_layer_0;
    vx_node org_khronos_nn_extension_activation_layer_0;
    vx_node org_khronos_nn_extension_pooling_layer_0;
    vx_node org_khronos_nn_extension_normalization_layer_0;
    vx_node org_khronos_nn_extension_convolution_layer_1;
    vx_node org_khronos_nn_extension_activation_layer_1;
    vx_node org_khronos_nn_extension_convolution_layer_2;
    vx_node org_khronos_nn_extension_activation_layer_2;
    vx_node org_khronos_nn_extension_normalization_layer_1;
    vx_node org_khronos_nn_extension_pooling_layer_1;
    vx_node org_khronos_nn_extension_convolution_layer_8;
    vx_node org_khronos_nn_extension_convolution_layer_6;
    vx_node org_khronos_nn_extension_convolution_layer_4;
    vx_node org_khronos_nn_extension_pooling_layer_2;
    vx_node org_khronos_nn_extension_activation_layer_8;
    vx_node org_khronos_nn_extension_activation_layer_6;
    vx_node org_khronos_nn_extension_activation_layer_4;
    vx_node org_khronos_nn_extension_convolution_layer_3;
    vx_node org_khronos_nn_extension_convolution_layer_7;
    vx_node org_khronos_nn_extension_convolution_layer_5;
    vx_node org_khronos_nn_extension_activation_layer_3;
    vx_node org_khronos_nn_extension_activation_layer_7;
    vx_node org_khronos_nn_extension_activation_layer_5;
    vx_node org_khronos_nn_extension_convolution_layer_14;
    vx_node org_khronos_nn_extension_convolution_layer_12;
    vx_node org_khronos_nn_extension_convolution_layer_10;
    vx_node org_khronos_nn_extension_pooling_layer_3;
    vx_node org_khronos_nn_extension_activation_layer_14;
    vx_node org_khronos_nn_extension_activation_layer_12;
    vx_node org_khronos_nn_extension_activation_layer_10;
    vx_node org_khronos_nn_extension_convolution_layer_9;
    vx_node org_khronos_nn_extension_convolution_layer_13;
    vx_node org_khronos_nn_extension_convolution_layer_11;
    vx_node org_khronos_nn_extension_activation_layer_9;
    vx_node org_khronos_nn_extension_activation_layer_13;
    vx_node org_khronos_nn_extension_activation_layer_11;
    vx_node org_khronos_nn_extension_pooling_layer_4;
    vx_node org_khronos_nn_extension_convolution_layer_20;
    vx_node org_khronos_nn_extension_convolution_layer_18;
    vx_node org_khronos_nn_extension_convolution_layer_16;
    vx_node org_khronos_nn_extension_pooling_layer_5;
    vx_node org_khronos_nn_extension_activation_layer_20;
    vx_node org_khronos_nn_extension_activation_layer_18;
    vx_node org_khronos_nn_extension_activation_layer_16;
    vx_node org_khronos_nn_extension_convolution_layer_15;
    vx_node org_khronos_nn_extension_convolution_layer_19;
    vx_node org_khronos_nn_extension_convolution_layer_17;
    vx_node org_khronos_nn_extension_activation_layer_15;
    vx_node org_khronos_nn_extension_activation_layer_19;
    vx_node org_khronos_nn_extension_activation_layer_17;
    vx_node org_khronos_nn_extension_convolution_layer_26;
    vx_node org_khronos_nn_extension_convolution_layer_24;
    vx_node org_khronos_nn_extension_convolution_layer_22;
    vx_node org_khronos_nn_extension_pooling_layer_6;
    vx_node org_khronos_nn_extension_activation_layer_26;
    vx_node org_khronos_nn_extension_activation_layer_24;
    vx_node org_khronos_nn_extension_activation_layer_22;
    vx_node org_khronos_nn_extension_convolution_layer_21;
    vx_node org_khronos_nn_extension_convolution_layer_25;
    vx_node org_khronos_nn_extension_convolution_layer_23;
    vx_node org_khronos_nn_extension_activation_layer_21;
    vx_node org_khronos_nn_extension_activation_layer_25;
    vx_node org_khronos_nn_extension_activation_layer_23;
    vx_node org_khronos_nn_extension_convolution_layer_32;
    vx_node org_khronos_nn_extension_convolution_layer_30;
    vx_node org_khronos_nn_extension_convolution_layer_28;
    vx_node org_khronos_nn_extension_pooling_layer_7;
    vx_node org_khronos_nn_extension_activation_layer_32;
    vx_node org_khronos_nn_extension_activation_layer_30;
    vx_node org_khronos_nn_extension_activation_layer_28;
    vx_node org_khronos_nn_extension_convolution_layer_27;
    vx_node org_khronos_nn_extension_convolution_layer_31;
    vx_node org_khronos_nn_extension_convolution_layer_29;
    vx_node org_khronos_nn_extension_activation_layer_27;
    vx_node org_khronos_nn_extension_activation_layer_31;
    vx_node org_khronos_nn_extension_activation_layer_29;
    vx_node org_khronos_nn_extension_convolution_layer_38;
    vx_node org_khronos_nn_extension_convolution_layer_36;
    vx_node org_khronos_nn_extension_convolution_layer_34;
    vx_node org_khronos_nn_extension_pooling_layer_8;
    vx_node org_khronos_nn_extension_activation_layer_38;
    vx_node org_khronos_nn_extension_activation_layer_36;
    vx_node org_khronos_nn_extension_activation_layer_34;
    vx_node org_khronos_nn_extension_convolution_layer_33;
    vx_node org_khronos_nn_extension_convolution_layer_37;
    vx_node org_khronos_nn_extension_convolution_layer_35;
    vx_node org_khronos_nn_extension_activation_layer_33;
    vx_node org_khronos_nn_extension_activation_layer_37;
    vx_node org_khronos_nn_extension_activation_layer_35;
    vx_node org_khronos_nn_extension_convolution_layer_44;
    vx_node org_khronos_nn_extension_convolution_layer_42;
    vx_node org_khronos_nn_extension_convolution_layer_40;
    vx_node org_khronos_nn_extension_pooling_layer_9;
    vx_node org_khronos_nn_extension_activation_layer_44;
    vx_node org_khronos_nn_extension_activation_layer_42;
    vx_node org_khronos_nn_extension_activation_layer_40;
    vx_node org_khronos_nn_extension_convolution_layer_39;
    vx_node org_khronos_nn_extension_convolution_layer_43;
    vx_node org_khronos_nn_extension_convolution_layer_41;
    vx_node org_khronos_nn_extension_activation_layer_39;
    vx_node org_khronos_nn_extension_activation_layer_43;
    vx_node org_khronos_nn_extension_activation_layer_41;
    vx_node org_khronos_nn_extension_pooling_layer_10;
    vx_node org_khronos_nn_extension_convolution_layer_50;
    vx_node org_khronos_nn_extension_convolution_layer_48;
    vx_node org_khronos_nn_extension_convolution_layer_46;
    vx_node org_khronos_nn_extension_pooling_layer_11;
    vx_node org_khronos_nn_extension_activation_layer_50;
    vx_node org_khronos_nn_extension_activation_layer_48;
    vx_node org_khronos_nn_extension_activation_layer_46;
    vx_node org_khronos_nn_extension_convolution_layer_45;
    vx_node org_khronos_nn_extension_convolution_layer_49;
    vx_node org_khronos_nn_extension_convolution_layer_47;
    vx_node org_khronos_nn_extension_activation_layer_45;
    vx_node org_khronos_nn_extension_activation_layer_49;
    vx_node org_khronos_nn_extension_activation_layer_47;
    vx_node org_khronos_nn_extension_convolution_layer_56;
    vx_node org_khronos_nn_extension_convolution_layer_54;
    vx_node org_khronos_nn_extension_convolution_layer_52;
    vx_node org_khronos_nn_extension_pooling_layer_12;
    vx_node org_khronos_nn_extension_activation_layer_56;
    vx_node org_khronos_nn_extension_activation_layer_54;
    vx_node org_khronos_nn_extension_activation_layer_52;
    vx_node org_khronos_nn_extension_convolution_layer_51;
    vx_node org_khronos_nn_extension_convolution_layer_55;
    vx_node org_khronos_nn_extension_convolution_layer_53;
    vx_node org_khronos_nn_extension_activation_layer_51;
    vx_node org_khronos_nn_extension_activation_layer_55;
    vx_node org_khronos_nn_extension_activation_layer_53;
    vx_node org_khronos_nn_extension_pooling_layer_13;
    vx_node org_khronos_nn_extension_fully_connected_layer_0;
    vx_node org_khronos_openvx_tensor_multiply_0;
    vx_node org_khronos_nn_extension_softmax_layer_0;

    //
    // Primitive Declarations
    //

    vx_tensor outputAllocators_MergeTensor_8_p0;
    vx_tensor outputAllocators_MergeTensor_7_p0;
    vx_tensor outputAllocators_MergeTensor_6_p0;
    vx_tensor outputAllocators_MergeTensor_5_p0;
    vx_tensor outputAllocators_MergeTensor_4_p0;
    vx_tensor outputAllocators_MergeTensor_3_p0;
    vx_tensor outputAllocators_MergeTensor_2_p0;
    vx_tensor outputAllocators_MergeTensor_1_p0;
    vx_tensor outputAllocators_MergeTensor_0_p0;
    vx_scalar org_khronos_nn_extension_activation_layer_0_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_0_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_0_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_0_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_0_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_0_p7;
    vx_scalar org_khronos_nn_extension_normalization_layer_0_p1;
    vx_scalar org_khronos_nn_extension_normalization_layer_0_p2;
    vx_scalar org_khronos_nn_extension_normalization_layer_0_p3;
    vx_scalar org_khronos_nn_extension_normalization_layer_0_p4;
    vx_tensor org_khronos_nn_extension_normalization_layer_0_p5;
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
    vx_tensor org_khronos_nn_extension_convolution_layer_2_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_2_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_2_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_2_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_2_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_2_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_2_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_2_p4;
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
    vx_tensor org_khronos_nn_extension_convolution_layer_8_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_8_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_8_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_8_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_8_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_8_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_8_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_8_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_6_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_6_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_6_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_6_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_4_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_4_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_4_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_4_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_2_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_2_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_8_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_8_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_8_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_8_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_6_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_6_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_6_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_6_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_4_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_4_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_4_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_4_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_3_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_3_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_3_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_3_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_7_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_7_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_7_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_7_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_5_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_5_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_5_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_5_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_3_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_3_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_3_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_3_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_7_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_7_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_7_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_7_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_5_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_5_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_5_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_5_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_14_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_14_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_14_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_14_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_14_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_14_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_14_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_14_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_12_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_12_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_12_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_12_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_12_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_12_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_12_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_12_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_10_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_10_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_10_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_10_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_10_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_10_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_10_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_10_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_3_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_3_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_3_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_3_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_3_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_3_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_3_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_14_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_14_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_14_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_14_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_12_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_12_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_12_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_12_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_10_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_10_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_10_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_10_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_9_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_9_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_9_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_9_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_9_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_9_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_9_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_9_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_13_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_13_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_13_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_13_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_13_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_13_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_13_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_13_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_11_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_11_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_11_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_11_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_11_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_11_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_11_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_11_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_9_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_9_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_9_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_9_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_13_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_13_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_13_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_13_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_11_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_11_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_11_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_11_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_4_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_4_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_4_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_4_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_4_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_4_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_4_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_20_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_20_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_20_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_20_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_20_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_20_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_20_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_20_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_18_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_18_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_18_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_18_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_18_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_18_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_18_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_18_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_16_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_16_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_16_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_16_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_16_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_16_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_16_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_16_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_5_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_5_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_5_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_5_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_5_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_5_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_5_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_20_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_20_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_20_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_20_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_18_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_18_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_18_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_18_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_16_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_16_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_16_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_16_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_15_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_15_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_15_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_15_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_15_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_15_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_15_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_15_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_19_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_19_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_19_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_19_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_19_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_19_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_19_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_19_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_17_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_17_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_17_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_17_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_17_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_17_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_17_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_17_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_15_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_15_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_15_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_15_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_19_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_19_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_19_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_19_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_17_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_17_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_17_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_17_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_26_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_26_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_26_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_26_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_26_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_26_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_26_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_26_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_24_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_24_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_24_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_24_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_24_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_24_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_24_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_24_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_22_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_22_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_22_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_22_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_22_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_22_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_22_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_22_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_6_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_6_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_6_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_6_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_6_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_6_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_6_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_26_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_26_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_26_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_26_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_24_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_24_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_24_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_24_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_22_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_22_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_22_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_22_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_21_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_21_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_21_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_21_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_21_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_21_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_21_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_21_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_25_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_25_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_25_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_25_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_25_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_25_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_25_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_25_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_23_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_23_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_23_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_23_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_23_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_23_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_23_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_23_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_21_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_21_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_21_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_21_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_25_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_25_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_25_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_25_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_23_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_23_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_23_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_23_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_32_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_32_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_32_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_32_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_32_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_32_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_32_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_32_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_30_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_30_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_30_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_30_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_30_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_30_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_30_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_30_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_28_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_28_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_28_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_28_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_28_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_28_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_28_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_28_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_7_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_7_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_7_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_7_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_7_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_7_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_7_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_32_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_32_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_32_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_32_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_30_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_30_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_30_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_30_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_28_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_28_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_28_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_28_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_27_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_27_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_27_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_27_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_27_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_27_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_27_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_27_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_31_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_31_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_31_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_31_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_31_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_31_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_31_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_31_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_29_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_29_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_29_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_29_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_29_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_29_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_29_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_29_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_27_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_27_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_27_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_27_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_31_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_31_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_31_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_31_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_29_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_29_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_29_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_29_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_38_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_38_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_38_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_38_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_38_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_38_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_38_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_38_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_36_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_36_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_36_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_36_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_36_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_36_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_36_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_36_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_34_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_34_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_34_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_34_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_34_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_34_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_34_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_34_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_8_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_8_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_8_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_8_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_8_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_8_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_8_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_38_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_38_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_38_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_38_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_36_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_36_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_36_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_36_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_34_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_34_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_34_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_34_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_33_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_33_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_33_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_33_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_33_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_33_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_33_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_33_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_37_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_37_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_37_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_37_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_37_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_37_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_37_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_37_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_35_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_35_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_35_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_35_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_35_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_35_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_35_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_35_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_33_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_33_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_33_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_33_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_37_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_37_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_37_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_37_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_35_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_35_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_35_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_35_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_44_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_44_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_44_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_44_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_44_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_44_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_44_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_44_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_42_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_42_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_42_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_42_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_42_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_42_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_42_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_42_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_40_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_40_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_40_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_40_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_40_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_40_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_40_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_40_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_9_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_9_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_9_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_9_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_9_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_9_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_9_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_44_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_44_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_44_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_44_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_42_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_42_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_42_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_42_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_40_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_40_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_40_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_40_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_39_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_39_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_39_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_39_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_39_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_39_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_39_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_39_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_43_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_43_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_43_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_43_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_43_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_43_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_43_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_43_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_41_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_41_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_41_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_41_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_41_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_41_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_41_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_41_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_39_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_39_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_39_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_39_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_43_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_43_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_43_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_43_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_41_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_41_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_41_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_41_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_10_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_10_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_10_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_10_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_10_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_10_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_10_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_50_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_50_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_50_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_50_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_50_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_50_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_50_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_50_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_48_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_48_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_48_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_48_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_48_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_48_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_48_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_48_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_46_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_46_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_46_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_46_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_46_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_46_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_46_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_46_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_11_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_11_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_11_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_11_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_11_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_11_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_11_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_50_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_50_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_50_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_50_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_48_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_48_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_48_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_48_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_46_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_46_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_46_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_46_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_45_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_45_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_45_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_45_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_45_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_45_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_45_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_45_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_49_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_49_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_49_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_49_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_49_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_49_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_49_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_49_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_47_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_47_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_47_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_47_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_47_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_47_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_47_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_47_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_45_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_45_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_45_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_45_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_49_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_49_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_49_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_49_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_47_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_47_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_47_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_47_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_56_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_56_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_56_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_56_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_56_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_56_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_56_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_56_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_54_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_54_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_54_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_54_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_54_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_54_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_54_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_54_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_52_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_52_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_52_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_52_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_52_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_52_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_52_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_52_p8;
    vx_scalar org_khronos_nn_extension_pooling_layer_12_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_12_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_12_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_12_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_12_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_12_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_12_p7;
    vx_scalar org_khronos_nn_extension_activation_layer_56_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_56_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_56_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_56_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_54_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_54_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_54_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_54_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_52_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_52_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_52_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_52_p4;
    vx_tensor org_khronos_nn_extension_convolution_layer_51_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_51_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_51_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_51_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_51_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_51_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_51_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_51_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_55_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_55_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_55_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_55_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_55_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_55_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_55_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_55_p8;
    vx_tensor org_khronos_nn_extension_convolution_layer_53_p1;
    vx_tensor org_khronos_nn_extension_convolution_layer_53_p2;
    vx_scalar org_khronos_nn_extension_convolution_layer_53_p3;
    vx_scalar org_khronos_nn_extension_convolution_layer_53_p4;
    vx_scalar org_khronos_nn_extension_convolution_layer_53_p5;
    vx_scalar org_khronos_nn_extension_convolution_layer_53_p6;
    vx_scalar org_khronos_nn_extension_convolution_layer_53_p7;
    vx_tensor org_khronos_nn_extension_convolution_layer_53_p8;
    vx_scalar org_khronos_nn_extension_activation_layer_51_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_51_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_51_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_51_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_55_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_55_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_55_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_55_p4;
    vx_scalar org_khronos_nn_extension_activation_layer_53_p1;
    vx_scalar org_khronos_nn_extension_activation_layer_53_p2;
    vx_scalar org_khronos_nn_extension_activation_layer_53_p3;
    vx_tensor org_khronos_nn_extension_activation_layer_53_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_13_p1;
    vx_scalar org_khronos_nn_extension_pooling_layer_13_p2;
    vx_scalar org_khronos_nn_extension_pooling_layer_13_p3;
    vx_scalar org_khronos_nn_extension_pooling_layer_13_p4;
    vx_scalar org_khronos_nn_extension_pooling_layer_13_p5;
    vx_scalar org_khronos_nn_extension_pooling_layer_13_p6;
    vx_tensor org_khronos_nn_extension_pooling_layer_13_p7;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_0_p1;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_0_p2;
    vx_scalar org_khronos_nn_extension_fully_connected_layer_0_p3;
    vx_scalar org_khronos_nn_extension_fully_connected_layer_0_p4;
    vx_tensor org_khronos_nn_extension_fully_connected_layer_0_p5;
    vx_tensor org_khronos_openvx_tensor_multiply_0_p1;
    vx_scalar org_khronos_openvx_tensor_multiply_0_p2;
    vx_scalar org_khronos_openvx_tensor_multiply_0_p3;
    vx_scalar org_khronos_openvx_tensor_multiply_0_p4;
    vx_tensor org_khronos_openvx_tensor_multiply_0_p5;
    vx_tensor org_khronos_nn_extension_softmax_layer_0_p1;

    //
    // Other Declarations
    //

    vx_size outputAllocators_MergeTensor_8_p0Dimensions[4] = {7,7,1024,1};
    vx_size outputAllocators_MergeTensor_7_p0Dimensions[4] = {7,7,832,1};
    vx_size outputAllocators_MergeTensor_6_p0Dimensions[4] = {14,14,832,1};
    vx_size outputAllocators_MergeTensor_5_p0Dimensions[4] = {14,14,528,1};
    vx_size outputAllocators_MergeTensor_4_p0Dimensions[4] = {14,14,512,1};
    vx_size outputAllocators_MergeTensor_3_p0Dimensions[4] = {14,14,512,1};
    vx_size outputAllocators_MergeTensor_2_p0Dimensions[4] = {14,14,512,1};
    vx_size outputAllocators_MergeTensor_1_p0Dimensions[4] = {28,28,480,1};
    vx_size outputAllocators_MergeTensor_0_p0Dimensions[4] = {28,28,256,1};
    vx_enum org_khronos_nn_extension_activation_layer_0_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_0_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_0_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_0_p4Dimensions[4] = {112,112,64,1};
    vx_enum org_khronos_nn_extension_pooling_layer_0_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_0_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_0_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_0_scalar_p4 = 0;
    vx_size org_khronos_nn_extension_pooling_layer_0_scalar_p5 = 0;
    vx_enum org_khronos_nn_extension_pooling_layer_0_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_0_p7Dimensions[4] = {56,56,64,1};
    vx_enum org_khronos_nn_extension_normalization_layer_0_scalar_p1 = VX_NN_NORMALIZATION_ACROSS_MAPS;
    vx_size org_khronos_nn_extension_normalization_layer_0_scalar_p2 = 5;
    vx_float32 org_khronos_nn_extension_normalization_layer_0_scalar_p3 = 0.0063999998;
    vx_float32 org_khronos_nn_extension_normalization_layer_0_scalar_p4 = 0.750000;
    vx_size org_khronos_nn_extension_normalization_layer_0_p5Dimensions[4] = {56,56,64,1};
    vx_size org_khronos_nn_extension_convolution_layer_1_p1Dimensions[4] = {1,1,64,64};
    vx_size org_khronos_nn_extension_convolution_layer_1_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_1_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_1_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_1_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_1_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_1_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_1_p8Dimensions[4] = {56,56,64,1};
    vx_enum org_khronos_nn_extension_activation_layer_1_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_1_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_1_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_1_p4Dimensions[4] = {56,56,64,1};
    vx_size org_khronos_nn_extension_convolution_layer_2_p1Dimensions[4] = {3,3,64,192};
    vx_size org_khronos_nn_extension_convolution_layer_2_p2Dimensions[1] = {192};
    vx_size org_khronos_nn_extension_convolution_layer_2_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_2_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_2_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_2_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_2_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_2_p8Dimensions[4] = {56,56,192,1};
    vx_enum org_khronos_nn_extension_activation_layer_2_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_2_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_2_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_2_p4Dimensions[4] = {56,56,192,1};
    vx_enum org_khronos_nn_extension_normalization_layer_1_scalar_p1 = VX_NN_NORMALIZATION_ACROSS_MAPS;
    vx_size org_khronos_nn_extension_normalization_layer_1_scalar_p2 = 5;
    vx_float32 org_khronos_nn_extension_normalization_layer_1_scalar_p3 = 0.0063999998;
    vx_float32 org_khronos_nn_extension_normalization_layer_1_scalar_p4 = 0.750000;
    vx_size org_khronos_nn_extension_normalization_layer_1_p5Dimensions[4] = {56,56,192,1};
    vx_enum org_khronos_nn_extension_pooling_layer_1_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_1_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_1_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_1_scalar_p4 = 0;
    vx_size org_khronos_nn_extension_pooling_layer_1_scalar_p5 = 0;
    vx_enum org_khronos_nn_extension_pooling_layer_1_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_1_p7Dimensions[4] = {28,28,192,1};
    vx_size org_khronos_nn_extension_convolution_layer_8_p1Dimensions[4] = {1,1,192,64};
    vx_size org_khronos_nn_extension_convolution_layer_8_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_8_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_8_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_8_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_8_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_8_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_8_p8Dimensions[4] = {28,28,64,1};
    vx_size org_khronos_nn_extension_convolution_layer_6_p1Dimensions[4] = {1,1,192,96};
    vx_size org_khronos_nn_extension_convolution_layer_6_p2Dimensions[1] = {96};
    vx_size org_khronos_nn_extension_convolution_layer_6_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_6_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_6_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_6_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_6_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_6_p8Dimensions[4] = {28,28,96,1};
    vx_size org_khronos_nn_extension_convolution_layer_4_p1Dimensions[4] = {1,1,192,16};
    vx_size org_khronos_nn_extension_convolution_layer_4_p2Dimensions[1] = {16};
    vx_size org_khronos_nn_extension_convolution_layer_4_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_4_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_4_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_4_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_4_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_4_p8Dimensions[4] = {28,28,16,1};
    vx_enum org_khronos_nn_extension_pooling_layer_2_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_2_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_2_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_2_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_2_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_2_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_2_p7Dimensions[4] = {28,28,192,1};
    vx_enum org_khronos_nn_extension_activation_layer_8_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_8_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_8_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_8_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_8_p4_view_view_end[4] = {28,28,64,1};
    vx_enum org_khronos_nn_extension_activation_layer_6_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_6_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_6_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_6_p4Dimensions[4] = {28,28,96,1};
    vx_enum org_khronos_nn_extension_activation_layer_4_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_4_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_4_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_4_p4Dimensions[4] = {28,28,16,1};
    vx_size org_khronos_nn_extension_convolution_layer_3_p1Dimensions[4] = {1,1,192,32};
    vx_size org_khronos_nn_extension_convolution_layer_3_p2Dimensions[1] = {32};
    vx_size org_khronos_nn_extension_convolution_layer_3_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_3_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_3_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_3_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_3_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_3_p8Dimensions[4] = {28,28,32,1};
    vx_size org_khronos_nn_extension_convolution_layer_7_p1Dimensions[4] = {3,3,96,128};
    vx_size org_khronos_nn_extension_convolution_layer_7_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_7_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_7_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_7_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_7_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_7_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_7_p8Dimensions[4] = {28,28,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_5_p1Dimensions[4] = {5,5,16,32};
    vx_size org_khronos_nn_extension_convolution_layer_5_p2Dimensions[1] = {32};
    vx_size org_khronos_nn_extension_convolution_layer_5_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_5_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_5_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_5_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_5_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_5_p8Dimensions[4] = {28,28,32,1};
    vx_enum org_khronos_nn_extension_activation_layer_3_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_3_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_3_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_3_p4_view_view_start[4] = {0,0,224,0};
    vx_size org_khronos_nn_extension_activation_layer_3_p4_view_view_end[4] = {28,28,256,1};
    vx_enum org_khronos_nn_extension_activation_layer_7_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_7_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_7_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_7_p4_view_view_start[4] = {0,0,64,0};
    vx_size org_khronos_nn_extension_activation_layer_7_p4_view_view_end[4] = {28,28,192,1};
    vx_enum org_khronos_nn_extension_activation_layer_5_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_5_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_5_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_5_p4_view_view_start[4] = {0,0,192,0};
    vx_size org_khronos_nn_extension_activation_layer_5_p4_view_view_end[4] = {28,28,224,1};
    vx_size org_khronos_nn_extension_convolution_layer_14_p1Dimensions[4] = {1,1,256,128};
    vx_size org_khronos_nn_extension_convolution_layer_14_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_14_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_14_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_14_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_14_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_14_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_14_p8Dimensions[4] = {28,28,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_12_p1Dimensions[4] = {1,1,256,128};
    vx_size org_khronos_nn_extension_convolution_layer_12_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_12_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_12_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_12_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_12_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_12_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_12_p8Dimensions[4] = {28,28,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_10_p1Dimensions[4] = {1,1,256,32};
    vx_size org_khronos_nn_extension_convolution_layer_10_p2Dimensions[1] = {32};
    vx_size org_khronos_nn_extension_convolution_layer_10_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_10_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_10_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_10_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_10_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_10_p8Dimensions[4] = {28,28,32,1};
    vx_enum org_khronos_nn_extension_pooling_layer_3_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_3_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_3_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_3_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_3_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_3_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_3_p7Dimensions[4] = {28,28,256,1};
    vx_enum org_khronos_nn_extension_activation_layer_14_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_14_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_14_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_14_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_14_p4_view_view_end[4] = {28,28,128,1};
    vx_enum org_khronos_nn_extension_activation_layer_12_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_12_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_12_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_12_p4Dimensions[4] = {28,28,128,1};
    vx_enum org_khronos_nn_extension_activation_layer_10_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_10_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_10_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_10_p4Dimensions[4] = {28,28,32,1};
    vx_size org_khronos_nn_extension_convolution_layer_9_p1Dimensions[4] = {1,1,256,64};
    vx_size org_khronos_nn_extension_convolution_layer_9_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_9_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_9_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_9_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_9_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_9_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_9_p8Dimensions[4] = {28,28,64,1};
    vx_size org_khronos_nn_extension_convolution_layer_13_p1Dimensions[4] = {3,3,128,192};
    vx_size org_khronos_nn_extension_convolution_layer_13_p2Dimensions[1] = {192};
    vx_size org_khronos_nn_extension_convolution_layer_13_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_13_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_13_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_13_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_13_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_13_p8Dimensions[4] = {28,28,192,1};
    vx_size org_khronos_nn_extension_convolution_layer_11_p1Dimensions[4] = {5,5,32,96};
    vx_size org_khronos_nn_extension_convolution_layer_11_p2Dimensions[1] = {96};
    vx_size org_khronos_nn_extension_convolution_layer_11_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_11_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_11_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_11_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_11_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_11_p8Dimensions[4] = {28,28,96,1};
    vx_enum org_khronos_nn_extension_activation_layer_9_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_9_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_9_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_9_p4_view_view_start[4] = {0,0,416,0};
    vx_size org_khronos_nn_extension_activation_layer_9_p4_view_view_end[4] = {28,28,480,1};
    vx_enum org_khronos_nn_extension_activation_layer_13_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_13_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_13_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_13_p4_view_view_start[4] = {0,0,128,0};
    vx_size org_khronos_nn_extension_activation_layer_13_p4_view_view_end[4] = {28,28,320,1};
    vx_enum org_khronos_nn_extension_activation_layer_11_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_11_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_11_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_11_p4_view_view_start[4] = {0,0,320,0};
    vx_size org_khronos_nn_extension_activation_layer_11_p4_view_view_end[4] = {28,28,416,1};
    vx_enum org_khronos_nn_extension_pooling_layer_4_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_4_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_4_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_4_scalar_p4 = 0;
    vx_size org_khronos_nn_extension_pooling_layer_4_scalar_p5 = 0;
    vx_enum org_khronos_nn_extension_pooling_layer_4_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_4_p7Dimensions[4] = {14,14,480,1};
    vx_size org_khronos_nn_extension_convolution_layer_20_p1Dimensions[4] = {1,1,480,192};
    vx_size org_khronos_nn_extension_convolution_layer_20_p2Dimensions[1] = {192};
    vx_size org_khronos_nn_extension_convolution_layer_20_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_20_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_20_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_20_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_20_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_20_p8Dimensions[4] = {14,14,192,1};
    vx_size org_khronos_nn_extension_convolution_layer_18_p1Dimensions[4] = {1,1,480,96};
    vx_size org_khronos_nn_extension_convolution_layer_18_p2Dimensions[1] = {96};
    vx_size org_khronos_nn_extension_convolution_layer_18_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_18_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_18_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_18_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_18_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_18_p8Dimensions[4] = {14,14,96,1};
    vx_size org_khronos_nn_extension_convolution_layer_16_p1Dimensions[4] = {1,1,480,16};
    vx_size org_khronos_nn_extension_convolution_layer_16_p2Dimensions[1] = {16};
    vx_size org_khronos_nn_extension_convolution_layer_16_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_16_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_16_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_16_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_16_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_16_p8Dimensions[4] = {14,14,16,1};
    vx_enum org_khronos_nn_extension_pooling_layer_5_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_5_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_5_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_5_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_5_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_5_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_5_p7Dimensions[4] = {14,14,480,1};
    vx_enum org_khronos_nn_extension_activation_layer_20_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_20_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_20_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_20_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_20_p4_view_view_end[4] = {14,14,192,1};
    vx_enum org_khronos_nn_extension_activation_layer_18_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_18_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_18_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_18_p4Dimensions[4] = {14,14,96,1};
    vx_enum org_khronos_nn_extension_activation_layer_16_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_16_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_16_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_16_p4Dimensions[4] = {14,14,16,1};
    vx_size org_khronos_nn_extension_convolution_layer_15_p1Dimensions[4] = {1,1,480,64};
    vx_size org_khronos_nn_extension_convolution_layer_15_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_15_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_15_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_15_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_15_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_15_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_15_p8Dimensions[4] = {14,14,64,1};
    vx_size org_khronos_nn_extension_convolution_layer_19_p1Dimensions[4] = {3,3,96,208};
    vx_size org_khronos_nn_extension_convolution_layer_19_p2Dimensions[1] = {208};
    vx_size org_khronos_nn_extension_convolution_layer_19_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_19_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_19_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_19_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_19_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_19_p8Dimensions[4] = {14,14,208,1};
    vx_size org_khronos_nn_extension_convolution_layer_17_p1Dimensions[4] = {5,5,16,48};
    vx_size org_khronos_nn_extension_convolution_layer_17_p2Dimensions[1] = {48};
    vx_size org_khronos_nn_extension_convolution_layer_17_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_17_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_17_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_17_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_17_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_17_p8Dimensions[4] = {14,14,48,1};
    vx_enum org_khronos_nn_extension_activation_layer_15_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_15_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_15_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_15_p4_view_view_start[4] = {0,0,448,0};
    vx_size org_khronos_nn_extension_activation_layer_15_p4_view_view_end[4] = {14,14,512,1};
    vx_enum org_khronos_nn_extension_activation_layer_19_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_19_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_19_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_19_p4_view_view_start[4] = {0,0,192,0};
    vx_size org_khronos_nn_extension_activation_layer_19_p4_view_view_end[4] = {14,14,400,1};
    vx_enum org_khronos_nn_extension_activation_layer_17_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_17_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_17_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_17_p4_view_view_start[4] = {0,0,400,0};
    vx_size org_khronos_nn_extension_activation_layer_17_p4_view_view_end[4] = {14,14,448,1};
    vx_size org_khronos_nn_extension_convolution_layer_26_p1Dimensions[4] = {1,1,512,160};
    vx_size org_khronos_nn_extension_convolution_layer_26_p2Dimensions[1] = {160};
    vx_size org_khronos_nn_extension_convolution_layer_26_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_26_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_26_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_26_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_26_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_26_p8Dimensions[4] = {14,14,160,1};
    vx_size org_khronos_nn_extension_convolution_layer_24_p1Dimensions[4] = {1,1,512,112};
    vx_size org_khronos_nn_extension_convolution_layer_24_p2Dimensions[1] = {112};
    vx_size org_khronos_nn_extension_convolution_layer_24_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_24_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_24_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_24_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_24_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_24_p8Dimensions[4] = {14,14,112,1};
    vx_size org_khronos_nn_extension_convolution_layer_22_p1Dimensions[4] = {1,1,512,24};
    vx_size org_khronos_nn_extension_convolution_layer_22_p2Dimensions[1] = {24};
    vx_size org_khronos_nn_extension_convolution_layer_22_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_22_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_22_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_22_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_22_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_22_p8Dimensions[4] = {14,14,24,1};
    vx_enum org_khronos_nn_extension_pooling_layer_6_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_6_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_6_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_6_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_6_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_6_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_6_p7Dimensions[4] = {14,14,512,1};
    vx_enum org_khronos_nn_extension_activation_layer_26_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_26_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_26_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_26_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_26_p4_view_view_end[4] = {14,14,160,1};
    vx_enum org_khronos_nn_extension_activation_layer_24_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_24_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_24_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_24_p4Dimensions[4] = {14,14,112,1};
    vx_enum org_khronos_nn_extension_activation_layer_22_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_22_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_22_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_22_p4Dimensions[4] = {14,14,24,1};
    vx_size org_khronos_nn_extension_convolution_layer_21_p1Dimensions[4] = {1,1,512,64};
    vx_size org_khronos_nn_extension_convolution_layer_21_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_21_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_21_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_21_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_21_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_21_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_21_p8Dimensions[4] = {14,14,64,1};
    vx_size org_khronos_nn_extension_convolution_layer_25_p1Dimensions[4] = {3,3,112,224};
    vx_size org_khronos_nn_extension_convolution_layer_25_p2Dimensions[1] = {224};
    vx_size org_khronos_nn_extension_convolution_layer_25_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_25_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_25_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_25_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_25_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_25_p8Dimensions[4] = {14,14,224,1};
    vx_size org_khronos_nn_extension_convolution_layer_23_p1Dimensions[4] = {5,5,24,64};
    vx_size org_khronos_nn_extension_convolution_layer_23_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_23_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_23_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_23_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_23_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_23_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_23_p8Dimensions[4] = {14,14,64,1};
    vx_enum org_khronos_nn_extension_activation_layer_21_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_21_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_21_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_21_p4_view_view_start[4] = {0,0,448,0};
    vx_size org_khronos_nn_extension_activation_layer_21_p4_view_view_end[4] = {14,14,512,1};
    vx_enum org_khronos_nn_extension_activation_layer_25_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_25_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_25_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_25_p4_view_view_start[4] = {0,0,160,0};
    vx_size org_khronos_nn_extension_activation_layer_25_p4_view_view_end[4] = {14,14,384,1};
    vx_enum org_khronos_nn_extension_activation_layer_23_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_23_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_23_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_23_p4_view_view_start[4] = {0,0,384,0};
    vx_size org_khronos_nn_extension_activation_layer_23_p4_view_view_end[4] = {14,14,448,1};
    vx_size org_khronos_nn_extension_convolution_layer_32_p1Dimensions[4] = {1,1,512,128};
    vx_size org_khronos_nn_extension_convolution_layer_32_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_32_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_32_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_32_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_32_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_32_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_32_p8Dimensions[4] = {14,14,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_30_p1Dimensions[4] = {1,1,512,128};
    vx_size org_khronos_nn_extension_convolution_layer_30_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_30_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_30_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_30_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_30_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_30_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_30_p8Dimensions[4] = {14,14,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_28_p1Dimensions[4] = {1,1,512,24};
    vx_size org_khronos_nn_extension_convolution_layer_28_p2Dimensions[1] = {24};
    vx_size org_khronos_nn_extension_convolution_layer_28_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_28_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_28_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_28_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_28_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_28_p8Dimensions[4] = {14,14,24,1};
    vx_enum org_khronos_nn_extension_pooling_layer_7_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_7_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_7_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_7_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_7_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_7_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_7_p7Dimensions[4] = {14,14,512,1};
    vx_enum org_khronos_nn_extension_activation_layer_32_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_32_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_32_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_32_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_32_p4_view_view_end[4] = {14,14,128,1};
    vx_enum org_khronos_nn_extension_activation_layer_30_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_30_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_30_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_30_p4Dimensions[4] = {14,14,128,1};
    vx_enum org_khronos_nn_extension_activation_layer_28_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_28_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_28_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_28_p4Dimensions[4] = {14,14,24,1};
    vx_size org_khronos_nn_extension_convolution_layer_27_p1Dimensions[4] = {1,1,512,64};
    vx_size org_khronos_nn_extension_convolution_layer_27_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_27_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_27_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_27_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_27_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_27_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_27_p8Dimensions[4] = {14,14,64,1};
    vx_size org_khronos_nn_extension_convolution_layer_31_p1Dimensions[4] = {3,3,128,256};
    vx_size org_khronos_nn_extension_convolution_layer_31_p2Dimensions[1] = {256};
    vx_size org_khronos_nn_extension_convolution_layer_31_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_31_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_31_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_31_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_31_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_31_p8Dimensions[4] = {14,14,256,1};
    vx_size org_khronos_nn_extension_convolution_layer_29_p1Dimensions[4] = {5,5,24,64};
    vx_size org_khronos_nn_extension_convolution_layer_29_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_29_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_29_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_29_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_29_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_29_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_29_p8Dimensions[4] = {14,14,64,1};
    vx_enum org_khronos_nn_extension_activation_layer_27_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_27_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_27_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_27_p4_view_view_start[4] = {0,0,448,0};
    vx_size org_khronos_nn_extension_activation_layer_27_p4_view_view_end[4] = {14,14,512,1};
    vx_enum org_khronos_nn_extension_activation_layer_31_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_31_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_31_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_31_p4_view_view_start[4] = {0,0,128,0};
    vx_size org_khronos_nn_extension_activation_layer_31_p4_view_view_end[4] = {14,14,384,1};
    vx_enum org_khronos_nn_extension_activation_layer_29_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_29_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_29_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_29_p4_view_view_start[4] = {0,0,384,0};
    vx_size org_khronos_nn_extension_activation_layer_29_p4_view_view_end[4] = {14,14,448,1};
    vx_size org_khronos_nn_extension_convolution_layer_38_p1Dimensions[4] = {1,1,512,112};
    vx_size org_khronos_nn_extension_convolution_layer_38_p2Dimensions[1] = {112};
    vx_size org_khronos_nn_extension_convolution_layer_38_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_38_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_38_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_38_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_38_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_38_p8Dimensions[4] = {14,14,112,1};
    vx_size org_khronos_nn_extension_convolution_layer_36_p1Dimensions[4] = {1,1,512,144};
    vx_size org_khronos_nn_extension_convolution_layer_36_p2Dimensions[1] = {144};
    vx_size org_khronos_nn_extension_convolution_layer_36_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_36_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_36_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_36_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_36_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_36_p8Dimensions[4] = {14,14,144,1};
    vx_size org_khronos_nn_extension_convolution_layer_34_p1Dimensions[4] = {1,1,512,32};
    vx_size org_khronos_nn_extension_convolution_layer_34_p2Dimensions[1] = {32};
    vx_size org_khronos_nn_extension_convolution_layer_34_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_34_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_34_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_34_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_34_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_34_p8Dimensions[4] = {14,14,32,1};
    vx_enum org_khronos_nn_extension_pooling_layer_8_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_8_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_8_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_8_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_8_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_8_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_8_p7Dimensions[4] = {14,14,512,1};
    vx_enum org_khronos_nn_extension_activation_layer_38_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_38_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_38_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_38_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_38_p4_view_view_end[4] = {14,14,112,1};
    vx_enum org_khronos_nn_extension_activation_layer_36_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_36_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_36_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_36_p4Dimensions[4] = {14,14,144,1};
    vx_enum org_khronos_nn_extension_activation_layer_34_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_34_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_34_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_34_p4Dimensions[4] = {14,14,32,1};
    vx_size org_khronos_nn_extension_convolution_layer_33_p1Dimensions[4] = {1,1,512,64};
    vx_size org_khronos_nn_extension_convolution_layer_33_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_33_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_33_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_33_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_33_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_33_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_33_p8Dimensions[4] = {14,14,64,1};
    vx_size org_khronos_nn_extension_convolution_layer_37_p1Dimensions[4] = {3,3,144,288};
    vx_size org_khronos_nn_extension_convolution_layer_37_p2Dimensions[1] = {288};
    vx_size org_khronos_nn_extension_convolution_layer_37_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_37_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_37_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_37_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_37_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_37_p8Dimensions[4] = {14,14,288,1};
    vx_size org_khronos_nn_extension_convolution_layer_35_p1Dimensions[4] = {5,5,32,64};
    vx_size org_khronos_nn_extension_convolution_layer_35_p2Dimensions[1] = {64};
    vx_size org_khronos_nn_extension_convolution_layer_35_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_35_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_35_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_35_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_35_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_35_p8Dimensions[4] = {14,14,64,1};
    vx_enum org_khronos_nn_extension_activation_layer_33_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_33_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_33_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_33_p4_view_view_start[4] = {0,0,464,0};
    vx_size org_khronos_nn_extension_activation_layer_33_p4_view_view_end[4] = {14,14,528,1};
    vx_enum org_khronos_nn_extension_activation_layer_37_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_37_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_37_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_37_p4_view_view_start[4] = {0,0,112,0};
    vx_size org_khronos_nn_extension_activation_layer_37_p4_view_view_end[4] = {14,14,400,1};
    vx_enum org_khronos_nn_extension_activation_layer_35_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_35_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_35_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_35_p4_view_view_start[4] = {0,0,400,0};
    vx_size org_khronos_nn_extension_activation_layer_35_p4_view_view_end[4] = {14,14,464,1};
    vx_size org_khronos_nn_extension_convolution_layer_44_p1Dimensions[4] = {1,1,528,256};
    vx_size org_khronos_nn_extension_convolution_layer_44_p2Dimensions[1] = {256};
    vx_size org_khronos_nn_extension_convolution_layer_44_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_44_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_44_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_44_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_44_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_44_p8Dimensions[4] = {14,14,256,1};
    vx_size org_khronos_nn_extension_convolution_layer_42_p1Dimensions[4] = {1,1,528,160};
    vx_size org_khronos_nn_extension_convolution_layer_42_p2Dimensions[1] = {160};
    vx_size org_khronos_nn_extension_convolution_layer_42_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_42_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_42_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_42_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_42_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_42_p8Dimensions[4] = {14,14,160,1};
    vx_size org_khronos_nn_extension_convolution_layer_40_p1Dimensions[4] = {1,1,528,32};
    vx_size org_khronos_nn_extension_convolution_layer_40_p2Dimensions[1] = {32};
    vx_size org_khronos_nn_extension_convolution_layer_40_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_40_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_40_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_40_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_40_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_40_p8Dimensions[4] = {14,14,32,1};
    vx_enum org_khronos_nn_extension_pooling_layer_9_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_9_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_9_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_9_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_9_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_9_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_9_p7Dimensions[4] = {14,14,528,1};
    vx_enum org_khronos_nn_extension_activation_layer_44_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_44_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_44_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_44_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_44_p4_view_view_end[4] = {14,14,256,1};
    vx_enum org_khronos_nn_extension_activation_layer_42_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_42_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_42_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_42_p4Dimensions[4] = {14,14,160,1};
    vx_enum org_khronos_nn_extension_activation_layer_40_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_40_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_40_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_40_p4Dimensions[4] = {14,14,32,1};
    vx_size org_khronos_nn_extension_convolution_layer_39_p1Dimensions[4] = {1,1,528,128};
    vx_size org_khronos_nn_extension_convolution_layer_39_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_39_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_39_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_39_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_39_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_39_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_39_p8Dimensions[4] = {14,14,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_43_p1Dimensions[4] = {3,3,160,320};
    vx_size org_khronos_nn_extension_convolution_layer_43_p2Dimensions[1] = {320};
    vx_size org_khronos_nn_extension_convolution_layer_43_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_43_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_43_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_43_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_43_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_43_p8Dimensions[4] = {14,14,320,1};
    vx_size org_khronos_nn_extension_convolution_layer_41_p1Dimensions[4] = {5,5,32,128};
    vx_size org_khronos_nn_extension_convolution_layer_41_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_41_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_41_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_41_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_41_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_41_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_41_p8Dimensions[4] = {14,14,128,1};
    vx_enum org_khronos_nn_extension_activation_layer_39_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_39_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_39_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_39_p4_view_view_start[4] = {0,0,704,0};
    vx_size org_khronos_nn_extension_activation_layer_39_p4_view_view_end[4] = {14,14,832,1};
    vx_enum org_khronos_nn_extension_activation_layer_43_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_43_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_43_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_43_p4_view_view_start[4] = {0,0,256,0};
    vx_size org_khronos_nn_extension_activation_layer_43_p4_view_view_end[4] = {14,14,576,1};
    vx_enum org_khronos_nn_extension_activation_layer_41_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_41_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_41_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_41_p4_view_view_start[4] = {0,0,576,0};
    vx_size org_khronos_nn_extension_activation_layer_41_p4_view_view_end[4] = {14,14,704,1};
    vx_enum org_khronos_nn_extension_pooling_layer_10_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_10_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_10_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_10_scalar_p4 = 0;
    vx_size org_khronos_nn_extension_pooling_layer_10_scalar_p5 = 0;
    vx_enum org_khronos_nn_extension_pooling_layer_10_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_10_p7Dimensions[4] = {7,7,832,1};
    vx_size org_khronos_nn_extension_convolution_layer_50_p1Dimensions[4] = {1,1,832,256};
    vx_size org_khronos_nn_extension_convolution_layer_50_p2Dimensions[1] = {256};
    vx_size org_khronos_nn_extension_convolution_layer_50_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_50_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_50_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_50_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_50_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_50_p8Dimensions[4] = {7,7,256,1};
    vx_size org_khronos_nn_extension_convolution_layer_48_p1Dimensions[4] = {1,1,832,160};
    vx_size org_khronos_nn_extension_convolution_layer_48_p2Dimensions[1] = {160};
    vx_size org_khronos_nn_extension_convolution_layer_48_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_48_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_48_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_48_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_48_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_48_p8Dimensions[4] = {7,7,160,1};
    vx_size org_khronos_nn_extension_convolution_layer_46_p1Dimensions[4] = {1,1,832,32};
    vx_size org_khronos_nn_extension_convolution_layer_46_p2Dimensions[1] = {32};
    vx_size org_khronos_nn_extension_convolution_layer_46_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_46_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_46_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_46_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_46_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_46_p8Dimensions[4] = {7,7,32,1};
    vx_enum org_khronos_nn_extension_pooling_layer_11_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_11_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_11_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_11_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_11_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_11_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_11_p7Dimensions[4] = {7,7,832,1};
    vx_enum org_khronos_nn_extension_activation_layer_50_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_50_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_50_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_50_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_50_p4_view_view_end[4] = {7,7,256,1};
    vx_enum org_khronos_nn_extension_activation_layer_48_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_48_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_48_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_48_p4Dimensions[4] = {7,7,160,1};
    vx_enum org_khronos_nn_extension_activation_layer_46_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_46_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_46_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_46_p4Dimensions[4] = {7,7,32,1};
    vx_size org_khronos_nn_extension_convolution_layer_45_p1Dimensions[4] = {1,1,832,128};
    vx_size org_khronos_nn_extension_convolution_layer_45_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_45_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_45_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_45_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_45_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_45_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_45_p8Dimensions[4] = {7,7,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_49_p1Dimensions[4] = {3,3,160,320};
    vx_size org_khronos_nn_extension_convolution_layer_49_p2Dimensions[1] = {320};
    vx_size org_khronos_nn_extension_convolution_layer_49_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_49_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_49_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_49_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_49_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_49_p8Dimensions[4] = {7,7,320,1};
    vx_size org_khronos_nn_extension_convolution_layer_47_p1Dimensions[4] = {5,5,32,128};
    vx_size org_khronos_nn_extension_convolution_layer_47_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_47_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_47_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_47_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_47_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_47_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_47_p8Dimensions[4] = {7,7,128,1};
    vx_enum org_khronos_nn_extension_activation_layer_45_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_45_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_45_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_45_p4_view_view_start[4] = {0,0,704,0};
    vx_size org_khronos_nn_extension_activation_layer_45_p4_view_view_end[4] = {7,7,832,1};
    vx_enum org_khronos_nn_extension_activation_layer_49_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_49_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_49_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_49_p4_view_view_start[4] = {0,0,256,0};
    vx_size org_khronos_nn_extension_activation_layer_49_p4_view_view_end[4] = {7,7,576,1};
    vx_enum org_khronos_nn_extension_activation_layer_47_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_47_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_47_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_47_p4_view_view_start[4] = {0,0,576,0};
    vx_size org_khronos_nn_extension_activation_layer_47_p4_view_view_end[4] = {7,7,704,1};
    vx_size org_khronos_nn_extension_convolution_layer_56_p1Dimensions[4] = {1,1,832,384};
    vx_size org_khronos_nn_extension_convolution_layer_56_p2Dimensions[1] = {384};
    vx_size org_khronos_nn_extension_convolution_layer_56_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_56_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_56_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_56_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_56_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_56_p8Dimensions[4] = {7,7,384,1};
    vx_size org_khronos_nn_extension_convolution_layer_54_p1Dimensions[4] = {1,1,832,192};
    vx_size org_khronos_nn_extension_convolution_layer_54_p2Dimensions[1] = {192};
    vx_size org_khronos_nn_extension_convolution_layer_54_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_54_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_54_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_54_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_54_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_54_p8Dimensions[4] = {7,7,192,1};
    vx_size org_khronos_nn_extension_convolution_layer_52_p1Dimensions[4] = {1,1,832,48};
    vx_size org_khronos_nn_extension_convolution_layer_52_p2Dimensions[1] = {48};
    vx_size org_khronos_nn_extension_convolution_layer_52_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_52_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_52_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_52_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_52_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_52_p8Dimensions[4] = {7,7,48,1};
    vx_enum org_khronos_nn_extension_pooling_layer_12_scalar_p1 = VX_NN_POOLING_MAX;
    vx_size org_khronos_nn_extension_pooling_layer_12_scalar_p2 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_12_scalar_p3 = 3;
    vx_size org_khronos_nn_extension_pooling_layer_12_scalar_p4 = 1;
    vx_size org_khronos_nn_extension_pooling_layer_12_scalar_p5 = 1;
    vx_enum org_khronos_nn_extension_pooling_layer_12_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_12_p7Dimensions[4] = {7,7,832,1};
    vx_enum org_khronos_nn_extension_activation_layer_56_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_56_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_56_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_56_p4_view_view_start[4] = {0,0,0,0};
    vx_size org_khronos_nn_extension_activation_layer_56_p4_view_view_end[4] = {7,7,384,1};
    vx_enum org_khronos_nn_extension_activation_layer_54_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_54_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_54_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_54_p4Dimensions[4] = {7,7,192,1};
    vx_enum org_khronos_nn_extension_activation_layer_52_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_52_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_52_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_52_p4Dimensions[4] = {7,7,48,1};
    vx_size org_khronos_nn_extension_convolution_layer_51_p1Dimensions[4] = {1,1,832,128};
    vx_size org_khronos_nn_extension_convolution_layer_51_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_51_scalar_p3 = 0;
    vx_size org_khronos_nn_extension_convolution_layer_51_scalar_p4 = 0;
    vx_enum org_khronos_nn_extension_convolution_layer_51_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_51_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_51_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_51_p8Dimensions[4] = {7,7,128,1};
    vx_size org_khronos_nn_extension_convolution_layer_55_p1Dimensions[4] = {3,3,192,384};
    vx_size org_khronos_nn_extension_convolution_layer_55_p2Dimensions[1] = {384};
    vx_size org_khronos_nn_extension_convolution_layer_55_scalar_p3 = 1;
    vx_size org_khronos_nn_extension_convolution_layer_55_scalar_p4 = 1;
    vx_enum org_khronos_nn_extension_convolution_layer_55_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_55_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_55_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_55_p8Dimensions[4] = {7,7,384,1};
    vx_size org_khronos_nn_extension_convolution_layer_53_p1Dimensions[4] = {5,5,48,128};
    vx_size org_khronos_nn_extension_convolution_layer_53_p2Dimensions[1] = {128};
    vx_size org_khronos_nn_extension_convolution_layer_53_scalar_p3 = 2;
    vx_size org_khronos_nn_extension_convolution_layer_53_scalar_p4 = 2;
    vx_enum org_khronos_nn_extension_convolution_layer_53_scalar_p5 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_convolution_layer_53_scalar_p6 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_enum org_khronos_nn_extension_convolution_layer_53_scalar_p7 = VX_NN_DS_SIZE_ROUNDING_FLOOR;
    vx_size org_khronos_nn_extension_convolution_layer_53_p8Dimensions[4] = {7,7,128,1};
    vx_enum org_khronos_nn_extension_activation_layer_51_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_51_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_51_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_51_p4_view_view_start[4] = {0,0,896,0};
    vx_size org_khronos_nn_extension_activation_layer_51_p4_view_view_end[4] = {7,7,1024,1};
    vx_enum org_khronos_nn_extension_activation_layer_55_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_55_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_55_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_55_p4_view_view_start[4] = {0,0,384,0};
    vx_size org_khronos_nn_extension_activation_layer_55_p4_view_view_end[4] = {7,7,768,1};
    vx_enum org_khronos_nn_extension_activation_layer_53_scalar_p1 = VX_NN_ACTIVATION_RELU;
    vx_float32 org_khronos_nn_extension_activation_layer_53_scalar_p2 = 1.0;
    vx_float32 org_khronos_nn_extension_activation_layer_53_scalar_p3 = 0.0;
    vx_size org_khronos_nn_extension_activation_layer_53_p4_view_view_start[4] = {0,0,768,0};
    vx_size org_khronos_nn_extension_activation_layer_53_p4_view_view_end[4] = {7,7,896,1};
    vx_enum org_khronos_nn_extension_pooling_layer_13_scalar_p1 = VX_NN_POOLING_AVG;
    vx_size org_khronos_nn_extension_pooling_layer_13_scalar_p2 = 7;
    vx_size org_khronos_nn_extension_pooling_layer_13_scalar_p3 = 7;
    vx_size org_khronos_nn_extension_pooling_layer_13_scalar_p4 = 0;
    vx_size org_khronos_nn_extension_pooling_layer_13_scalar_p5 = 0;
    vx_enum org_khronos_nn_extension_pooling_layer_13_scalar_p6 = VX_NN_DS_SIZE_ROUNDING_CEILING;
    vx_size org_khronos_nn_extension_pooling_layer_13_p7Dimensions[4] = {1,1,1024,1};
    vx_size org_khronos_nn_extension_fully_connected_layer_0_p1Dimensions[4] = {1,1,1024,1000};
    vx_size org_khronos_nn_extension_fully_connected_layer_0_p2Dimensions[1] = {1000};
    vx_enum org_khronos_nn_extension_fully_connected_layer_0_scalar_p3 = VX_CONVERT_POLICY_WRAP;
    vx_enum org_khronos_nn_extension_fully_connected_layer_0_scalar_p4 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_size org_khronos_nn_extension_fully_connected_layer_0_p5Dimensions[2] = {1000,1};
    vx_size org_khronos_openvx_tensor_multiply_0_p1Dimensions[2] = {1,1};
    vx_float32 org_khronos_openvx_tensor_multiply_0_scalar_p2 = 8;
    vx_enum org_khronos_openvx_tensor_multiply_0_scalar_p3 = VX_CONVERT_POLICY_SATURATE;
    vx_enum org_khronos_openvx_tensor_multiply_0_scalar_p4 = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_size org_khronos_openvx_tensor_multiply_0_p5Dimensions[2] = {1000,1};
    vx_size org_khronos_nn_extension_softmax_layer_0_p1Dimensions[2] = {1000,1};
    vx_nn_convolution_params_t org_khronos_nn_extension_convolution_0_p3 = {0,0,VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0};
    vx_nn_convolution_params_t org_khronos_nn_extension_convolution_1_p3 = {1,1,VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0};
    vx_nn_convolution_params_t org_khronos_nn_extension_convolution_2_p3 = {2,2,VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, VX_NN_DS_SIZE_ROUNDING_FLOOR, 0, 0};


    //
    // Kernel Assignments
    //
//    org_khronos_nn_extension_convolution_layer_Kernel = vxGetKernelByName(context, "org.khronos.nn_extension.convolution_layer");
//    if(!org_khronos_nn_extension_convolution_layer_Kernel)
//    {
//        WriteLog("ERROR: cannot get kernel org.khronos.nn_extension.convolution_layer\n");
//        return VX_FAILURE;
//    }
//    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_Kernel, VX_TYPE_KERNEL, 0);
    org_khronos_nn_extension_activation_layer_Kernel = vxGetKernelByName(context, "org.khronos.nn_extension.activation_layer");
    if(!org_khronos_nn_extension_activation_layer_Kernel)
    {
        WriteLog("ERROR: cannot get kernel org.khronos.nn_extension.activation_layer\n");
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_Kernel, VX_TYPE_KERNEL, 0);
    org_khronos_nn_extension_pooling_layer_Kernel = vxGetKernelByName(context, "org.khronos.nn_extension.pooling_layer");
    if(!org_khronos_nn_extension_pooling_layer_Kernel)
    {
        WriteLog("ERROR: cannot get kernel org.khronos.nn_extension.pooling_layer\n");
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_Kernel, VX_TYPE_KERNEL, 0);
    org_khronos_nn_extension_normalization_layer_Kernel = vxGetKernelByName(context, "org.khronos.nn_extension.normalization_layer");
    if(!org_khronos_nn_extension_normalization_layer_Kernel)
    {
        WriteLog("ERROR: cannot get kernel org.khronos.nn_extension.normalization_layer\n");
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_Kernel, VX_TYPE_KERNEL, 0);
    org_khronos_nn_extension_fully_connected_layer_Kernel = vxGetKernelByName(context, "org.khronos.nn_extension.fully_connected_layer");
    if(!org_khronos_nn_extension_fully_connected_layer_Kernel)
    {
        WriteLog("ERROR: cannot get kernel org.khronos.nn_extension.fully_connected_layer\n");
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_Kernel, VX_TYPE_KERNEL, 0);
    org_khronos_openvx_tensor_multiply_Kernel = vxGetKernelByName(context, "org.khronos.openvx.tensor_multiply");
    if(!org_khronos_openvx_tensor_multiply_Kernel)
    {
        WriteLog("ERROR: cannot get kernel org.khronos.openvx.tensor_multiply\n");
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_openvx_tensor_multiply_Kernel, VX_TYPE_KERNEL, 0);
    org_khronos_nn_extension_softmax_layer_Kernel = vxGetKernelByName(context, "org.khronos.nn_extension.softmax_layer");
    if(!org_khronos_nn_extension_softmax_layer_Kernel)
    {
        WriteLog("ERROR: cannot get kernel org.khronos.nn_extension.softmax_layer\n");
        return VX_FAILURE;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_softmax_layer_Kernel, VX_TYPE_KERNEL, 0);


    //
    // Primitive Assignments
    //
    outputAllocators_MergeTensor_8_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_8_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_8_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_8_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_8_p0, VX_TYPE_TENSOR, "pool5_7x7_s1_0");

    outputAllocators_MergeTensor_7_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_7_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_7_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_7_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_7_p0, VX_TYPE_TENSOR, "inception_5b_1x1_0");

    outputAllocators_MergeTensor_6_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_6_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_6_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_6_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_6_p0, VX_TYPE_TENSOR, "pool4_3x3_s2_0");

    outputAllocators_MergeTensor_5_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_5_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_5_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_5_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_5_p0, VX_TYPE_TENSOR, "inception_4e_1x1_0");

    outputAllocators_MergeTensor_4_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_4_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_4_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_4_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_4_p0, VX_TYPE_TENSOR, "inception_4d_1x1_0");

    outputAllocators_MergeTensor_3_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_3_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_3_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_3_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_3_p0, VX_TYPE_TENSOR, "inception_4c_1x1_0");

    outputAllocators_MergeTensor_2_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_2_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_2_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_2_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_2_p0, VX_TYPE_TENSOR, "inception_4b_1x1_0");

    outputAllocators_MergeTensor_1_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_1_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_1_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_1_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_1_p0, VX_TYPE_TENSOR, "pool3_3x3_s2_0");

    outputAllocators_MergeTensor_0_p0 = vxCreateTensor(context, 4, outputAllocators_MergeTensor_0_p0Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)outputAllocators_MergeTensor_0_p0);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter outputAllocators_MergeTensor_0_p0 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)outputAllocators_MergeTensor_0_p0, VX_TYPE_TENSOR, "inception_3b_1x1_0");

    org_khronos_nn_extension_activation_layer_0_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_0_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0_p1, VX_TYPE_SCALAR, "conv1_relu_7x7_1");

    org_khronos_nn_extension_activation_layer_0_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_0_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0_p2, VX_TYPE_SCALAR, "conv1_relu_7x7_2");

    org_khronos_nn_extension_activation_layer_0_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0_p3, VX_TYPE_SCALAR, "conv1_relu_7x7_2");

    org_khronos_nn_extension_activation_layer_0_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_0_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_0_p4, VX_TYPE_TENSOR, "conv1_relu_7x7_4");

    org_khronos_nn_extension_pooling_layer_0_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p1, VX_TYPE_SCALAR, "pool1_3x3_s2_1");

    org_khronos_nn_extension_pooling_layer_0_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p2, VX_TYPE_SCALAR, "pool1_3x3_s2_2");

    org_khronos_nn_extension_pooling_layer_0_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p3, VX_TYPE_SCALAR, "pool1_3x3_s2_3");

    org_khronos_nn_extension_pooling_layer_0_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p4, VX_TYPE_SCALAR, "pool1_3x3_s2_4");

    org_khronos_nn_extension_pooling_layer_0_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p5, VX_TYPE_SCALAR, "pool1_3x3_s2_5");

    org_khronos_nn_extension_pooling_layer_0_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_0_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p6, VX_TYPE_SCALAR, "pool1_3x3_s2_6");

    org_khronos_nn_extension_pooling_layer_0_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_0_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_0_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_0_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p7, VX_TYPE_TENSOR, "pool1_3x3_s2_7");

    org_khronos_nn_extension_normalization_layer_0_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_normalization_layer_0_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p1, VX_TYPE_SCALAR, "pool1_norm1_1");

    org_khronos_nn_extension_normalization_layer_0_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_normalization_layer_0_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p2, VX_TYPE_SCALAR, "pool1_norm1_2");

    org_khronos_nn_extension_normalization_layer_0_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_normalization_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p3, VX_TYPE_SCALAR, "pool1_norm1_3");

    org_khronos_nn_extension_normalization_layer_0_p4 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_normalization_layer_0_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p4, VX_TYPE_SCALAR, "pool1_norm1_4");

    org_khronos_nn_extension_normalization_layer_0_p5 = vxCreateTensor(context, 4, org_khronos_nn_extension_normalization_layer_0_p5Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_0_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_0_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p5, VX_TYPE_TENSOR, "pool1_norm1_5");

    org_khronos_nn_extension_convolution_layer_1_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_1_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p1, VX_TYPE_TENSOR, "conv2_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_1_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_1_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p2, VX_TYPE_TENSOR, "conv2_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_1_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p3, VX_TYPE_SCALAR, "conv2_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_1_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p4, VX_TYPE_SCALAR, "conv2_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_1_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p5, VX_TYPE_SCALAR, "conv2_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_1_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p6, VX_TYPE_SCALAR, "conv2_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_1_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_1_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p7, VX_TYPE_SCALAR, "conv2_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_1_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_1_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_1_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p8, VX_TYPE_TENSOR, "conv2_3x3_reduce_8");

    org_khronos_nn_extension_activation_layer_1_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_1_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1_p1, VX_TYPE_SCALAR, "conv2_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_1_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_1_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1_p2, VX_TYPE_SCALAR, "conv2_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_1_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1_p3, VX_TYPE_SCALAR, "conv2_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_1_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_1_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_1_p4, VX_TYPE_TENSOR, "conv2_relu_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_2_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_2_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p1, VX_TYPE_TENSOR, "conv2_3x3_weights");

    org_khronos_nn_extension_convolution_layer_2_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_2_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p2, VX_TYPE_TENSOR, "conv2_3x3_bias");

    org_khronos_nn_extension_convolution_layer_2_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p3, VX_TYPE_SCALAR, "conv2_3x3_3");

    org_khronos_nn_extension_convolution_layer_2_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p4, VX_TYPE_SCALAR, "conv2_3x3_4");

    org_khronos_nn_extension_convolution_layer_2_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p5, VX_TYPE_SCALAR, "conv2_3x3_5");

    org_khronos_nn_extension_convolution_layer_2_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p6, VX_TYPE_SCALAR, "conv2_3x3_6");

    org_khronos_nn_extension_convolution_layer_2_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_2_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p7, VX_TYPE_SCALAR, "conv2_3x3_7");

    org_khronos_nn_extension_convolution_layer_2_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_2_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_2_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p8, VX_TYPE_TENSOR, "conv2_3x3_8");

    org_khronos_nn_extension_activation_layer_2_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_2_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_2_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_2_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2_p1, VX_TYPE_SCALAR, "conv2_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_2_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_2_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_2_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_2_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2_p2, VX_TYPE_SCALAR, "conv2_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_2_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_2_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_2_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_2_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2_p3, VX_TYPE_SCALAR, "conv2_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_2_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_2_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_2_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_2_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_2_p4, VX_TYPE_TENSOR, "conv2_relu_3x3_4");

    org_khronos_nn_extension_normalization_layer_1_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_normalization_layer_1_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p1, VX_TYPE_SCALAR, "conv2_norm2_1");

    org_khronos_nn_extension_normalization_layer_1_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_normalization_layer_1_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p2, VX_TYPE_SCALAR, "conv2_norm2_2");

    org_khronos_nn_extension_normalization_layer_1_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_normalization_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p3, VX_TYPE_SCALAR, "conv2_norm2_3");

    org_khronos_nn_extension_normalization_layer_1_p4 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_normalization_layer_1_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p4, VX_TYPE_SCALAR, "conv2_norm2_4");

    org_khronos_nn_extension_normalization_layer_1_p5 = vxCreateTensor(context, 4, org_khronos_nn_extension_normalization_layer_1_p5Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_normalization_layer_1_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_normalization_layer_1_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p5, VX_TYPE_TENSOR, "conv2_norm2_5");

    org_khronos_nn_extension_pooling_layer_1_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p1, VX_TYPE_SCALAR, "pool2_3x3_s2_1");

    org_khronos_nn_extension_pooling_layer_1_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p2, VX_TYPE_SCALAR, "pool2_3x3_s2_2");

    org_khronos_nn_extension_pooling_layer_1_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p3, VX_TYPE_SCALAR, "pool2_3x3_s2_3");

    org_khronos_nn_extension_pooling_layer_1_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p4, VX_TYPE_SCALAR, "pool2_3x3_s2_4");

    org_khronos_nn_extension_pooling_layer_1_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p5, VX_TYPE_SCALAR, "pool2_3x3_s2_5");

    org_khronos_nn_extension_pooling_layer_1_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_1_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p6, VX_TYPE_SCALAR, "pool2_3x3_s2_6");

    org_khronos_nn_extension_pooling_layer_1_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_1_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_1_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_1_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p7, VX_TYPE_TENSOR, "pool2_3x3_s2_7");

    org_khronos_nn_extension_convolution_layer_8_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_8_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_8_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p1, VX_TYPE_TENSOR, "inception_3a_1x1_weights");

    org_khronos_nn_extension_convolution_layer_8_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_8_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_8_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p2, VX_TYPE_TENSOR, "inception_3a_1x1_bias");

    org_khronos_nn_extension_convolution_layer_8_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_8_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_8_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p3, VX_TYPE_SCALAR, "inception_3a_1x1_3");

    org_khronos_nn_extension_convolution_layer_8_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_8_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_8_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p4, VX_TYPE_SCALAR, "inception_3a_1x1_4");

    org_khronos_nn_extension_convolution_layer_8_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_8_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_8_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p5, VX_TYPE_SCALAR, "inception_3a_1x1_5");

    org_khronos_nn_extension_convolution_layer_8_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_8_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_8_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p6, VX_TYPE_SCALAR, "inception_3a_1x1_6");

    org_khronos_nn_extension_convolution_layer_8_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_8_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_8_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p7, VX_TYPE_SCALAR, "inception_3a_1x1_7");

    org_khronos_nn_extension_convolution_layer_8_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_8_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_8_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p8, VX_TYPE_TENSOR, "inception_3a_1x1_8");

    org_khronos_nn_extension_convolution_layer_6_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_6_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p1, VX_TYPE_TENSOR, "inception_3a_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_6_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_6_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p2, VX_TYPE_TENSOR, "inception_3a_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_6_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p3, VX_TYPE_SCALAR, "inception_3a_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_6_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p4, VX_TYPE_SCALAR, "inception_3a_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_6_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p5, VX_TYPE_SCALAR, "inception_3a_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_6_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p6, VX_TYPE_SCALAR, "inception_3a_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_6_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_6_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p7, VX_TYPE_SCALAR, "inception_3a_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_6_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_6_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_6_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p8, VX_TYPE_TENSOR, "inception_3a_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_4_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_4_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p1, VX_TYPE_TENSOR, "inception_3a_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_4_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_4_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p2, VX_TYPE_TENSOR, "inception_3a_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_4_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p3, VX_TYPE_SCALAR, "inception_3a_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_4_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p4, VX_TYPE_SCALAR, "inception_3a_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_4_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p5, VX_TYPE_SCALAR, "inception_3a_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_4_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p6, VX_TYPE_SCALAR, "inception_3a_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_4_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_4_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p7, VX_TYPE_SCALAR, "inception_3a_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_4_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_4_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_4_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p8, VX_TYPE_TENSOR, "inception_3a_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_2_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p1, VX_TYPE_SCALAR, "inception_3a_pool_1");

    org_khronos_nn_extension_pooling_layer_2_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p2, VX_TYPE_SCALAR, "inception_3a_pool_2");

    org_khronos_nn_extension_pooling_layer_2_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p3, VX_TYPE_SCALAR, "inception_3a_pool_3");

    org_khronos_nn_extension_pooling_layer_2_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p4, VX_TYPE_SCALAR, "inception_3a_pool_4");

    org_khronos_nn_extension_pooling_layer_2_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p5, VX_TYPE_SCALAR, "inception_3a_pool_5");

    org_khronos_nn_extension_pooling_layer_2_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_2_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p6, VX_TYPE_SCALAR, "inception_3a_pool_6");

    org_khronos_nn_extension_pooling_layer_2_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_2_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_2_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_2_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p7, VX_TYPE_TENSOR, "inception_3a_pool_7");

    org_khronos_nn_extension_activation_layer_8_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_8_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_8_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_8_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_8_p1, VX_TYPE_SCALAR, "inception_3a_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_8_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_8_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_8_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_8_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_8_p2, VX_TYPE_SCALAR, "inception_3a_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_8_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_8_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_8_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_8_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_8_p3, VX_TYPE_SCALAR, "inception_3a_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_8_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_0_p0, 4, org_khronos_nn_extension_activation_layer_8_p4_view_view_start, org_khronos_nn_extension_activation_layer_8_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_8_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_8_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_8_p4, VX_TYPE_TENSOR, "inception_3a_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_6_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_6_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_6_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_6_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6_p1, VX_TYPE_SCALAR, "inception_3a_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_6_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_6_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_6_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_6_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6_p2, VX_TYPE_SCALAR, "inception_3a_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_6_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_6_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_6_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_6_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6_p3, VX_TYPE_SCALAR, "inception_3a_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_6_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_6_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_6_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_6_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_6_p4, VX_TYPE_TENSOR, "inception_3a_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_4_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_4_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_4_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_4_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4_p1, VX_TYPE_SCALAR, "inception_3a_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_4_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_4_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_4_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_4_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4_p2, VX_TYPE_SCALAR, "inception_3a_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_4_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_4_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_4_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_4_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4_p3, VX_TYPE_SCALAR, "inception_3a_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_4_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_4_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_4_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_4_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_4_p4, VX_TYPE_TENSOR, "inception_3a_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_3_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_3_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p1, VX_TYPE_TENSOR, "inception_3a_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_3_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_3_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p2, VX_TYPE_TENSOR, "inception_3a_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_3_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p3, VX_TYPE_SCALAR, "inception_3a_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_3_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p4, VX_TYPE_SCALAR, "inception_3a_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_3_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p5, VX_TYPE_SCALAR, "inception_3a_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_3_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p6, VX_TYPE_SCALAR, "inception_3a_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_3_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_3_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p7, VX_TYPE_SCALAR, "inception_3a_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_3_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_3_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_3_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p8, VX_TYPE_TENSOR, "inception_3a_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_7_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_7_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p1, VX_TYPE_TENSOR, "inception_3a_3x3_weights");

    org_khronos_nn_extension_convolution_layer_7_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_7_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p2, VX_TYPE_TENSOR, "inception_3a_3x3_bias");

    org_khronos_nn_extension_convolution_layer_7_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p3, VX_TYPE_SCALAR, "inception_3a_3x3_3");

    org_khronos_nn_extension_convolution_layer_7_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p4, VX_TYPE_SCALAR, "inception_3a_3x3_4");

    org_khronos_nn_extension_convolution_layer_7_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p5, VX_TYPE_SCALAR, "inception_3a_3x3_5");

    org_khronos_nn_extension_convolution_layer_7_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p6, VX_TYPE_SCALAR, "inception_3a_3x3_6");

    org_khronos_nn_extension_convolution_layer_7_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_7_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p7, VX_TYPE_SCALAR, "inception_3a_3x3_7");

    org_khronos_nn_extension_convolution_layer_7_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_7_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_7_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p8, VX_TYPE_TENSOR, "inception_3a_3x3_8");

    org_khronos_nn_extension_convolution_layer_5_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_5_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p1, VX_TYPE_TENSOR, "inception_3a_5x5_weights");

    org_khronos_nn_extension_convolution_layer_5_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_5_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p2, VX_TYPE_TENSOR, "inception_3a_5x5_bias");

    org_khronos_nn_extension_convolution_layer_5_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p3, VX_TYPE_SCALAR, "inception_3a_5x5_3");

    org_khronos_nn_extension_convolution_layer_5_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p4, VX_TYPE_SCALAR, "inception_3a_5x5_4");

    org_khronos_nn_extension_convolution_layer_5_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p5, VX_TYPE_SCALAR, "inception_3a_5x5_5");

    org_khronos_nn_extension_convolution_layer_5_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p6, VX_TYPE_SCALAR, "inception_3a_5x5_6");

    org_khronos_nn_extension_convolution_layer_5_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_5_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p7, VX_TYPE_SCALAR, "inception_3a_5x5_7");

    org_khronos_nn_extension_convolution_layer_5_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_5_p8Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_5_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p8, VX_TYPE_TENSOR, "inception_3a_5x5_8");

    org_khronos_nn_extension_activation_layer_3_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_3_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_3_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_3_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3_p1, VX_TYPE_SCALAR, "inception_3a_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_3_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_3_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_3_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_3_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3_p2, VX_TYPE_SCALAR, "inception_3a_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_3_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_3_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_3_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_3_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3_p3, VX_TYPE_SCALAR, "inception_3a_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_3_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_0_p0, 4, org_khronos_nn_extension_activation_layer_3_p4_view_view_start, org_khronos_nn_extension_activation_layer_3_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_3_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_3_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_3_p4, VX_TYPE_TENSOR, "inception_3a_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_7_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_7_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_7_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_7_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_7_p1, VX_TYPE_SCALAR, "inception_3a_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_7_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_7_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_7_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_7_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_7_p2, VX_TYPE_SCALAR, "inception_3a_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_7_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_7_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_7_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_7_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_7_p3, VX_TYPE_SCALAR, "inception_3a_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_7_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_0_p0, 4, org_khronos_nn_extension_activation_layer_7_p4_view_view_start, org_khronos_nn_extension_activation_layer_7_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_7_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_7_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_7_p4, VX_TYPE_TENSOR, "inception_3a_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_5_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_5_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_5_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_5_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5_p1, VX_TYPE_SCALAR, "inception_3a_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_5_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_5_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_5_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_5_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5_p2, VX_TYPE_SCALAR, "inception_3a_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_5_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_5_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_5_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_5_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5_p3, VX_TYPE_SCALAR, "inception_3a_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_5_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_0_p0, 4, org_khronos_nn_extension_activation_layer_5_p4_view_view_start, org_khronos_nn_extension_activation_layer_5_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_5_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_5_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_5_p4, VX_TYPE_TENSOR, "inception_3a_relu_5x5_4");

    org_khronos_nn_extension_convolution_layer_14_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_14_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_14_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p1, VX_TYPE_TENSOR, "inception_3b_1x1_weights");

    org_khronos_nn_extension_convolution_layer_14_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_14_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_14_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p2, VX_TYPE_TENSOR, "inception_3b_1x1_bias");

    org_khronos_nn_extension_convolution_layer_14_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_14_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_14_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p3, VX_TYPE_SCALAR, "inception_3b_1x1_3");

    org_khronos_nn_extension_convolution_layer_14_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_14_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_14_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p4, VX_TYPE_SCALAR, "inception_3b_1x1_4");

    org_khronos_nn_extension_convolution_layer_14_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_14_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_14_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p5, VX_TYPE_SCALAR, "inception_3b_1x1_5");

    org_khronos_nn_extension_convolution_layer_14_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_14_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_14_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p6, VX_TYPE_SCALAR, "inception_3b_1x1_6");

    org_khronos_nn_extension_convolution_layer_14_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_14_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_14_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p7, VX_TYPE_SCALAR, "inception_3b_1x1_7");

    org_khronos_nn_extension_convolution_layer_14_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_14_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_14_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p8, VX_TYPE_TENSOR, "inception_3b_1x1_8");

    org_khronos_nn_extension_convolution_layer_12_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_12_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_12_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p1, VX_TYPE_TENSOR, "inception_3b_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_12_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_12_p2Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_12_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p2, VX_TYPE_TENSOR, "inception_3b_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_12_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_12_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_12_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p3, VX_TYPE_SCALAR, "inception_3b_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_12_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_12_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_12_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p4, VX_TYPE_SCALAR, "inception_3b_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_12_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_12_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_12_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p5, VX_TYPE_SCALAR, "inception_3b_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_12_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_12_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_12_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p6, VX_TYPE_SCALAR, "inception_3b_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_12_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_12_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_12_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p7, VX_TYPE_SCALAR, "inception_3b_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_12_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_12_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_12_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p8, VX_TYPE_TENSOR, "inception_3b_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_10_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_10_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_10_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p1, VX_TYPE_TENSOR, "inception_3b_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_10_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_10_p2Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_10_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p2, VX_TYPE_TENSOR, "inception_3b_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_10_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_10_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_10_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p3, VX_TYPE_SCALAR, "inception_3b_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_10_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_10_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_10_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p4, VX_TYPE_SCALAR, "inception_3b_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_10_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_10_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_10_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p5, VX_TYPE_SCALAR, "inception_3b_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_10_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_10_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_10_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p6, VX_TYPE_SCALAR, "inception_3b_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_10_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_10_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_10_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p7, VX_TYPE_SCALAR, "inception_3b_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_10_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_10_p8Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_10_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p8, VX_TYPE_TENSOR, "inception_3b_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_3_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_3_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_3_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_3_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p1, VX_TYPE_SCALAR, "inception_3b_pool_1");

    org_khronos_nn_extension_pooling_layer_3_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_3_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_3_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_3_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p2, VX_TYPE_SCALAR, "inception_3b_pool_2");

    org_khronos_nn_extension_pooling_layer_3_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_3_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_3_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_3_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p3, VX_TYPE_SCALAR, "inception_3b_pool_3");

    org_khronos_nn_extension_pooling_layer_3_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_3_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_3_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_3_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p4, VX_TYPE_SCALAR, "inception_3b_pool_4");

    org_khronos_nn_extension_pooling_layer_3_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_3_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_3_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_3_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p5, VX_TYPE_SCALAR, "inception_3b_pool_5");

    org_khronos_nn_extension_pooling_layer_3_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_3_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_3_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_3_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p6, VX_TYPE_SCALAR, "inception_3b_pool_6");

    org_khronos_nn_extension_pooling_layer_3_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_3_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_3_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_3_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p7, VX_TYPE_TENSOR, "inception_3b_pool_7");

    org_khronos_nn_extension_activation_layer_14_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_14_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_14_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_14_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_14_p1, VX_TYPE_SCALAR, "inception_3b_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_14_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_14_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_14_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_14_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_14_p2, VX_TYPE_SCALAR, "inception_3b_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_14_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_14_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_14_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_14_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_14_p3, VX_TYPE_SCALAR, "inception_3b_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_14_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_1_p0, 4, org_khronos_nn_extension_activation_layer_14_p4_view_view_start, org_khronos_nn_extension_activation_layer_14_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_14_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_14_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_14_p4, VX_TYPE_TENSOR, "inception_3b_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_12_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_12_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_12_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_12_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_12_p1, VX_TYPE_SCALAR, "inception_3b_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_12_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_12_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_12_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_12_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_12_p2, VX_TYPE_SCALAR, "inception_3b_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_12_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_12_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_12_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_12_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_12_p3, VX_TYPE_SCALAR, "inception_3b_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_12_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_12_p4Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_12_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_12_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_12_p4, VX_TYPE_TENSOR, "inception_3b_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_10_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_10_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_10_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_10_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_10_p1, VX_TYPE_SCALAR, "inception_3b_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_10_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_10_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_10_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_10_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_10_p2, VX_TYPE_SCALAR, "inception_3b_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_10_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_10_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_10_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_10_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_10_p3, VX_TYPE_SCALAR, "inception_3b_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_10_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_10_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_10_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_10_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_10_p4, VX_TYPE_TENSOR, "inception_3b_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_9_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_9_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_9_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p1, VX_TYPE_TENSOR, "inception_3b_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_9_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_9_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_9_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p2, VX_TYPE_TENSOR, "inception_3b_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_9_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_9_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_9_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p3, VX_TYPE_SCALAR, "inception_3b_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_9_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_9_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_9_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p4, VX_TYPE_SCALAR, "inception_3b_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_9_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_9_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_9_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p5, VX_TYPE_SCALAR, "inception_3b_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_9_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_9_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_9_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p6, VX_TYPE_SCALAR, "inception_3b_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_9_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_9_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_9_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p7, VX_TYPE_SCALAR, "inception_3b_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_9_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_9_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_9_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p8, VX_TYPE_TENSOR, "inception_3b_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_13_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_13_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_13_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p1, VX_TYPE_TENSOR, "inception_3b_3x3_weights");

    org_khronos_nn_extension_convolution_layer_13_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_13_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_13_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p2, VX_TYPE_TENSOR, "inception_3b_3x3_bias");

    org_khronos_nn_extension_convolution_layer_13_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_13_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_13_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p3, VX_TYPE_SCALAR, "inception_3b_3x3_3");

    org_khronos_nn_extension_convolution_layer_13_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_13_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_13_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p4, VX_TYPE_SCALAR, "inception_3b_3x3_4");

    org_khronos_nn_extension_convolution_layer_13_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_13_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_13_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p5, VX_TYPE_SCALAR, "inception_3b_3x3_5");

    org_khronos_nn_extension_convolution_layer_13_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_13_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_13_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p6, VX_TYPE_SCALAR, "inception_3b_3x3_6");

    org_khronos_nn_extension_convolution_layer_13_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_13_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_13_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p7, VX_TYPE_SCALAR, "inception_3b_3x3_7");

    org_khronos_nn_extension_convolution_layer_13_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_13_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_13_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p8, VX_TYPE_TENSOR, "inception_3b_3x3_8");

    org_khronos_nn_extension_convolution_layer_11_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_11_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_11_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p1, VX_TYPE_TENSOR, "inception_3b_5x5_weights");

    org_khronos_nn_extension_convolution_layer_11_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_11_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_11_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p2, VX_TYPE_TENSOR, "inception_3b_5x5_bias");

    org_khronos_nn_extension_convolution_layer_11_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_11_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_11_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p3, VX_TYPE_SCALAR, "inception_3b_5x5_3");

    org_khronos_nn_extension_convolution_layer_11_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_11_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_11_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p4, VX_TYPE_SCALAR, "inception_3b_5x5_4");

    org_khronos_nn_extension_convolution_layer_11_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_11_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_11_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p5, VX_TYPE_SCALAR, "inception_3b_5x5_5");

    org_khronos_nn_extension_convolution_layer_11_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_11_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_11_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p6, VX_TYPE_SCALAR, "inception_3b_5x5_6");

    org_khronos_nn_extension_convolution_layer_11_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_11_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_11_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p7, VX_TYPE_SCALAR, "inception_3b_5x5_7");

    org_khronos_nn_extension_convolution_layer_11_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_11_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_11_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p8, VX_TYPE_TENSOR, "inception_3b_5x5_8");

    org_khronos_nn_extension_activation_layer_9_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_9_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_9_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_9_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_9_p1, VX_TYPE_SCALAR, "inception_3b_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_9_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_9_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_9_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_9_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_9_p2, VX_TYPE_SCALAR, "inception_3b_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_9_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_9_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_9_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_9_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_9_p3, VX_TYPE_SCALAR, "inception_3b_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_9_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_1_p0, 4, org_khronos_nn_extension_activation_layer_9_p4_view_view_start, org_khronos_nn_extension_activation_layer_9_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_9_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_9_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_9_p4, VX_TYPE_TENSOR, "inception_3b_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_13_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_13_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_13_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_13_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_13_p1, VX_TYPE_SCALAR, "inception_3b_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_13_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_13_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_13_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_13_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_13_p2, VX_TYPE_SCALAR, "inception_3b_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_13_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_13_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_13_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_13_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_13_p3, VX_TYPE_SCALAR, "inception_3b_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_13_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_1_p0, 4, org_khronos_nn_extension_activation_layer_13_p4_view_view_start, org_khronos_nn_extension_activation_layer_13_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_13_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_13_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_13_p4, VX_TYPE_TENSOR, "inception_3b_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_11_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_11_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_11_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_11_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_11_p1, VX_TYPE_SCALAR, "inception_3b_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_11_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_11_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_11_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_11_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_11_p2, VX_TYPE_SCALAR, "inception_3b_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_11_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_11_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_11_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_11_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_11_p3, VX_TYPE_SCALAR, "inception_3b_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_11_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_1_p0, 4, org_khronos_nn_extension_activation_layer_11_p4_view_view_start, org_khronos_nn_extension_activation_layer_11_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_11_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_11_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_11_p4, VX_TYPE_TENSOR, "inception_3b_relu_5x5_4");

    org_khronos_nn_extension_pooling_layer_4_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_4_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_4_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_4_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p1, VX_TYPE_SCALAR, "pool3_3x3_s2_1");

    org_khronos_nn_extension_pooling_layer_4_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_4_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_4_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_4_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p2, VX_TYPE_SCALAR, "pool3_3x3_s2_2");

    org_khronos_nn_extension_pooling_layer_4_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_4_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_4_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_4_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p3, VX_TYPE_SCALAR, "pool3_3x3_s2_3");

    org_khronos_nn_extension_pooling_layer_4_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_4_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_4_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_4_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p4, VX_TYPE_SCALAR, "pool3_3x3_s2_4");

    org_khronos_nn_extension_pooling_layer_4_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_4_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_4_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_4_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p5, VX_TYPE_SCALAR, "pool3_3x3_s2_5");

    org_khronos_nn_extension_pooling_layer_4_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_4_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_4_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_4_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p6, VX_TYPE_SCALAR, "pool3_3x3_s2_6");

    org_khronos_nn_extension_pooling_layer_4_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_4_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_4_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_4_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p7, VX_TYPE_TENSOR, "pool3_3x3_s2_7");

    org_khronos_nn_extension_convolution_layer_20_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_20_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_20_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p1, VX_TYPE_TENSOR, "inception_4a_1x1_weights");

    org_khronos_nn_extension_convolution_layer_20_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_20_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_20_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p2, VX_TYPE_TENSOR, "inception_4a_1x1_bias");

    org_khronos_nn_extension_convolution_layer_20_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_20_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_20_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p3, VX_TYPE_SCALAR, "inception_4a_1x1_3");

    org_khronos_nn_extension_convolution_layer_20_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_20_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_20_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p4, VX_TYPE_SCALAR, "inception_4a_1x1_4");

    org_khronos_nn_extension_convolution_layer_20_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_20_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_20_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p5, VX_TYPE_SCALAR, "inception_4a_1x1_5");

    org_khronos_nn_extension_convolution_layer_20_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_20_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_20_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p6, VX_TYPE_SCALAR, "inception_4a_1x1_6");

    org_khronos_nn_extension_convolution_layer_20_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_20_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_20_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p7, VX_TYPE_SCALAR, "inception_4a_1x1_7");

    org_khronos_nn_extension_convolution_layer_20_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_20_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_20_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p8, VX_TYPE_TENSOR, "inception_4a_1x1_8");

    org_khronos_nn_extension_convolution_layer_18_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_18_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_18_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p1, VX_TYPE_TENSOR, "inception_4a_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_18_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_18_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_18_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p2, VX_TYPE_TENSOR, "inception_4a_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_18_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_18_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_18_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p3, VX_TYPE_SCALAR, "inception_4a_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_18_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_18_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_18_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p4, VX_TYPE_SCALAR, "inception_4a_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_18_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_18_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_18_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p5, VX_TYPE_SCALAR, "inception_4a_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_18_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_18_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_18_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p6, VX_TYPE_SCALAR, "inception_4a_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_18_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_18_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_18_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p7, VX_TYPE_SCALAR, "inception_4a_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_18_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_18_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_18_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p8, VX_TYPE_TENSOR, "inception_4a_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_16_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_16_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_16_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p1, VX_TYPE_TENSOR, "inception_4a_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_16_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_16_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_16_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p2, VX_TYPE_TENSOR, "inception_4a_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_16_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_16_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_16_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p3, VX_TYPE_SCALAR, "inception_4a_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_16_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_16_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_16_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p4, VX_TYPE_SCALAR, "inception_4a_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_16_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_16_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_16_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p5, VX_TYPE_SCALAR, "inception_4a_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_16_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_16_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_16_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p6, VX_TYPE_SCALAR, "inception_4a_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_16_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_16_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_16_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p7, VX_TYPE_SCALAR, "inception_4a_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_16_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_16_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_16_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p8, VX_TYPE_TENSOR, "inception_4a_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_5_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_5_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_5_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_5_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p1, VX_TYPE_SCALAR, "inception_4a_pool_1");

    org_khronos_nn_extension_pooling_layer_5_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_5_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_5_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_5_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p2, VX_TYPE_SCALAR, "inception_4a_pool_2");

    org_khronos_nn_extension_pooling_layer_5_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_5_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_5_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_5_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p3, VX_TYPE_SCALAR, "inception_4a_pool_3");

    org_khronos_nn_extension_pooling_layer_5_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_5_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_5_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_5_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p4, VX_TYPE_SCALAR, "inception_4a_pool_4");

    org_khronos_nn_extension_pooling_layer_5_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_5_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_5_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_5_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p5, VX_TYPE_SCALAR, "inception_4a_pool_5");

    org_khronos_nn_extension_pooling_layer_5_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_5_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_5_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_5_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p6, VX_TYPE_SCALAR, "inception_4a_pool_6");

    org_khronos_nn_extension_pooling_layer_5_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_5_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_5_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_5_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p7, VX_TYPE_TENSOR, "inception_4a_pool_7");

    org_khronos_nn_extension_activation_layer_20_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_20_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_20_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_20_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_20_p1, VX_TYPE_SCALAR, "inception_4a_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_20_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_20_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_20_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_20_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_20_p2, VX_TYPE_SCALAR, "inception_4a_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_20_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_20_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_20_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_20_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_20_p3, VX_TYPE_SCALAR, "inception_4a_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_20_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_2_p0, 4, org_khronos_nn_extension_activation_layer_20_p4_view_view_start, org_khronos_nn_extension_activation_layer_20_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_20_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_20_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_20_p4, VX_TYPE_TENSOR, "inception_4a_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_18_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_18_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_18_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_18_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_18_p1, VX_TYPE_SCALAR, "inception_4a_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_18_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_18_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_18_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_18_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_18_p2, VX_TYPE_SCALAR, "inception_4a_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_18_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_18_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_18_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_18_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_18_p3, VX_TYPE_SCALAR, "inception_4a_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_18_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_18_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_18_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_18_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_18_p4, VX_TYPE_TENSOR, "inception_4a_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_16_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_16_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_16_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_16_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_16_p1, VX_TYPE_SCALAR, "inception_4a_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_16_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_16_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_16_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_16_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_16_p2, VX_TYPE_SCALAR, "inception_4a_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_16_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_16_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_16_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_16_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_16_p3, VX_TYPE_SCALAR, "inception_4a_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_16_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_16_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_16_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_16_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_16_p4, VX_TYPE_TENSOR, "inception_4a_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_15_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_15_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_15_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p1, VX_TYPE_TENSOR, "inception_4a_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_15_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_15_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_15_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p2, VX_TYPE_TENSOR, "inception_4a_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_15_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_15_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_15_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p3, VX_TYPE_SCALAR, "inception_4a_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_15_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_15_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_15_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p4, VX_TYPE_SCALAR, "inception_4a_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_15_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_15_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_15_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p5, VX_TYPE_SCALAR, "inception_4a_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_15_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_15_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_15_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p6, VX_TYPE_SCALAR, "inception_4a_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_15_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_15_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_15_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p7, VX_TYPE_SCALAR, "inception_4a_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_15_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_15_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_15_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p8, VX_TYPE_TENSOR, "inception_4a_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_19_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_19_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_19_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p1, VX_TYPE_TENSOR, "inception_4a_3x3_weights");

    org_khronos_nn_extension_convolution_layer_19_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_19_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_19_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p2, VX_TYPE_TENSOR, "inception_4a_3x3_bias");

    org_khronos_nn_extension_convolution_layer_19_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_19_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_19_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p3, VX_TYPE_SCALAR, "inception_4a_3x3_3");

    org_khronos_nn_extension_convolution_layer_19_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_19_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_19_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p4, VX_TYPE_SCALAR, "inception_4a_3x3_4");

    org_khronos_nn_extension_convolution_layer_19_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_19_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_19_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p5, VX_TYPE_SCALAR, "inception_4a_3x3_5");

    org_khronos_nn_extension_convolution_layer_19_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_19_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_19_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p6, VX_TYPE_SCALAR, "inception_4a_3x3_6");

    org_khronos_nn_extension_convolution_layer_19_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_19_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_19_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p7, VX_TYPE_SCALAR, "inception_4a_3x3_7");

    org_khronos_nn_extension_convolution_layer_19_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_19_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_19_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p8, VX_TYPE_TENSOR, "inception_4a_3x3_8");

    org_khronos_nn_extension_convolution_layer_17_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_17_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_17_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p1, VX_TYPE_TENSOR, "inception_4a_5x5_weights");

    org_khronos_nn_extension_convolution_layer_17_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_17_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_17_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p2, VX_TYPE_TENSOR, "inception_4a_5x5_bias");

    org_khronos_nn_extension_convolution_layer_17_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_17_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_17_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p3, VX_TYPE_SCALAR, "inception_4a_5x5_3");

    org_khronos_nn_extension_convolution_layer_17_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_17_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_17_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p4, VX_TYPE_SCALAR, "inception_4a_5x5_4");

    org_khronos_nn_extension_convolution_layer_17_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_17_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_17_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p5, VX_TYPE_SCALAR, "inception_4a_5x5_5");

    org_khronos_nn_extension_convolution_layer_17_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_17_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_17_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p6, VX_TYPE_SCALAR, "inception_4a_5x5_6");

    org_khronos_nn_extension_convolution_layer_17_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_17_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_17_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p7, VX_TYPE_SCALAR, "inception_4a_5x5_7");

    org_khronos_nn_extension_convolution_layer_17_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_17_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_17_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p8, VX_TYPE_TENSOR, "inception_4a_5x5_8");

    org_khronos_nn_extension_activation_layer_15_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_15_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_15_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_15_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_15_p1, VX_TYPE_SCALAR, "inception_4a_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_15_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_15_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_15_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_15_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_15_p2, VX_TYPE_SCALAR, "inception_4a_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_15_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_15_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_15_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_15_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_15_p3, VX_TYPE_SCALAR, "inception_4a_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_15_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_2_p0, 4, org_khronos_nn_extension_activation_layer_15_p4_view_view_start, org_khronos_nn_extension_activation_layer_15_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_15_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_15_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_15_p4, VX_TYPE_TENSOR, "inception_4a_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_19_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_19_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_19_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_19_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_19_p1, VX_TYPE_SCALAR, "inception_4a_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_19_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_19_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_19_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_19_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_19_p2, VX_TYPE_SCALAR, "inception_4a_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_19_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_19_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_19_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_19_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_19_p3, VX_TYPE_SCALAR, "inception_4a_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_19_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_2_p0, 4, org_khronos_nn_extension_activation_layer_19_p4_view_view_start, org_khronos_nn_extension_activation_layer_19_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_19_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_19_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_19_p4, VX_TYPE_TENSOR, "inception_4a_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_17_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_17_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_17_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_17_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_17_p1, VX_TYPE_SCALAR, "inception_4a_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_17_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_17_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_17_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_17_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_17_p2, VX_TYPE_SCALAR, "inception_4a_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_17_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_17_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_17_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_17_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_17_p3, VX_TYPE_SCALAR, "inception_4a_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_17_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_2_p0, 4, org_khronos_nn_extension_activation_layer_17_p4_view_view_start, org_khronos_nn_extension_activation_layer_17_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_17_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_17_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_17_p4, VX_TYPE_TENSOR, "inception_4a_relu_5x5_4");

    org_khronos_nn_extension_convolution_layer_26_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_26_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_26_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p1, VX_TYPE_TENSOR, "inception_4b_1x1_weights");

    org_khronos_nn_extension_convolution_layer_26_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_26_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_26_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p2, VX_TYPE_TENSOR, "inception_4b_1x1_bias");

    org_khronos_nn_extension_convolution_layer_26_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_26_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_26_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p3, VX_TYPE_SCALAR, "inception_4b_1x1_3");

    org_khronos_nn_extension_convolution_layer_26_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_26_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_26_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p4, VX_TYPE_SCALAR, "inception_4b_1x1_4");

    org_khronos_nn_extension_convolution_layer_26_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_26_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_26_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p5, VX_TYPE_SCALAR, "inception_4b_1x1_5");

    org_khronos_nn_extension_convolution_layer_26_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_26_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_26_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p6, VX_TYPE_SCALAR, "inception_4b_1x1_6");

    org_khronos_nn_extension_convolution_layer_26_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_26_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_26_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p7, VX_TYPE_SCALAR, "inception_4b_1x1_7");

    org_khronos_nn_extension_convolution_layer_26_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_26_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_26_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p8, VX_TYPE_TENSOR, "inception_4b_1x1_8");

    org_khronos_nn_extension_convolution_layer_24_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_24_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_24_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p1, VX_TYPE_TENSOR, "inception_4b_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_24_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_24_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_24_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p2, VX_TYPE_TENSOR, "inception_4b_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_24_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_24_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_24_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p3, VX_TYPE_SCALAR, "inception_4b_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_24_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_24_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_24_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p4, VX_TYPE_SCALAR, "inception_4b_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_24_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_24_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_24_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p5, VX_TYPE_SCALAR, "inception_4b_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_24_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_24_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_24_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p6, VX_TYPE_SCALAR, "inception_4b_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_24_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_24_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_24_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p7, VX_TYPE_SCALAR, "inception_4b_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_24_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_24_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_24_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p8, VX_TYPE_TENSOR, "inception_4b_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_22_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_22_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_22_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p1, VX_TYPE_TENSOR, "inception_4b_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_22_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_22_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_22_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p2, VX_TYPE_TENSOR, "inception_4b_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_22_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_22_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_22_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p3, VX_TYPE_SCALAR, "inception_4b_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_22_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_22_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_22_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p4, VX_TYPE_SCALAR, "inception_4b_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_22_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_22_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_22_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p5, VX_TYPE_SCALAR, "inception_4b_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_22_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_22_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_22_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p6, VX_TYPE_SCALAR, "inception_4b_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_22_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_22_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_22_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p7, VX_TYPE_SCALAR, "inception_4b_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_22_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_22_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_22_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p8, VX_TYPE_TENSOR, "inception_4b_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_6_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_6_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_6_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_6_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p1, VX_TYPE_SCALAR, "inception_4b_pool_1");

    org_khronos_nn_extension_pooling_layer_6_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_6_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_6_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_6_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p2, VX_TYPE_SCALAR, "inception_4b_pool_2");

    org_khronos_nn_extension_pooling_layer_6_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_6_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_6_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_6_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p3, VX_TYPE_SCALAR, "inception_4b_pool_3");

    org_khronos_nn_extension_pooling_layer_6_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_6_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_6_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_6_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p4, VX_TYPE_SCALAR, "inception_4b_pool_4");

    org_khronos_nn_extension_pooling_layer_6_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_6_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_6_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_6_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p5, VX_TYPE_SCALAR, "inception_4b_pool_5");

    org_khronos_nn_extension_pooling_layer_6_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_6_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_6_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_6_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p6, VX_TYPE_SCALAR, "inception_4b_pool_6");

    org_khronos_nn_extension_pooling_layer_6_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_6_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_6_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_6_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p7, VX_TYPE_TENSOR, "inception_4b_pool_7");

    org_khronos_nn_extension_activation_layer_26_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_26_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_26_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_26_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_26_p1, VX_TYPE_SCALAR, "inception_4b_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_26_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_26_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_26_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_26_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_26_p2, VX_TYPE_SCALAR, "inception_4b_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_26_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_26_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_26_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_26_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_26_p3, VX_TYPE_SCALAR, "inception_4b_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_26_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_3_p0, 4, org_khronos_nn_extension_activation_layer_26_p4_view_view_start, org_khronos_nn_extension_activation_layer_26_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_26_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_26_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_26_p4, VX_TYPE_TENSOR, "inception_4b_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_24_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_24_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_24_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_24_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_24_p1, VX_TYPE_SCALAR, "inception_4b_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_24_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_24_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_24_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_24_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_24_p2, VX_TYPE_SCALAR, "inception_4b_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_24_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_24_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_24_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_24_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_24_p3, VX_TYPE_SCALAR, "inception_4b_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_24_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_24_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_24_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_24_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_24_p4, VX_TYPE_TENSOR, "inception_4b_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_22_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_22_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_22_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_22_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_22_p1, VX_TYPE_SCALAR, "inception_4b_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_22_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_22_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_22_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_22_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_22_p2, VX_TYPE_SCALAR, "inception_4b_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_22_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_22_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_22_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_22_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_22_p3, VX_TYPE_SCALAR, "inception_4b_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_22_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_22_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_22_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_22_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_22_p4, VX_TYPE_TENSOR, "inception_4b_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_21_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_21_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_21_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p1, VX_TYPE_TENSOR, "inception_4b_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_21_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_21_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_21_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p2, VX_TYPE_TENSOR, "inception_4b_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_21_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_21_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_21_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p3, VX_TYPE_SCALAR, "inception_4b_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_21_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_21_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_21_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p4, VX_TYPE_SCALAR, "inception_4b_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_21_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_21_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_21_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p5, VX_TYPE_SCALAR, "inception_4b_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_21_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_21_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_21_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p6, VX_TYPE_SCALAR, "inception_4b_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_21_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_21_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_21_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p7, VX_TYPE_SCALAR, "inception_4b_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_21_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_21_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_21_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p8, VX_TYPE_TENSOR, "inception_4b_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_25_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_25_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_25_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p1, VX_TYPE_TENSOR, "inception_4b_3x3_weights");

    org_khronos_nn_extension_convolution_layer_25_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_25_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_25_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p2, VX_TYPE_TENSOR, "inception_4b_3x3_bias");

    org_khronos_nn_extension_convolution_layer_25_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_25_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_25_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p3, VX_TYPE_SCALAR, "inception_4b_3x3_3");

    org_khronos_nn_extension_convolution_layer_25_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_25_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_25_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p4, VX_TYPE_SCALAR, "inception_4b_3x3_4");

    org_khronos_nn_extension_convolution_layer_25_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_25_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_25_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p5, VX_TYPE_SCALAR, "inception_4b_3x3_5");

    org_khronos_nn_extension_convolution_layer_25_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_25_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_25_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p6, VX_TYPE_SCALAR, "inception_4b_3x3_6");

    org_khronos_nn_extension_convolution_layer_25_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_25_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_25_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p7, VX_TYPE_SCALAR, "inception_4b_3x3_7");

    org_khronos_nn_extension_convolution_layer_25_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_25_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_25_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p8, VX_TYPE_TENSOR, "inception_4b_3x3_8");

    org_khronos_nn_extension_convolution_layer_23_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_23_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_23_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p1, VX_TYPE_TENSOR, "inception_4b_5x5_weights");

    org_khronos_nn_extension_convolution_layer_23_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_23_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_23_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p2, VX_TYPE_TENSOR, "inception_4b_5x5_bias");

    org_khronos_nn_extension_convolution_layer_23_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_23_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_23_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p3, VX_TYPE_SCALAR, "inception_4b_5x5_3");

    org_khronos_nn_extension_convolution_layer_23_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_23_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_23_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p4, VX_TYPE_SCALAR, "inception_4b_5x5_4");

    org_khronos_nn_extension_convolution_layer_23_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_23_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_23_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p5, VX_TYPE_SCALAR, "inception_4b_5x5_5");

    org_khronos_nn_extension_convolution_layer_23_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_23_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_23_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p6, VX_TYPE_SCALAR, "inception_4b_5x5_6");

    org_khronos_nn_extension_convolution_layer_23_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_23_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_23_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p7, VX_TYPE_SCALAR, "inception_4b_5x5_7");

    org_khronos_nn_extension_convolution_layer_23_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_23_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_23_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p8, VX_TYPE_TENSOR, "inception_4b_5x5_8");

    org_khronos_nn_extension_activation_layer_21_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_21_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_21_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_21_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_21_p1, VX_TYPE_SCALAR, "inception_4b_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_21_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_21_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_21_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_21_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_21_p2, VX_TYPE_SCALAR, "inception_4b_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_21_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_21_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_21_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_21_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_21_p3, VX_TYPE_SCALAR, "inception_4b_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_21_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_3_p0, 4, org_khronos_nn_extension_activation_layer_21_p4_view_view_start, org_khronos_nn_extension_activation_layer_21_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_21_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_21_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_21_p4, VX_TYPE_TENSOR, "inception_4b_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_25_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_25_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_25_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_25_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_25_p1, VX_TYPE_SCALAR, "inception_4b_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_25_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_25_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_25_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_25_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_25_p2, VX_TYPE_SCALAR, "inception_4b_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_25_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_25_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_25_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_25_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_25_p3, VX_TYPE_SCALAR, "inception_4b_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_25_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_3_p0, 4, org_khronos_nn_extension_activation_layer_25_p4_view_view_start, org_khronos_nn_extension_activation_layer_25_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_25_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_25_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_25_p4, VX_TYPE_TENSOR, "inception_4b_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_23_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_23_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_23_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_23_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_23_p1, VX_TYPE_SCALAR, "inception_4b_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_23_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_23_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_23_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_23_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_23_p2, VX_TYPE_SCALAR, "inception_4b_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_23_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_23_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_23_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_23_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_23_p3, VX_TYPE_SCALAR, "inception_4b_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_23_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_3_p0, 4, org_khronos_nn_extension_activation_layer_23_p4_view_view_start, org_khronos_nn_extension_activation_layer_23_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_23_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_23_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_23_p4, VX_TYPE_TENSOR, "inception_4b_relu_5x5_4");

    org_khronos_nn_extension_convolution_layer_32_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_32_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_32_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p1, VX_TYPE_TENSOR, "inception_4c_1x1_weights");

    org_khronos_nn_extension_convolution_layer_32_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_32_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_32_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p2, VX_TYPE_TENSOR, "inception_4c_1x1_bias");

    org_khronos_nn_extension_convolution_layer_32_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_32_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_32_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p3, VX_TYPE_SCALAR, "inception_4c_1x1_3");

    org_khronos_nn_extension_convolution_layer_32_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_32_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_32_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p4, VX_TYPE_SCALAR, "inception_4c_1x1_4");

    org_khronos_nn_extension_convolution_layer_32_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_32_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_32_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p5, VX_TYPE_SCALAR, "inception_4c_1x1_5");

    org_khronos_nn_extension_convolution_layer_32_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_32_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_32_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p6, VX_TYPE_SCALAR, "inception_4c_1x1_6");

    org_khronos_nn_extension_convolution_layer_32_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_32_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_32_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p7, VX_TYPE_SCALAR, "inception_4c_1x1_7");

    org_khronos_nn_extension_convolution_layer_32_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_32_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_32_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p8, VX_TYPE_TENSOR, "inception_4c_1x1_8");

    org_khronos_nn_extension_convolution_layer_30_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_30_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_30_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p1, VX_TYPE_TENSOR, "inception_4c_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_30_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_30_p2Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_30_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p2, VX_TYPE_TENSOR, "inception_4c_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_30_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_30_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_30_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p3, VX_TYPE_SCALAR, "inception_4c_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_30_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_30_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_30_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p4, VX_TYPE_SCALAR, "inception_4c_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_30_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_30_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_30_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p5, VX_TYPE_SCALAR, "inception_4c_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_30_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_30_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_30_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p6, VX_TYPE_SCALAR, "inception_4c_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_30_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_30_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_30_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p7, VX_TYPE_SCALAR, "inception_4c_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_30_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_30_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_30_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p8, VX_TYPE_TENSOR, "inception_4c_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_28_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_28_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_28_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p1, VX_TYPE_TENSOR, "inception_4c_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_28_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_28_p2Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_28_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p2, VX_TYPE_TENSOR, "inception_4c_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_28_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_28_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_28_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p3, VX_TYPE_SCALAR, "inception_4c_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_28_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_28_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_28_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p4, VX_TYPE_SCALAR, "inception_4c_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_28_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_28_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_28_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p5, VX_TYPE_SCALAR, "inception_4c_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_28_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_28_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_28_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p6, VX_TYPE_SCALAR, "inception_4c_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_28_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_28_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_28_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p7, VX_TYPE_SCALAR, "inception_4c_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_28_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_28_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_28_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p8, VX_TYPE_TENSOR, "inception_4c_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_7_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_7_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_7_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_7_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p1, VX_TYPE_SCALAR, "inception_4c_pool_1");

    org_khronos_nn_extension_pooling_layer_7_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_7_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_7_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_7_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p2, VX_TYPE_SCALAR, "inception_4c_pool_2");

    org_khronos_nn_extension_pooling_layer_7_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_7_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_7_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_7_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p3, VX_TYPE_SCALAR, "inception_4c_pool_3");

    org_khronos_nn_extension_pooling_layer_7_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_7_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_7_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_7_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p4, VX_TYPE_SCALAR, "inception_4c_pool_4");

    org_khronos_nn_extension_pooling_layer_7_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_7_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_7_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_7_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p5, VX_TYPE_SCALAR, "inception_4c_pool_5");

    org_khronos_nn_extension_pooling_layer_7_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_7_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_7_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_7_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p6, VX_TYPE_SCALAR, "inception_4c_pool_6");

    org_khronos_nn_extension_pooling_layer_7_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_7_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_7_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_7_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p7, VX_TYPE_TENSOR, "inception_4c_pool_7");

    org_khronos_nn_extension_activation_layer_32_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_32_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_32_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_32_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_32_p1, VX_TYPE_SCALAR, "inception_4c_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_32_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_32_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_32_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_32_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_32_p2, VX_TYPE_SCALAR, "inception_4c_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_32_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_32_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_32_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_32_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_32_p3, VX_TYPE_SCALAR, "inception_4c_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_32_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_4_p0, 4, org_khronos_nn_extension_activation_layer_32_p4_view_view_start, org_khronos_nn_extension_activation_layer_32_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_32_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_32_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_32_p4, VX_TYPE_TENSOR, "inception_4c_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_30_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_30_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_30_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_30_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_30_p1, VX_TYPE_SCALAR, "inception_4c_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_30_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_30_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_30_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_30_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_30_p2, VX_TYPE_SCALAR, "inception_4c_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_30_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_30_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_30_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_30_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_30_p3, VX_TYPE_SCALAR, "inception_4c_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_30_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_30_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_30_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_30_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_30_p4, VX_TYPE_TENSOR, "inception_4c_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_28_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_28_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_28_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_28_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_28_p1, VX_TYPE_SCALAR, "inception_4c_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_28_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_28_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_28_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_28_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_28_p2, VX_TYPE_SCALAR, "inception_4c_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_28_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_28_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_28_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_28_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_28_p3, VX_TYPE_SCALAR, "inception_4c_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_28_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_28_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_28_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_28_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_28_p4, VX_TYPE_TENSOR, "inception_4c_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_27_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_27_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_27_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p1, VX_TYPE_TENSOR, "inception_4c_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_27_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_27_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_27_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p2, VX_TYPE_TENSOR, "inception_4c_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_27_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_27_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_27_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p3, VX_TYPE_SCALAR, "inception_4c_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_27_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_27_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_27_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p4, VX_TYPE_SCALAR, "inception_4c_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_27_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_27_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_27_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p5, VX_TYPE_SCALAR, "inception_4c_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_27_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_27_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_27_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p6, VX_TYPE_SCALAR, "inception_4c_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_27_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_27_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_27_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p7, VX_TYPE_SCALAR, "inception_4c_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_27_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_27_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_27_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p8, VX_TYPE_TENSOR, "inception_4c_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_31_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_31_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_31_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p1, VX_TYPE_TENSOR, "inception_4c_3x3_weights");

    org_khronos_nn_extension_convolution_layer_31_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_31_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_31_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p2, VX_TYPE_TENSOR, "inception_4c_3x3_bias");

    org_khronos_nn_extension_convolution_layer_31_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_31_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_31_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p3, VX_TYPE_SCALAR, "inception_4c_3x3_3");

    org_khronos_nn_extension_convolution_layer_31_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_31_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_31_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p4, VX_TYPE_SCALAR, "inception_4c_3x3_4");

    org_khronos_nn_extension_convolution_layer_31_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_31_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_31_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p5, VX_TYPE_SCALAR, "inception_4c_3x3_5");

    org_khronos_nn_extension_convolution_layer_31_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_31_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_31_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p6, VX_TYPE_SCALAR, "inception_4c_3x3_6");

    org_khronos_nn_extension_convolution_layer_31_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_31_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_31_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p7, VX_TYPE_SCALAR, "inception_4c_3x3_7");

    org_khronos_nn_extension_convolution_layer_31_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_31_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_31_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p8, VX_TYPE_TENSOR, "inception_4c_3x3_8");

    org_khronos_nn_extension_convolution_layer_29_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_29_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_29_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p1, VX_TYPE_TENSOR, "inception_4c_5x5_weights");

    org_khronos_nn_extension_convolution_layer_29_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_29_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_29_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p2, VX_TYPE_TENSOR, "inception_4c_5x5_bias");

    org_khronos_nn_extension_convolution_layer_29_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_29_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_29_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p3, VX_TYPE_SCALAR, "inception_4c_5x5_3");

    org_khronos_nn_extension_convolution_layer_29_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_29_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_29_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p4, VX_TYPE_SCALAR, "inception_4c_5x5_4");

    org_khronos_nn_extension_convolution_layer_29_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_29_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_29_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p5, VX_TYPE_SCALAR, "inception_4c_5x5_5");

    org_khronos_nn_extension_convolution_layer_29_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_29_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_29_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p6, VX_TYPE_SCALAR, "inception_4c_5x5_6");

    org_khronos_nn_extension_convolution_layer_29_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_29_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_29_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p7, VX_TYPE_SCALAR, "inception_4c_5x5_7");

    org_khronos_nn_extension_convolution_layer_29_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_29_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_29_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p8, VX_TYPE_TENSOR, "inception_4c_5x5_8");

    org_khronos_nn_extension_activation_layer_27_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_27_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_27_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_27_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_27_p1, VX_TYPE_SCALAR, "inception_4c_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_27_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_27_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_27_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_27_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_27_p2, VX_TYPE_SCALAR, "inception_4c_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_27_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_27_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_27_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_27_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_27_p3, VX_TYPE_SCALAR, "inception_4c_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_27_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_4_p0, 4, org_khronos_nn_extension_activation_layer_27_p4_view_view_start, org_khronos_nn_extension_activation_layer_27_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_27_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_27_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_27_p4, VX_TYPE_TENSOR, "inception_4c_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_31_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_31_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_31_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_31_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_31_p1, VX_TYPE_SCALAR, "inception_4c_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_31_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_31_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_31_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_31_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_31_p2, VX_TYPE_SCALAR, "inception_4c_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_31_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_31_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_31_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_31_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_31_p3, VX_TYPE_SCALAR, "inception_4c_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_31_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_4_p0, 4, org_khronos_nn_extension_activation_layer_31_p4_view_view_start, org_khronos_nn_extension_activation_layer_31_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_31_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_31_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_31_p4, VX_TYPE_TENSOR, "inception_4c_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_29_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_29_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_29_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_29_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_29_p1, VX_TYPE_SCALAR, "inception_4c_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_29_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_29_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_29_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_29_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_29_p2, VX_TYPE_SCALAR, "inception_4c_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_29_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_29_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_29_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_29_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_29_p3, VX_TYPE_SCALAR, "inception_4c_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_29_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_4_p0, 4, org_khronos_nn_extension_activation_layer_29_p4_view_view_start, org_khronos_nn_extension_activation_layer_29_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_29_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_29_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_29_p4, VX_TYPE_TENSOR, "inception_4c_relu_5x5_4");

    org_khronos_nn_extension_convolution_layer_38_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_38_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_38_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p1, VX_TYPE_TENSOR, "inception_4d_1x1_weights");

    org_khronos_nn_extension_convolution_layer_38_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_38_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_38_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p2, VX_TYPE_TENSOR, "inception_4d_1x1_bias");

    org_khronos_nn_extension_convolution_layer_38_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_38_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_38_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p3, VX_TYPE_SCALAR, "inception_4d_1x1_3");

    org_khronos_nn_extension_convolution_layer_38_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_38_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_38_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p4, VX_TYPE_SCALAR, "inception_4d_1x1_4");

    org_khronos_nn_extension_convolution_layer_38_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_38_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_38_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p5, VX_TYPE_SCALAR, "inception_4d_1x1_5");

    org_khronos_nn_extension_convolution_layer_38_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_38_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_38_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p6, VX_TYPE_SCALAR, "inception_4d_1x1_6");

    org_khronos_nn_extension_convolution_layer_38_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_38_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_38_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p7, VX_TYPE_SCALAR, "inception_4d_1x1_7");

    org_khronos_nn_extension_convolution_layer_38_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_38_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_38_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p8, VX_TYPE_TENSOR, "inception_4d_1x1_8");

    org_khronos_nn_extension_convolution_layer_36_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_36_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_36_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p1, VX_TYPE_TENSOR, "inception_4d_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_36_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_36_p2Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_36_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p2, VX_TYPE_TENSOR, "inception_4d_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_36_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_36_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_36_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p3, VX_TYPE_SCALAR, "inception_4d_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_36_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_36_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_36_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p4, VX_TYPE_SCALAR, "inception_4d_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_36_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_36_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_36_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p5, VX_TYPE_SCALAR, "inception_4d_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_36_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_36_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_36_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p6, VX_TYPE_SCALAR, "inception_4d_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_36_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_36_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_36_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p7, VX_TYPE_SCALAR, "inception_4d_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_36_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_36_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_36_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p8, VX_TYPE_TENSOR, "inception_4d_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_34_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_34_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_34_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p1, VX_TYPE_TENSOR, "inception_4d_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_34_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_34_p2Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_34_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p2, VX_TYPE_TENSOR, "inception_4d_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_34_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_34_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_34_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p3, VX_TYPE_SCALAR, "inception_4d_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_34_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_34_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_34_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p4, VX_TYPE_SCALAR, "inception_4d_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_34_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_34_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_34_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p5, VX_TYPE_SCALAR, "inception_4d_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_34_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_34_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_34_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p6, VX_TYPE_SCALAR, "inception_4d_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_34_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_34_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_34_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p7, VX_TYPE_SCALAR, "inception_4d_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_34_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_34_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_34_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p8, VX_TYPE_TENSOR, "inception_4d_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_8_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_8_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_8_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_8_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p1, VX_TYPE_SCALAR, "inception_4d_pool_1");

    org_khronos_nn_extension_pooling_layer_8_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_8_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_8_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_8_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p2, VX_TYPE_SCALAR, "inception_4d_pool_2");

    org_khronos_nn_extension_pooling_layer_8_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_8_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_8_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_8_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p3, VX_TYPE_SCALAR, "inception_4d_pool_3");

    org_khronos_nn_extension_pooling_layer_8_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_8_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_8_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_8_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p4, VX_TYPE_SCALAR, "inception_4d_pool_4");

    org_khronos_nn_extension_pooling_layer_8_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_8_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_8_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_8_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p5, VX_TYPE_SCALAR, "inception_4d_pool_5");

    org_khronos_nn_extension_pooling_layer_8_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_8_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_8_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_8_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p6, VX_TYPE_SCALAR, "inception_4d_pool_6");

    org_khronos_nn_extension_pooling_layer_8_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_8_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_8_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_8_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p7, VX_TYPE_TENSOR, "inception_4d_pool_7");

    org_khronos_nn_extension_activation_layer_38_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_38_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_38_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_38_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_38_p1, VX_TYPE_SCALAR, "inception_4d_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_38_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_38_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_38_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_38_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_38_p2, VX_TYPE_SCALAR, "inception_4d_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_38_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_38_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_38_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_38_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_38_p3, VX_TYPE_SCALAR, "inception_4d_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_38_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_5_p0, 4, org_khronos_nn_extension_activation_layer_38_p4_view_view_start, org_khronos_nn_extension_activation_layer_38_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_38_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_38_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_38_p4, VX_TYPE_TENSOR, "inception_4d_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_36_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_36_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_36_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_36_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_36_p1, VX_TYPE_SCALAR, "inception_4d_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_36_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_36_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_36_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_36_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_36_p2, VX_TYPE_SCALAR, "inception_4d_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_36_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_36_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_36_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_36_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_36_p3, VX_TYPE_SCALAR, "inception_4d_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_36_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_36_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_36_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_36_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_36_p4, VX_TYPE_TENSOR, "inception_4d_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_34_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_34_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_34_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_34_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_34_p1, VX_TYPE_SCALAR, "inception_4d_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_34_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_34_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_34_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_34_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_34_p2, VX_TYPE_SCALAR, "inception_4d_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_34_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_34_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_34_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_34_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_34_p3, VX_TYPE_SCALAR, "inception_4d_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_34_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_34_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_34_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_34_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_34_p4, VX_TYPE_TENSOR, "inception_4d_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_33_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_33_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_33_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p1, VX_TYPE_TENSOR, "inception_4d_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_33_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_33_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_33_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p2, VX_TYPE_TENSOR, "inception_4d_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_33_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_33_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_33_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p3, VX_TYPE_SCALAR, "inception_4d_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_33_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_33_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_33_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p4, VX_TYPE_SCALAR, "inception_4d_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_33_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_33_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_33_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p5, VX_TYPE_SCALAR, "inception_4d_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_33_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_33_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_33_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p6, VX_TYPE_SCALAR, "inception_4d_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_33_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_33_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_33_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p7, VX_TYPE_SCALAR, "inception_4d_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_33_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_33_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_33_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p8, VX_TYPE_TENSOR, "inception_4d_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_37_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_37_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_37_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p1, VX_TYPE_TENSOR, "inception_4d_3x3_weights");

    org_khronos_nn_extension_convolution_layer_37_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_37_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_37_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p2, VX_TYPE_TENSOR, "inception_4d_3x3_bias");

    org_khronos_nn_extension_convolution_layer_37_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_37_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_37_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p3, VX_TYPE_SCALAR, "inception_4d_3x3_3");

    org_khronos_nn_extension_convolution_layer_37_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_37_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_37_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p4, VX_TYPE_SCALAR, "inception_4d_3x3_4");

    org_khronos_nn_extension_convolution_layer_37_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_37_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_37_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p5, VX_TYPE_SCALAR, "inception_4d_3x3_5");

    org_khronos_nn_extension_convolution_layer_37_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_37_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_37_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p6, VX_TYPE_SCALAR, "inception_4d_3x3_6");

    org_khronos_nn_extension_convolution_layer_37_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_37_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_37_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p7, VX_TYPE_SCALAR, "inception_4d_3x3_7");

    org_khronos_nn_extension_convolution_layer_37_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_37_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_37_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p8, VX_TYPE_TENSOR, "inception_4d_3x3_8");

    org_khronos_nn_extension_convolution_layer_35_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_35_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_35_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p1, VX_TYPE_TENSOR, "inception_4d_5x5_weights");

    org_khronos_nn_extension_convolution_layer_35_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_35_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_35_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p2, VX_TYPE_TENSOR, "inception_4d_5x5_bias");

    org_khronos_nn_extension_convolution_layer_35_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_35_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_35_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p3, VX_TYPE_SCALAR, "inception_4d_5x5_3");

    org_khronos_nn_extension_convolution_layer_35_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_35_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_35_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p4, VX_TYPE_SCALAR, "inception_4d_5x5_4");

    org_khronos_nn_extension_convolution_layer_35_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_35_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_35_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p5, VX_TYPE_SCALAR, "inception_4d_5x5_5");

    org_khronos_nn_extension_convolution_layer_35_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_35_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_35_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p6, VX_TYPE_SCALAR, "inception_4d_5x5_6");

    org_khronos_nn_extension_convolution_layer_35_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_35_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_35_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p7, VX_TYPE_SCALAR, "inception_4d_5x5_7");

    org_khronos_nn_extension_convolution_layer_35_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_35_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_35_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p8, VX_TYPE_TENSOR, "inception_4d_5x5_8");

    org_khronos_nn_extension_activation_layer_33_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_33_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_33_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_33_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_33_p1, VX_TYPE_SCALAR, "inception_4d_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_33_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_33_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_33_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_33_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_33_p2, VX_TYPE_SCALAR, "inception_4d_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_33_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_33_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_33_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_33_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_33_p3, VX_TYPE_SCALAR, "inception_4d_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_33_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_5_p0, 4, org_khronos_nn_extension_activation_layer_33_p4_view_view_start, org_khronos_nn_extension_activation_layer_33_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_33_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_33_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_33_p4, VX_TYPE_TENSOR, "inception_4d_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_37_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_37_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_37_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_37_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_37_p1, VX_TYPE_SCALAR, "inception_4d_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_37_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_37_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_37_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_37_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_37_p2, VX_TYPE_SCALAR, "inception_4d_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_37_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_37_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_37_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_37_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_37_p3, VX_TYPE_SCALAR, "inception_4d_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_37_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_5_p0, 4, org_khronos_nn_extension_activation_layer_37_p4_view_view_start, org_khronos_nn_extension_activation_layer_37_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_37_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_37_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_37_p4, VX_TYPE_TENSOR, "inception_4d_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_35_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_35_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_35_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_35_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_35_p1, VX_TYPE_SCALAR, "inception_4d_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_35_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_35_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_35_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_35_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_35_p2, VX_TYPE_SCALAR, "inception_4d_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_35_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_35_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_35_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_35_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_35_p3, VX_TYPE_SCALAR, "inception_4d_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_35_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_5_p0, 4, org_khronos_nn_extension_activation_layer_35_p4_view_view_start, org_khronos_nn_extension_activation_layer_35_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_35_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_35_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_35_p4, VX_TYPE_TENSOR, "inception_4d_relu_5x5_4");

    org_khronos_nn_extension_convolution_layer_44_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_44_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_44_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p1, VX_TYPE_TENSOR, "inception_4e_1x1_weights");

    org_khronos_nn_extension_convolution_layer_44_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_44_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_44_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p2, VX_TYPE_TENSOR, "inception_4e_1x1_bias");

    org_khronos_nn_extension_convolution_layer_44_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_44_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_44_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p3, VX_TYPE_SCALAR, "inception_4e_1x1_3");

    org_khronos_nn_extension_convolution_layer_44_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_44_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_44_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p4, VX_TYPE_SCALAR, "inception_4e_1x1_4");

    org_khronos_nn_extension_convolution_layer_44_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_44_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_44_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p5, VX_TYPE_SCALAR, "inception_4e_1x1_5");

    org_khronos_nn_extension_convolution_layer_44_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_44_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_44_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p6, VX_TYPE_SCALAR, "inception_4e_1x1_6");

    org_khronos_nn_extension_convolution_layer_44_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_44_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_44_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p7, VX_TYPE_SCALAR, "inception_4e_1x1_7");

    org_khronos_nn_extension_convolution_layer_44_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_44_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_44_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p8, VX_TYPE_TENSOR, "inception_4e_1x1_8");

    org_khronos_nn_extension_convolution_layer_42_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_42_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_42_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p1, VX_TYPE_TENSOR, "inception_4e_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_42_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_42_p2Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_42_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p2, VX_TYPE_TENSOR, "inception_4e_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_42_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_42_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_42_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p3, VX_TYPE_SCALAR, "inception_4e_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_42_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_42_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_42_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p4, VX_TYPE_SCALAR, "inception_4e_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_42_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_42_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_42_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p5, VX_TYPE_SCALAR, "inception_4e_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_42_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_42_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_42_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p6, VX_TYPE_SCALAR, "inception_4e_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_42_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_42_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_42_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p7, VX_TYPE_SCALAR, "inception_4e_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_42_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_42_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_42_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p8, VX_TYPE_TENSOR, "inception_4e_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_40_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_40_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_40_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p1, VX_TYPE_TENSOR, "inception_4e_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_40_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_40_p2Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_40_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p2, VX_TYPE_TENSOR, "inception_4e_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_40_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_40_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_40_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p3, VX_TYPE_SCALAR, "inception_4e_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_40_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_40_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_40_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p4, VX_TYPE_SCALAR, "inception_4e_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_40_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_40_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_40_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p5, VX_TYPE_SCALAR, "inception_4e_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_40_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_40_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_40_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p6, VX_TYPE_SCALAR, "inception_4e_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_40_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_40_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_40_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p7, VX_TYPE_SCALAR, "inception_4e_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_40_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_40_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_40_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p8, VX_TYPE_TENSOR, "inception_4e_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_9_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_9_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_9_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_9_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p1, VX_TYPE_SCALAR, "inception_4e_pool_1");

    org_khronos_nn_extension_pooling_layer_9_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_9_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_9_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_9_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p2, VX_TYPE_SCALAR, "inception_4e_pool_2");

    org_khronos_nn_extension_pooling_layer_9_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_9_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_9_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_9_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p3, VX_TYPE_SCALAR, "inception_4e_pool_3");

    org_khronos_nn_extension_pooling_layer_9_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_9_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_9_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_9_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p4, VX_TYPE_SCALAR, "inception_4e_pool_4");

    org_khronos_nn_extension_pooling_layer_9_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_9_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_9_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_9_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p5, VX_TYPE_SCALAR, "inception_4e_pool_5");

    org_khronos_nn_extension_pooling_layer_9_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_9_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_9_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_9_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p6, VX_TYPE_SCALAR, "inception_4e_pool_6");

    org_khronos_nn_extension_pooling_layer_9_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_9_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_9_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_9_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p7, VX_TYPE_TENSOR, "inception_4e_pool_7");

    org_khronos_nn_extension_activation_layer_44_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_44_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_44_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_44_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_44_p1, VX_TYPE_SCALAR, "inception_4e_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_44_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_44_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_44_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_44_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_44_p2, VX_TYPE_SCALAR, "inception_4e_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_44_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_44_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_44_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_44_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_44_p3, VX_TYPE_SCALAR, "inception_4e_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_44_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_6_p0, 4, org_khronos_nn_extension_activation_layer_44_p4_view_view_start, org_khronos_nn_extension_activation_layer_44_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_44_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_44_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_44_p4, VX_TYPE_TENSOR, "inception_4e_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_42_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_42_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_42_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_42_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_42_p1, VX_TYPE_SCALAR, "inception_4e_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_42_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_42_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_42_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_42_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_42_p2, VX_TYPE_SCALAR, "inception_4e_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_42_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_42_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_42_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_42_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_42_p3, VX_TYPE_SCALAR, "inception_4e_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_42_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_42_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_42_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_42_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_42_p4, VX_TYPE_TENSOR, "inception_4e_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_40_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_40_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_40_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_40_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_40_p1, VX_TYPE_SCALAR, "inception_4e_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_40_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_40_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_40_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_40_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_40_p2, VX_TYPE_SCALAR, "inception_4e_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_40_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_40_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_40_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_40_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_40_p3, VX_TYPE_SCALAR, "inception_4e_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_40_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_40_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_40_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_40_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_40_p4, VX_TYPE_TENSOR, "inception_4e_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_39_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_39_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_39_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p1, VX_TYPE_TENSOR, "inception_4e_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_39_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_39_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_39_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p2, VX_TYPE_TENSOR, "inception_4e_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_39_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_39_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_39_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p3, VX_TYPE_SCALAR, "inception_4e_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_39_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_39_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_39_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p4, VX_TYPE_SCALAR, "inception_4e_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_39_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_39_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_39_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p5, VX_TYPE_SCALAR, "inception_4e_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_39_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_39_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_39_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p6, VX_TYPE_SCALAR, "inception_4e_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_39_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_39_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_39_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p7, VX_TYPE_SCALAR, "inception_4e_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_39_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_39_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_39_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p8, VX_TYPE_TENSOR, "inception_4e_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_43_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_43_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_43_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p1, VX_TYPE_TENSOR, "inception_4e_3x3_weights");

    org_khronos_nn_extension_convolution_layer_43_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_43_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_43_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p2, VX_TYPE_TENSOR, "inception_4e_3x3_bias");

    org_khronos_nn_extension_convolution_layer_43_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_43_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_43_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p3, VX_TYPE_SCALAR, "inception_4e_3x3_3");

    org_khronos_nn_extension_convolution_layer_43_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_43_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_43_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p4, VX_TYPE_SCALAR, "inception_4e_3x3_4");

    org_khronos_nn_extension_convolution_layer_43_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_43_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_43_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p5, VX_TYPE_SCALAR, "inception_4e_3x3_5");

    org_khronos_nn_extension_convolution_layer_43_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_43_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_43_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p6, VX_TYPE_SCALAR, "inception_4e_3x3_6");

    org_khronos_nn_extension_convolution_layer_43_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_43_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_43_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p7, VX_TYPE_SCALAR, "inception_4e_3x3_7");

    org_khronos_nn_extension_convolution_layer_43_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_43_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_43_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p8, VX_TYPE_TENSOR, "inception_4e_3x3_8");

    org_khronos_nn_extension_convolution_layer_41_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_41_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_41_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p1, VX_TYPE_TENSOR, "inception_4e_5x5_weights");

    org_khronos_nn_extension_convolution_layer_41_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_41_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_41_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p2, VX_TYPE_TENSOR, "inception_4e_5x5_bias");

    org_khronos_nn_extension_convolution_layer_41_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_41_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_41_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p3, VX_TYPE_SCALAR, "inception_4e_5x5_3");

    org_khronos_nn_extension_convolution_layer_41_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_41_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_41_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p4, VX_TYPE_SCALAR, "inception_4e_5x5_4");

    org_khronos_nn_extension_convolution_layer_41_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_41_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_41_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p5, VX_TYPE_SCALAR, "inception_4e_5x5_5");

    org_khronos_nn_extension_convolution_layer_41_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_41_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_41_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p6, VX_TYPE_SCALAR, "inception_4e_5x5_6");

    org_khronos_nn_extension_convolution_layer_41_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_41_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_41_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p7, VX_TYPE_SCALAR, "inception_4e_5x5_7");

    org_khronos_nn_extension_convolution_layer_41_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_41_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_41_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p8, VX_TYPE_TENSOR, "inception_4e_5x5_8");

    org_khronos_nn_extension_activation_layer_39_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_39_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_39_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_39_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_39_p1, VX_TYPE_SCALAR, "inception_4e_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_39_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_39_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_39_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_39_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_39_p2, VX_TYPE_SCALAR, "inception_4e_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_39_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_39_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_39_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_39_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_39_p3, VX_TYPE_SCALAR, "inception_4e_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_39_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_6_p0, 4, org_khronos_nn_extension_activation_layer_39_p4_view_view_start, org_khronos_nn_extension_activation_layer_39_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_39_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_39_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_39_p4, VX_TYPE_TENSOR, "inception_4e_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_43_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_43_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_43_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_43_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_43_p1, VX_TYPE_SCALAR, "inception_4e_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_43_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_43_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_43_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_43_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_43_p2, VX_TYPE_SCALAR, "inception_4e_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_43_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_43_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_43_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_43_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_43_p3, VX_TYPE_SCALAR, "inception_4e_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_43_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_6_p0, 4, org_khronos_nn_extension_activation_layer_43_p4_view_view_start, org_khronos_nn_extension_activation_layer_43_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_43_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_43_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_43_p4, VX_TYPE_TENSOR, "inception_4e_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_41_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_41_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_41_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_41_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_41_p1, VX_TYPE_SCALAR, "inception_4e_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_41_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_41_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_41_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_41_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_41_p2, VX_TYPE_SCALAR, "inception_4e_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_41_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_41_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_41_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_41_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_41_p3, VX_TYPE_SCALAR, "inception_4e_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_41_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_6_p0, 4, org_khronos_nn_extension_activation_layer_41_p4_view_view_start, org_khronos_nn_extension_activation_layer_41_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_41_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_41_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_41_p4, VX_TYPE_TENSOR, "inception_4e_relu_5x5_4");

    org_khronos_nn_extension_pooling_layer_10_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_10_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_10_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_10_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p1, VX_TYPE_SCALAR, "pool4_3x3_s2_1");

    org_khronos_nn_extension_pooling_layer_10_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_10_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_10_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_10_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p2, VX_TYPE_SCALAR, "pool4_3x3_s2_2");

    org_khronos_nn_extension_pooling_layer_10_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_10_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_10_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_10_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p3, VX_TYPE_SCALAR, "pool4_3x3_s2_3");

    org_khronos_nn_extension_pooling_layer_10_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_10_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_10_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_10_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p4, VX_TYPE_SCALAR, "pool4_3x3_s2_4");

    org_khronos_nn_extension_pooling_layer_10_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_10_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_10_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_10_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p5, VX_TYPE_SCALAR, "pool4_3x3_s2_5");

    org_khronos_nn_extension_pooling_layer_10_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_10_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_10_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_10_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p6, VX_TYPE_SCALAR, "pool4_3x3_s2_6");

    org_khronos_nn_extension_pooling_layer_10_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_10_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_10_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_10_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p7, VX_TYPE_TENSOR, "pool4_3x3_s2_7");

    org_khronos_nn_extension_convolution_layer_50_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_50_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_50_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p1, VX_TYPE_TENSOR, "inception_5a_1x1_weights");

    org_khronos_nn_extension_convolution_layer_50_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_50_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_50_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p2, VX_TYPE_TENSOR, "inception_5a_1x1_bias");

    org_khronos_nn_extension_convolution_layer_50_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_50_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_50_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p3, VX_TYPE_SCALAR, "inception_5a_1x1_3");

    org_khronos_nn_extension_convolution_layer_50_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_50_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_50_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p4, VX_TYPE_SCALAR, "inception_5a_1x1_4");

    org_khronos_nn_extension_convolution_layer_50_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_50_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_50_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p5, VX_TYPE_SCALAR, "inception_5a_1x1_5");

    org_khronos_nn_extension_convolution_layer_50_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_50_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_50_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p6, VX_TYPE_SCALAR, "inception_5a_1x1_6");

    org_khronos_nn_extension_convolution_layer_50_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_50_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_50_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p7, VX_TYPE_SCALAR, "inception_5a_1x1_7");

    org_khronos_nn_extension_convolution_layer_50_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_50_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_50_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p8, VX_TYPE_TENSOR, "inception_5a_1x1_8");

    org_khronos_nn_extension_convolution_layer_48_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_48_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_48_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p1, VX_TYPE_TENSOR, "inception_5a_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_48_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_48_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_48_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p2, VX_TYPE_TENSOR, "inception_5a_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_48_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_48_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_48_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p3, VX_TYPE_SCALAR, "inception_5a_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_48_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_48_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_48_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p4, VX_TYPE_SCALAR, "inception_5a_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_48_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_48_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_48_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p5, VX_TYPE_SCALAR, "inception_5a_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_48_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_48_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_48_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p6, VX_TYPE_SCALAR, "inception_5a_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_48_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_48_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_48_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p7, VX_TYPE_SCALAR, "inception_5a_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_48_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_48_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_48_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p8, VX_TYPE_TENSOR, "inception_5a_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_46_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_46_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_46_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p1, VX_TYPE_TENSOR, "inception_5a_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_46_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_46_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_46_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p2, VX_TYPE_TENSOR, "inception_5a_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_46_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_46_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_46_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p3, VX_TYPE_SCALAR, "inception_5a_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_46_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_46_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_46_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p4, VX_TYPE_SCALAR, "inception_5a_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_46_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_46_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_46_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p5, VX_TYPE_SCALAR, "inception_5a_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_46_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_46_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_46_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p6, VX_TYPE_SCALAR, "inception_5a_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_46_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_46_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_46_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p7, VX_TYPE_SCALAR, "inception_5a_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_46_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_46_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_46_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p8, VX_TYPE_TENSOR, "inception_5a_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_11_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_11_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_11_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_11_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p1, VX_TYPE_SCALAR, "inception_5a_pool_1");

    org_khronos_nn_extension_pooling_layer_11_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_11_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_11_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_11_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p2, VX_TYPE_SCALAR, "inception_5a_pool_2");

    org_khronos_nn_extension_pooling_layer_11_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_11_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_11_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_11_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p3, VX_TYPE_SCALAR, "inception_5a_pool_3");

    org_khronos_nn_extension_pooling_layer_11_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_11_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_11_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_11_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p4, VX_TYPE_SCALAR, "inception_5a_pool_4");

    org_khronos_nn_extension_pooling_layer_11_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_11_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_11_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_11_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p5, VX_TYPE_SCALAR, "inception_5a_pool_5");

    org_khronos_nn_extension_pooling_layer_11_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_11_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_11_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_11_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p6, VX_TYPE_SCALAR, "inception_5a_pool_6");

    org_khronos_nn_extension_pooling_layer_11_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_11_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_11_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_11_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p7, VX_TYPE_TENSOR, "inception_5a_pool_7");

    org_khronos_nn_extension_activation_layer_50_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_50_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_50_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_50_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_50_p1, VX_TYPE_SCALAR, "inception_5a_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_50_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_50_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_50_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_50_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_50_p2, VX_TYPE_SCALAR, "inception_5a_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_50_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_50_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_50_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_50_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_50_p3, VX_TYPE_SCALAR, "inception_5a_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_50_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_7_p0, 4, org_khronos_nn_extension_activation_layer_50_p4_view_view_start, org_khronos_nn_extension_activation_layer_50_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_50_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_50_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_50_p4, VX_TYPE_TENSOR, "inception_5a_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_48_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_48_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_48_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_48_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_48_p1, VX_TYPE_SCALAR, "inception_5a_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_48_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_48_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_48_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_48_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_48_p2, VX_TYPE_SCALAR, "inception_5a_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_48_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_48_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_48_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_48_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_48_p3, VX_TYPE_SCALAR, "inception_5a_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_48_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_48_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_48_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_48_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_48_p4, VX_TYPE_TENSOR, "inception_5a_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_46_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_46_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_46_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_46_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_46_p1, VX_TYPE_SCALAR, "inception_5a_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_46_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_46_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_46_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_46_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_46_p2, VX_TYPE_SCALAR, "inception_5a_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_46_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_46_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_46_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_46_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_46_p3, VX_TYPE_SCALAR, "inception_5a_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_46_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_46_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_46_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_46_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_46_p4, VX_TYPE_TENSOR, "inception_5a_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_45_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_45_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_45_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p1, VX_TYPE_TENSOR, "inception_5a_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_45_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_45_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_45_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p2, VX_TYPE_TENSOR, "inception_5a_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_45_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_45_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_45_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p3, VX_TYPE_SCALAR, "inception_5a_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_45_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_45_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_45_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p4, VX_TYPE_SCALAR, "inception_5a_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_45_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_45_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_45_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p5, VX_TYPE_SCALAR, "inception_5a_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_45_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_45_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_45_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p6, VX_TYPE_SCALAR, "inception_5a_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_45_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_45_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_45_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p7, VX_TYPE_SCALAR, "inception_5a_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_45_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_45_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_45_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p8, VX_TYPE_TENSOR, "inception_5a_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_49_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_49_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_49_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p1, VX_TYPE_TENSOR, "inception_5a_3x3_weights");

    org_khronos_nn_extension_convolution_layer_49_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_49_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_49_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p2, VX_TYPE_TENSOR, "inception_5a_3x3_bias");

    org_khronos_nn_extension_convolution_layer_49_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_49_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_49_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p3, VX_TYPE_SCALAR, "inception_5a_3x3_3");

    org_khronos_nn_extension_convolution_layer_49_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_49_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_49_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p4, VX_TYPE_SCALAR, "inception_5a_3x3_4");

    org_khronos_nn_extension_convolution_layer_49_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_49_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_49_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p5, VX_TYPE_SCALAR, "inception_5a_3x3_5");

    org_khronos_nn_extension_convolution_layer_49_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_49_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_49_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p6, VX_TYPE_SCALAR, "inception_5a_3x3_6");

    org_khronos_nn_extension_convolution_layer_49_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_49_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_49_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p7, VX_TYPE_SCALAR, "inception_5a_3x3_7");

    org_khronos_nn_extension_convolution_layer_49_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_49_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_49_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p8, VX_TYPE_TENSOR, "inception_5a_3x3_8");

    org_khronos_nn_extension_convolution_layer_47_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_47_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_47_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p1, VX_TYPE_TENSOR, "inception_5a_5x5_weights");

    org_khronos_nn_extension_convolution_layer_47_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_47_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_47_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p2, VX_TYPE_TENSOR, "inception_5a_5x5_bias");

    org_khronos_nn_extension_convolution_layer_47_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_47_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_47_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p3, VX_TYPE_SCALAR, "inception_5a_5x5_3");

    org_khronos_nn_extension_convolution_layer_47_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_47_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_47_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p4, VX_TYPE_SCALAR, "inception_5a_5x5_4");

    org_khronos_nn_extension_convolution_layer_47_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_47_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_47_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p5, VX_TYPE_SCALAR, "inception_5a_5x5_5");

    org_khronos_nn_extension_convolution_layer_47_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_47_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_47_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p6, VX_TYPE_SCALAR, "inception_5a_5x5_6");

    org_khronos_nn_extension_convolution_layer_47_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_47_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_47_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p7, VX_TYPE_SCALAR, "inception_5a_5x5_7");

    org_khronos_nn_extension_convolution_layer_47_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_47_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_47_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p8, VX_TYPE_TENSOR, "inception_5a_5x5_8");

    org_khronos_nn_extension_activation_layer_45_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_45_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_45_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_45_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_45_p1, VX_TYPE_SCALAR, "inception_5a_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_45_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_45_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_45_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_45_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_45_p2, VX_TYPE_SCALAR, "inception_5a_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_45_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_45_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_45_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_45_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_45_p3, VX_TYPE_SCALAR, "inception_5a_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_45_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_7_p0, 4, org_khronos_nn_extension_activation_layer_45_p4_view_view_start, org_khronos_nn_extension_activation_layer_45_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_45_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_45_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_45_p4, VX_TYPE_TENSOR, "inception_5a_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_49_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_49_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_49_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_49_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_49_p1, VX_TYPE_SCALAR, "inception_5a_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_49_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_49_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_49_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_49_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_49_p2, VX_TYPE_SCALAR, "inception_5a_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_49_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_49_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_49_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_49_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_49_p3, VX_TYPE_SCALAR, "inception_5a_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_49_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_7_p0, 4, org_khronos_nn_extension_activation_layer_49_p4_view_view_start, org_khronos_nn_extension_activation_layer_49_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_49_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_49_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_49_p4, VX_TYPE_TENSOR, "inception_5a_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_47_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_47_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_47_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_47_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_47_p1, VX_TYPE_SCALAR, "inception_5a_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_47_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_47_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_47_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_47_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_47_p2, VX_TYPE_SCALAR, "inception_5a_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_47_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_47_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_47_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_47_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_47_p3, VX_TYPE_SCALAR, "inception_5a_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_47_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_7_p0, 4, org_khronos_nn_extension_activation_layer_47_p4_view_view_start, org_khronos_nn_extension_activation_layer_47_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_47_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_47_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_47_p4, VX_TYPE_TENSOR, "inception_5a_relu_5x5_4");

    org_khronos_nn_extension_convolution_layer_56_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_56_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_56_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p1, VX_TYPE_TENSOR, "inception_5b_1x1_weights");

    org_khronos_nn_extension_convolution_layer_56_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_56_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_56_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p2, VX_TYPE_TENSOR, "inception_5b_1x1_bias");

    org_khronos_nn_extension_convolution_layer_56_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_56_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_56_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p3, VX_TYPE_SCALAR, "inception_5b_1x1_3");

    org_khronos_nn_extension_convolution_layer_56_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_56_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_56_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p4, VX_TYPE_SCALAR, "inception_5b_1x1_4");

    org_khronos_nn_extension_convolution_layer_56_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_56_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_56_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p5, VX_TYPE_SCALAR, "inception_5b_1x1_5");

    org_khronos_nn_extension_convolution_layer_56_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_56_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_56_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p6, VX_TYPE_SCALAR, "inception_5b_1x1_6");

    org_khronos_nn_extension_convolution_layer_56_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_56_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_56_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p7, VX_TYPE_SCALAR, "inception_5b_1x1_7");

    org_khronos_nn_extension_convolution_layer_56_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_56_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_56_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p8, VX_TYPE_TENSOR, "inception_5b_1x1_8");

    org_khronos_nn_extension_convolution_layer_54_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_54_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_54_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p1, VX_TYPE_TENSOR, "inception_5b_3x3_reduce_weights");

    org_khronos_nn_extension_convolution_layer_54_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_54_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_54_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p2, VX_TYPE_TENSOR, "inception_5b_3x3_reduce_bias");

    org_khronos_nn_extension_convolution_layer_54_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_54_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_54_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p3, VX_TYPE_SCALAR, "inception_5b_3x3_reduce_3");

    org_khronos_nn_extension_convolution_layer_54_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_54_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_54_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p4, VX_TYPE_SCALAR, "inception_5b_3x3_reduce_4");

    org_khronos_nn_extension_convolution_layer_54_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_54_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_54_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p5, VX_TYPE_SCALAR, "inception_5b_3x3_reduce_5");

    org_khronos_nn_extension_convolution_layer_54_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_54_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_54_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p6, VX_TYPE_SCALAR, "inception_5b_3x3_reduce_6");

    org_khronos_nn_extension_convolution_layer_54_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_54_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_54_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p7, VX_TYPE_SCALAR, "inception_5b_3x3_reduce_7");

    org_khronos_nn_extension_convolution_layer_54_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_54_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_54_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p8, VX_TYPE_TENSOR, "inception_5b_3x3_reduce_8");

    org_khronos_nn_extension_convolution_layer_52_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_52_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_52_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p1, VX_TYPE_TENSOR, "inception_5b_5x5_reduce_weights");

    org_khronos_nn_extension_convolution_layer_52_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_52_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_52_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p2, VX_TYPE_TENSOR, "inception_5b_5x5_reduce_bias");

    org_khronos_nn_extension_convolution_layer_52_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_52_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_52_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p3, VX_TYPE_SCALAR, "inception_5b_5x5_reduce_3");

    org_khronos_nn_extension_convolution_layer_52_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_52_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_52_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p4, VX_TYPE_SCALAR, "inception_5b_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_52_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_52_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_52_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p5, VX_TYPE_SCALAR, "inception_5b_5x5_reduce_5");

    org_khronos_nn_extension_convolution_layer_52_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_52_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_52_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p6, VX_TYPE_SCALAR, "inception_5b_5x5_reduce_6");

    org_khronos_nn_extension_convolution_layer_52_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_52_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_52_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p7, VX_TYPE_SCALAR, "inception_5b_5x5_reduce_7");

    org_khronos_nn_extension_convolution_layer_52_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_52_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_52_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p8, VX_TYPE_TENSOR, "inception_5b_5x5_reduce_8");

    org_khronos_nn_extension_pooling_layer_12_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_12_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_12_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_12_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p1, VX_TYPE_SCALAR, "inception_5b_pool_1");

    org_khronos_nn_extension_pooling_layer_12_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_12_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_12_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_12_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p2, VX_TYPE_SCALAR, "inception_5b_pool_2");

    org_khronos_nn_extension_pooling_layer_12_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_12_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_12_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_12_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p3, VX_TYPE_SCALAR, "inception_5b_pool_3");

    org_khronos_nn_extension_pooling_layer_12_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_12_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_12_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_12_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p4, VX_TYPE_SCALAR, "inception_5b_pool_4");

    org_khronos_nn_extension_pooling_layer_12_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_12_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_12_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_12_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p5, VX_TYPE_SCALAR, "inception_5b_pool_5");

    org_khronos_nn_extension_pooling_layer_12_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_12_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_12_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_12_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p6, VX_TYPE_SCALAR, "inception_5b_pool_6");

    org_khronos_nn_extension_pooling_layer_12_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_12_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_12_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_12_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p7, VX_TYPE_TENSOR, "inception_5b_pool_7");

    org_khronos_nn_extension_activation_layer_56_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_56_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_56_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_56_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_56_p1, VX_TYPE_SCALAR, "inception_5b_relu_1x1_1");

    org_khronos_nn_extension_activation_layer_56_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_56_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_56_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_56_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_56_p2, VX_TYPE_SCALAR, "inception_5b_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_56_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_56_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_56_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_56_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_56_p3, VX_TYPE_SCALAR, "inception_5b_relu_1x1_2");

    org_khronos_nn_extension_activation_layer_56_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_8_p0, 4, org_khronos_nn_extension_activation_layer_56_p4_view_view_start, org_khronos_nn_extension_activation_layer_56_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_56_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_56_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_56_p4, VX_TYPE_TENSOR, "inception_5b_relu_1x1_4");

    org_khronos_nn_extension_activation_layer_54_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_54_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_54_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_54_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_54_p1, VX_TYPE_SCALAR, "inception_5b_relu_3x3_reduce_1");

    org_khronos_nn_extension_activation_layer_54_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_54_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_54_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_54_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_54_p2, VX_TYPE_SCALAR, "inception_5b_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_54_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_54_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_54_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_54_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_54_p3, VX_TYPE_SCALAR, "inception_5b_relu_3x3_reduce_2");

    org_khronos_nn_extension_activation_layer_54_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_54_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_54_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_54_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_54_p4, VX_TYPE_TENSOR, "inception_5b_relu_3x3_reduce_4");

    org_khronos_nn_extension_activation_layer_52_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_52_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_52_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_52_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_52_p1, VX_TYPE_SCALAR, "inception_5b_relu_5x5_reduce_1");

    org_khronos_nn_extension_activation_layer_52_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_52_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_52_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_52_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_52_p2, VX_TYPE_SCALAR, "inception_5b_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_52_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_52_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_52_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_52_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_52_p3, VX_TYPE_SCALAR, "inception_5b_relu_5x5_reduce_2");

    org_khronos_nn_extension_activation_layer_52_p4 = vxCreateTensor(context, 4, org_khronos_nn_extension_activation_layer_52_p4Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_52_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_52_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_52_p4, VX_TYPE_TENSOR, "inception_5b_relu_5x5_reduce_4");

    org_khronos_nn_extension_convolution_layer_51_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_51_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_51_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p1, VX_TYPE_TENSOR, "inception_5b_pool_proj_weights");

    org_khronos_nn_extension_convolution_layer_51_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_51_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_51_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p2, VX_TYPE_TENSOR, "inception_5b_pool_proj_bias");

    org_khronos_nn_extension_convolution_layer_51_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_51_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_51_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p3, VX_TYPE_SCALAR, "inception_5b_pool_proj_3");

    org_khronos_nn_extension_convolution_layer_51_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_51_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_51_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p4, VX_TYPE_SCALAR, "inception_5b_pool_proj_4");

    org_khronos_nn_extension_convolution_layer_51_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_51_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_51_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p5, VX_TYPE_SCALAR, "inception_5b_pool_proj_5");

    org_khronos_nn_extension_convolution_layer_51_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_51_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_51_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p6, VX_TYPE_SCALAR, "inception_5b_pool_proj_6");

    org_khronos_nn_extension_convolution_layer_51_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_51_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_51_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p7, VX_TYPE_SCALAR, "inception_5b_pool_proj_7");

    org_khronos_nn_extension_convolution_layer_51_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_51_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_51_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p8, VX_TYPE_TENSOR, "inception_5b_pool_proj_8");

    org_khronos_nn_extension_convolution_layer_55_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_55_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_55_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p1, VX_TYPE_TENSOR, "inception_5b_3x3_weights");

    org_khronos_nn_extension_convolution_layer_55_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_55_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_55_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p2, VX_TYPE_TENSOR, "inception_5b_3x3_bias");

    org_khronos_nn_extension_convolution_layer_55_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_55_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_55_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p3, VX_TYPE_SCALAR, "inception_5b_3x3_3");

    org_khronos_nn_extension_convolution_layer_55_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_55_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_55_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p4, VX_TYPE_SCALAR, "inception_5b_3x3_4");

    org_khronos_nn_extension_convolution_layer_55_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_55_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_55_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p5, VX_TYPE_SCALAR, "inception_5b_3x3_5");

    org_khronos_nn_extension_convolution_layer_55_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_55_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_55_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p6, VX_TYPE_SCALAR, "inception_5b_3x3_6");

    org_khronos_nn_extension_convolution_layer_55_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_55_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_55_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p7, VX_TYPE_SCALAR, "inception_5b_3x3_7");

    org_khronos_nn_extension_convolution_layer_55_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_55_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_55_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p8, VX_TYPE_TENSOR, "inception_5b_3x3_8");

    org_khronos_nn_extension_convolution_layer_53_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_53_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_53_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p1, VX_TYPE_TENSOR, "inception_5b_5x5_weights");

    org_khronos_nn_extension_convolution_layer_53_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_convolution_layer_53_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_53_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p2, VX_TYPE_TENSOR, "inception_5b_5x5_bias");

    org_khronos_nn_extension_convolution_layer_53_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_53_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_53_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p3, VX_TYPE_SCALAR, "inception_5b_5x5_3");

    org_khronos_nn_extension_convolution_layer_53_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_convolution_layer_53_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_53_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p4, VX_TYPE_SCALAR, "inception_5b_5x5_4");

    org_khronos_nn_extension_convolution_layer_53_p5 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_53_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_53_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p5, VX_TYPE_SCALAR, "inception_5b_5x5_5");

    org_khronos_nn_extension_convolution_layer_53_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_53_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_53_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p6, VX_TYPE_SCALAR, "inception_5b_5x5_6");

    org_khronos_nn_extension_convolution_layer_53_p7 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_convolution_layer_53_scalar_p7);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_53_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p7, VX_TYPE_SCALAR, "inception_5b_5x5_7");

    org_khronos_nn_extension_convolution_layer_53_p8 = vxCreateTensor(context, 4, org_khronos_nn_extension_convolution_layer_53_p8Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53_p8);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_convolution_layer_53_p8 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p8, VX_TYPE_TENSOR, "inception_5b_5x5_8");

    org_khronos_nn_extension_activation_layer_51_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_51_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_51_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_51_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_51_p1, VX_TYPE_SCALAR, "inception_5b_relu_pool_proj_1");

    org_khronos_nn_extension_activation_layer_51_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_51_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_51_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_51_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_51_p2, VX_TYPE_SCALAR, "inception_5b_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_51_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_51_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_51_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_51_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_51_p3, VX_TYPE_SCALAR, "inception_5b_relu_pool_proj_2");

    org_khronos_nn_extension_activation_layer_51_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_8_p0, 4, org_khronos_nn_extension_activation_layer_51_p4_view_view_start, org_khronos_nn_extension_activation_layer_51_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_51_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_51_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_51_p4, VX_TYPE_TENSOR, "inception_5b_relu_pool_proj_4");

    org_khronos_nn_extension_activation_layer_55_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_55_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_55_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_55_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_55_p1, VX_TYPE_SCALAR, "inception_5b_relu_3x3_1");

    org_khronos_nn_extension_activation_layer_55_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_55_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_55_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_55_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_55_p2, VX_TYPE_SCALAR, "inception_5b_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_55_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_55_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_55_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_55_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_55_p3, VX_TYPE_SCALAR, "inception_5b_relu_3x3_2");

    org_khronos_nn_extension_activation_layer_55_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_8_p0, 4, org_khronos_nn_extension_activation_layer_55_p4_view_view_start, org_khronos_nn_extension_activation_layer_55_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_55_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_55_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_55_p4, VX_TYPE_TENSOR, "inception_5b_relu_3x3_4");

    org_khronos_nn_extension_activation_layer_53_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_activation_layer_53_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_53_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_53_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_53_p1, VX_TYPE_SCALAR, "inception_5b_relu_5x5_1");

    org_khronos_nn_extension_activation_layer_53_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_53_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_53_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_53_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_53_p2, VX_TYPE_SCALAR, "inception_5b_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_53_p3 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_nn_extension_activation_layer_53_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_53_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_53_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_53_p3, VX_TYPE_SCALAR, "inception_5b_relu_5x5_2");

    org_khronos_nn_extension_activation_layer_53_p4 = vxCreateTensorFromView(outputAllocators_MergeTensor_8_p0, 4, org_khronos_nn_extension_activation_layer_53_p4_view_view_start, org_khronos_nn_extension_activation_layer_53_p4_view_view_end);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_activation_layer_53_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_activation_layer_53_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_activation_layer_53_p4, VX_TYPE_TENSOR, "inception_5b_relu_5x5_4");

    org_khronos_nn_extension_pooling_layer_13_p1 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_13_scalar_p1);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_13_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_13_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p1, VX_TYPE_SCALAR, "pool5_7x7_s1_1");

    org_khronos_nn_extension_pooling_layer_13_p2 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_13_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_13_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_13_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p2, VX_TYPE_SCALAR, "pool5_7x7_s1_2");

    org_khronos_nn_extension_pooling_layer_13_p3 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_13_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_13_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_13_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p3, VX_TYPE_SCALAR, "pool5_7x7_s1_3");

    org_khronos_nn_extension_pooling_layer_13_p4 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_13_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_13_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_13_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p4, VX_TYPE_SCALAR, "pool5_7x7_s1_4");

    org_khronos_nn_extension_pooling_layer_13_p5 = vxCreateScalar(context, VX_TYPE_SIZE, (void*)&org_khronos_nn_extension_pooling_layer_13_scalar_p5);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_13_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_13_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p5, VX_TYPE_SCALAR, "pool5_7x7_s1_5");

    org_khronos_nn_extension_pooling_layer_13_p6 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_pooling_layer_13_scalar_p6);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_13_p6);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_13_p6 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p6, VX_TYPE_SCALAR, "pool5_7x7_s1_6");

    org_khronos_nn_extension_pooling_layer_13_p7 = vxCreateTensor(context, 4, org_khronos_nn_extension_pooling_layer_13_p7Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_pooling_layer_13_p7);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_pooling_layer_13_p7 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p7, VX_TYPE_TENSOR, "pool5_7x7_s1_7");

    org_khronos_nn_extension_fully_connected_layer_0_p1 = vxCreateTensor(context, 4, org_khronos_nn_extension_fully_connected_layer_0_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p1, VX_TYPE_TENSOR, "loss3_classifier_weights");

    org_khronos_nn_extension_fully_connected_layer_0_p2 = vxCreateTensor(context, 1, org_khronos_nn_extension_fully_connected_layer_0_p2Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p2, VX_TYPE_TENSOR, "loss3_classifier_bias");

    org_khronos_nn_extension_fully_connected_layer_0_p3 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_fully_connected_layer_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p3, VX_TYPE_SCALAR, "loss3_classifier_3");

    org_khronos_nn_extension_fully_connected_layer_0_p4 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_nn_extension_fully_connected_layer_0_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p4, VX_TYPE_SCALAR, "loss3_classifier_4");

    org_khronos_nn_extension_fully_connected_layer_0_p5 = vxCreateTensor(context, 2, org_khronos_nn_extension_fully_connected_layer_0_p5Dimensions ,VX_TYPE_INT16, 8 );
    
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_nn_extension_fully_connected_layer_0_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p5, VX_TYPE_TENSOR, "loss3_classifier_5");

    org_khronos_openvx_tensor_multiply_0_p1 = vxCreateTensor(context, 2, org_khronos_openvx_tensor_multiply_0_p1Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_openvx_tensor_multiply_0_p1);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_openvx_tensor_multiply_0_p1 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_openvx_tensor_multiply_0_p1, VX_TYPE_TENSOR, "Power0_scale");

    org_khronos_openvx_tensor_multiply_0_p2 = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*)&org_khronos_openvx_tensor_multiply_0_scalar_p2);
    status = vxGetStatus((vx_reference)org_khronos_openvx_tensor_multiply_0_p2);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_openvx_tensor_multiply_0_p2 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_openvx_tensor_multiply_0_p2, VX_TYPE_SCALAR, "Power0_2");

    org_khronos_openvx_tensor_multiply_0_p3 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_openvx_tensor_multiply_0_scalar_p3);
    status = vxGetStatus((vx_reference)org_khronos_openvx_tensor_multiply_0_p3);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_openvx_tensor_multiply_0_p3 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_openvx_tensor_multiply_0_p3, VX_TYPE_SCALAR, "Power0_3");

    org_khronos_openvx_tensor_multiply_0_p4 = vxCreateScalar(context, VX_TYPE_ENUM, (void*)&org_khronos_openvx_tensor_multiply_0_scalar_p4);
    status = vxGetStatus((vx_reference)org_khronos_openvx_tensor_multiply_0_p4);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_openvx_tensor_multiply_0_p4 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_openvx_tensor_multiply_0_p4, VX_TYPE_SCALAR, "Power0_4");

    org_khronos_openvx_tensor_multiply_0_p5 = vxCreateTensor(context, 2, org_khronos_openvx_tensor_multiply_0_p5Dimensions ,VX_TYPE_INT16, 8 );

    status = vxGetStatus((vx_reference)org_khronos_openvx_tensor_multiply_0_p5);
    if(status != VX_SUCCESS)
    {
        WriteLog("ERROR: cannot create parameter org_khronos_openvx_tensor_multiply_0_p5 (vx_status=%s)\n", getVxStatusDesc(status));
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_openvx_tensor_multiply_0_p5, VX_TYPE_TENSOR, "Power0_5");

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
//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_0);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_0 = vxConvolutionLayer(graph, org_khronos_nn_extension_convolution_layer_0_p0, org_khronos_nn_extension_convolution_layer_0_p1,
        org_khronos_nn_extension_convolution_layer_0_p2, &org_khronos_nn_extension_convolution_layer_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_0_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_0);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_0\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_0, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_0");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_0);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_0);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_normalization_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_normalization_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_normalization_layer_0);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_1);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_1 = vxConvolutionLayer(graph, org_khronos_nn_extension_normalization_layer_0_p5, org_khronos_nn_extension_convolution_layer_1_p1,
            org_khronos_nn_extension_convolution_layer_1_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_1_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_1);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_1\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_1, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_1");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_1);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_2", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_2);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_2 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_1_p4, org_khronos_nn_extension_convolution_layer_2_p1,
        org_khronos_nn_extension_convolution_layer_2_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_2_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_2);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_2\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_2, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_2");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_2", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_2);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_normalization_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_normalization_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_normalization_layer_1);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_1", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_1);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_8", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_8);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_8 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_1_p7, org_khronos_nn_extension_convolution_layer_8_p1,
        org_khronos_nn_extension_convolution_layer_8_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_8_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_8);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_8\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_8, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_8");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_6", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_6);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_6 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_1_p7, org_khronos_nn_extension_convolution_layer_6_p1,
        org_khronos_nn_extension_convolution_layer_6_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_6_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_6);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_6\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_6, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_6");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_4", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_4);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_4 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_1_p7, org_khronos_nn_extension_convolution_layer_4_p1,
        org_khronos_nn_extension_convolution_layer_4_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_4_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_4);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_4\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_4, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_4");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_2", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_2);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_8", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_8);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_6", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_6);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_4", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_4);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_3", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_3);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_3 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_2_p7, org_khronos_nn_extension_convolution_layer_3_p1,
        org_khronos_nn_extension_convolution_layer_3_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_3_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_3);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_3\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_3, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_3");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_7", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_7);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_7 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_6_p4, org_khronos_nn_extension_convolution_layer_7_p1,
        org_khronos_nn_extension_convolution_layer_7_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_7_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_7);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_7\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_7, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_7");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_5", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_5);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_5 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_4_p4, org_khronos_nn_extension_convolution_layer_5_p1,
        org_khronos_nn_extension_convolution_layer_5_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_5_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_5);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_5\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_5, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_5");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_3", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_3);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_7", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_7);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_5", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_5);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_14", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_14);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_14 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_0_p0, org_khronos_nn_extension_convolution_layer_14_p1,
        org_khronos_nn_extension_convolution_layer_14_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_14_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_14);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_14\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_14, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_14");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_12", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_12);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_12 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_0_p0, org_khronos_nn_extension_convolution_layer_12_p1,
        org_khronos_nn_extension_convolution_layer_12_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_12_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_12);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_12\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_12, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_12");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_10", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_10);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_10 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_0_p0, org_khronos_nn_extension_convolution_layer_10_p1,
        org_khronos_nn_extension_convolution_layer_10_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_10_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_10);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_10\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_10, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_10");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_3", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_3);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_14", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_14);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_12", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_12);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_10", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_10);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_9", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_9);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_9 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_3_p7, org_khronos_nn_extension_convolution_layer_9_p1,
        org_khronos_nn_extension_convolution_layer_9_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_9_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_9);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_9\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_9, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_9");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_13", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_13);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_13 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_12_p4, org_khronos_nn_extension_convolution_layer_13_p1,
        org_khronos_nn_extension_convolution_layer_13_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_13_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_13);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_13\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_13, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_13");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_11", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_11);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_11 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_10_p4, org_khronos_nn_extension_convolution_layer_11_p1,
        org_khronos_nn_extension_convolution_layer_11_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_11_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_11);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_11\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_11, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_11");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_9", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_9);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_13", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_13);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_11", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_11);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_4", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_4);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_20", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_20);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_20 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_4_p7, org_khronos_nn_extension_convolution_layer_20_p1,
        org_khronos_nn_extension_convolution_layer_20_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_20_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_20);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_20\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_20, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_20");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_18", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_18);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_18 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_4_p7, org_khronos_nn_extension_convolution_layer_18_p1,
        org_khronos_nn_extension_convolution_layer_18_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_18_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_18);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_18\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_18, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_18");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_16", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_16);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_16 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_4_p7, org_khronos_nn_extension_convolution_layer_16_p1,
        org_khronos_nn_extension_convolution_layer_16_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_16_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_16);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_16\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_16, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_16");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_5", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_5);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_20", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_20);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_18", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_18);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_16", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_16);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_15", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_15);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_15 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_5_p7, org_khronos_nn_extension_convolution_layer_15_p1,
        org_khronos_nn_extension_convolution_layer_15_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_15_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_15);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_15\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_15, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_15");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_19", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_19);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_19 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_18_p4, org_khronos_nn_extension_convolution_layer_19_p1,
        org_khronos_nn_extension_convolution_layer_19_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_19_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_19);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_19\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_19, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_19");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_17", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_17);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_17 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_16_p4, org_khronos_nn_extension_convolution_layer_17_p1,
        org_khronos_nn_extension_convolution_layer_17_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_17_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_17);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_17\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_17, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_17");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_15", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_15);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_19", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_19);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_17", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_17);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_26", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_26);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_26 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_2_p0, org_khronos_nn_extension_convolution_layer_26_p1,
        org_khronos_nn_extension_convolution_layer_26_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_26_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_26);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_26\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_26, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_26");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_24", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_24);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_24 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_2_p0, org_khronos_nn_extension_convolution_layer_24_p1,
        org_khronos_nn_extension_convolution_layer_24_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_24_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_24);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_24\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_24, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_24");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_22", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_22);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_22 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_2_p0, org_khronos_nn_extension_convolution_layer_22_p1,
        org_khronos_nn_extension_convolution_layer_22_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_22_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_22);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_22\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_22, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_22");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_6", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_6);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_26", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_26);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_24", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_24);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_22", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_22);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_21", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_21);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_21 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_6_p7, org_khronos_nn_extension_convolution_layer_21_p1,
        org_khronos_nn_extension_convolution_layer_21_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_21_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_21);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_21\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_21, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_21");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_25", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_25);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_25 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_24_p4, org_khronos_nn_extension_convolution_layer_25_p1,
        org_khronos_nn_extension_convolution_layer_25_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_25_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_25);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_25\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_25, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_25");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_23", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_23);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_23 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_22_p4, org_khronos_nn_extension_convolution_layer_23_p1,
        org_khronos_nn_extension_convolution_layer_23_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_23_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_23);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_23\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_23, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_23");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_21", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_21);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_25", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_25);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_23", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_23);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_32", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_32);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_32 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_3_p0, org_khronos_nn_extension_convolution_layer_32_p1,
        org_khronos_nn_extension_convolution_layer_32_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_32_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_32);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_32\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_32, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_32");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_30", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_30);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_30 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_3_p0, org_khronos_nn_extension_convolution_layer_30_p1,
        org_khronos_nn_extension_convolution_layer_30_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_30_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_30);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_30\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_30, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_30");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_28", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_28);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_28 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_3_p0, org_khronos_nn_extension_convolution_layer_28_p1,
        org_khronos_nn_extension_convolution_layer_28_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_28_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_28);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_28\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_28, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_28");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_7", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_7);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_32", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_32);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_30", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_30);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_28", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_28);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_27", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_27);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_27 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_7_p7, org_khronos_nn_extension_convolution_layer_27_p1,
        org_khronos_nn_extension_convolution_layer_27_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_27_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_27);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_27\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_27, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_27");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_31", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_31);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_31 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_30_p4, org_khronos_nn_extension_convolution_layer_31_p1,
        org_khronos_nn_extension_convolution_layer_31_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_31_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_31);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_31\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_31, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_31");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_29", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_29);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_29 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_28_p4, org_khronos_nn_extension_convolution_layer_29_p1,
        org_khronos_nn_extension_convolution_layer_29_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_29_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_29);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_29\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_29, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_29");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_27", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_27);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_31", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_31);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_29", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_29);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_38", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_38);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_38 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_4_p0, org_khronos_nn_extension_convolution_layer_38_p1,
        org_khronos_nn_extension_convolution_layer_38_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_38_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_38);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_38\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_38, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_38");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_36", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_36);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_36 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_4_p0, org_khronos_nn_extension_convolution_layer_36_p1,
        org_khronos_nn_extension_convolution_layer_36_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_36_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_36);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_36\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_36, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_36");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_34", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_34);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_34 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_4_p0, org_khronos_nn_extension_convolution_layer_34_p1,
        org_khronos_nn_extension_convolution_layer_34_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_34_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_34);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_34\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_34, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_34");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_8", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_8);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_38", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_38);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_36", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_36);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_34", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_34);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_33", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_33);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_33 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_8_p7, org_khronos_nn_extension_convolution_layer_33_p1,
        org_khronos_nn_extension_convolution_layer_33_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_33_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_33);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_33\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_33, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_33");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_37", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_37);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_37 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_36_p4, org_khronos_nn_extension_convolution_layer_37_p1,
        org_khronos_nn_extension_convolution_layer_37_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_37_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_37);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_37\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_37, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_37");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_35", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_35);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_35 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_34_p4, org_khronos_nn_extension_convolution_layer_35_p1,
        org_khronos_nn_extension_convolution_layer_35_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_35_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_35);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_35\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_35, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_35");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_33", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_33);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_37", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_37);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_35", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_35);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_44", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_44);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_44 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_5_p0, org_khronos_nn_extension_convolution_layer_44_p1,
        org_khronos_nn_extension_convolution_layer_44_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_44_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_44);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_44\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_44, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_44");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_42", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_42);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_42 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_5_p0, org_khronos_nn_extension_convolution_layer_42_p1,
        org_khronos_nn_extension_convolution_layer_42_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_42_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_42);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_42\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_42, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_42");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_40", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_40);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_40 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_5_p0, org_khronos_nn_extension_convolution_layer_40_p1,
        org_khronos_nn_extension_convolution_layer_40_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_40_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_40);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_40\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_40, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_40");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_9", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_9);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_44", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_44);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_42", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_42);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_40", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_40);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_39", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_39);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_39 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_9_p7, org_khronos_nn_extension_convolution_layer_39_p1,
        org_khronos_nn_extension_convolution_layer_39_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_39_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_39);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_39\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_39, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_39");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_43", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_43);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_43 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_42_p4, org_khronos_nn_extension_convolution_layer_43_p1,
        org_khronos_nn_extension_convolution_layer_43_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_43_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_43);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_43\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_43, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_43");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_41", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_41);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_41 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_40_p4, org_khronos_nn_extension_convolution_layer_41_p1,
        org_khronos_nn_extension_convolution_layer_41_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_41_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_41);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_41\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_41, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_41");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_39", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_39);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_43", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_43);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_41", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_41);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_10", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_10);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_50", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_50);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_50 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_10_p7, org_khronos_nn_extension_convolution_layer_50_p1,
        org_khronos_nn_extension_convolution_layer_50_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_50_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_50);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_50\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_50, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_50");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_48", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_48);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_48 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_10_p7, org_khronos_nn_extension_convolution_layer_48_p1,
        org_khronos_nn_extension_convolution_layer_48_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_48_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_48);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_48\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_48, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_48");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_46", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_46);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_46 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_10_p7, org_khronos_nn_extension_convolution_layer_46_p1,
        org_khronos_nn_extension_convolution_layer_46_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_46_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_46);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_46\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_46, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_46");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_11", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_11);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_50", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_50);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_48", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_48);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_46", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_46);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_45", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_45);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_45 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_11_p7, org_khronos_nn_extension_convolution_layer_45_p1,
        org_khronos_nn_extension_convolution_layer_45_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_45_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_45);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_45\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_45, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_45");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_49", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_49);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_49 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_48_p4, org_khronos_nn_extension_convolution_layer_49_p1,
        org_khronos_nn_extension_convolution_layer_49_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_49_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_49);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_49\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_49, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_49");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_47", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_47);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_47 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_46_p4, org_khronos_nn_extension_convolution_layer_47_p1,
        org_khronos_nn_extension_convolution_layer_47_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_47_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_47);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_47\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_47, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_47");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_45", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_45);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_49", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_49);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_47", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_47);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_56", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_56);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_56 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_7_p0, org_khronos_nn_extension_convolution_layer_56_p1,
        org_khronos_nn_extension_convolution_layer_56_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_56_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_56);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_56\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_56, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_56");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_54", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_54);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_54 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_7_p0, org_khronos_nn_extension_convolution_layer_54_p1,
        org_khronos_nn_extension_convolution_layer_54_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_54_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_54);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_54\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_54, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_54");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_52", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_52);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_52 = vxConvolutionLayer(graph, outputAllocators_MergeTensor_7_p0, org_khronos_nn_extension_convolution_layer_52_p1,
        org_khronos_nn_extension_convolution_layer_52_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_52_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_52);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_52\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_52, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_52");

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_12", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_12);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_56", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_56);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_54", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_54);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_52", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_52);
    if(status != VX_SUCCESS)
        return status;

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_51", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_51);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_51 = vxConvolutionLayer(graph, org_khronos_nn_extension_pooling_layer_12_p7, org_khronos_nn_extension_convolution_layer_51_p1,
        org_khronos_nn_extension_convolution_layer_51_p2, &org_khronos_nn_extension_convolution_0_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_51_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_51);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_51\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_51, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_51");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_55", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_55);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_55 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_54_p4, org_khronos_nn_extension_convolution_layer_55_p1,
        org_khronos_nn_extension_convolution_layer_55_p2, &org_khronos_nn_extension_convolution_1_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_55_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_55);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_55\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_55, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_55");

//    status = CreateNode(graph, org_khronos_nn_extension_convolution_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_convolution_layer_53", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_convolution_layer_53);
//    if(status != VX_SUCCESS)
//        return status;
    org_khronos_nn_extension_convolution_layer_53 = vxConvolutionLayer(graph, org_khronos_nn_extension_activation_layer_52_p4, org_khronos_nn_extension_convolution_layer_53_p1,
        org_khronos_nn_extension_convolution_layer_53_p2, &org_khronos_nn_extension_convolution_2_p3, sizeof(vx_nn_convolution_params_t), org_khronos_nn_extension_convolution_layer_53_p8);
    status = vxGetStatus((vx_reference)org_khronos_nn_extension_convolution_layer_53);
    if (status != VX_SUCCESS)
    {
        WriteLog("ERROR: failed to create node org_khronos_nn_extension_convolution_layer_53\n");
        return status;
    }
    AddVXObject(pObjectContainer, (vx_reference)org_khronos_nn_extension_convolution_layer_53, VX_TYPE_NODE, "org_khronos_nn_extension_convolution_layer_53");

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_51", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_51);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_55", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_55);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_activation_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_activation_layer_53", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_activation_layer_53);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_pooling_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_pooling_layer_13", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_pooling_layer_13);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_fully_connected_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_fully_connected_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_fully_connected_layer_0);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_openvx_tensor_multiply_Kernel, pObjectContainer, "org_khronos_openvx_tensor_multiply_0", filteredNodeList, filteredNodeCount, &org_khronos_openvx_tensor_multiply_0);
    if(status != VX_SUCCESS)
        return status;

    status = CreateNode(graph, org_khronos_nn_extension_softmax_layer_Kernel, pObjectContainer, "org_khronos_nn_extension_softmax_layer_0", filteredNodeList, filteredNodeCount, &org_khronos_nn_extension_softmax_layer_0);
    if(status != VX_SUCCESS)
        return status;


    //
    // Assign Primitives to nodes
    //
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_0, "org_khronos_nn_extension_convolution_layer_0", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_0, "org_khronos_nn_extension_activation_layer_0", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_0_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_0, "org_khronos_nn_extension_activation_layer_0", 1, (vx_reference)org_khronos_nn_extension_activation_layer_0_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_0, "org_khronos_nn_extension_activation_layer_0", 2, (vx_reference)org_khronos_nn_extension_activation_layer_0_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_0, "org_khronos_nn_extension_activation_layer_0", 3, (vx_reference)org_khronos_nn_extension_activation_layer_0_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_0, "org_khronos_nn_extension_activation_layer_0", 4, (vx_reference)org_khronos_nn_extension_activation_layer_0_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_0, "org_khronos_nn_extension_pooling_layer_0", 0, (vx_reference)org_khronos_nn_extension_activation_layer_0_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_0, "org_khronos_nn_extension_pooling_layer_0", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_0, "org_khronos_nn_extension_pooling_layer_0", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_0, "org_khronos_nn_extension_pooling_layer_0", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_0, "org_khronos_nn_extension_pooling_layer_0", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_0, "org_khronos_nn_extension_pooling_layer_0", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_0, "org_khronos_nn_extension_pooling_layer_0", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_0, "org_khronos_nn_extension_pooling_layer_0", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_0, "org_khronos_nn_extension_normalization_layer_0", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_0_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_0, "org_khronos_nn_extension_normalization_layer_0", 1, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_0, "org_khronos_nn_extension_normalization_layer_0", 2, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_0, "org_khronos_nn_extension_normalization_layer_0", 3, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_0, "org_khronos_nn_extension_normalization_layer_0", 4, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_0, "org_khronos_nn_extension_normalization_layer_0", 5, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p5);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 0, (vx_reference)org_khronos_nn_extension_normalization_layer_0_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_1, "org_khronos_nn_extension_convolution_layer_1", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_1, "org_khronos_nn_extension_activation_layer_1", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_1_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_1, "org_khronos_nn_extension_activation_layer_1", 1, (vx_reference)org_khronos_nn_extension_activation_layer_1_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_1, "org_khronos_nn_extension_activation_layer_1", 2, (vx_reference)org_khronos_nn_extension_activation_layer_1_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_1, "org_khronos_nn_extension_activation_layer_1", 3, (vx_reference)org_khronos_nn_extension_activation_layer_1_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_1, "org_khronos_nn_extension_activation_layer_1", 4, (vx_reference)org_khronos_nn_extension_activation_layer_1_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 0, (vx_reference)org_khronos_nn_extension_activation_layer_1_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_2, "org_khronos_nn_extension_convolution_layer_2", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_2, "org_khronos_nn_extension_activation_layer_2", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_2_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_2, "org_khronos_nn_extension_activation_layer_2", 1, (vx_reference)org_khronos_nn_extension_activation_layer_2_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_2, "org_khronos_nn_extension_activation_layer_2", 2, (vx_reference)org_khronos_nn_extension_activation_layer_2_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_2, "org_khronos_nn_extension_activation_layer_2", 3, (vx_reference)org_khronos_nn_extension_activation_layer_2_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_2, "org_khronos_nn_extension_activation_layer_2", 4, (vx_reference)org_khronos_nn_extension_activation_layer_2_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_1, "org_khronos_nn_extension_normalization_layer_1", 0, (vx_reference)org_khronos_nn_extension_activation_layer_2_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_1, "org_khronos_nn_extension_normalization_layer_1", 1, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_1, "org_khronos_nn_extension_normalization_layer_1", 2, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_1, "org_khronos_nn_extension_normalization_layer_1", 3, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_1, "org_khronos_nn_extension_normalization_layer_1", 4, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_normalization_layer_1, "org_khronos_nn_extension_normalization_layer_1", 5, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_1, "org_khronos_nn_extension_pooling_layer_1", 0, (vx_reference)org_khronos_nn_extension_normalization_layer_1_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_1, "org_khronos_nn_extension_pooling_layer_1", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_1, "org_khronos_nn_extension_pooling_layer_1", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_1, "org_khronos_nn_extension_pooling_layer_1", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_1, "org_khronos_nn_extension_pooling_layer_1", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_1, "org_khronos_nn_extension_pooling_layer_1", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_1, "org_khronos_nn_extension_pooling_layer_1", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_1, "org_khronos_nn_extension_pooling_layer_1", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p7);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_8, "org_khronos_nn_extension_convolution_layer_8", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_6, "org_khronos_nn_extension_convolution_layer_6", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_4, "org_khronos_nn_extension_convolution_layer_4", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_2, "org_khronos_nn_extension_pooling_layer_2", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_1_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_2, "org_khronos_nn_extension_pooling_layer_2", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_2, "org_khronos_nn_extension_pooling_layer_2", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_2, "org_khronos_nn_extension_pooling_layer_2", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_2, "org_khronos_nn_extension_pooling_layer_2", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_2, "org_khronos_nn_extension_pooling_layer_2", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_2, "org_khronos_nn_extension_pooling_layer_2", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_2, "org_khronos_nn_extension_pooling_layer_2", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_8, "org_khronos_nn_extension_activation_layer_8", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_8_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_8, "org_khronos_nn_extension_activation_layer_8", 1, (vx_reference)org_khronos_nn_extension_activation_layer_8_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_8, "org_khronos_nn_extension_activation_layer_8", 2, (vx_reference)org_khronos_nn_extension_activation_layer_8_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_8, "org_khronos_nn_extension_activation_layer_8", 3, (vx_reference)org_khronos_nn_extension_activation_layer_8_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_8, "org_khronos_nn_extension_activation_layer_8", 4, (vx_reference)org_khronos_nn_extension_activation_layer_8_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_6, "org_khronos_nn_extension_activation_layer_6", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_6_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_6, "org_khronos_nn_extension_activation_layer_6", 1, (vx_reference)org_khronos_nn_extension_activation_layer_6_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_6, "org_khronos_nn_extension_activation_layer_6", 2, (vx_reference)org_khronos_nn_extension_activation_layer_6_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_6, "org_khronos_nn_extension_activation_layer_6", 3, (vx_reference)org_khronos_nn_extension_activation_layer_6_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_6, "org_khronos_nn_extension_activation_layer_6", 4, (vx_reference)org_khronos_nn_extension_activation_layer_6_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_4, "org_khronos_nn_extension_activation_layer_4", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_4_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_4, "org_khronos_nn_extension_activation_layer_4", 1, (vx_reference)org_khronos_nn_extension_activation_layer_4_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_4, "org_khronos_nn_extension_activation_layer_4", 2, (vx_reference)org_khronos_nn_extension_activation_layer_4_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_4, "org_khronos_nn_extension_activation_layer_4", 3, (vx_reference)org_khronos_nn_extension_activation_layer_4_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_4, "org_khronos_nn_extension_activation_layer_4", 4, (vx_reference)org_khronos_nn_extension_activation_layer_4_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_2_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_3, "org_khronos_nn_extension_convolution_layer_3", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 0, (vx_reference)org_khronos_nn_extension_activation_layer_6_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_7, "org_khronos_nn_extension_convolution_layer_7", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 0, (vx_reference)org_khronos_nn_extension_activation_layer_4_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_5, "org_khronos_nn_extension_convolution_layer_5", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_3, "org_khronos_nn_extension_activation_layer_3", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_3_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_3, "org_khronos_nn_extension_activation_layer_3", 1, (vx_reference)org_khronos_nn_extension_activation_layer_3_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_3, "org_khronos_nn_extension_activation_layer_3", 2, (vx_reference)org_khronos_nn_extension_activation_layer_3_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_3, "org_khronos_nn_extension_activation_layer_3", 3, (vx_reference)org_khronos_nn_extension_activation_layer_3_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_3, "org_khronos_nn_extension_activation_layer_3", 4, (vx_reference)org_khronos_nn_extension_activation_layer_3_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_7, "org_khronos_nn_extension_activation_layer_7", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_7_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_7, "org_khronos_nn_extension_activation_layer_7", 1, (vx_reference)org_khronos_nn_extension_activation_layer_7_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_7, "org_khronos_nn_extension_activation_layer_7", 2, (vx_reference)org_khronos_nn_extension_activation_layer_7_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_7, "org_khronos_nn_extension_activation_layer_7", 3, (vx_reference)org_khronos_nn_extension_activation_layer_7_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_7, "org_khronos_nn_extension_activation_layer_7", 4, (vx_reference)org_khronos_nn_extension_activation_layer_7_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_5, "org_khronos_nn_extension_activation_layer_5", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_5_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_5, "org_khronos_nn_extension_activation_layer_5", 1, (vx_reference)org_khronos_nn_extension_activation_layer_5_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_5, "org_khronos_nn_extension_activation_layer_5", 2, (vx_reference)org_khronos_nn_extension_activation_layer_5_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_5, "org_khronos_nn_extension_activation_layer_5", 3, (vx_reference)org_khronos_nn_extension_activation_layer_5_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_5, "org_khronos_nn_extension_activation_layer_5", 4, (vx_reference)org_khronos_nn_extension_activation_layer_5_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 0, (vx_reference)outputAllocators_MergeTensor_0_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_14, "org_khronos_nn_extension_convolution_layer_14", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 0, (vx_reference)outputAllocators_MergeTensor_0_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_12, "org_khronos_nn_extension_convolution_layer_12", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 0, (vx_reference)outputAllocators_MergeTensor_0_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_10, "org_khronos_nn_extension_convolution_layer_10", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_3, "org_khronos_nn_extension_pooling_layer_3", 0, (vx_reference)outputAllocators_MergeTensor_0_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_3, "org_khronos_nn_extension_pooling_layer_3", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_3, "org_khronos_nn_extension_pooling_layer_3", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_3, "org_khronos_nn_extension_pooling_layer_3", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_3, "org_khronos_nn_extension_pooling_layer_3", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_3, "org_khronos_nn_extension_pooling_layer_3", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_3, "org_khronos_nn_extension_pooling_layer_3", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_3, "org_khronos_nn_extension_pooling_layer_3", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_14, "org_khronos_nn_extension_activation_layer_14", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_14_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_14, "org_khronos_nn_extension_activation_layer_14", 1, (vx_reference)org_khronos_nn_extension_activation_layer_14_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_14, "org_khronos_nn_extension_activation_layer_14", 2, (vx_reference)org_khronos_nn_extension_activation_layer_14_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_14, "org_khronos_nn_extension_activation_layer_14", 3, (vx_reference)org_khronos_nn_extension_activation_layer_14_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_14, "org_khronos_nn_extension_activation_layer_14", 4, (vx_reference)org_khronos_nn_extension_activation_layer_14_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_12, "org_khronos_nn_extension_activation_layer_12", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_12_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_12, "org_khronos_nn_extension_activation_layer_12", 1, (vx_reference)org_khronos_nn_extension_activation_layer_12_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_12, "org_khronos_nn_extension_activation_layer_12", 2, (vx_reference)org_khronos_nn_extension_activation_layer_12_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_12, "org_khronos_nn_extension_activation_layer_12", 3, (vx_reference)org_khronos_nn_extension_activation_layer_12_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_12, "org_khronos_nn_extension_activation_layer_12", 4, (vx_reference)org_khronos_nn_extension_activation_layer_12_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_10, "org_khronos_nn_extension_activation_layer_10", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_10_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_10, "org_khronos_nn_extension_activation_layer_10", 1, (vx_reference)org_khronos_nn_extension_activation_layer_10_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_10, "org_khronos_nn_extension_activation_layer_10", 2, (vx_reference)org_khronos_nn_extension_activation_layer_10_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_10, "org_khronos_nn_extension_activation_layer_10", 3, (vx_reference)org_khronos_nn_extension_activation_layer_10_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_10, "org_khronos_nn_extension_activation_layer_10", 4, (vx_reference)org_khronos_nn_extension_activation_layer_10_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_3_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_9, "org_khronos_nn_extension_convolution_layer_9", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 0, (vx_reference)org_khronos_nn_extension_activation_layer_12_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_13, "org_khronos_nn_extension_convolution_layer_13", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 0, (vx_reference)org_khronos_nn_extension_activation_layer_10_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_11, "org_khronos_nn_extension_convolution_layer_11", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p8);
//    if(status != VX_SUCCESS)
//        return status;

    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_9, "org_khronos_nn_extension_activation_layer_9", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_9_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_9, "org_khronos_nn_extension_activation_layer_9", 1, (vx_reference)org_khronos_nn_extension_activation_layer_9_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_9, "org_khronos_nn_extension_activation_layer_9", 2, (vx_reference)org_khronos_nn_extension_activation_layer_9_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_9, "org_khronos_nn_extension_activation_layer_9", 3, (vx_reference)org_khronos_nn_extension_activation_layer_9_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_9, "org_khronos_nn_extension_activation_layer_9", 4, (vx_reference)org_khronos_nn_extension_activation_layer_9_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_13, "org_khronos_nn_extension_activation_layer_13", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_13_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_13, "org_khronos_nn_extension_activation_layer_13", 1, (vx_reference)org_khronos_nn_extension_activation_layer_13_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_13, "org_khronos_nn_extension_activation_layer_13", 2, (vx_reference)org_khronos_nn_extension_activation_layer_13_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_13, "org_khronos_nn_extension_activation_layer_13", 3, (vx_reference)org_khronos_nn_extension_activation_layer_13_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_13, "org_khronos_nn_extension_activation_layer_13", 4, (vx_reference)org_khronos_nn_extension_activation_layer_13_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_11, "org_khronos_nn_extension_activation_layer_11", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_11_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_11, "org_khronos_nn_extension_activation_layer_11", 1, (vx_reference)org_khronos_nn_extension_activation_layer_11_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_11, "org_khronos_nn_extension_activation_layer_11", 2, (vx_reference)org_khronos_nn_extension_activation_layer_11_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_11, "org_khronos_nn_extension_activation_layer_11", 3, (vx_reference)org_khronos_nn_extension_activation_layer_11_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_11, "org_khronos_nn_extension_activation_layer_11", 4, (vx_reference)org_khronos_nn_extension_activation_layer_11_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_4, "org_khronos_nn_extension_pooling_layer_4", 0, (vx_reference)outputAllocators_MergeTensor_1_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_4, "org_khronos_nn_extension_pooling_layer_4", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_4, "org_khronos_nn_extension_pooling_layer_4", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_4, "org_khronos_nn_extension_pooling_layer_4", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_4, "org_khronos_nn_extension_pooling_layer_4", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_4, "org_khronos_nn_extension_pooling_layer_4", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_4, "org_khronos_nn_extension_pooling_layer_4", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_4, "org_khronos_nn_extension_pooling_layer_4", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p7);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_20, "org_khronos_nn_extension_convolution_layer_20", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_18, "org_khronos_nn_extension_convolution_layer_18", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_16, "org_khronos_nn_extension_convolution_layer_16", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_5, "org_khronos_nn_extension_pooling_layer_5", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_4_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_5, "org_khronos_nn_extension_pooling_layer_5", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_5, "org_khronos_nn_extension_pooling_layer_5", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_5, "org_khronos_nn_extension_pooling_layer_5", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_5, "org_khronos_nn_extension_pooling_layer_5", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_5, "org_khronos_nn_extension_pooling_layer_5", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_5, "org_khronos_nn_extension_pooling_layer_5", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_5, "org_khronos_nn_extension_pooling_layer_5", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_20, "org_khronos_nn_extension_activation_layer_20", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_20_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_20, "org_khronos_nn_extension_activation_layer_20", 1, (vx_reference)org_khronos_nn_extension_activation_layer_20_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_20, "org_khronos_nn_extension_activation_layer_20", 2, (vx_reference)org_khronos_nn_extension_activation_layer_20_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_20, "org_khronos_nn_extension_activation_layer_20", 3, (vx_reference)org_khronos_nn_extension_activation_layer_20_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_20, "org_khronos_nn_extension_activation_layer_20", 4, (vx_reference)org_khronos_nn_extension_activation_layer_20_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_18, "org_khronos_nn_extension_activation_layer_18", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_18_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_18, "org_khronos_nn_extension_activation_layer_18", 1, (vx_reference)org_khronos_nn_extension_activation_layer_18_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_18, "org_khronos_nn_extension_activation_layer_18", 2, (vx_reference)org_khronos_nn_extension_activation_layer_18_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_18, "org_khronos_nn_extension_activation_layer_18", 3, (vx_reference)org_khronos_nn_extension_activation_layer_18_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_18, "org_khronos_nn_extension_activation_layer_18", 4, (vx_reference)org_khronos_nn_extension_activation_layer_18_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_16, "org_khronos_nn_extension_activation_layer_16", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_16_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_16, "org_khronos_nn_extension_activation_layer_16", 1, (vx_reference)org_khronos_nn_extension_activation_layer_16_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_16, "org_khronos_nn_extension_activation_layer_16", 2, (vx_reference)org_khronos_nn_extension_activation_layer_16_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_16, "org_khronos_nn_extension_activation_layer_16", 3, (vx_reference)org_khronos_nn_extension_activation_layer_16_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_16, "org_khronos_nn_extension_activation_layer_16", 4, (vx_reference)org_khronos_nn_extension_activation_layer_16_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_5_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_15, "org_khronos_nn_extension_convolution_layer_15", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 0, (vx_reference)org_khronos_nn_extension_activation_layer_18_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_19, "org_khronos_nn_extension_convolution_layer_19", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 0, (vx_reference)org_khronos_nn_extension_activation_layer_16_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_17, "org_khronos_nn_extension_convolution_layer_17", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_15, "org_khronos_nn_extension_activation_layer_15", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_15_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_15, "org_khronos_nn_extension_activation_layer_15", 1, (vx_reference)org_khronos_nn_extension_activation_layer_15_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_15, "org_khronos_nn_extension_activation_layer_15", 2, (vx_reference)org_khronos_nn_extension_activation_layer_15_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_15, "org_khronos_nn_extension_activation_layer_15", 3, (vx_reference)org_khronos_nn_extension_activation_layer_15_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_15, "org_khronos_nn_extension_activation_layer_15", 4, (vx_reference)org_khronos_nn_extension_activation_layer_15_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_19, "org_khronos_nn_extension_activation_layer_19", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_19_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_19, "org_khronos_nn_extension_activation_layer_19", 1, (vx_reference)org_khronos_nn_extension_activation_layer_19_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_19, "org_khronos_nn_extension_activation_layer_19", 2, (vx_reference)org_khronos_nn_extension_activation_layer_19_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_19, "org_khronos_nn_extension_activation_layer_19", 3, (vx_reference)org_khronos_nn_extension_activation_layer_19_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_19, "org_khronos_nn_extension_activation_layer_19", 4, (vx_reference)org_khronos_nn_extension_activation_layer_19_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_17, "org_khronos_nn_extension_activation_layer_17", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_17_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_17, "org_khronos_nn_extension_activation_layer_17", 1, (vx_reference)org_khronos_nn_extension_activation_layer_17_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_17, "org_khronos_nn_extension_activation_layer_17", 2, (vx_reference)org_khronos_nn_extension_activation_layer_17_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_17, "org_khronos_nn_extension_activation_layer_17", 3, (vx_reference)org_khronos_nn_extension_activation_layer_17_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_17, "org_khronos_nn_extension_activation_layer_17", 4, (vx_reference)org_khronos_nn_extension_activation_layer_17_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 0, (vx_reference)outputAllocators_MergeTensor_2_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_26, "org_khronos_nn_extension_convolution_layer_26", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 0, (vx_reference)outputAllocators_MergeTensor_2_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_24, "org_khronos_nn_extension_convolution_layer_24", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 0, (vx_reference)outputAllocators_MergeTensor_2_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_22, "org_khronos_nn_extension_convolution_layer_22", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_6, "org_khronos_nn_extension_pooling_layer_6", 0, (vx_reference)outputAllocators_MergeTensor_2_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_6, "org_khronos_nn_extension_pooling_layer_6", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_6, "org_khronos_nn_extension_pooling_layer_6", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_6, "org_khronos_nn_extension_pooling_layer_6", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_6, "org_khronos_nn_extension_pooling_layer_6", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_6, "org_khronos_nn_extension_pooling_layer_6", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_6, "org_khronos_nn_extension_pooling_layer_6", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_6, "org_khronos_nn_extension_pooling_layer_6", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_26, "org_khronos_nn_extension_activation_layer_26", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_26_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_26, "org_khronos_nn_extension_activation_layer_26", 1, (vx_reference)org_khronos_nn_extension_activation_layer_26_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_26, "org_khronos_nn_extension_activation_layer_26", 2, (vx_reference)org_khronos_nn_extension_activation_layer_26_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_26, "org_khronos_nn_extension_activation_layer_26", 3, (vx_reference)org_khronos_nn_extension_activation_layer_26_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_26, "org_khronos_nn_extension_activation_layer_26", 4, (vx_reference)org_khronos_nn_extension_activation_layer_26_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_24, "org_khronos_nn_extension_activation_layer_24", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_24_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_24, "org_khronos_nn_extension_activation_layer_24", 1, (vx_reference)org_khronos_nn_extension_activation_layer_24_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_24, "org_khronos_nn_extension_activation_layer_24", 2, (vx_reference)org_khronos_nn_extension_activation_layer_24_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_24, "org_khronos_nn_extension_activation_layer_24", 3, (vx_reference)org_khronos_nn_extension_activation_layer_24_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_24, "org_khronos_nn_extension_activation_layer_24", 4, (vx_reference)org_khronos_nn_extension_activation_layer_24_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_22, "org_khronos_nn_extension_activation_layer_22", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_22_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_22, "org_khronos_nn_extension_activation_layer_22", 1, (vx_reference)org_khronos_nn_extension_activation_layer_22_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_22, "org_khronos_nn_extension_activation_layer_22", 2, (vx_reference)org_khronos_nn_extension_activation_layer_22_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_22, "org_khronos_nn_extension_activation_layer_22", 3, (vx_reference)org_khronos_nn_extension_activation_layer_22_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_22, "org_khronos_nn_extension_activation_layer_22", 4, (vx_reference)org_khronos_nn_extension_activation_layer_22_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_6_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_21, "org_khronos_nn_extension_convolution_layer_21", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 0, (vx_reference)org_khronos_nn_extension_activation_layer_24_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_25, "org_khronos_nn_extension_convolution_layer_25", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 0, (vx_reference)org_khronos_nn_extension_activation_layer_22_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_23, "org_khronos_nn_extension_convolution_layer_23", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_21, "org_khronos_nn_extension_activation_layer_21", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_21_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_21, "org_khronos_nn_extension_activation_layer_21", 1, (vx_reference)org_khronos_nn_extension_activation_layer_21_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_21, "org_khronos_nn_extension_activation_layer_21", 2, (vx_reference)org_khronos_nn_extension_activation_layer_21_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_21, "org_khronos_nn_extension_activation_layer_21", 3, (vx_reference)org_khronos_nn_extension_activation_layer_21_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_21, "org_khronos_nn_extension_activation_layer_21", 4, (vx_reference)org_khronos_nn_extension_activation_layer_21_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_25, "org_khronos_nn_extension_activation_layer_25", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_25_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_25, "org_khronos_nn_extension_activation_layer_25", 1, (vx_reference)org_khronos_nn_extension_activation_layer_25_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_25, "org_khronos_nn_extension_activation_layer_25", 2, (vx_reference)org_khronos_nn_extension_activation_layer_25_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_25, "org_khronos_nn_extension_activation_layer_25", 3, (vx_reference)org_khronos_nn_extension_activation_layer_25_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_25, "org_khronos_nn_extension_activation_layer_25", 4, (vx_reference)org_khronos_nn_extension_activation_layer_25_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_23, "org_khronos_nn_extension_activation_layer_23", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_23_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_23, "org_khronos_nn_extension_activation_layer_23", 1, (vx_reference)org_khronos_nn_extension_activation_layer_23_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_23, "org_khronos_nn_extension_activation_layer_23", 2, (vx_reference)org_khronos_nn_extension_activation_layer_23_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_23, "org_khronos_nn_extension_activation_layer_23", 3, (vx_reference)org_khronos_nn_extension_activation_layer_23_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_23, "org_khronos_nn_extension_activation_layer_23", 4, (vx_reference)org_khronos_nn_extension_activation_layer_23_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 0, (vx_reference)outputAllocators_MergeTensor_3_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_32, "org_khronos_nn_extension_convolution_layer_32", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 0, (vx_reference)outputAllocators_MergeTensor_3_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_30, "org_khronos_nn_extension_convolution_layer_30", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 0, (vx_reference)outputAllocators_MergeTensor_3_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_28, "org_khronos_nn_extension_convolution_layer_28", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_7, "org_khronos_nn_extension_pooling_layer_7", 0, (vx_reference)outputAllocators_MergeTensor_3_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_7, "org_khronos_nn_extension_pooling_layer_7", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_7, "org_khronos_nn_extension_pooling_layer_7", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_7, "org_khronos_nn_extension_pooling_layer_7", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_7, "org_khronos_nn_extension_pooling_layer_7", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_7, "org_khronos_nn_extension_pooling_layer_7", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_7, "org_khronos_nn_extension_pooling_layer_7", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_7, "org_khronos_nn_extension_pooling_layer_7", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_32, "org_khronos_nn_extension_activation_layer_32", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_32_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_32, "org_khronos_nn_extension_activation_layer_32", 1, (vx_reference)org_khronos_nn_extension_activation_layer_32_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_32, "org_khronos_nn_extension_activation_layer_32", 2, (vx_reference)org_khronos_nn_extension_activation_layer_32_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_32, "org_khronos_nn_extension_activation_layer_32", 3, (vx_reference)org_khronos_nn_extension_activation_layer_32_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_32, "org_khronos_nn_extension_activation_layer_32", 4, (vx_reference)org_khronos_nn_extension_activation_layer_32_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_30, "org_khronos_nn_extension_activation_layer_30", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_30_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_30, "org_khronos_nn_extension_activation_layer_30", 1, (vx_reference)org_khronos_nn_extension_activation_layer_30_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_30, "org_khronos_nn_extension_activation_layer_30", 2, (vx_reference)org_khronos_nn_extension_activation_layer_30_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_30, "org_khronos_nn_extension_activation_layer_30", 3, (vx_reference)org_khronos_nn_extension_activation_layer_30_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_30, "org_khronos_nn_extension_activation_layer_30", 4, (vx_reference)org_khronos_nn_extension_activation_layer_30_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_28, "org_khronos_nn_extension_activation_layer_28", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_28_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_28, "org_khronos_nn_extension_activation_layer_28", 1, (vx_reference)org_khronos_nn_extension_activation_layer_28_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_28, "org_khronos_nn_extension_activation_layer_28", 2, (vx_reference)org_khronos_nn_extension_activation_layer_28_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_28, "org_khronos_nn_extension_activation_layer_28", 3, (vx_reference)org_khronos_nn_extension_activation_layer_28_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_28, "org_khronos_nn_extension_activation_layer_28", 4, (vx_reference)org_khronos_nn_extension_activation_layer_28_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_7_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_27, "org_khronos_nn_extension_convolution_layer_27", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 0, (vx_reference)org_khronos_nn_extension_activation_layer_30_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_31, "org_khronos_nn_extension_convolution_layer_31", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 0, (vx_reference)org_khronos_nn_extension_activation_layer_28_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_29, "org_khronos_nn_extension_convolution_layer_29", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_27, "org_khronos_nn_extension_activation_layer_27", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_27_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_27, "org_khronos_nn_extension_activation_layer_27", 1, (vx_reference)org_khronos_nn_extension_activation_layer_27_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_27, "org_khronos_nn_extension_activation_layer_27", 2, (vx_reference)org_khronos_nn_extension_activation_layer_27_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_27, "org_khronos_nn_extension_activation_layer_27", 3, (vx_reference)org_khronos_nn_extension_activation_layer_27_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_27, "org_khronos_nn_extension_activation_layer_27", 4, (vx_reference)org_khronos_nn_extension_activation_layer_27_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_31, "org_khronos_nn_extension_activation_layer_31", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_31_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_31, "org_khronos_nn_extension_activation_layer_31", 1, (vx_reference)org_khronos_nn_extension_activation_layer_31_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_31, "org_khronos_nn_extension_activation_layer_31", 2, (vx_reference)org_khronos_nn_extension_activation_layer_31_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_31, "org_khronos_nn_extension_activation_layer_31", 3, (vx_reference)org_khronos_nn_extension_activation_layer_31_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_31, "org_khronos_nn_extension_activation_layer_31", 4, (vx_reference)org_khronos_nn_extension_activation_layer_31_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_29, "org_khronos_nn_extension_activation_layer_29", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_29_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_29, "org_khronos_nn_extension_activation_layer_29", 1, (vx_reference)org_khronos_nn_extension_activation_layer_29_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_29, "org_khronos_nn_extension_activation_layer_29", 2, (vx_reference)org_khronos_nn_extension_activation_layer_29_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_29, "org_khronos_nn_extension_activation_layer_29", 3, (vx_reference)org_khronos_nn_extension_activation_layer_29_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_29, "org_khronos_nn_extension_activation_layer_29", 4, (vx_reference)org_khronos_nn_extension_activation_layer_29_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 0, (vx_reference)outputAllocators_MergeTensor_4_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_38, "org_khronos_nn_extension_convolution_layer_38", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 0, (vx_reference)outputAllocators_MergeTensor_4_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_36, "org_khronos_nn_extension_convolution_layer_36", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 0, (vx_reference)outputAllocators_MergeTensor_4_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_34, "org_khronos_nn_extension_convolution_layer_34", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_8, "org_khronos_nn_extension_pooling_layer_8", 0, (vx_reference)outputAllocators_MergeTensor_4_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_8, "org_khronos_nn_extension_pooling_layer_8", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_8, "org_khronos_nn_extension_pooling_layer_8", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_8, "org_khronos_nn_extension_pooling_layer_8", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_8, "org_khronos_nn_extension_pooling_layer_8", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_8, "org_khronos_nn_extension_pooling_layer_8", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_8, "org_khronos_nn_extension_pooling_layer_8", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_8, "org_khronos_nn_extension_pooling_layer_8", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_38, "org_khronos_nn_extension_activation_layer_38", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_38_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_38, "org_khronos_nn_extension_activation_layer_38", 1, (vx_reference)org_khronos_nn_extension_activation_layer_38_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_38, "org_khronos_nn_extension_activation_layer_38", 2, (vx_reference)org_khronos_nn_extension_activation_layer_38_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_38, "org_khronos_nn_extension_activation_layer_38", 3, (vx_reference)org_khronos_nn_extension_activation_layer_38_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_38, "org_khronos_nn_extension_activation_layer_38", 4, (vx_reference)org_khronos_nn_extension_activation_layer_38_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_36, "org_khronos_nn_extension_activation_layer_36", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_36_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_36, "org_khronos_nn_extension_activation_layer_36", 1, (vx_reference)org_khronos_nn_extension_activation_layer_36_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_36, "org_khronos_nn_extension_activation_layer_36", 2, (vx_reference)org_khronos_nn_extension_activation_layer_36_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_36, "org_khronos_nn_extension_activation_layer_36", 3, (vx_reference)org_khronos_nn_extension_activation_layer_36_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_36, "org_khronos_nn_extension_activation_layer_36", 4, (vx_reference)org_khronos_nn_extension_activation_layer_36_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_34, "org_khronos_nn_extension_activation_layer_34", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_34_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_34, "org_khronos_nn_extension_activation_layer_34", 1, (vx_reference)org_khronos_nn_extension_activation_layer_34_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_34, "org_khronos_nn_extension_activation_layer_34", 2, (vx_reference)org_khronos_nn_extension_activation_layer_34_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_34, "org_khronos_nn_extension_activation_layer_34", 3, (vx_reference)org_khronos_nn_extension_activation_layer_34_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_34, "org_khronos_nn_extension_activation_layer_34", 4, (vx_reference)org_khronos_nn_extension_activation_layer_34_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_8_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_33, "org_khronos_nn_extension_convolution_layer_33", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 0, (vx_reference)org_khronos_nn_extension_activation_layer_36_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_37, "org_khronos_nn_extension_convolution_layer_37", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 0, (vx_reference)org_khronos_nn_extension_activation_layer_34_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_35, "org_khronos_nn_extension_convolution_layer_35", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p8);
//    if(status != VX_SUCCESS)
//        return status;

    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_33, "org_khronos_nn_extension_activation_layer_33", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_33_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_33, "org_khronos_nn_extension_activation_layer_33", 1, (vx_reference)org_khronos_nn_extension_activation_layer_33_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_33, "org_khronos_nn_extension_activation_layer_33", 2, (vx_reference)org_khronos_nn_extension_activation_layer_33_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_33, "org_khronos_nn_extension_activation_layer_33", 3, (vx_reference)org_khronos_nn_extension_activation_layer_33_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_33, "org_khronos_nn_extension_activation_layer_33", 4, (vx_reference)org_khronos_nn_extension_activation_layer_33_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_37, "org_khronos_nn_extension_activation_layer_37", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_37_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_37, "org_khronos_nn_extension_activation_layer_37", 1, (vx_reference)org_khronos_nn_extension_activation_layer_37_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_37, "org_khronos_nn_extension_activation_layer_37", 2, (vx_reference)org_khronos_nn_extension_activation_layer_37_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_37, "org_khronos_nn_extension_activation_layer_37", 3, (vx_reference)org_khronos_nn_extension_activation_layer_37_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_37, "org_khronos_nn_extension_activation_layer_37", 4, (vx_reference)org_khronos_nn_extension_activation_layer_37_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_35, "org_khronos_nn_extension_activation_layer_35", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_35_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_35, "org_khronos_nn_extension_activation_layer_35", 1, (vx_reference)org_khronos_nn_extension_activation_layer_35_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_35, "org_khronos_nn_extension_activation_layer_35", 2, (vx_reference)org_khronos_nn_extension_activation_layer_35_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_35, "org_khronos_nn_extension_activation_layer_35", 3, (vx_reference)org_khronos_nn_extension_activation_layer_35_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_35, "org_khronos_nn_extension_activation_layer_35", 4, (vx_reference)org_khronos_nn_extension_activation_layer_35_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 0, (vx_reference)outputAllocators_MergeTensor_5_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_44, "org_khronos_nn_extension_convolution_layer_44", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 0, (vx_reference)outputAllocators_MergeTensor_5_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_42, "org_khronos_nn_extension_convolution_layer_42", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 0, (vx_reference)outputAllocators_MergeTensor_5_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_40, "org_khronos_nn_extension_convolution_layer_40", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p8);
//    if(status != VX_SUCCESS)
//        return status;

    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_9, "org_khronos_nn_extension_pooling_layer_9", 0, (vx_reference)outputAllocators_MergeTensor_5_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_9, "org_khronos_nn_extension_pooling_layer_9", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_9, "org_khronos_nn_extension_pooling_layer_9", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_9, "org_khronos_nn_extension_pooling_layer_9", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_9, "org_khronos_nn_extension_pooling_layer_9", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_9, "org_khronos_nn_extension_pooling_layer_9", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_9, "org_khronos_nn_extension_pooling_layer_9", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_9, "org_khronos_nn_extension_pooling_layer_9", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_44, "org_khronos_nn_extension_activation_layer_44", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_44_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_44, "org_khronos_nn_extension_activation_layer_44", 1, (vx_reference)org_khronos_nn_extension_activation_layer_44_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_44, "org_khronos_nn_extension_activation_layer_44", 2, (vx_reference)org_khronos_nn_extension_activation_layer_44_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_44, "org_khronos_nn_extension_activation_layer_44", 3, (vx_reference)org_khronos_nn_extension_activation_layer_44_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_44, "org_khronos_nn_extension_activation_layer_44", 4, (vx_reference)org_khronos_nn_extension_activation_layer_44_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_42, "org_khronos_nn_extension_activation_layer_42", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_42_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_42, "org_khronos_nn_extension_activation_layer_42", 1, (vx_reference)org_khronos_nn_extension_activation_layer_42_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_42, "org_khronos_nn_extension_activation_layer_42", 2, (vx_reference)org_khronos_nn_extension_activation_layer_42_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_42, "org_khronos_nn_extension_activation_layer_42", 3, (vx_reference)org_khronos_nn_extension_activation_layer_42_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_42, "org_khronos_nn_extension_activation_layer_42", 4, (vx_reference)org_khronos_nn_extension_activation_layer_42_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_40, "org_khronos_nn_extension_activation_layer_40", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_40_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_40, "org_khronos_nn_extension_activation_layer_40", 1, (vx_reference)org_khronos_nn_extension_activation_layer_40_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_40, "org_khronos_nn_extension_activation_layer_40", 2, (vx_reference)org_khronos_nn_extension_activation_layer_40_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_40, "org_khronos_nn_extension_activation_layer_40", 3, (vx_reference)org_khronos_nn_extension_activation_layer_40_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_40, "org_khronos_nn_extension_activation_layer_40", 4, (vx_reference)org_khronos_nn_extension_activation_layer_40_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_9_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_39, "org_khronos_nn_extension_convolution_layer_39", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 0, (vx_reference)org_khronos_nn_extension_activation_layer_42_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_43, "org_khronos_nn_extension_convolution_layer_43", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 0, (vx_reference)org_khronos_nn_extension_activation_layer_40_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_41, "org_khronos_nn_extension_convolution_layer_41", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p8);
//    if(status != VX_SUCCESS)
//        return status;

    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_39, "org_khronos_nn_extension_activation_layer_39", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_39_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_39, "org_khronos_nn_extension_activation_layer_39", 1, (vx_reference)org_khronos_nn_extension_activation_layer_39_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_39, "org_khronos_nn_extension_activation_layer_39", 2, (vx_reference)org_khronos_nn_extension_activation_layer_39_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_39, "org_khronos_nn_extension_activation_layer_39", 3, (vx_reference)org_khronos_nn_extension_activation_layer_39_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_39, "org_khronos_nn_extension_activation_layer_39", 4, (vx_reference)org_khronos_nn_extension_activation_layer_39_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_43, "org_khronos_nn_extension_activation_layer_43", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_43_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_43, "org_khronos_nn_extension_activation_layer_43", 1, (vx_reference)org_khronos_nn_extension_activation_layer_43_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_43, "org_khronos_nn_extension_activation_layer_43", 2, (vx_reference)org_khronos_nn_extension_activation_layer_43_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_43, "org_khronos_nn_extension_activation_layer_43", 3, (vx_reference)org_khronos_nn_extension_activation_layer_43_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_43, "org_khronos_nn_extension_activation_layer_43", 4, (vx_reference)org_khronos_nn_extension_activation_layer_43_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_41, "org_khronos_nn_extension_activation_layer_41", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_41_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_41, "org_khronos_nn_extension_activation_layer_41", 1, (vx_reference)org_khronos_nn_extension_activation_layer_41_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_41, "org_khronos_nn_extension_activation_layer_41", 2, (vx_reference)org_khronos_nn_extension_activation_layer_41_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_41, "org_khronos_nn_extension_activation_layer_41", 3, (vx_reference)org_khronos_nn_extension_activation_layer_41_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_41, "org_khronos_nn_extension_activation_layer_41", 4, (vx_reference)org_khronos_nn_extension_activation_layer_41_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_10, "org_khronos_nn_extension_pooling_layer_10", 0, (vx_reference)outputAllocators_MergeTensor_6_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_10, "org_khronos_nn_extension_pooling_layer_10", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_10, "org_khronos_nn_extension_pooling_layer_10", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_10, "org_khronos_nn_extension_pooling_layer_10", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_10, "org_khronos_nn_extension_pooling_layer_10", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_10, "org_khronos_nn_extension_pooling_layer_10", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_10, "org_khronos_nn_extension_pooling_layer_10", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_10, "org_khronos_nn_extension_pooling_layer_10", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p7);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_50, "org_khronos_nn_extension_convolution_layer_50", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_48, "org_khronos_nn_extension_convolution_layer_48", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_46, "org_khronos_nn_extension_convolution_layer_46", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p8);
//    if(status != VX_SUCCESS)
//        return status;

    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_11, "org_khronos_nn_extension_pooling_layer_11", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_10_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_11, "org_khronos_nn_extension_pooling_layer_11", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_11, "org_khronos_nn_extension_pooling_layer_11", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_11, "org_khronos_nn_extension_pooling_layer_11", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_11, "org_khronos_nn_extension_pooling_layer_11", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_11, "org_khronos_nn_extension_pooling_layer_11", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_11, "org_khronos_nn_extension_pooling_layer_11", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_11, "org_khronos_nn_extension_pooling_layer_11", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_50, "org_khronos_nn_extension_activation_layer_50", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_50_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_50, "org_khronos_nn_extension_activation_layer_50", 1, (vx_reference)org_khronos_nn_extension_activation_layer_50_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_50, "org_khronos_nn_extension_activation_layer_50", 2, (vx_reference)org_khronos_nn_extension_activation_layer_50_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_50, "org_khronos_nn_extension_activation_layer_50", 3, (vx_reference)org_khronos_nn_extension_activation_layer_50_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_50, "org_khronos_nn_extension_activation_layer_50", 4, (vx_reference)org_khronos_nn_extension_activation_layer_50_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_48, "org_khronos_nn_extension_activation_layer_48", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_48_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_48, "org_khronos_nn_extension_activation_layer_48", 1, (vx_reference)org_khronos_nn_extension_activation_layer_48_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_48, "org_khronos_nn_extension_activation_layer_48", 2, (vx_reference)org_khronos_nn_extension_activation_layer_48_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_48, "org_khronos_nn_extension_activation_layer_48", 3, (vx_reference)org_khronos_nn_extension_activation_layer_48_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_48, "org_khronos_nn_extension_activation_layer_48", 4, (vx_reference)org_khronos_nn_extension_activation_layer_48_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_46, "org_khronos_nn_extension_activation_layer_46", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_46_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_46, "org_khronos_nn_extension_activation_layer_46", 1, (vx_reference)org_khronos_nn_extension_activation_layer_46_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_46, "org_khronos_nn_extension_activation_layer_46", 2, (vx_reference)org_khronos_nn_extension_activation_layer_46_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_46, "org_khronos_nn_extension_activation_layer_46", 3, (vx_reference)org_khronos_nn_extension_activation_layer_46_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_46, "org_khronos_nn_extension_activation_layer_46", 4, (vx_reference)org_khronos_nn_extension_activation_layer_46_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_11_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_45, "org_khronos_nn_extension_convolution_layer_45", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 0, (vx_reference)org_khronos_nn_extension_activation_layer_48_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_49, "org_khronos_nn_extension_convolution_layer_49", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 0, (vx_reference)org_khronos_nn_extension_activation_layer_46_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_47, "org_khronos_nn_extension_convolution_layer_47", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_45, "org_khronos_nn_extension_activation_layer_45", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_45_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_45, "org_khronos_nn_extension_activation_layer_45", 1, (vx_reference)org_khronos_nn_extension_activation_layer_45_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_45, "org_khronos_nn_extension_activation_layer_45", 2, (vx_reference)org_khronos_nn_extension_activation_layer_45_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_45, "org_khronos_nn_extension_activation_layer_45", 3, (vx_reference)org_khronos_nn_extension_activation_layer_45_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_45, "org_khronos_nn_extension_activation_layer_45", 4, (vx_reference)org_khronos_nn_extension_activation_layer_45_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_49, "org_khronos_nn_extension_activation_layer_49", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_49_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_49, "org_khronos_nn_extension_activation_layer_49", 1, (vx_reference)org_khronos_nn_extension_activation_layer_49_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_49, "org_khronos_nn_extension_activation_layer_49", 2, (vx_reference)org_khronos_nn_extension_activation_layer_49_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_49, "org_khronos_nn_extension_activation_layer_49", 3, (vx_reference)org_khronos_nn_extension_activation_layer_49_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_49, "org_khronos_nn_extension_activation_layer_49", 4, (vx_reference)org_khronos_nn_extension_activation_layer_49_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_47, "org_khronos_nn_extension_activation_layer_47", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_47_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_47, "org_khronos_nn_extension_activation_layer_47", 1, (vx_reference)org_khronos_nn_extension_activation_layer_47_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_47, "org_khronos_nn_extension_activation_layer_47", 2, (vx_reference)org_khronos_nn_extension_activation_layer_47_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_47, "org_khronos_nn_extension_activation_layer_47", 3, (vx_reference)org_khronos_nn_extension_activation_layer_47_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_47, "org_khronos_nn_extension_activation_layer_47", 4, (vx_reference)org_khronos_nn_extension_activation_layer_47_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 0, (vx_reference)outputAllocators_MergeTensor_7_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_56, "org_khronos_nn_extension_convolution_layer_56", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 0, (vx_reference)outputAllocators_MergeTensor_7_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_54, "org_khronos_nn_extension_convolution_layer_54", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 0, (vx_reference)outputAllocators_MergeTensor_7_p0);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_52, "org_khronos_nn_extension_convolution_layer_52", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p8);
//    if(status != VX_SUCCESS)
//        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_12, "org_khronos_nn_extension_pooling_layer_12", 0, (vx_reference)outputAllocators_MergeTensor_7_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_12, "org_khronos_nn_extension_pooling_layer_12", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_12, "org_khronos_nn_extension_pooling_layer_12", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_12, "org_khronos_nn_extension_pooling_layer_12", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_12, "org_khronos_nn_extension_pooling_layer_12", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_12, "org_khronos_nn_extension_pooling_layer_12", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_12, "org_khronos_nn_extension_pooling_layer_12", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_12, "org_khronos_nn_extension_pooling_layer_12", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_56, "org_khronos_nn_extension_activation_layer_56", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_56_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_56, "org_khronos_nn_extension_activation_layer_56", 1, (vx_reference)org_khronos_nn_extension_activation_layer_56_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_56, "org_khronos_nn_extension_activation_layer_56", 2, (vx_reference)org_khronos_nn_extension_activation_layer_56_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_56, "org_khronos_nn_extension_activation_layer_56", 3, (vx_reference)org_khronos_nn_extension_activation_layer_56_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_56, "org_khronos_nn_extension_activation_layer_56", 4, (vx_reference)org_khronos_nn_extension_activation_layer_56_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_54, "org_khronos_nn_extension_activation_layer_54", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_54_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_54, "org_khronos_nn_extension_activation_layer_54", 1, (vx_reference)org_khronos_nn_extension_activation_layer_54_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_54, "org_khronos_nn_extension_activation_layer_54", 2, (vx_reference)org_khronos_nn_extension_activation_layer_54_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_54, "org_khronos_nn_extension_activation_layer_54", 3, (vx_reference)org_khronos_nn_extension_activation_layer_54_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_54, "org_khronos_nn_extension_activation_layer_54", 4, (vx_reference)org_khronos_nn_extension_activation_layer_54_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_52, "org_khronos_nn_extension_activation_layer_52", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_52_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_52, "org_khronos_nn_extension_activation_layer_52", 1, (vx_reference)org_khronos_nn_extension_activation_layer_52_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_52, "org_khronos_nn_extension_activation_layer_52", 2, (vx_reference)org_khronos_nn_extension_activation_layer_52_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_52, "org_khronos_nn_extension_activation_layer_52", 3, (vx_reference)org_khronos_nn_extension_activation_layer_52_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_52, "org_khronos_nn_extension_activation_layer_52", 4, (vx_reference)org_khronos_nn_extension_activation_layer_52_p4);
    if(status != VX_SUCCESS)
        return status;
        
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_12_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_51, "org_khronos_nn_extension_convolution_layer_51", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 0, (vx_reference)org_khronos_nn_extension_activation_layer_54_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_55, "org_khronos_nn_extension_convolution_layer_55", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p8);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 0, (vx_reference)org_khronos_nn_extension_activation_layer_52_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 1, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p1);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 2, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p2);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 3, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p3);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 4, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p4);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 5, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p5);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 6, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p6);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 7, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p7);
//    if(status != VX_SUCCESS)
//        return status;
//
//    status = AssignNodeParameter(org_khronos_nn_extension_convolution_layer_53, "org_khronos_nn_extension_convolution_layer_53", 8, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p8);
//    if(status != VX_SUCCESS)
//        return status;

    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_51, "org_khronos_nn_extension_activation_layer_51", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_51_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_51, "org_khronos_nn_extension_activation_layer_51", 1, (vx_reference)org_khronos_nn_extension_activation_layer_51_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_51, "org_khronos_nn_extension_activation_layer_51", 2, (vx_reference)org_khronos_nn_extension_activation_layer_51_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_51, "org_khronos_nn_extension_activation_layer_51", 3, (vx_reference)org_khronos_nn_extension_activation_layer_51_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_51, "org_khronos_nn_extension_activation_layer_51", 4, (vx_reference)org_khronos_nn_extension_activation_layer_51_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_55, "org_khronos_nn_extension_activation_layer_55", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_55_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_55, "org_khronos_nn_extension_activation_layer_55", 1, (vx_reference)org_khronos_nn_extension_activation_layer_55_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_55, "org_khronos_nn_extension_activation_layer_55", 2, (vx_reference)org_khronos_nn_extension_activation_layer_55_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_55, "org_khronos_nn_extension_activation_layer_55", 3, (vx_reference)org_khronos_nn_extension_activation_layer_55_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_55, "org_khronos_nn_extension_activation_layer_55", 4, (vx_reference)org_khronos_nn_extension_activation_layer_55_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_53, "org_khronos_nn_extension_activation_layer_53", 0, (vx_reference)org_khronos_nn_extension_convolution_layer_53_p8);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_53, "org_khronos_nn_extension_activation_layer_53", 1, (vx_reference)org_khronos_nn_extension_activation_layer_53_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_53, "org_khronos_nn_extension_activation_layer_53", 2, (vx_reference)org_khronos_nn_extension_activation_layer_53_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_53, "org_khronos_nn_extension_activation_layer_53", 3, (vx_reference)org_khronos_nn_extension_activation_layer_53_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_activation_layer_53, "org_khronos_nn_extension_activation_layer_53", 4, (vx_reference)org_khronos_nn_extension_activation_layer_53_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_13, "org_khronos_nn_extension_pooling_layer_13", 0, (vx_reference)outputAllocators_MergeTensor_8_p0);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_13, "org_khronos_nn_extension_pooling_layer_13", 1, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_13, "org_khronos_nn_extension_pooling_layer_13", 2, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_13, "org_khronos_nn_extension_pooling_layer_13", 3, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_13, "org_khronos_nn_extension_pooling_layer_13", 4, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_13, "org_khronos_nn_extension_pooling_layer_13", 5, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_13, "org_khronos_nn_extension_pooling_layer_13", 6, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p6);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_pooling_layer_13, "org_khronos_nn_extension_pooling_layer_13", 7, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_fully_connected_layer_0, "org_khronos_nn_extension_fully_connected_layer_0", 0, (vx_reference)org_khronos_nn_extension_pooling_layer_13_p7);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_fully_connected_layer_0, "org_khronos_nn_extension_fully_connected_layer_0", 1, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_fully_connected_layer_0, "org_khronos_nn_extension_fully_connected_layer_0", 2, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_fully_connected_layer_0, "org_khronos_nn_extension_fully_connected_layer_0", 3, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_fully_connected_layer_0, "org_khronos_nn_extension_fully_connected_layer_0", 4, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_fully_connected_layer_0, "org_khronos_nn_extension_fully_connected_layer_0", 5, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_openvx_tensor_multiply_0, "org_khronos_openvx_tensor_multiply_0", 0, (vx_reference)org_khronos_nn_extension_fully_connected_layer_0_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_openvx_tensor_multiply_0, "org_khronos_openvx_tensor_multiply_0", 1, (vx_reference)org_khronos_openvx_tensor_multiply_0_p1);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_openvx_tensor_multiply_0, "org_khronos_openvx_tensor_multiply_0", 2, (vx_reference)org_khronos_openvx_tensor_multiply_0_p2);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_openvx_tensor_multiply_0, "org_khronos_openvx_tensor_multiply_0", 3, (vx_reference)org_khronos_openvx_tensor_multiply_0_p3);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_openvx_tensor_multiply_0, "org_khronos_openvx_tensor_multiply_0", 4, (vx_reference)org_khronos_openvx_tensor_multiply_0_p4);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_openvx_tensor_multiply_0, "org_khronos_openvx_tensor_multiply_0", 5, (vx_reference)org_khronos_openvx_tensor_multiply_0_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_softmax_layer_0, "org_khronos_nn_extension_softmax_layer_0", 0, (vx_reference)org_khronos_openvx_tensor_multiply_0_p5);
    if(status != VX_SUCCESS)
        return status;
        
    status = AssignNodeParameter(org_khronos_nn_extension_softmax_layer_0, "org_khronos_nn_extension_softmax_layer_0", 1, (vx_reference)org_khronos_nn_extension_softmax_layer_0_p1);
    if(status != VX_SUCCESS)
        return status;
        
 

    return status;
}
#endif
