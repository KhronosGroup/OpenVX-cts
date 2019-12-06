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

#ifdef OPENVX_CONFORMANCE_NEURAL_NETWORKS
#ifdef OPENVX_USE_NN_16

#include "test_tensor_util.h"

#include "Networks/src/common.h"
#include "Networks/src/graph.h"
#include "Networks/src/load_weights.h"
#include "Networks/src/graph_process.h"

#include <stdio.h>


TESTCASE(TensorNetworks, CT_VXContext, ct_setup_vx_context, 0)


/****************************************************************************
 *                                                                          *
 *                              Reference Values                            *
 *                                                                          *
 ***************************************************************************/

typedef struct
{
    const char * file;
    int alexnet_classification;
} ref_t;

const ref_t refs[] = {
	{ "00000000.jpg", 89 },
	{ "00000001.jpg", 89 },
	{ "00000002.jpg", 89 },
	{ "00000003.jpg", 89 },
	{ "00000004.jpg", 109 },
	{ "00000005.jpg", 109 },
	{ "00000006.jpg", 109 },
	{ "00000007.jpg", 109 },
	{ "00000008.jpg", 109 },
	{ "00000009.jpg", 125 },
	{ "00000010.jpg", 132 },
	{ "00000011.jpg", 215 },
	{ "00000012.jpg", 345 },
	{ "00000013.jpg", 366 },
	{ "00000014.jpg", 366 },
	{ "00000015.jpg", 393 },
	{ "00000016.jpg", 397 },
	{ "00000017.jpg", 397 },
	{ "00000018.jpg", 401 },
	{ "00000019.jpg", 401 },
	{ "00000020.jpg", 401 },
	{ "00000021.jpg", 404 },
	{ "00000022.jpg", 406 },
	{ "00000023.jpg", 406 },
	{ "00000024.jpg", 409 },
	{ "00000025.jpg", 412 },
	{ "00000026.jpg", 417 },
	{ "00000027.jpg", 417 },
	{ "00000028.jpg", 418 },
	{ "00000029.jpg", 418 },
	{ "00000030.jpg", 426 },
	{ "00000031.jpg", 434 },
	{ "00000032.jpg", 441 },
	{ "00000033.jpg", 460 },
	{ "00000034.jpg", 470 },
	{ "00000035.jpg", 470 },
	{ "00000036.jpg", 479 },
	{ "00000037.jpg", 484 },
	{ "00000038.jpg", 484 },
	{ "00000039.jpg", 506 },
	{ "00000040.jpg", 507 },
	{ "00000041.jpg", 520 },
	{ "00000042.jpg", 520 },
	{ "00000043.jpg", 527 },
	{ "00000044.jpg", 528 },
	{ "00000045.jpg", 528 },
	{ "00000046.jpg", 530 },
	{ "00000047.jpg", 531 },
	{ "00000048.jpg", 567 },
	{ "00000049.jpg", 574 },
	{ "00000050.jpg", 574 },
	{ "00000051.jpg", 574 },
	{ "00000052.jpg", 576 },
	{ "00000053.jpg", 611 },
	{ "00000054.jpg", 611 },
	{ "00000055.jpg", 611 },
	{ "00000056.jpg", 625 },
	{ "00000057.jpg", 632 },
	{ "00000058.jpg", 632 },
	{ "00000059.jpg", 632 },
	{ "00000060.jpg", 632 },
	{ "00000061.jpg", 651 },
	{ "00000062.jpg", 651 },
	{ "00000063.jpg", 673 },
	{ "00000064.jpg", 681 },
	{ "00000065.jpg", 703 },
	{ "00000066.jpg", 703 },
	{ "00000067.jpg", 703 },
	{ "00000068.jpg", 722 },
	{ "00000069.jpg", 726 },
	{ "00000070.jpg", 726 },
	{ "00000071.jpg", 726 },
	{ "00000072.jpg", 726 },
	{ "00000073.jpg", 727 },
	{ "00000074.jpg", 733 },
	{ "00000075.jpg", 745 },
	{ "00000076.jpg", 752 },
	{ "00000077.jpg", 752 },
	{ "00000078.jpg", 759 },
	{ "00000079.jpg", 761 },
	{ "00000080.jpg", 809 },
	{ "00000081.jpg", 809 },
	{ "00000082.jpg", 823 },
	{ "00000083.jpg", 823 },
	{ "00000084.jpg", 823 },
	{ "00000085.jpg", 823 },
	{ "00000086.jpg", 823 },
	{ "00000087.jpg", 840 },
	{ "00000088.jpg", 845 },
	{ "00000089.jpg", 845 },
	{ "00000090.jpg", 845 },
	{ "00000091.jpg", 845 },
	{ "00000092.jpg", 847 },
	{ "00000093.jpg", 847 },
	{ "00000094.jpg", 852 },
	{ "00000095.jpg", 852 },
	{ "00000096.jpg", 852 },
	{ "00000097.jpg", 852 },
	{ "00000098.jpg", 852 },
	{ "00000099.jpg", 852 },
	{ "00000100.jpg", 852 },
	{ "00000101.jpg", 867 },
	{ "00000102.jpg", 870 },
	{ "00000103.jpg", 894 },
	{ "00000104.jpg", 894 },
	{ "00000105.jpg", 894 },
	{ "00000106.jpg", 896 },
	{ "00000107.jpg", 896 },
	{ "00000108.jpg", 898 },
	{ "00000109.jpg", 935 },
	{ "00000110.jpg", 937 },
	{ "00000111.jpg", 938 },
	{ "00000112.jpg", 971 },
	{ "00000113.jpg", 971 },
	{ "00000114.jpg", 971 },
	{ "00000115.jpg", 971 },
	{ "00000116.jpg", 972 },
	{ "00000117.jpg", 972 },
	{ "00000118.jpg", 973 },
	{ "00000119.jpg", 990 },
	{ "00000120.jpg", 999 },
};


const int refs_count = sizeof(refs) / sizeof(refs[0]);

const int alexnet_correct_detections = 95;
const int min_correct_alexnet = 83; //alexnet_correct_detections - refs_count / 10;


/****************************************************************************
 *                                                                          *
 *                             Common Network Code                          *
 *                                                                          *
 ***************************************************************************/


/****************************************************************************
 *                                                                          *
 *                                Test Code                                 *
 *                                                                          *
 ***************************************************************************/


/****************************************************************************
 *                                                                          *
 *                                 AlexNet                                  *
 *                                                                          *
 ***************************************************************************/

// NOTE: Most of the following is taken from the auto generated MO code

TEST(TensorNetworks, AlexNetTestNetwork)
{
    vx_status status   = VX_SUCCESS;
    vx_context context = context_->vx_context_;
    vx_graph graph     = NULL;
    char weights_path_full[MAXPATHLENGTH];

    ObjectRefType            vxObjects[MAX_REFERENCE_OBJECTS];
    ObjectRefContainerType   vxObjectsContainer;

    vxObjectsContainer.count    = 0;
    vxObjectsContainer.pObjects = &vxObjects[0];

    const char * images_path = "images";
    const char * weights_path = "../test_conformance/Networks/Binaries/Alexnet";
    snprintf(weights_path_full, MAXPATHLENGTH, "%s/%s", ct_get_test_file_path(), weights_path);

    // Register OpenVX log callback
    vxRegisterLogCallback(context, (vx_log_callback_f)VXLog, vx_true_e);

    int correct_detections = 0;

    // Create the OpenVX graph instance
    graph = vxCreateGraph(context);
    status |= vxGetStatus((vx_reference)graph);
    if(status == VX_SUCCESS)
    {
        /*
         * List of nodes to define a graph partition to create (use for debug purposes)
         * Note: 1) If the list is empty, the entire graph will be created
         *       2) The developer is responsible to initialize the graph partition input and process the outputs
         */
        char** includeNodesList = NULL; //[] = { }; // List of nodes to include (specified by node name). If empty, all graph nodes will be created.
        size_t filteredNodeCount = 0;//sizeof(includeNodesList) / sizeof(char *);
        // Call the graph factory to construct the graph structure
        status = _GraphFactoryAlexnet(context, graph, &vxObjectsContainer, includeNodesList, filteredNodeCount);
        if(status == VX_SUCCESS)
        {
            // Initialize graph weights and biases
            status = initAllWeightsAlexnet(&vxObjectsContainer, weights_path_full);
            if(status == VX_SUCCESS)
            {
                // Get reference to the graph input
                vx_tensor input = (vx_tensor)GetObjectRef(&vxObjectsContainer, "cnn_input");

                // Verify OpenVX graph integrity
                status = vxVerifyGraph(graph);
                if(status == VX_SUCCESS)
                {
                    for (int image_num = 0; image_num < refs_count; ++image_num)
                    {
                        char image_file[255] = "";
                        int n = snprintf(image_file, sizeof(image_file), "%s/%s/%s",ct_get_test_file_path(), images_path, refs[image_num].file);
                        EXPECT_EQ_INT(n >= 0, 1);
                        EXPECT_EQ_INT(n < sizeof(image_file), 1);
                        if (n < 0 || n >= sizeof(image_file)) continue;

                        // Initialize graph input
                        status = preprocess(input, image_file);
                        if(status == VX_SUCCESS)
                        {
                            // Process the OpenVX graph
                            status = vxProcessGraph(graph);
                            if(status == VX_SUCCESS)
                            {
                                // Get reference to the graph output
                                vx_tensor output = (vx_tensor)GetObjectRef(&vxObjectsContainer, "cnn_output");

                                // Process graph output
                                int detected_class;
                                status = postprocess(output, &detected_class);
                                if(status != VX_SUCCESS)
                                {
                                    WriteLog("ERROR: failed to process graph execution results (vx_status=%s)\n", getVxStatusDesc(status));
                                }
                                printf("predicted class: %d, expected class: %d\n", detected_class, refs[image_num].alexnet_classification);

                                if (detected_class == refs[image_num].alexnet_classification)
                                {
                                    ++correct_detections;
                                }

                                status = debugDumpLayers(&vxObjectsContainer);
                                if(status != VX_SUCCESS)
                                {
                                    WriteLog("ERROR: failed to dump all layers post graph execution results (vx_status=%s)\n", getVxStatusDesc(status));
                                }
                            }
                            else
                            {
                                WriteLog("ERROR: failed to process graph outputs (vx_status=%s)\n", getVxStatusDesc(status));
                            }
                        }
                        else
                        {
                            WriteLog("ERROR: failed to process graph inputs (vx_status=%s)\n", getVxStatusDesc(status));
                        }
                    }
                }
                else
                {
                    WriteLog("ERROR: failed to verify graph (vx_status=%s)\n", getVxStatusDesc(status));
                }
            }
            else
            {
                WriteLog("ERROR: failed to load weights\n");
            }
        }
        else
        {
            WriteLog("ERROR: failed to build graph (vx_status=%s)\n", getVxStatusDesc(status));
        }
    }
    else
    {
        WriteLog("ERROR: failed to create graph (vx_status=%s)\n", getVxStatusDesc(status));
    }

    // Release all OpenVX objects
    ReleaseObjects(&vxObjectsContainer);

    if(graph)
    {
        // Release OpenVX graph
        status = vxReleaseGraph(&graph);
        if(status != VX_SUCCESS)
        {
            WriteLog("ERROR: failed to release graph (vx_status=%s)\n", getVxStatusDesc(status));
        }
    }

    VX_CALL(status);
    if (correct_detections < min_correct_alexnet)
    {
        printf("correct detections: %d out of %d (ref has %d correct) min required to pass: %d\n",
               correct_detections, refs_count, alexnet_correct_detections, min_correct_alexnet);
        EXPECT_EQ_INT(correct_detections >= min_correct_alexnet, 1);
    }
}

/****************************************************************************
 *                                                                          *
 *                                 GoogleNet                                *
 *                                                                          *
 ***************************************************************************/

//TEST(TensorNetworks, GoogleNet)
//{
//}


/****************************************************************************
 *                                                                          *
 *                                    FCN                                   *
 *                                                                          *
 ***************************************************************************/

//TEST(TensorNetworks, FCN)
//{
//}

TESTCASE_TESTS(TensorNetworks,
    AlexNetTestNetwork
//    GoogleNet,
//    FCN
)
#endif
#endif//OPENVX_CONFORMANCE_NEURAL_NETWORKS