/** @file load_weights.c
 *  @brief 
 *  This file contains the implementation of the weight and biases loading functions
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <VX/vx.h>
#include <VX/vx_khr_nn.h>
#define VX_MAX_TENSOR_DIMS_CT 6

#include "common.h"

#include "load_weights.h"

vx_status initAllWeightsAlexnet(ObjectRefContainerType* pContainer, const char* pFileDir)
{
    vx_status status = VX_SUCCESS;

    WriteLog("Loading weights and biases from '%s'...\n", pFileDir);

    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv1_weights"), pFileDir, "conv1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv1_bias"), pFileDir, "conv1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv2_0_weights"), pFileDir, "conv2_0_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv2_0_bias"), pFileDir, "conv2_0_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv2_1_weights"), pFileDir, "conv2_1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv2_1_bias"), pFileDir, "conv2_1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv3_weights"), pFileDir, "conv3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv3_bias"), pFileDir, "conv3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv4_0_weights"), pFileDir, "conv4_0_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv4_0_bias"), pFileDir, "conv4_0_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv4_1_weights"), pFileDir, "conv4_1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv4_1_bias"), pFileDir, "conv4_1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv5_0_weights"), pFileDir, "conv5_0_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv5_0_bias"), pFileDir, "conv5_0_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv5_1_weights"), pFileDir, "conv5_1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv5_1_bias"), pFileDir, "conv5_1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "fc6_weights"), pFileDir, "fc6_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"fc6_bias"), pFileDir, "fc6_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "fc7_weights"), pFileDir, "fc7_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"fc7_bias"), pFileDir, "fc7_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "fc8_weights"), pFileDir, "fc8_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"fc8_bias"), pFileDir, "fc8_bias.bin");

    return status;
}

vx_status initAllWeightsGooglenet(ObjectRefContainerType* pContainer, const char* pFileDir)
{
    vx_status status = VX_SUCCESS;

    WriteLog("Loading weights and biases from '%s'...\n", pFileDir);

    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "Power0_scale"), pFileDir, "Power0_scale.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv1_7x7_s2_weights"), pFileDir, "conv1_7x7_s2_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv1_7x7_s2_bias"), pFileDir, "conv1_7x7_s2_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv2_3x3_weights"), pFileDir, "conv2_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv2_3x3_bias"), pFileDir, "conv2_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "conv2_3x3_reduce_weights"), pFileDir, "conv2_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"conv2_3x3_reduce_bias"), pFileDir, "conv2_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3a_1x1_weights"), pFileDir, "inception_3a_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3a_1x1_bias"), pFileDir, "inception_3a_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3a_3x3_weights"), pFileDir, "inception_3a_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3a_3x3_bias"), pFileDir, "inception_3a_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3a_3x3_reduce_weights"), pFileDir, "inception_3a_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3a_3x3_reduce_bias"), pFileDir, "inception_3a_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3a_5x5_weights"), pFileDir, "inception_3a_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3a_5x5_bias"), pFileDir, "inception_3a_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3a_5x5_reduce_weights"), pFileDir, "inception_3a_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3a_5x5_reduce_bias"), pFileDir, "inception_3a_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3a_pool_proj_weights"), pFileDir, "inception_3a_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3a_pool_proj_bias"), pFileDir, "inception_3a_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3b_1x1_weights"), pFileDir, "inception_3b_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3b_1x1_bias"), pFileDir, "inception_3b_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3b_3x3_weights"), pFileDir, "inception_3b_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3b_3x3_bias"), pFileDir, "inception_3b_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3b_3x3_reduce_weights"), pFileDir, "inception_3b_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3b_3x3_reduce_bias"), pFileDir, "inception_3b_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3b_5x5_weights"), pFileDir, "inception_3b_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3b_5x5_bias"), pFileDir, "inception_3b_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3b_5x5_reduce_weights"), pFileDir, "inception_3b_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3b_5x5_reduce_bias"), pFileDir, "inception_3b_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_3b_pool_proj_weights"), pFileDir, "inception_3b_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_3b_pool_proj_bias"), pFileDir, "inception_3b_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4a_1x1_weights"), pFileDir, "inception_4a_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4a_1x1_bias"), pFileDir, "inception_4a_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4a_3x3_weights"), pFileDir, "inception_4a_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4a_3x3_bias"), pFileDir, "inception_4a_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4a_3x3_reduce_weights"), pFileDir, "inception_4a_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4a_3x3_reduce_bias"), pFileDir, "inception_4a_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4a_5x5_weights"), pFileDir, "inception_4a_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4a_5x5_bias"), pFileDir, "inception_4a_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4a_5x5_reduce_weights"), pFileDir, "inception_4a_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4a_5x5_reduce_bias"), pFileDir, "inception_4a_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4a_pool_proj_weights"), pFileDir, "inception_4a_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4a_pool_proj_bias"), pFileDir, "inception_4a_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4b_1x1_weights"), pFileDir, "inception_4b_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4b_1x1_bias"), pFileDir, "inception_4b_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4b_3x3_weights"), pFileDir, "inception_4b_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4b_3x3_bias"), pFileDir, "inception_4b_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4b_3x3_reduce_weights"), pFileDir, "inception_4b_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4b_3x3_reduce_bias"), pFileDir, "inception_4b_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4b_5x5_weights"), pFileDir, "inception_4b_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4b_5x5_bias"), pFileDir, "inception_4b_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4b_5x5_reduce_weights"), pFileDir, "inception_4b_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4b_5x5_reduce_bias"), pFileDir, "inception_4b_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4b_pool_proj_weights"), pFileDir, "inception_4b_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4b_pool_proj_bias"), pFileDir, "inception_4b_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4c_1x1_weights"), pFileDir, "inception_4c_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4c_1x1_bias"), pFileDir, "inception_4c_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4c_3x3_weights"), pFileDir, "inception_4c_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4c_3x3_bias"), pFileDir, "inception_4c_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4c_3x3_reduce_weights"), pFileDir, "inception_4c_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4c_3x3_reduce_bias"), pFileDir, "inception_4c_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4c_5x5_weights"), pFileDir, "inception_4c_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4c_5x5_bias"), pFileDir, "inception_4c_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4c_5x5_reduce_weights"), pFileDir, "inception_4c_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4c_5x5_reduce_bias"), pFileDir, "inception_4c_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4c_pool_proj_weights"), pFileDir, "inception_4c_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4c_pool_proj_bias"), pFileDir, "inception_4c_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4d_1x1_weights"), pFileDir, "inception_4d_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4d_1x1_bias"), pFileDir, "inception_4d_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4d_3x3_weights"), pFileDir, "inception_4d_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4d_3x3_bias"), pFileDir, "inception_4d_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4d_3x3_reduce_weights"), pFileDir, "inception_4d_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4d_3x3_reduce_bias"), pFileDir, "inception_4d_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4d_5x5_weights"), pFileDir, "inception_4d_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4d_5x5_bias"), pFileDir, "inception_4d_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4d_5x5_reduce_weights"), pFileDir, "inception_4d_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4d_5x5_reduce_bias"), pFileDir, "inception_4d_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4d_pool_proj_weights"), pFileDir, "inception_4d_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4d_pool_proj_bias"), pFileDir, "inception_4d_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4e_1x1_weights"), pFileDir, "inception_4e_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4e_1x1_bias"), pFileDir, "inception_4e_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4e_3x3_weights"), pFileDir, "inception_4e_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4e_3x3_bias"), pFileDir, "inception_4e_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4e_3x3_reduce_weights"), pFileDir, "inception_4e_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4e_3x3_reduce_bias"), pFileDir, "inception_4e_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4e_5x5_weights"), pFileDir, "inception_4e_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4e_5x5_bias"), pFileDir, "inception_4e_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4e_5x5_reduce_weights"), pFileDir, "inception_4e_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4e_5x5_reduce_bias"), pFileDir, "inception_4e_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_4e_pool_proj_weights"), pFileDir, "inception_4e_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_4e_pool_proj_bias"), pFileDir, "inception_4e_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5a_1x1_weights"), pFileDir, "inception_5a_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5a_1x1_bias"), pFileDir, "inception_5a_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5a_3x3_weights"), pFileDir, "inception_5a_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5a_3x3_bias"), pFileDir, "inception_5a_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5a_3x3_reduce_weights"), pFileDir, "inception_5a_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5a_3x3_reduce_bias"), pFileDir, "inception_5a_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5a_5x5_weights"), pFileDir, "inception_5a_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5a_5x5_bias"), pFileDir, "inception_5a_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5a_5x5_reduce_weights"), pFileDir, "inception_5a_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5a_5x5_reduce_bias"), pFileDir, "inception_5a_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5a_pool_proj_weights"), pFileDir, "inception_5a_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5a_pool_proj_bias"), pFileDir, "inception_5a_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5b_1x1_weights"), pFileDir, "inception_5b_1x1_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5b_1x1_bias"), pFileDir, "inception_5b_1x1_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5b_3x3_weights"), pFileDir, "inception_5b_3x3_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5b_3x3_bias"), pFileDir, "inception_5b_3x3_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5b_3x3_reduce_weights"), pFileDir, "inception_5b_3x3_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5b_3x3_reduce_bias"), pFileDir, "inception_5b_3x3_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5b_5x5_weights"), pFileDir, "inception_5b_5x5_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5b_5x5_bias"), pFileDir, "inception_5b_5x5_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5b_5x5_reduce_weights"), pFileDir, "inception_5b_5x5_reduce_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5b_5x5_reduce_bias"), pFileDir, "inception_5b_5x5_reduce_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "inception_5b_pool_proj_weights"), pFileDir, "inception_5b_pool_proj_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"inception_5b_pool_proj_bias"), pFileDir, "inception_5b_pool_proj_bias.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer, "loss3_classifier_weights"), pFileDir, "loss3_classifier_weights.bin");
    status |= loadTensorFromFile((vx_tensor)GetObjectRef(pContainer,"loss3_classifier_bias"), pFileDir, "loss3_classifier_bias.bin");

    return status;
}

	vx_status loadTensorFromFile(vx_tensor tensor, const char* pFileDir, const char* pFileName)
	{
		vx_status status = VX_SUCCESS;

		char filePath[1024];
		strcpy(filePath, pFileDir);
		strcat(filePath, "/");
		strcat(filePath, pFileName);

		if (tensor == NULL)
		{
			WriteLog("ERROR: invalid tensor object!\n");
			return VX_FAILURE;
		}

		WriteLog("    - %s\n", pFileName);
		FILE* fp = fopen(filePath, "rb");
		if (fp == NULL)
		{
			WriteLog("ERROR: cannot open weights file\n");
			return VX_FAILURE;
		}

		fseek(fp, 0, SEEK_END); // seek to end of file
		long fileSize = ftell(fp); // get current file pointer
		fseek(fp, 0, SEEK_SET); // seek back to beginning of file

		void* ptr = malloc(fileSize);

		if (ptr == NULL)
		{
			WriteLog("ERROR: cannot allocate memory for reading the weights file\n");
			fclose(fp);
			return VX_FAILURE;
		}

		size_t nRead = fread(ptr, 1, fileSize, fp);
		fclose(fp);

    if(nRead != fileSize)
		{
			WriteLog("ERROR: read error on weights file\n");
			free(ptr);
			return VX_FAILURE;
		}

		vx_enum dataFormat;
		status |= vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &dataFormat, sizeof(dataFormat));

		vx_size tensorNumDims;
		status |= vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &tensorNumDims, sizeof(tensorNumDims));

		vx_size tensorDims[VX_MAX_TENSOR_DIMS_CT];
		status |= vxQueryTensor(tensor, VX_TENSOR_DIMS, tensorDims, sizeof(tensorDims));

		if (status != VX_SUCCESS)
		{
			WriteLog("ERROR: cannot query tensor properties\n");
			free(ptr);
			return VX_FAILURE;
		}

		vx_size viewStart[VX_MAX_TENSOR_DIMS_CT];
		memset(viewStart, 0, VX_MAX_TENSOR_DIMS_CT * sizeof(vx_size));

		vx_size userStrides[VX_MAX_TENSOR_DIMS_CT];
		userStrides[0] = dataFormat == VX_TYPE_FLOAT32 ? sizeof(vx_float32) : sizeof(vx_int16);

		vx_size i = 1;
		for (; i < VX_MAX_TENSOR_DIMS_CT; i++)
		{
			userStrides[i] = userStrides[i - 1] * tensorDims[i - 1];
		}

		if (tensorDims[tensorNumDims - 1] * userStrides[tensorNumDims - 1] != fileSize)
		{
			WriteLog("ERROR: inconsistent tensor and weight file sizes!\n");
			free(ptr);
			return status;
		}

		vx_size vxSizeTensorDims[VX_MAX_TENSOR_DIMS_CT];
		i = 0;
		for (; i < tensorNumDims; i++)
		{
			vxSizeTensorDims[i] = tensorDims[i];
		}

		status |= vxCopyTensorPatch(tensor, (vx_size)tensorNumDims, viewStart, vxSizeTensorDims, userStrides, ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

		if (status != VX_SUCCESS)
		{
			WriteLog("ERROR: cannot copy tensor patch!\n");
			free(ptr);
			return status;
		}

		free(ptr);

		return VX_SUCCESS;
	}
