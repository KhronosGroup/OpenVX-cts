/** @file common.c
 *  @brief
 *  This file contains the implementation of the generated utility functions
 */
#ifdef OPENVX_CONFORMANCE_NEURAL_NETWORKS
#ifdef OPENVX_USE_NN_16

#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

#include <VX/vx_khr_nn.h>
#include <VX/vx_types.h>

void ReleaseObjects(ObjectRefContainerType* pObjectContainer)
{
    if(pObjectContainer)
    {
        for(unsigned int i = 0; i < pObjectContainer->count; ++i)
        {
            ObjectRefType* pObj = pObjectContainer->pObjects + i;

            if(pObj->ref)
            {
                if(pObj->type == VX_TYPE_TENSOR)
                    vxReleaseTensor((vx_tensor*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_SCALAR)
                    vxReleaseScalar((vx_scalar*)&(pObj->ref));
#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION
                else if(pObj->type == VX_TYPE_IMAGE)
                    vxReleaseImage((vx_image*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_ARRAY)
                    vxReleaseArray((vx_array*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_REMAP)
                    vxReleaseRemap((vx_remap*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_CONVOLUTION)
                    vxReleaseConvolution((vx_convolution*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_MATRIX)
                    vxReleaseMatrix((vx_matrix*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_THRESHOLD)
                    vxReleaseThreshold((vx_threshold*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_PYRAMID)
                    vxReleasePyramid((vx_pyramid*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_DISTRIBUTION)
                    vxReleaseDistribution((vx_distribution*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_LUT)
                    vxReleaseLUT((vx_lut*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_OBJECT_ARRAY)
                    vxReleaseObjectArray((vx_object_array*)&(pObj->ref));
                else if(pObj->type == VX_TYPE_DELAY)
                    vxReleaseDelay((vx_delay*)&(pObj->ref));
#endif
            }
            else if(pObj->pMem)
            {
                free(pObj->pMem);
            }

            if(pObj->uniqueRef)
                free(pObj->uniqueRef);

            // Release of VX_TYPE_KERNEL and VX_TYPE_NODE ignored for now
        }
    }
}

void AddVXObject(ObjectRefContainerType* pObjectContainer, vx_reference ref, vx_enum type, const char* uniqueRef)
{
    if(MAX_REFERENCE_OBJECTS == pObjectContainer->count)
    {
        WriteLog("ERROR: cannot add object to reference pool. Max items [%d] reached\n", MAX_REFERENCE_OBJECTS);
        return;
    }

    ObjectRefType* pNewObj = &pObjectContainer->pObjects[pObjectContainer->count];
    pNewObj->type = type;
    pNewObj->ref  = ref;
    pNewObj->pMem = 0;

    if(uniqueRef)
    {
        size_t buflen = strlen(uniqueRef)+1;
        pNewObj->uniqueRef = (char*)malloc(buflen);
        memcpy(pNewObj->uniqueRef, uniqueRef, buflen);
    }
    else
    {
        pNewObj->uniqueRef = 0;
    }
    pObjectContainer->count++;
}

void AddObject(ObjectRefContainerType* pObjectContainer, void* pMem)
{
    if(MAX_REFERENCE_OBJECTS == pObjectContainer->count)
    {
        WriteLog("ERROR: cannot add object to reference pool. Max items [%d] reached\n", MAX_REFERENCE_OBJECTS);
        return;
    }

    (pObjectContainer->pObjects + pObjectContainer->count)->type = (vx_enum)0;
    (pObjectContainer->pObjects + pObjectContainer->count)->ref  = 0;
    (pObjectContainer->pObjects + pObjectContainer->count)->pMem = pMem;
    (pObjectContainer->pObjects + pObjectContainer->count)->uniqueRef = 0;

    pObjectContainer->count++;
}

vx_reference GetObjectRef(ObjectRefContainerType* pObjectContainer, const char* uniqueRef)
{
    unsigned int i;

    if(pObjectContainer)
    {
        for(i = 0; i < pObjectContainer->count; ++i)
        {
            ObjectRefType* pObj = pObjectContainer->pObjects + i;

            if(pObj->ref && pObj->uniqueRef && (0 == strcmp(uniqueRef, pObj->uniqueRef)))
            {
                return pObj->ref;
            }
        }
    }

    return 0;
}

void VXLog(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    WriteLog("VXLOG: %s (sts=%s) \n", string, getVxStatusDesc(status));
}

vx_status AssignNodeParameter(vx_node node, const char* nodeName, vx_uint32 index, vx_reference parameter)
{
    if(node)
    {
        vx_status status = vxSetParameterByIndex(node, index, parameter);
        if(status != VX_SUCCESS)
        {
            WriteLog("ERROR: cannot set parameter at index %d for '%s' node (vx_status=%s)\n", index, nodeName, getVxStatusDesc(status));
            return status;
        }
        return status;
    }
    return VX_SUCCESS;
}

vx_status CreateNode(vx_graph graph, vx_kernel kernel, ObjectRefContainerType* pObjectContainer, const char* nodeName, char* filteredNodesList[], size_t filteredNodesCount, vx_node* node)
{
    bool bCreateNode = true;
    // Check if the node is filtered out
    if(filteredNodesList && filteredNodesCount > 0)
    {
        bCreateNode = false;
        for(unsigned int i = 0; i < filteredNodesCount; i++)
        {
            if(0 == strcmp(nodeName, filteredNodesList[i]))
            {
                bCreateNode = true;
                break;
            }
        }
    }

    if(bCreateNode)
    {
        WriteLog("    - %s\n", nodeName);
        *node = vxCreateGenericNode(graph, kernel);
        vx_status status = vxGetStatus((vx_reference)*node);
        if(status != VX_SUCCESS)
        {
            WriteLog("ERROR: failed to create node '%s'\n", nodeName);
            return status;
        }
        AddVXObject(pObjectContainer, (vx_reference)*node, VX_TYPE_NODE, nodeName);
    }
    else
    {
        *node = NULL;
    }
    return VX_SUCCESS;
}

int WriteLog(const char* format, ...)
{
    char buffer[1024];
    va_list args;
    va_start(args, format);
    int retVal = vsprintf (buffer,format, args);
    printf("%s", buffer);
    va_end(args);
    return retVal;
}

char** getAllVxStatusEnums()
{
    static char allVxStatusEnumsContainer[1024];
    static char* statusArray[VX_SUCCESS - VX_STATUS_MIN + 1];
    static bool firstTime = true;
    if (firstTime)
    {
        size_t index = 0;
        for (int i = VX_STATUS_MIN; i <= VX_SUCCESS; ++i)
        {
            if (i == VX_ERROR_REFERENCE_NONZERO) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_REFERENCE_NONZERO");
            else if (i == VX_ERROR_MULTIPLE_WRITERS) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_MULTIPLE_WRITERS");
            else if (i == VX_ERROR_GRAPH_ABANDONED) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_GRAPH_ABANDONED");
            else if (i == VX_ERROR_GRAPH_SCHEDULED) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_GRAPH_SCHEDULED");
            else if (i == VX_ERROR_INVALID_SCOPE) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_SCOPE");
            else if (i == VX_ERROR_INVALID_NODE) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_NODE");
            else if (i == VX_ERROR_INVALID_GRAPH) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_GRAPH");
            else if (i == VX_ERROR_INVALID_TYPE) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_TYPE");
            else if (i == VX_ERROR_INVALID_VALUE) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_VALUE");
            else if (i == VX_ERROR_INVALID_DIMENSION) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_DIMENSION");
            else if (i == VX_ERROR_INVALID_FORMAT) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_FORMAT");
            else if (i == VX_ERROR_INVALID_LINK) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_LINK");
            else if (i == VX_ERROR_INVALID_REFERENCE) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_REFERENCE");
            else if (i == VX_ERROR_INVALID_MODULE) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_MODULE");
            else if (i == VX_ERROR_INVALID_PARAMETERS) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_INVALID_PARAMETERS");
            else if (i == VX_ERROR_OPTIMIZED_AWAY) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_OPTIMIZED_AWAY");
            else if (i == VX_ERROR_NO_MEMORY) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_NO_MEMORY");
            else if (i == VX_ERROR_NO_RESOURCES) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_NO_RESOURCES");
            else if (i == VX_ERROR_NOT_COMPATIBLE) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_NOT_COMPATIBLE");
            else if (i == VX_ERROR_NOT_ALLOCATED) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_NOT_ALLOCATED");
            else if (i == VX_ERROR_NOT_SUFFICIENT) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_NOT_SUFFICIENT");
            else if (i == VX_ERROR_NOT_SUPPORTED) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_NOT_SUPPORTED");
            else if (i == VX_ERROR_NOT_IMPLEMENTED) strcpy(allVxStatusEnumsContainer + index, "VX_ERROR_NOT_IMPLEMENTED");
            else if (i == VX_FAILURE) strcpy(allVxStatusEnumsContainer + index, "VX_FAILURE");
            else if (i == VX_SUCCESS) strcpy(allVxStatusEnumsContainer + index, "VX_SUCCESS");
            else strcpy(allVxStatusEnumsContainer + index, "Unknown error");
            statusArray[i - VX_STATUS_MIN] = allVxStatusEnumsContainer + index;
            index += strlen(allVxStatusEnumsContainer + index) + 1;
        }
        firstTime = false;
    }
    return statusArray;
}

const char* getVxStatusDesc(int vxStatus)
{
    char** statusContainer = getAllVxStatusEnums();
    static char invalidStatus[30];
    if (vxStatus < VX_STATUS_MIN || vxStatus > VX_SUCCESS)
    {
        sprintf(invalidStatus, "%d - unknown error", vxStatus);
        return invalidStatus;
    }
    return statusContainer[vxStatus - VX_STATUS_MIN];
}

#endif
#endif
