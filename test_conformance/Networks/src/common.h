/** @file common.h
 *  @brief 
 *  This file contains the definitions of the generated utility functions and common structures
 */
#ifndef COMMON_H
#define COMMON_H

#include <VX/vx.h>

/** The number of references to be created in the graph (generated automatically) */
#define MAX_REFERENCE_OBJECTS 963

#ifdef __cplusplus 
extern "C" {
#endif

/** Defines a record that will be used to access and to release an OpenVX reference */ 
typedef struct {
    vx_enum type;
    vx_reference   ref;
    void*          pMem;
    char*          uniqueRef;
} ObjectRefType;

/** Define a container of OpenVX references created in graph */
typedef struct {
    unsigned int   count;
    ObjectRefType* pObjects;
} ObjectRefContainerType;

typedef struct _pooling_params
{
	vx_size pooling_size_x;
	vx_size pooling_size_y;
	vx_size pooling_padding_x;
	vx_size pooling_padding_y;
	vx_enum rounding;
} pooling_params;


typedef struct _activation_params
{
	vx_enum function;
	vx_float32 a;
	vx_float32 b;
} activation_params;
typedef struct _normalization_params
{
	vx_enum type;
	vx_size normalization_size;
	vx_float32 alpha;
	vx_float32 beta;
} normalization_params;
/** @brief Releases all OpenVX references in graph.
 *
 *  @param pObjectContainer The pointer to object container.
 *  @return Void.
 */
void ReleaseObjects(ObjectRefContainerType* pObjectContainer);

/** @brief Add OpenVX reference to a container
 *
 *  @param pObjectContainer The pointer to object container.
 *  @param ref The OpenVX reference to add
 *  @param type The type of the OpenVX reference
 *  @param uniqueRef Unique string ID of the reference (used to access the reference)
 *  @return Void.
 */
void AddVXObject(ObjectRefContainerType* pObjectContainer, vx_reference ref, vx_enum type, const char* uniqueRef);

/** @brief Add object to container
 *
 *  @param pObjectContainer The pointer to object container.
 *  @param pMem The pointer to the object memory to add
 *  @return Void.
 */
void AddObject(ObjectRefContainerType* pObjectContainer, void* pMem);

/** @brief Get OpenVX reference by its unique ID
 *
 *  @param pObjectContainer The pointer to object container
 *  @param uniqueRef Unique string ID of the reference
 *  @return OpenVX reference.
 */
vx_reference GetObjectRef(ObjectRefContainerType* pObjectContainer, const char* uniqueRef);

/** @brief Callback function for OpenVX log
 *
 *  @param context The OpenVX context
 *  @param ref OpenVX reference that the log message is associated with
 *  @param status vx_status associated with the log message
 *  @param string The log messages
 *  @return Void.
 */
void VXLog(vx_context context, vx_reference ref, vx_status status, const vx_char string[]);

/** @brief Factory method to create a graph node
 *
 *  @param graph The OpenVX graph
 *  @param kernel The kernel to instantiate
 *  @param pObjectContainer The pointer to object container.
 *  @param nodeName The node name
 *  @param filteredNodesList The list of filtered nodes to create in the graph (can be empty)
 *  @param filteredNodesCount The number of filtered nodes to create in the graph
 *  @param node Pointer to the node to create
 *  @return vx_status code.
 */
vx_status CreateNode(vx_graph graph, vx_kernel kernel, ObjectRefContainerType* pObjectContainer, const char* nodeName, char* filteredNodesList[], size_t filteredNodesCount, vx_node* node);

/** @brief Assign node parameter
 *
 *  @param node The node to assign to
 *  @param nodeName The unique name of the node
 *  @param index The index of the port to assign to
 *  @param nodeName The parameter to assign
 *  @return vx_status code.
*/ 
vx_status AssignNodeParameter(vx_node node, const char* nodeName, vx_uint32 index, vx_reference parameter);


/** @brief Write log message
 *
 *  This message might be modified to redirect all log messages to any destination.
 *  By default, all messages are printed to stdout
 *
 *  @param format The format of the log message
 *  @return Number of logged arguments
 */
int WriteLog(const char* format, ...);

/** @brief Get enum description of vx_status
 *
 *  @param vxStatus vx_status value
 *  @return enum description of the vxStatus
 */
const char* getVxStatusDesc(int vxStatus);

#ifdef __cplusplus 
}
#endif

#endif /* COMMON_H */
