/** @file graph.h
 *  @brief 
 *  This file contains the definition of the generated graph factory function
 */
 
#ifndef GRAPH_H
#define GRAPH_H

#include <VX/vx.h>
#include "common.h"

#ifdef __cplusplus 
extern "C" {
#endif

/** @brief Constructs OpenVX graph
 *
 *  @param context The OpenVX context
 *  @param graph The OpenVX graph
 *  @param pContainer The pointer to object container.
 *  @param filteredNodesList The list of filtered nodes to create in the graph (can be empty)
 *  @param filteredNodesCount The number of filtered nodes to create in the graph
 *  @return vx_status code.
 */
vx_status _GraphFactoryAlexnet(vx_context context, vx_graph graph, ObjectRefContainerType* pContainer, char* filteredNodesList[], size_t filteredNodesCount);

#ifdef __cplusplus 
}
#endif

#endif /* GRAPH_H */
