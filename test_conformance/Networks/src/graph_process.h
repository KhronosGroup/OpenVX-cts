/** @file graph_process.h
 *  @brief
 *  This file contains the definitions of the graph inputs/outputs processing functions
 */
#ifndef GRAPH_PROCESS_H
#define GRAPH_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif


/** @brief Pre-process the OpenVX graph inputs
 *
 *  @param input The input Tensor obejct to initialize
 *  @param path The jpeg filename to load
 *  @return vx_status code.
 */
vx_status preprocess(vx_tensor input, const char * path);

/** @brief Post-process the OpenVX graph outputs
 *
 *  @param output The output Tensor obejct to process
 *  @param detected_class The class detected for the image
 *  @return vx_status code.
 */
vx_status postprocess(vx_tensor output, /*OUT*/ int* detected_class);

vx_status debugDumpLayers(ObjectRefContainerType * vxObjectsContainer);

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_PROCESS_H */
