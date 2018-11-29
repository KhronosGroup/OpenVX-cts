
/** @file load_weights.h
 *  @brief 
 *  This file contains the definitions of the weight and biases loading functions
 */
#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

#ifdef __cplusplus
extern "C" {
#endif


/** @brief Init all weights and biases
 *
 *  @param pObjectContainer The pointer to object container.
 *  @param pFileDir The path to the binary files location
 *  @return vx_status code.
 */
vx_status loadTensorFromFile(vx_tensor input, const char* pFileDir, const char* pFileName);
/** @brief Load Tensor object from file
 *
 *  @param input The Tensor reference to load.
 *  @param pFileDir The path to the binary files location
 *  @param pFileName The file name
 *  @return vx_status code.
 */

vx_status initAllWeightsAlexnet(ObjectRefContainerType* pContainer, const char* pFileDir);
vx_status initAllWeightsGooglenet(ObjectRefContainerType* pContainer, const char* pFileDir);


#ifdef __cplusplus
}
#endif

#endif /* LOAD_WEIGHTS_H */
