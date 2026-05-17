# OpenVX 1.3.2 Conformance Test Suite -- API Coverage Report

This document provides a comprehensive mapping of CTS test coverage for every OpenVX 1.3.2 API function. It covers all core API functions, node creation functions, immediate-mode utility functions, and extension APIs.

## Contents

* [Summary](#summary)
* [Conformance Test Modes](#conformance-test-modes)
* [Test File Inventory](#test-file-inventory)
* [Core API Coverage](#core-api-coverage)
  * [Context Functions](#context-functions)
  * [Image Functions](#image-functions)
  * [Kernel Functions](#kernel-functions)
  * [Graph Functions](#graph-functions)
  * [Node Functions](#node-functions)
  * [Parameter Functions](#parameter-functions)
  * [Scalar Functions](#scalar-functions)
  * [Reference Functions](#reference-functions)
  * [Delay Functions](#delay-functions)
  * [Logging Functions](#logging-functions)
  * [LUT Functions](#lut-functions)
  * [Distribution Functions](#distribution-functions)
  * [Threshold Functions](#threshold-functions)
  * [Matrix Functions](#matrix-functions)
  * [Convolution Functions](#convolution-functions)
  * [Pyramid Functions](#pyramid-functions)
  * [Remap Functions](#remap-functions)
  * [Array Functions](#array-functions)
  * [Object Array Functions](#object-array-functions)
  * [Tensor Functions](#tensor-functions)
  * [Meta Format Functions](#meta-format-functions)
* [Node Creation Function Coverage](#node-creation-function-coverage)
* [Immediate Mode Function Coverage](#immediate-mode-function-coverage)
* [Extension API Coverage](#extension-api-coverage)
  * [Import/Export (IX)](#importexport-ix)
  * [Neural Networks (NN)](#neural-networks-nn)
  * [User Data Object](#user-data-object)
  * [NNEF Import Kernel](#nnef-import-kernel)
  * [Pipelining](#pipelining)
  * [Streaming](#streaming)
  * [Event Queue](#event-queue)
  * [Compatibility/Deprecated](#compatibilitydeprecated)
* [Extensions Without CTS Coverage](#extensions-without-cts-coverage)
* [CI Pipeline Modes](#ci-pipeline-modes)

---

## Summary

| Category | Total Functions | CTS Covered | Coverage |
|----------|----------------|-------------|----------|
| Core API (`vx_api.h`) | 155 | 155 | 100% |
| Node Creation (`vx_nodes.h`) | 61 | 61 | 100% |
| Immediate Mode (`vxu.h`) | 59 | 59 | 100% |
| **Core Total** | **275** | **275** | **100%** |
| Import/Export (IX) | 5 | 5 | 100% |
| Neural Networks (NN) | 8 | 8 | 100% |
| User Data Object | 8 | 6 | 75% |
| NNEF Import Kernel | 1 | 1 | 100% |
| Pipelining | 7 | 4 | 57% |
| Streaming | 3 | 3 | 100% |
| Event Queue | 10 | 5 | 50% |
| Compatibility/Deprecated | 26 | 4 | 15% |

> **Note:** 7 node/vxu functions (`vxAbsDiffNode`, `vxXorNode`, `vxuAbsDiff`, `vxuAnd`, `vxuOr`, `vxuXor`, plus `vxAndNode`/`vxOrNode`) are tested through C preprocessor token-pasting macros (e.g., `FUNC_ARG(AbsDiff)` expands to both `vxAbsDiffNode` and `vxuAbsDiff`), not direct string references.

---

## Conformance Test Modes

The CTS supports multiple conformance modes, controlled by CMake options:

| Mode | CMake Options | Description |
|------|--------------|-------------|
| 1 | `-DOPENVX_CONFORMANCE_VISION=ON` | Vision conformance |
| 2 | `-DOPENVX_CONFORMANCE_VISION=ON -DOPENVX_USE_ENHANCED_VISION=ON` | Vision & Enhanced Vision |
| 3 | `-DOPENVX_CONFORMANCE_NEURAL_NETWORKS=ON` | Neural Networks |
| 4 | `-DOPENVX_CONFORMANCE_NNEF_IMPORT=ON` | NNEF Import |
| 5 | `-DOPENVX_CONFORMANCE_VISION=ON -DOPENVX_USE_ENHANCED_VISION=ON -DOPENVX_CONFORMANCE_NEURAL_NETWORKS=ON -DOPENVX_USE_NN=ON -DOPENVX_USE_IX=ON -DOPENVX_USE_U1=ON` | Combined |
| 6 | `-DOPENVX_USE_USER_DATA_OBJECT=ON` | User Data Object |
| 7 | `-DOPENVX_CONFORMANCE_VISION=ON -DOPENVX_USE_ENHANCED_VISION=ON -DOPENVX_USE_PIPELINING=ON -DOPENVX_USE_STREAMING=ON` | Pipelining & Streaming |

---

## Test File Inventory

The CTS contains 73 test source files organized by functional area:

### Core Framework Tests
| Test File | Area | Key APIs Tested |
|-----------|------|-----------------|
| `test_smoke.c` | Framework | Context, kernel, reference, array lifecycle |
| `test_graph.c` | Graph | Graph creation, verification, execution, parameters, user kernels |
| `test_graph_callbacks.c` | Graph | Node callbacks (`vxAssignNodeCallback`, `vxRetrieveNodeCallback`) |
| `test_graph_delay.c` | Graph | Delay objects, auto-aging, graph parameters |
| `test_graph_roi.c` | Graph | Image ROI in graph pipelines |
| `test_logging.c` | Logging | `vxAddLogEntry`, `vxRegisterLogCallback` |
| `test_target.c` | Target | `vxSetNodeTarget`, `vxSetImmediateModeTarget` |
| `test_usernode.c` | User Kernel | User-defined kernels, validators, parameter queries |

### Data Object Tests
| Test File | Area | Key APIs Tested |
|-----------|------|-----------------|
| `test_vximage.c` | Image | Create, query, copy, map, uniform, ROI, channel, handle |
| `test_vxtensor.c` | Tensor | Create, query, copy, map, view, handle, object array |
| `test_scalar.c` | Scalar | Create, query, copy, virtual, sized |
| `test_array.c` | Array | Create, query, add, copy, map, user structs |
| `test_object_array.c` | Object Array | Create, query, get items, virtual |
| `test_matrix.c` | Matrix | Create, query, copy, pattern, virtual |
| `test_distribution.c` | Distribution | Create, query, virtual |
| `test_convolution.c` | Convolution | Create, query, set attribute, copy, virtual |
| `test_copy.c` | Copy/Access | Copy and map operations across all data objects |
| `test_controlflow.c` | Control Flow | Select, scalar operations, data object lifecycle |

### Vision Kernel Tests
| Test File | Kernel | Node + vxu |
|-----------|--------|------------|
| `test_addsub.c` | Add/Subtract | `vxAddNode`, `vxuAdd` |
| `test_binop8u.c` | Binary Ops (U8) | AbsDiff, And, Or, Xor (via macro) |
| `test_binop16s.c` | Binary Ops (S16) | AbsDiff (via macro) |
| `test_binop1u.c` | Binary Ops (U1) | And, Or, Xor, ConvertDepth (via macro) |
| `test_box3x3.c` | Box Filter | `vxBox3x3Node`, `vxuBox3x3` |
| `test_canny.c` | Canny Edge | `vxCannyEdgeDetectorNode`, `vxuCannyEdgeDetector` |
| `test_channelcombine.c` | Channel Combine | `vxChannelCombineNode`, `vxuChannelCombine` |
| `test_channelextract.c` | Channel Extract | `vxChannelExtractNode`, `vxuChannelExtract` |
| `test_convertcolor.c` | Color Convert | `vxColorConvertNode`, `vxuColorConvert` |
| `test_convertdepth.c` | Convert Depth | `vxConvertDepthNode`, `vxuConvertDepth` |
| `test_convolve.c` | Custom Convolution | `vxConvolveNode`, `vxuConvolve` |
| `test_dilate3x3.c` | Dilate | `vxDilate3x3Node`, `vxuDilate3x3` |
| `test_eqhist.c` | Equalize Histogram | `vxEqualizeHistNode`, `vxuEqualizeHist` |
| `test_erode3x3.c` | Erode | `vxErode3x3Node`, `vxuErode3x3` |
| `test_fast.c` | FAST Corners | `vxFastCornersNode`, `vxuFastCorners` |
| `test_gaussian3x3.c` | Gaussian Filter | `vxGaussian3x3Node`, `vxuGaussian3x3` |
| `test_gaussianpyramid.c` | Gaussian Pyramid | `vxGaussianPyramidNode`, `vxuGaussianPyramid` |
| `test_halfscalegaussian.c` | Half-Scale Gaussian | `vxHalfScaleGaussianNode`, `vxuHalfScaleGaussian` |
| `test_harriscorners.c` | Harris Corners | `vxHarrisCornersNode`, `vxuHarrisCorners` |
| `test_histogram.c` | Histogram | `vxHistogramNode`, `vxuHistogram` |
| `test_hog.c` | HOG | `vxHOGCellsNode`, `vxHOGFeaturesNode`, `vxuHOGCells`, `vxuHOGFeatures` |
| `test_houghlinesp.c` | Hough Lines | `vxHoughLinesPNode`, `vxuHoughLinesP` |
| `test_integral.c` | Integral Image | `vxIntegralImageNode`, `vxuIntegralImage` |
| `test_laplacianpyramid.c` | Laplacian Pyramid | `vxLaplacianPyramidNode`, `vxLaplacianReconstructNode` |
| `test_lbp.c` | LBP | `vxLBPNode`, `vxuLBP` |
| `test_lut.c` | Table Lookup | `vxTableLookupNode`, `vxuTableLookup` |
| `test_magnitude.c` | Magnitude | `vxMagnitudeNode`, `vxuMagnitude` |
| `test_matchtemplate.c` | Match Template | `vxMatchTemplateNode`, `vxuMatchTemplate` |
| `test_max.c` | Max | `vxMaxNode`, `vxuMax` |
| `test_meanstddev.c` | Mean/StdDev | `vxMeanStdDevNode`, `vxuMeanStdDev` |
| `test_median3x3.c` | Median Filter | `vxMedian3x3Node`, `vxuMedian3x3` |
| `test_min.c` | Min | `vxMinNode`, `vxuMin` |
| `test_minmaxloc.c` | MinMaxLoc | `vxMinMaxLocNode`, `vxuMinMaxLoc` |
| `test_multiply.c` | Multiply | `vxMultiplyNode`, `vxuMultiply` |
| `test_nonlinearfilter.c` | Non-Linear Filter | `vxNonLinearFilterNode`, `vxuNonLinearFilter` |
| `test_nonmaxsuppression.c` | Non-Max Suppression | `vxNonMaxSuppressionNode`, `vxuNonMaxSuppression` |
| `test_not.c` | Bitwise Not | `vxNotNode`, `vxuNot` |
| `test_optflowpyrlk.c` | Optical Flow | `vxOpticalFlowPyrLKNode`, `vxuOpticalFlowPyrLK` |
| `test_phase.c` | Phase | `vxPhaseNode`, `vxuPhase` |
| `test_bilateralfilter.c` | Bilateral Filter | `vxBilateralFilterNode`, `vxuBilateralFilter` |
| `test_remap.c` | Remap | `vxRemapNode`, `vxuRemap` |
| `test_scale.c` | Scale Image | `vxScaleImageNode`, `vxuScaleImage` |
| `test_sobel3x3.c` | Sobel | `vxSobel3x3Node`, `vxuSobel3x3` |
| `test_threshold.c` | Threshold | `vxThresholdNode`, `vxuThreshold` |
| `test_warpaffine.c` | Warp Affine | `vxWarpAffineNode`, `vxuWarpAffine` |
| `test_warpperspective.c` | Warp Perspective | `vxWarpPerspectiveNode`, `vxuWarpPerspective` |
| `test_weighted_average.c` | Weighted Average | `vxWeightedAverageNode`, `vxuWeightedAverage` |

### Tensor Operation Tests
| Test File | Area | Key APIs Tested |
|-----------|------|-----------------|
| `test_tensor_op.c` | Tensor Operations | Add, subtract, multiply, LUT, transpose, convert depth, matrix multiply |
| `test_tensor_nn.c` | Neural Network Layers | Convolution, pooling, fully-connected, softmax, activation, ROI pooling, deconvolution |
| `test_tensor_networks.c` | NN Networks | End-to-end network validation (AlexNet, GoogleNet) |

### Extension Tests
| Test File | Extension | Key APIs Tested |
|-----------|-----------|-----------------|
| `test_export_import_extension.c` | Import/Export (IX) | Export, import, release, get by name |
| `test_nnef_import.c` | NNEF Import | `vxImportKernelFromURL` |
| `test_user_data_object.c` | User Data Object | Create, release, query, copy, map, unmap |
| `test_graph_pipeline.c` | Pipelining + Events | Schedule config, enqueue, dequeue, check, events |
| `test_graph_streaming.c` | Streaming | Enable, start, stop graph streaming |

---

## Core API Coverage

All 155 functions declared in `vx_api.h` have CTS test coverage.

### Context Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateContext` | test_export_import_extension.c, test_logging.c, test_target.c |
| `vxReleaseContext` | test_export_import_extension.c, test_target.c |
| `vxGetContext` | test_graph_pipeline.c, test_graph.c, test_smoke.c |
| `vxQueryContext` | test_bilateralfilter.c, test_convolve.c, test_graph.c, test_smoke.c, test_target.c, test_tensor_nn.c |
| `vxSetContextAttribute` | test_bilateralfilter.c, test_box3x3.c, test_canny.c, test_convolve.c, test_target.c |
| `vxDirective` | test_graph.c, test_logging.c |
| `vxGetStatus` | test_export_import_extension.c, test_graph.c, test_smoke.c, test_vximage.c |
| `vxRegisterUserStruct` | test_array.c, test_graph.c, test_smoke.c |
| `vxRegisterUserStructWithName` | test_array.c, test_graph.c, test_smoke.c |
| `vxGetUserStructEnumByName` | test_graph.c |
| `vxGetUserStructNameByEnum` | test_graph.c |
| `vxAllocateUserKernelId` | test_graph_pipeline.c, test_graph.c |
| `vxAllocateUserKernelLibraryId` | test_graph.c |
| `vxSetImmediateModeTarget` | test_target.c |
| `vxHint` | test_smoke.c |
| `vxAddLogEntry` | test_logging.c |
| `vxRegisterLogCallback` | test_logging.c, test_tensor_networks.c |

### Image Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateImage` | 56+ test files |
| `vxCreateImageFromROI` | test_graph_roi.c, test_graph.c, test_vximage.c |
| `vxCreateUniformImage` | test_addsub.c, test_graph_pipeline.c, test_graph.c, test_vximage.c |
| `vxCreateVirtualImage` | test_addsub.c, test_graph_pipeline.c, test_graph.c, test_vximage.c |
| `vxCreateImageFromHandle` | test_vximage.c |
| `vxCreateImageFromChannel` | test_vximage.c |
| `vxCreateImageObjectArrayFromTensor` | test_vxtensor.c |
| `vxSwapImageHandle` | test_vximage.c |
| `vxQueryImage` | test_addsub.c, test_graph_delay.c, test_object_array.c, test_vximage.c |
| `vxSetImageAttribute` | test_convertcolor.c |
| `vxSetImagePixelValues` | test_vximage.c |
| `vxReleaseImage` | 56+ test files |
| `vxFormatImagePatchAddress1d` | test_copy.c, test_vximage.c |
| `vxFormatImagePatchAddress2d` | test_addsub.c, test_graph.c, test_vximage.c |
| `vxGetValidRegionImage` | test_copy.c, test_dilate3x3.c, test_erode3x3.c, test_vximage.c |
| `vxCopyImagePatch` | test_vximage.c |
| `vxMapImagePatch` | test_addsub.c, test_copy.c, test_graph_pipeline.c, test_graph.c, test_vximage.c |
| `vxUnmapImagePatch` | test_addsub.c, test_copy.c, test_graph_pipeline.c, test_graph.c, test_vximage.c |
| `vxSetImageValidRectangle` | test_dilate3x3.c, test_erode3x3.c, test_meanstddev.c, test_threshold.c |

### Kernel Functions

| Function | Test Files |
|----------|-----------|
| `vxLoadKernels` | test_smoke.c |
| `vxUnloadKernels` | test_smoke.c |
| `vxGetKernelByName` | test_graph_pipeline.c, test_graph.c, test_smoke.c, test_usernode.c |
| `vxGetKernelByEnum` | test_graph.c, test_smoke.c, test_usernode.c |
| `vxGetKernelParameterByIndex` | test_graph.c, test_nnef_import.c, test_smoke.c, test_usernode.c |
| `vxQueryKernel` | test_graph.c, test_nnef_import.c, test_smoke.c |
| `vxReleaseKernel` | test_graph_pipeline.c, test_graph.c, test_nnef_import.c, test_smoke.c, test_usernode.c |
| `vxAddUserKernel` | test_graph_pipeline.c, test_graph.c, test_smoke.c, test_usernode.c |
| `vxFinalizeKernel` | test_graph_pipeline.c, test_graph.c, test_smoke.c, test_usernode.c |
| `vxAddParameterToKernel` | test_graph_pipeline.c, test_graph.c, test_smoke.c, test_usernode.c |
| `vxRemoveKernel` | test_graph_pipeline.c, test_graph.c, test_smoke.c, test_usernode.c |
| `vxSetKernelAttribute` | test_graph.c, test_usernode.c |
| `vxRegisterKernelLibrary` | test_smoke.c |

### Graph Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateGraph` | 69 test files |
| `vxReleaseGraph` | 69 test files |
| `vxVerifyGraph` | 62 test files |
| `vxProcessGraph` | 64 test files |
| `vxScheduleGraph` | test_addsub.c, test_graph_pipeline.c, test_graph.c |
| `vxWaitGraph` | test_addsub.c, test_graph_pipeline.c, test_graph.c |
| `vxIsGraphVerified` | test_graph.c, test_threshold.c |
| `vxQueryGraph` | test_graph.c |
| `vxSetGraphAttribute` | test_graph.c |
| `vxAddParameterToGraph` | test_graph_pipeline.c, test_graph.c |
| `vxSetGraphParameterByIndex` | test_graph_pipeline.c, test_graph.c |
| `vxGetGraphParameterByIndex` | test_graph.c |

### Node Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateGenericNode` | test_graph_pipeline.c, test_graph.c, test_nnef_import.c, test_smoke.c, test_usernode.c |
| `vxQueryNode` | test_graph.c, test_smoke.c, test_usernode.c |
| `vxSetNodeAttribute` | test_bilateralfilter.c, test_box3x3.c, test_canny.c, test_graph.c, test_usernode.c |
| `vxReleaseNode` | 60 test files |
| `vxRemoveNode` | test_graph.c |
| `vxAssignNodeCallback` | test_graph_callbacks.c, test_graph_roi.c, test_graph.c |
| `vxRetrieveNodeCallback` | test_graph_callbacks.c |
| `vxSetNodeTarget` | test_target.c |
| `vxReplicateNode` | test_graph_pipeline.c, test_graph.c |

### Parameter Functions

| Function | Test Files |
|----------|-----------|
| `vxGetParameterByIndex` | test_graph_delay.c, test_graph_pipeline.c, test_graph.c, test_smoke.c |
| `vxReleaseParameter` | test_graph_delay.c, test_graph_pipeline.c, test_graph.c, test_smoke.c, test_usernode.c |
| `vxSetParameterByIndex` | test_graph_pipeline.c, test_graph.c, test_nnef_import.c, test_smoke.c, test_usernode.c |
| `vxSetParameterByReference` | test_graph.c, test_smoke.c |
| `vxQueryParameter` | test_graph_delay.c, test_graph.c, test_nnef_import.c, test_smoke.c, test_usernode.c |

### Scalar Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateScalar` | test_controlflow.c, test_copy.c, test_graph_pipeline.c, test_graph.c, test_scalar.c, test_smoke.c |
| `vxCreateScalarWithSize` | test_scalar.c |
| `vxCreateVirtualScalar` | test_scalar.c |
| `vxReleaseScalar` | test_controlflow.c, test_graph_pipeline.c, test_graph.c, test_scalar.c, test_smoke.c |
| `vxCopyScalar` | test_controlflow.c, test_copy.c, test_graph_pipeline.c, test_graph.c, test_scalar.c |
| `vxCopyScalarWithSize` | test_scalar.c |
| `vxQueryScalar` | test_graph_delay.c, test_graph_pipeline.c, test_object_array.c, test_scalar.c |

### Reference Functions

| Function | Test Files |
|----------|-----------|
| `vxQueryReference` | test_copy.c, test_graph_delay.c, test_graph.c, test_smoke.c, test_user_data_object.c |
| `vxReleaseReference` | test_controlflow.c, test_copy.c, test_graph_delay.c, test_graph.c, test_smoke.c |
| `vxRetainReference` | test_graph.c, test_smoke.c |
| `vxSetReferenceName` | test_export_import_extension.c, test_smoke.c |

### Delay Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateDelay` | test_export_import_extension.c, test_graph_delay.c, test_graph_pipeline.c |
| `vxReleaseDelay` | test_export_import_extension.c, test_graph_delay.c, test_graph_pipeline.c |
| `vxQueryDelay` | test_graph_delay.c |
| `vxGetReferenceFromDelay` | test_graph_delay.c, test_graph_pipeline.c |
| `vxAgeDelay` | test_graph_delay.c |
| `vxRegisterAutoAging` | test_graph_delay.c, test_graph_pipeline.c |

### Logging Functions

| Function | Test Files |
|----------|-----------|
| `vxAddLogEntry` | test_logging.c |
| `vxRegisterLogCallback` | test_logging.c, test_tensor_networks.c |

### LUT Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateLUT` | test_controlflow.c, test_copy.c, test_graph.c, test_lut.c, test_smoke.c |
| `vxCreateVirtualLUT` | test_lut.c |
| `vxReleaseLUT` | test_graph.c, test_lut.c, test_smoke.c |
| `vxQueryLUT` | test_graph_delay.c, test_lut.c, test_object_array.c |
| `vxCopyLUT` | test_copy.c, test_lut.c, test_tensor_op.c |
| `vxMapLUT` | test_copy.c, test_graph.c, test_lut.c |
| `vxUnmapLUT` | test_copy.c, test_graph.c, test_lut.c |

### Distribution Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateDistribution` | test_controlflow.c, test_copy.c, test_histogram.c, test_smoke.c |
| `vxCreateVirtualDistribution` | test_distribution.c |
| `vxReleaseDistribution` | test_distribution.c, test_histogram.c, test_smoke.c |
| `vxQueryDistribution` | test_distribution.c, test_histogram.c, test_object_array.c |
| `vxCopyDistribution` | test_histogram.c |
| `vxMapDistribution` | test_histogram.c |
| `vxUnmapDistribution` | test_histogram.c |

### Threshold Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateThresholdForImage` | test_canny.c, test_controlflow.c, test_copy.c, test_graph.c, test_threshold.c |
| `vxCreateVirtualThresholdForImage` | test_threshold.c |
| `vxReleaseThreshold` | test_canny.c, test_graph.c, test_threshold.c |
| `vxQueryThreshold` | test_canny.c, test_graph_delay.c, test_threshold.c |
| `vxSetThresholdAttribute` | test_threshold.c |
| `vxCopyThresholdValue` | test_copy.c, test_threshold.c |
| `vxCopyThresholdRange` | test_canny.c, test_graph.c, test_threshold.c |
| `vxCopyThresholdOutput` | test_threshold.c |

### Matrix Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateMatrix` | test_controlflow.c, test_copy.c, test_matrix.c, test_nonlinearfilter.c, test_warpaffine.c |
| `vxCreateMatrixFromPattern` | test_matrix.c, test_nonlinearfilter.c |
| `vxCreateMatrixFromPatternAndOrigin` | test_matrix.c, test_nonlinearfilter.c |
| `vxCreateVirtualMatrix` | test_matrix.c |
| `vxReleaseMatrix` | test_matrix.c, test_nonlinearfilter.c, test_warpaffine.c |
| `vxQueryMatrix` | test_graph_delay.c, test_matrix.c, test_object_array.c |
| `vxCopyMatrix` | test_copy.c, test_matrix.c, test_warpaffine.c, test_warpperspective.c |

### Convolution Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateConvolution` | test_controlflow.c, test_convolution.c, test_convolve.c, test_copy.c |
| `vxCreateVirtualConvolution` | test_convolution.c |
| `vxReleaseConvolution` | test_convolution.c, test_convolve.c |
| `vxQueryConvolution` | test_convolution.c, test_convolve.c, test_laplacianpyramid.c |
| `vxSetConvolutionAttribute` | test_convolution.c, test_convolve.c |
| `vxCopyConvolutionCoefficients` | test_convolution.c, test_convolve.c, test_copy.c |

### Pyramid Functions

| Function | Test Files |
|----------|-----------|
| `vxCreatePyramid` | test_controlflow.c, test_gaussianpyramid.c, test_graph.c, test_laplacianpyramid.c |
| `vxCreateVirtualPyramid` | test_graph.c, test_optflowpyrlk.c |
| `vxReleasePyramid` | test_gaussianpyramid.c, test_graph.c, test_laplacianpyramid.c |
| `vxQueryPyramid` | test_gaussianpyramid.c, test_graph.c, test_laplacianpyramid.c |
| `vxGetPyramidLevel` | test_copy.c, test_gaussianpyramid.c, test_graph.c, test_laplacianpyramid.c |

### Remap Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateRemap` | test_controlflow.c, test_copy.c, test_remap.c, test_smoke.c |
| `vxCreateVirtualRemap` | test_remap.c |
| `vxReleaseRemap` | test_remap.c, test_smoke.c |
| `vxQueryRemap` | test_graph_delay.c, test_object_array.c, test_remap.c |
| `vxCopyRemapPatch` | test_controlflow.c, test_copy.c, test_remap.c |
| `vxMapRemapPatch` | test_copy.c, test_remap.c |
| `vxUnmapRemapPatch` | test_copy.c, test_remap.c |

### Array Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateArray` | test_array.c, test_controlflow.c, test_fast.c, test_graph.c, test_harriscorners.c |
| `vxCreateVirtualArray` | test_graph.c |
| `vxReleaseArray` | test_array.c, test_fast.c, test_graph.c, test_harriscorners.c |
| `vxQueryArray` | test_array.c, test_graph.c, test_harriscorners.c, test_object_array.c |
| `vxAddArrayItems` | test_array.c, test_copy.c, test_graph.c |
| `vxTruncateArray` | test_graph.c |
| `vxCopyArrayRange` | test_array.c |
| `vxMapArrayRange` | test_array.c, test_copy.c, test_graph.c, test_harriscorners.c |
| `vxUnmapArrayRange` | test_array.c, test_copy.c, test_graph.c, test_harriscorners.c |

### Object Array Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateObjectArray` | test_controlflow.c, test_copy.c, test_graph_pipeline.c, test_graph.c, test_object_array.c |
| `vxCreateVirtualObjectArray` | test_object_array.c |
| `vxReleaseObjectArray` | test_graph_delay.c, test_graph_pipeline.c, test_graph.c, test_object_array.c |
| `vxGetObjectArrayItem` | test_copy.c, test_graph_delay.c, test_graph_pipeline.c, test_graph.c, test_object_array.c |
| `vxQueryObjectArray` | test_graph_delay.c, test_graph.c, test_object_array.c |

### Tensor Functions

| Function | Test Files |
|----------|-----------|
| `vxCreateTensor` | test_bilateralfilter.c, test_copy.c, test_graph.c, test_tensor_nn.c, test_tensor_op.c, test_vxtensor.c |
| `vxCreateTensorFromHandle` | test_vxtensor.c |
| `vxCreateTensorFromView` | test_vxtensor.c |
| `vxCreateVirtualTensor` | test_vxtensor.c |
| `vxSwapTensorHandle` | test_vxtensor.c |
| `vxReleaseTensor` | test_bilateralfilter.c, test_graph.c, test_tensor_nn.c, test_tensor_op.c, test_vxtensor.c |
| `vxQueryTensor` | test_graph_delay.c, test_graph.c, test_vxtensor.c |
| `vxCopyTensorPatch` | test_bilateralfilter.c, test_copy.c, test_tensor_nn.c, test_tensor_op.c, test_vxtensor.c |
| `vxMapTensorPatch` | test_vxtensor.c |
| `vxUnmapTensorPatch` | test_vxtensor.c |

### Meta Format Functions

| Function | Test Files |
|----------|-----------|
| `vxSetMetaFormatAttribute` | test_graph_pipeline.c, test_graph.c, test_user_data_object.c, test_usernode.c |
| `vxSetMetaFormatFromReference` | test_graph.c, test_user_data_object.c, test_usernode.c |
| `vxQueryMetaFormatAttribute` | test_graph.c, test_nnef_import.c |

---

## Node Creation Function Coverage

All 61 node creation functions from `vx_nodes.h` have CTS coverage.

| Function | Test Files |
|----------|-----------|
| `vxColorConvertNode` | test_convertcolor.c |
| `vxChannelExtractNode` | test_channelextract.c |
| `vxChannelCombineNode` | test_channelcombine.c |
| `vxSobel3x3Node` | test_sobel3x3.c |
| `vxMagnitudeNode` | test_magnitude.c |
| `vxPhaseNode` | test_phase.c |
| `vxScaleImageNode` | test_scale.c |
| `vxTableLookupNode` | test_graph.c, test_lut.c |
| `vxHistogramNode` | test_histogram.c |
| `vxEqualizeHistNode` | test_eqhist.c |
| `vxAbsDiffNode` | test_binop8u.c, test_binop16s.c (via `FUNC_ARG` macro) |
| `vxMeanStdDevNode` | test_graph_pipeline.c, test_graph.c, test_meanstddev.c |
| `vxThresholdNode` | test_threshold.c |
| `vxNonMaxSuppressionNode` | test_nonmaxsuppression.c |
| `vxIntegralImageNode` | test_graph.c, test_integral.c |
| `vxErode3x3Node` | test_erode3x3.c |
| `vxDilate3x3Node` | test_dilate3x3.c |
| `vxMedian3x3Node` | test_graph.c, test_median3x3.c |
| `vxBox3x3Node` | test_graph.c, test_box3x3.c |
| `vxGaussian3x3Node` | test_gaussian3x3.c |
| `vxNonLinearFilterNode` | test_nonlinearfilter.c |
| `vxConvolveNode` | test_convolve.c |
| `vxGaussianPyramidNode` | test_graph.c, test_gaussianpyramid.c |
| `vxLaplacianPyramidNode` | test_graph.c, test_laplacianpyramid.c |
| `vxLaplacianReconstructNode` | test_graph.c, test_laplacianpyramid.c |
| `vxWeightedAverageNode` | test_graph.c, test_weighted_average.c |
| `vxMinMaxLocNode` | test_minmaxloc.c, test_matchtemplate.c |
| `vxMinNode` | test_min.c |
| `vxMaxNode` | test_max.c |
| `vxAndNode` | test_graph_pipeline.c, test_binop8u.c (via macro) |
| `vxOrNode` | test_graph_pipeline.c, test_binop8u.c (via macro) |
| `vxXorNode` | test_binop8u.c, test_binop1u.c (via macro) |
| `vxNotNode` | test_graph_pipeline.c, test_graph.c, test_not.c |
| `vxScalarOperationNode` | test_controlflow.c |
| `vxSelectNode` | test_controlflow.c |
| `vxMultiplyNode` | test_graph.c, test_multiply.c |
| `vxAddNode` | test_addsub.c, test_graph_pipeline.c, test_graph.c, test_smoke.c |
| `vxSubtractNode` | test_graph.c |
| `vxConvertDepthNode` | test_convertdepth.c, test_not.c |
| `vxCannyEdgeDetectorNode` | test_canny.c, test_graph.c |
| `vxWarpAffineNode` | test_warpaffine.c |
| `vxWarpPerspectiveNode` | test_warpperspective.c |
| `vxHarrisCornersNode` | test_graph.c, test_harriscorners.c |
| `vxFastCornersNode` | test_graph.c, test_fast.c |
| `vxOpticalFlowPyrLKNode` | test_graph.c, test_optflowpyrlk.c |
| `vxRemapNode` | test_remap.c |
| `vxHalfScaleGaussianNode` | test_graph.c, test_halfscalegaussian.c |
| `vxMatchTemplateNode` | test_matchtemplate.c |
| `vxLBPNode` | test_lbp.c |
| `vxHOGCellsNode` | test_hog.c |
| `vxHOGFeaturesNode` | test_hog.c |
| `vxHoughLinesPNode` | test_houghlinesp.c |
| `vxBilateralFilterNode` | test_bilateralfilter.c |
| `vxCopyNode` | test_copy.c |
| `vxAccumulateImageNode` | test_graph.c |
| `vxAccumulateWeightedImageNode` | test_graph.c |
| `vxAccumulateSquareImageNode` | test_graph.c |
| `vxTensorMultiplyNode` | test_tensor_op.c |
| `vxTensorAddNode` | test_tensor_op.c |
| `vxTensorSubtractNode` | test_tensor_op.c |
| `vxTensorTableLookupNode` | test_tensor_op.c |
| `vxTensorTransposeNode` | test_tensor_op.c |
| `vxTensorConvertDepthNode` | test_tensor_op.c |
| `vxTensorMatrixMultiplyNode` | test_tensor_op.c |

---

## Immediate Mode Function Coverage

All 59 immediate-mode functions from `vxu.h` have CTS coverage.

| Function | Test Files |
|----------|-----------|
| `vxuColorConvert` | test_convertcolor.c |
| `vxuChannelExtract` | test_channelextract.c, test_vximage.c |
| `vxuChannelCombine` | test_channelcombine.c |
| `vxuSobel3x3` | test_sobel3x3.c |
| `vxuMagnitude` | test_magnitude.c |
| `vxuPhase` | test_phase.c |
| `vxuScaleImage` | test_scale.c |
| `vxuTableLookup` | test_graph.c, test_lut.c |
| `vxuHistogram` | test_histogram.c |
| `vxuEqualizeHist` | test_eqhist.c |
| `vxuAbsDiff` | test_binop8u.c, test_binop16s.c (via `FUNC_ARG` macro) |
| `vxuMeanStdDev` | test_meanstddev.c |
| `vxuThreshold` | test_threshold.c |
| `vxuIntegralImage` | test_integral.c |
| `vxuErode3x3` | test_erode3x3.c |
| `vxuDilate3x3` | test_dilate3x3.c |
| `vxuMedian3x3` | test_median3x3.c |
| `vxuBox3x3` | test_box3x3.c |
| `vxuGaussian3x3` | test_gaussian3x3.c |
| `vxuNonLinearFilter` | test_nonlinearfilter.c |
| `vxuConvolve` | test_convolve.c |
| `vxuGaussianPyramid` | test_gaussianpyramid.c, test_laplacianpyramid.c |
| `vxuLaplacianPyramid` | test_laplacianpyramid.c |
| `vxuLaplacianReconstruct` | test_laplacianpyramid.c |
| `vxuWeightedAverage` | test_weighted_average.c |
| `vxuMin` | test_min.c, test_matchtemplate.c |
| `vxuMax` | test_max.c |
| `vxuMinMaxLoc` | test_minmaxloc.c, test_matchtemplate.c |
| `vxuAnd` | test_binop8u.c, test_binop1u.c (via `FUNC_ARG` macro) |
| `vxuOr` | test_binop8u.c, test_binop1u.c (via `FUNC_ARG` macro) |
| `vxuXor` | test_binop8u.c, test_binop1u.c (via `FUNC_ARG` macro) |
| `vxuNot` | test_canny.c, test_not.c |
| `vxuMultiply` | test_graph.c, test_multiply.c |
| `vxuAdd` | test_addsub.c, test_laplacianpyramid.c, test_target.c |
| `vxuSubtract` | test_graph.c, test_laplacianpyramid.c |
| `vxuConvertDepth` | test_convertdepth.c |
| `vxuCannyEdgeDetector` | test_canny.c |
| `vxuWarpAffine` | test_warpaffine.c |
| `vxuWarpPerspective` | test_warpperspective.c |
| `vxuHarrisCorners` | test_harriscorners.c |
| `vxuFastCorners` | test_fast.c |
| `vxuOpticalFlowPyrLK` | test_optflowpyrlk.c |
| `vxuRemap` | test_remap.c |
| `vxuHalfScaleGaussian` | test_halfscalegaussian.c |
| `vxuMatchTemplate` | test_matchtemplate.c |
| `vxuLBP` | test_lbp.c |
| `vxuBilateralFilter` | test_bilateralfilter.c |
| `vxuHOGCells` | test_hog.c |
| `vxuHOGFeatures` | test_hog.c |
| `vxuHoughLinesP` | test_houghlinesp.c |
| `vxuNonMaxSuppression` | test_nonmaxsuppression.c |
| `vxuCopy` | test_copy.c |
| `vxuTensorMultiply` | test_tensor_op.c |
| `vxuTensorAdd` | test_tensor_op.c |
| `vxuTensorSubtract` | test_tensor_op.c |
| `vxuTensorTableLookup` | test_tensor_op.c |
| `vxuTensorTranspose` | test_tensor_op.c |
| `vxuTensorConvertDepth` | test_tensor_op.c |
| `vxuTensorMatrixMultiply` | test_tensor_op.c |

---

## Extension API Coverage

### Import/Export (IX)

**CTS Test File:** `test_export_import_extension.c` | **Coverage: 5/5 (100%)**

| Function | Covered |
|----------|:-------:|
| `vxExportObjectsToMemory` | Yes |
| `vxReleaseExportedMemory` | Yes |
| `vxImportObjectsFromMemory` | Yes |
| `vxReleaseImport` | Yes |
| `vxGetImportReferenceByName` | Yes |

### Neural Networks (NN)

**CTS Test File:** `test_tensor_nn.c`, `test_tensor_networks.c` | **Coverage: 8/8 (100%)**

| Function | Test File |
|----------|-----------|
| `vxConvolutionLayer` | test_tensor_nn.c |
| `vxFullyConnectedLayer` | test_tensor_nn.c |
| `vxPoolingLayer` | test_tensor_nn.c |
| `vxSoftmaxLayer` | test_tensor_nn.c |
| `vxLocalResponseNormalizationLayer` | Networks/src/graph_alexnet.c |
| `vxActivationLayer` | test_tensor_nn.c |
| `vxROIPoolingLayer` | test_tensor_nn.c |
| `vxDeconvolutionLayer` | test_tensor_nn.c |

### User Data Object

**CTS Test File:** `test_user_data_object.c` | **Coverage: 6/8 (75%)**

| Function | Covered | Notes |
|----------|:-------:|-------|
| `vxCreateUserDataObject` | Yes | |
| `vxCreateVirtualUserDataObject` | No | Not tested |
| `vxReleaseUserDataObject` | Yes | |
| `vxQueryUserDataObject` | Yes | |
| `vxSetUserDataObjectAttribute` | No | Not tested |
| `vxCopyUserDataObject` | Yes | |
| `vxMapUserDataObject` | Yes | |
| `vxUnmapUserDataObject` | Yes | |

### NNEF Import Kernel

**CTS Test File:** `test_nnef_import.c` | **Coverage: 1/1 (100%)**

| Function | Covered |
|----------|:-------:|
| `vxImportKernelFromURL` | Yes |

### Pipelining

**CTS Test File:** `test_graph_pipeline.c` | **Coverage: 4/7 (57%)**

| Function | Covered | Notes |
|----------|:-------:|-------|
| `vxSetGraphScheduleConfig` | Yes | |
| `vxGraphParameterEnqueueReadyRef` | Yes | |
| `vxGraphParameterDequeueDoneRef` | Yes | |
| `vxGraphParameterCheckDoneRef` | Yes | |
| `vxGetGraphParameterRefsList` | No | Not tested |
| `vxAddReferencesToGraphParameterList` | No | Not tested |
| `vxGetKernelParameterConfig` | No | Not tested |

### Streaming

**CTS Test File:** `test_graph_streaming.c` | **Coverage: 3/3 (100%)**

| Function | Covered |
|----------|:-------:|
| `vxEnableGraphStreaming` | Yes |
| `vxStartGraphStreaming` | Yes |
| `vxStopGraphStreaming` | Yes |

### Event Queue

**CTS Test File:** `test_graph_pipeline.c` | **Coverage: 5/10 (50%)**

| Function | Covered | Notes |
|----------|:-------:|-------|
| `vxEnableEvents` | Yes | |
| `vxDisableEvents` | Yes | |
| `vxSendUserEvent` | Yes | |
| `vxWaitEvent` | Yes | |
| `vxRegisterEvent` | Yes | |
| `vxRegisterGraphEvent` | No | Not tested |
| `vxWaitGraphEvent` | No | Not tested |
| `vxEnableGraphEvents` | No | Not tested |
| `vxDisableGraphEvents` | No | Not tested |
| `vxSendUserGraphEvent` | No | Not tested |

### Compatibility/Deprecated

**Coverage: 4/26 (15%)**

| Function | Covered | Test File |
|----------|:-------:|-----------|
| `vxAccessImagePatch` | Yes | test_vximage.c |
| `vxCommitImagePatch` | Yes | test_vximage.c |
| `vxAccessArrayRange` | Yes | test_optflowpyrlk.c |
| `vxCommitArrayRange` | Yes | test_optflowpyrlk.c |
| All other deprecated functions | No | Superseded by modern equivalents |

> Deprecated functions are superseded by their modern equivalents (e.g., `vxCopyImagePatch` replaces `vxAccessImagePatch`/`vxCommitImagePatch`). The modern equivalents all have full CTS coverage.

---

## Extensions Without CTS Coverage

The following optional KHR extensions have API headers but no CTS tests. These are not required for conformance.

| Extension | Header | Functions |
|-----------|--------|-----------|
| Node Send Command | `vx_khr_node_send_command.h` | `vxAddCommandToKernel`, `vxNodeSendCommand` |
| Bidirectional Parameters | `vx_khr_bidirectional_parameters.h` | `vxAccumulateWeightedImageNodeX`, `vxAccumulateSquareImageNodeX` |
| Buffer Aliasing | `vx_khr_buffer_aliasing.h` | `vxAliasParameterIndexHint`, `vxIsParameterAliased` |
| Swap/Move | `vx_khr_swap_move.h` | `vxSwapNode`, `vxMoveNode`, `vxuSwap`, `vxuMove` |
| OpenCL Interop | `vx_khr_opencl_interop.h` | `vxCreateContextFromCL` |
| Raw Image | `vx_khr_raw_image.h` | `vxCreateRawImage`, `vxCreateVirtualRawImage`, `vxCopyImagePatchWithFlags` |
| Safe Casts | `vx_khr_safe_casts.h` | Macro-generated cast/get functions |
| ICD Loader | `vx_khr_icd.h` | `vxIcdGetPlatforms`, `vxQueryPlatform`, `vxCreateContextFromPlatform` |
| Sub-Image Object Array | `vx_khr_sub_image_object_array.h` | `vxCreateObjectArrayFromROI`, `vxCreateObjectArrayFromChannel` |
| Supplementary Data | `vx_khr_supplementary_data.h` | `vxSetSupplementaryUserDataObject`, `vxExtendSupplementaryUserDataObject`, `vxGetSupplementaryUserDataObject` |
| Target Kernel | `vx_khr_target_kernel.h` | 8 functions |
| Tensor From Image | `vx_khr_tensor_from_image.h` | 5 functions |
| Tiling | `vx_khr_tiling.h` | `vxAddTilingKernel` |
| XML | `vx_khr_xml.h` | `vxExportToXML`, `vxImportFromXML`, `vxGetImportReferenceByIndex`, `vxQueryImport` |
| Classifier | `vx_khr_class.h` | `vxImportClassifierModel`, `vxReleaseClassifierModel`, `vxScanClassifierNode` |

---

## CI Pipeline Modes

The CTS CI pipeline (`.gitlab-ci.yml`) runs 7 conformance modes against the OpenVX sample implementation:

| Job | Mode | CMake Options | Status |
|-----|------|--------------|--------|
| Build CTS | -- | (default) | Build only |
| Conformance Vision | 1 | `OPENVX_CONFORMANCE_VISION=ON` | Required pass |
| Conformance Vision Enhanced Vision | 2 | `OPENVX_CONFORMANCE_VISION=ON OPENVX_USE_ENHANCED_VISION=ON` | Required pass |
| Conformance Neural Net | 3 | `OPENVX_CONFORMANCE_NEURAL_NETWORKS=ON` | Required pass |
| Conformance NNEF Import | 4 | `OPENVX_CONFORMANCE_NNEF_IMPORT=ON` | Required pass |
| Conformance Combined | 5 | Vision + Enhanced Vision + NN + IX + U1 | Required pass |
| Conformance User Data Object | 6 | `OPENVX_USE_USER_DATA_OBJECT=ON` | Required pass |
| Conformance Pipelining Streaming | 7 | Vision + Enhanced Vision + Pipelining + Streaming | `allow_failure` |
