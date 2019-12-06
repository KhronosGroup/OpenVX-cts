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

/* Base Feature Set Tests (for all conformance feature sets) */
TESTCASE(GraphBase)
TESTCASE(Logging)
TESTCASE(SmokeTestBase)
TESTCASE(TargetBase)

#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION
TESTCASE(Graph)
TESTCASE(GraphCallback)
TESTCASE(GraphDelay)
TESTCASE(GraphROI)

TESTCASE(Array)
TESTCASE(ObjectArray)
TESTCASE(Image)
TESTCASE(vxCreateImageFromChannel)
TESTCASE(vxCopyImagePatch)
TESTCASE(vxMapImagePatch)
TESTCASE(Distribution)

TESTCASE(vxCopyRemapPatch)
TESTCASE(vxMapRemapPatch)

TESTCASE(UserNode)
TESTCASE(SmokeTest)
TESTCASE(Target)
TESTCASE(Convolution)
TESTCASE(Matrix)
TESTCASE(vxuConvertDepth)
TESTCASE(vxConvertDepth)
TESTCASE(ChannelCombine)
TESTCASE(ChannelExtract)
TESTCASE(ColorConvert)
TESTCASE(vxuAddSub)
TESTCASE(vxAddSub)
TESTCASE(vxuNot)
TESTCASE(vxNot)

#ifdef OPENVX_USE_U1
TESTCASE(vxuBinOp1u)
TESTCASE(vxBinOp1u)
#endif

TESTCASE(vxuBinOp8u)
TESTCASE(vxBinOp8u)

TESTCASE(vxuBinOp16s)
TESTCASE(vxBinOp16s)

TESTCASE(vxuMultiply)
TESTCASE(vxMultiply)
TESTCASE(Histogram)
TESTCASE(EqualizeHistogram)
TESTCASE(MeanStdDev)
TESTCASE(MinMaxLoc)

TESTCASE(WeightedAverage)
TESTCASE(Threshold)
TESTCASE(Box3x3)
TESTCASE(Convolve)
TESTCASE(Dilate3x3)
TESTCASE(Erode3x3)

TESTCASE(Gaussian3x3)
TESTCASE(Median3x3)
TESTCASE(Sobel3x3)
TESTCASE(NonLinearFilter)
TESTCASE(Integral)

TESTCASE(Magnitude)
TESTCASE(Phase)
TESTCASE(FastCorners)
TESTCASE(HarrisCorners)
TESTCASE(Scale)
TESTCASE(WarpAffine)
TESTCASE(WarpPerspective)
TESTCASE(Remap)
TESTCASE(Scalar)

TESTCASE(GaussianPyramid)
TESTCASE(HalfScaleGaussian)
TESTCASE(LaplacianPyramid)
TESTCASE(LaplacianReconstruct)
TESTCASE(vxuCanny)
TESTCASE(vxCanny)
TESTCASE(OptFlowPyrLK)
TESTCASE(LUT)
#endif

#ifdef OPENVX_USE_ENHANCED_VISION
TESTCASE(GraphEnhanced)
TESTCASE(GraphDelayTensor)
TESTCASE(Min)
TESTCASE(Max)
TESTCASE(Nonmaxsuppression)
TESTCASE(TensorOp)
TESTCASE(LBP)
TESTCASE(BilateralFilter)
TESTCASE(MatchTemplate)
TESTCASE(Houghlinesp)
TESTCASE(Copy)
TESTCASE(HogCells)
TESTCASE(HogFeatures)
TESTCASE(ControlFlow)
TESTCASE(TensorEnhanced)
#endif

#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_NEURAL_NETWORKS || OPENVX_CONFORMANCE_NNEF_IMPORT
TESTCASE(Tensor)
#endif

#if defined OPENVX_CONFORMANCE_NEURAL_NETWORKS || OPENVX_CONFORMANCE_NNEF_IMPORT
TESTCASE(VxKernelOfNNAndNNEF)
TESTCASE(VxParameterOfNNAndNNEF)
TESTCASE(UserKernelsOfNNAndNNEF)
TESTCASE(MetaFormatOfNNAndNNEF)
#endif

#ifdef OPENVX_USE_IX
TESTCASE(ExtensionObject)
#endif

#ifdef OPENVX_CONFORMANCE_NEURAL_NETWORKS
#ifdef OPENVX_USE_NN
TESTCASE(TensorNN)
#endif
#ifdef OPENVX_USE_NN_16
TESTCASE(TensorNetworks)
#endif
#endif

#ifdef OPENVX_CONFORMANCE_NNEF_IMPORT
TESTCASE(TensorNNEFImport)
#endif

#ifdef OPENVX_USE_PIPELINING
TESTCASE(GraphPipeline)
#endif

#ifdef OPENVX_USE_STREAMING
TESTCASE(GraphStreaming)
#endif

#ifdef OPENVX_USE_USER_DATA_OBJECT
TESTCASE(UserDataObject)
#endif

