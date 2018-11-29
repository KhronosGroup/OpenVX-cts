# 

# Copyright (c) 2012-2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

project(test_data_generator)

if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
endif()
#TODO: enable C99 features for non-gnu platforms (MSVC - ???)

if(MSVC_IDE OR CMAKE_GENERATOR MATCHES Xcode)
  option(ENABLE_SOLUTION_FOLDERS "Solution folder in Visual Studio or in other IDEs" ON)
endif()

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "Build type is not specified, use: Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE INTERNAL "" FORCE)
endif()

# DEBUG/NDEBUG preprocessor definition
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-DDEBUG)
else()
  add_definitions(-DNDEBUG)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


option(BUILD_HC_TEST_DATA_GENERATOR "Build generator for test data for Harris Corners tests" OFF)

if(BUILD_HC_TEST_DATA_GENERATOR)
  # Harris Corners test data generator needs OpenCV 2.4.7 with patch
  add_subdirectory(gen_harris_corners)
else()
  # these test data generators needs clean OpenCV 2.4.7
  add_subdirectory(gen_optflow_pyrlk)
  add_subdirectory(gen_canny)
endif()

