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


cmake_minimum_required(VERSION 2.8.12) #2.8.11.2 fails

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE) #instead of manually added -fPIC

project(openvx_conformance C)

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

macro(status_separator)
  message(STATUS "============================================")
endmacro()

status_separator()
message(STATUS "=       Conformance Tests for OpenVX       =")
status_separator()

# Default options for the CMake GUI
option( OPENVX_USE_IX ON )
option( OPENVX_USE_NN ON )
option( OPENVX_USE_NN_16 ON )

message( "-- Selected OpenVX Extensions:")
if (OPENVX_USE_IX)
    add_definitions( -DOPENVX_USE_IX )
    message( "--    Offical Import-Export extension")
endif (OPENVX_USE_IX)
if (OPENVX_USE_NN)
    add_definitions( -DOPENVX_USE_NN )
    message( "--    Offical NeuralNet extension")
endif (OPENVX_USE_NN)
if (OPENVX_USE_NN_16)
    add_definitions( -DOPENVX_USE_NN_16 )
    message( "--    Offical NeuralNet 16 bits extension")
endif (OPENVX_USE_NN_16)
message( "--" )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_PDB_OUTPUT_DIRECTORY     ${CMAKE_BINARY_DIR}/lib)

if (CMAKE_VERSION VERSION_LESS "3.0.0")
  if (NOT EXISTS "${CMAKE_BINARY_DIR}/dummy.c")
    file(WRITE "${CMAKE_BINARY_DIR}/dummy.c" "")
  endif()
  add_library(openvx-interface STATIC "${CMAKE_BINARY_DIR}/dummy.c")
else()
  add_library(openvx-interface INTERFACE)
endif()

if (DEFINED OPENVX_LIBRARIES)
  message(STATUS "OpenVX: Prebuild Version")
  message(STATUS "Compiler: ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1} (ver ${CMAKE_C_COMPILER_VERSION})")
  if (CMAKE_BUILD_TYPE)
    message(STATUS "Configuration: ${CMAKE_BUILD_TYPE}")
  endif()
  message(STATUS)

  message(STATUS "OPENVX_LIBRARIES: \t${OPENVX_LIBRARIES}")
  set_target_properties(openvx-interface PROPERTIES INTERFACE_LINK_LIBRARIES "${OPENVX_LIBRARIES}")

  if (DEFINED OPENVX_INCLUDES)
    message(STATUS "OPENVX_INCLUDES: \t${OPENVX_INCLUDES}")
    set_target_properties(openvx-interface PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OPENVX_INCLUDES}")
  else()
    message(STATUS "OPENVX_INCLUDES: \t<empty>")
  endif()

  if (DEFINED OPENVX_DEFINITIONS)
    message(STATUS "OPENVX_DEFINITIONS: \t${OPENVX_DEFINITIONS}")
    set_target_properties(openvx-interface PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${OPENVX_DEFINITIONS}")
  else()
    message(STATUS "OPENVX_DEFINITIONS: \t<empty>")
  endif()

  if (DEFINED OPENVX_CFLAGS)
    message(STATUS "OPENVX_CFLAGS: \t${OPENVX_CFLAGS}")
    set_target_properties(openvx-interface PROPERTIES INTERFACE_COMPILE_OPTIONS "${OPENVX_CFLAGS}")
  else()
    message(STATUS "OPENVX_CFLAGS: \t<empty>")
  endif()
else()
  message(STATUS "OpenVX: Khronos Sample Implementation")
  message(STATUS "Compiler: ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1} (ver ${CMAKE_C_COMPILER_VERSION})")
  if (CMAKE_BUILD_TYPE)
    message(STATUS "Configuration: ${CMAKE_BUILD_TYPE}")
  endif()
  include(cmake/openvx.cmake)
  target_link_libraries(openvx-interface INTERFACE openvx vxu)
endif()
status_separator()

if(CMAKE_COMPILER_IS_GNUCC)
  string(REPLACE "-std=gnu99" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()
#TODO: enable C99 features for non-gnu platforms (MSVC - ???)

add_custom_target(generate_version_file
    ${CMAKE_COMMAND} -DOUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcs_version.cmake"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_definitions(-DHAVE_VCS_VERSION_INC)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/openvx_cts_version.inc")
  add_definitions(-DHAVE_VERSION_INC)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(NOT CT_DISABLE_TIME_SUPPORT)
  include(CheckIncludeFile)
  include(CheckCSourceCompiles)

  check_include_file(sys/time.h HAVE_SYS_TIME_H)
  if(HAVE_SYS_TIME_H)
    add_definitions(-DHAVE_SYS_TIME_H)
    set(CMAKE_REQUIRED_DEFINITIONS "-DHAVE_SYS_TIME_H")
  endif()

  # _XOPEN_SOURCE=700 -> use POSIX 2008 (SUS v4)
  add_definitions(-D_XOPEN_SOURCE=700)
  set(CMAKE_REQUIRED_DEFINITIONS "${CMAKE_REQUIRED_DEFINITIONS} -D_XOPEN_SOURCE=700")

  file(READ "${CMAKE_SOURCE_DIR}/cmake/checks/time_duration.c" TIME_DURATION_SOURCE)
  check_c_source_compiles("${TIME_DURATION_SOURCE}" CT_TEST_TIME)
  if(CT_TEST_TIME)
    add_definitions(-DCT_TEST_TIME)
  endif()
endif()

add_subdirectory(test_engine)
add_subdirectory(test_conformance)

option(BUILD_TEST_DATA_GENERATORS "Build generators for test data" OFF)
if(BUILD_TEST_DATA_GENERATORS)
  message(STATUS "Build with test data generators")
  add_subdirectory(test_data_generator)
endif()
