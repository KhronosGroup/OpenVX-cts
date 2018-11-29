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


project(vx_test_conformance C)
set(target vx_test_conformance)

file(GLOB SOURCES "./*.c")
file(GLOB HEADERS "./*.h")
file(GLOB NETWORK_SOURCES "./Networks/src/*")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(Networks)

add_executable(${target} ${SOURCES} ${HEADERS} ${NETWORK_SOURCES})
target_link_libraries(${target} test_engine openvx-interface)

set(TESTMODULE_FOLDER TestModule)

add_subdirectory(test_module)
