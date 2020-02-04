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


project(test_engine C)

set(target test_engine)
file(GLOB SOURCES "./*.c")
file(GLOB HEADERS "./*.h")

# Following is added for accessing prototype of usleep() in unistd.h
add_definitions( -D_XOPEN_SOURCE=700 -D_GNU_SOURCE=1 )

add_library(${target} STATIC ${SOURCES} ${HEADERS})
target_include_directories(${target} PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(${target} PUBLIC openvx-interface)
add_dependencies(${target} generate_version_file)

if (MSVC)
  target_compile_definitions(${target} PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()
