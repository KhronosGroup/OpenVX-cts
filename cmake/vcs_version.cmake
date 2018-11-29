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


message(STATUS "Get version information")
function(fill_version)
  find_package(Git QUIET)

  if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
      WORKING_DIRECTORY "."
      OUTPUT_VARIABLE VCSVERSION
      RESULT_VARIABLE GIT_RESULT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GIT_RESULT EQUAL 0)
      unset(VCSVERSION)
    endif()
  else()
    # We don't have git:
    unset(VCSVERSION)
  endif()

  if(DEFINED VCSVERSION)
    set(VERSION "${VCSVERSION}")
  else()
    set(VERSION "unknown")
  endif()

  message(STATUS "Version: ${VERSION}")

  set(RESULT "#define VCS_VERSION_STR \"${VERSION}\"")
  set(OUTPUT ${OUTPUT_DIR}/vcs_version.inc)

  if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" lines)
  endif()
  if("${lines}" STREQUAL "${RESULT}")
    #message(STATUS "${OUTPUT} contains same content")
  else()
    file(WRITE "${OUTPUT}" "${RESULT}")
  endif()
endfunction()
fill_version()
