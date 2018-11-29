#

# Copyright (c) 2011-2017 The Khronos Group Inc.
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


include_directories( BEFORE
                     ${CMAKE_CURRENT_SOURCE_DIR} 
                     ${CMAKE_SOURCE_DIR}/include
                     ${OPENVX_INCLUDES} )

# set target name
set( TARGET_NAME_TESTMODULE_LIB test-testmodule-lib )
# set target name
set( TARGET_NAME_TESTMODULE test-testmodule )

if(CYGWIN)
    set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll.a")
    set( OPENVX_BUILDING_EXPORT_DEF -DVX_API_ENTRY=__attribute__\(\(dllexport\)\) )
endif()

if ((WIN32) OR (CYGWIN))
   add_definitions( ${OPENVX_BUILDING_EXPORT_DEF} )
endif ((WIN32) OR (CYGWIN))

# add a target named ${TARGET_NAME}
add_library (${TARGET_NAME_TESTMODULE_LIB}  vx_testmodule_lib.c)

target_link_libraries( ${TARGET_NAME_TESTMODULE_LIB} openvx-interface )

if ((WIN32) OR (CYGWIN))
   set( DEF_FILE testmodule.def )
endif ((WIN32) OR (CYGWIN))
# add a target named ${TARGET_NAME}
add_library (${TARGET_NAME_TESTMODULE} SHARED vx_testmodule_module.c ${DEF_FILE})

if (CYGWIN)
   set_target_properties( ${TARGET_NAME_TESTMODULE} PROPERTIES LINK_FLAGS ${CMAKE_CURRENT_SOURCE_DIR}/${DEF_FILE} )
endif (CYGWIN)

target_link_libraries( ${TARGET_NAME_TESTMODULE} test-testmodule-lib openvx-interface )

install ( TARGETS ${TARGET_NAME_TESTMODULE_LIB} ${TARGET_NAME_TESTMODULE} ${TARGET_NAME_TESTMODULE}
          RUNTIME DESTINATION bin
          ARCHIVE DESTINATION bin
          LIBRARY DESTINATION bin )
		  
set_target_properties( ${TARGET_NAME_TESTMODULE_LIB} PROPERTIES FOLDER ${TESTMODULE_FOLDER} )
set_target_properties( ${TARGET_NAME_TESTMODULE} PROPERTIES FOLDER ${TESTMODULE_FOLDER} )
set_target_properties( ${TARGET_NAME_TESTMODULE} PROPERTIES FOLDER ${TESTMODULE_FOLDER} )
