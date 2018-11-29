cmake_minimum_required (VERSION 2.6)
project (networks)

set (OPENVX_LIB_NAME openvx)
set (OPENVXU_LIB_NAME vxu)
set (CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-std=c99") 


set (OPENVX_INCLUDE_FOLDER $ENV{OPENVX_FOLDER}/include)
set (OPENVX_LIB_FOLDER $ENV{OPENVX_FOLDER}/bin $ENV{OPENVX_FOLDER}/lib)


include_directories(${OPENVX_INCLUDE_FOLDER} .)
link_directories(${OPENVX_LIB_FOLDER})
 
#add_executable(networks main.c graph_alexnet.c graph_googlenet.c common.c load_weights.c graph_process.c utilities.c precisionConverter.c)

if( BUILD_ARCH STREQUAL "x86" )
    set_target_properties(networks PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32")
endif()

#target_link_libraries (networks -lm ${OPENVX_LIB_NAME} ${OPENVXU_LIB_NAME})

file(GLOB PROJECT_BINARY_FILES "../../Binaries/*.bin")
file(COPY ${PROJECT_BINARY_FILES} DESTINATION ${CMAKE_BINARY_DIR})
