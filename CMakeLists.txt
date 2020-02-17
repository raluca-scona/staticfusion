PROJECT(StaticFusion)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

set(CMAKE_CXX_STANDARD 11)

# custom cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

MESSAGE(STATUS "CMAKE_MODULE_PATH:   " ${CMAKE_MODULE_PATH})

find_package(Pangolin 0.1 REQUIRED)

#Find dependencies (Eigen is included in MRPT)
find_package(MRPT REQUIRED base obs)
find_package(OpenCV REQUIRED)
find_package(OpenNI2)

if(NOT OpenNI2_FOUND)
    message(STATUS "OpenNI2 not found. Will not use online live mode.")
endif()


set(sfusion_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")

include_directories(${PROJECT_SOURCE_DIR})
if(OpenNI2_FOUND)
    include_directories(${OpenNI2_INCLUDE_DIRS})
endif()

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(sf_utils_srcs
    "Utils/Datasets.cpp"
    "Utils/GPUTexture.cpp"
    "Utils/Parse.cpp"
)
if(OpenNI2_FOUND)
    list(APPEND sf_utils_srcs "Utils/RGBD_Camera.cpp")
endif()

file(GLOB sf_shader_srcs Shaders/*.cpp)

file(GLOB sf_srcs 
	Reconstruction.cpp 
	GlobalModel.cpp 
	IndexMap.cpp
	IndexMap.h
	KMeans.cpp
	FrontEnd.cpp
	StaticFusion.h
	SegmentationBackground.cpp
	)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

# we have to fetch the full path to libraries such that they are properly exported
get_target_property(pangolin_lib_path pangolin IMPORTED_LOCATION_RELEASE)
get_target_property(pangolin_intf_libs pangolin INTERFACE_LINK_LIBRARIES)

ADD_LIBRARY(sf_lib
   ${sf_srcs}
   ${sf_utils_srcs}
   ${sf_shader_srcs}
   ${sf_containers}	
)
	
TARGET_LINK_LIBRARIES(sf_lib
	${MRPT_LIBS}
	${OpenCV_LIBS}
	${pangolin_lib_path}
	${pangolin_intf_libs}
)

if(OpenNI2_FOUND)
    TARGET_LINK_LIBRARIES(sf_lib ${OpenNI2_LIBRARY})
endif()
		
#To run online with an RGB-D camera
if(OpenNI2_FOUND)
    ADD_EXECUTABLE(StaticFusion-Camera 	StaticFusion-camera.cpp)
    TARGET_LINK_LIBRARIES(StaticFusion-Camera 	sf_lib)
endif()
		
#To test it with the TUM dataset		
ADD_EXECUTABLE(StaticFusion-Datasets 	StaticFusion-datasets.cpp)
TARGET_LINK_LIBRARIES(StaticFusion-Datasets 	sf_lib)	

#To test it with pre-recorded RGB-D sequences		
ADD_EXECUTABLE(StaticFusion-ImageSeqAssoc 	StaticFusion-imagesequenceassoc.cpp)
TARGET_LINK_LIBRARIES(StaticFusion-ImageSeqAssoc 	sf_lib)

add_definitions(-DSHADER_DIR="${sfusion_SHADER_DIR}")
		
set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3")

# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

include(GNUInstallDirs)

# TODO: according to CMake documentation
# https://cmake.org/cmake/help/latest/prop_tgt/INTERFACE_INCLUDE_DIRECTORIES.html,
# we should not include "MRPT_INCLUDE_DIRS" and "Pangolin_INCLUDE_DIRS" here
target_include_directories(sf_lib PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/${PROJECT_NAME};${MRPT_INCLUDE_DIRS};${Pangolin_INCLUDE_DIRS}")

set_property(TARGET sf_lib PROPERTY INTERFACE_LINK_DIRECTORIES ${Pangolin_LIBRARY_DIR})

set_target_properties(sf_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_INSTALL_LIBDIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_LIBDIR}
)

# install library
install(TARGETS sf_lib EXPORT staticfusion
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS StaticFusion-Datasets StaticFusion-ImageSeqAssoc
    RUNTIME DESTINATION bin
)

if(OpenNI2_FOUND)
    install(TARGETS StaticFusion-Camera RUNTIME DESTINATION bin)
endif()

# install public headers
set(PUBLIC_HEADERS
    "GlobalModel.h"
    "IndexMap.h"
    "Reconstruction.h"
    "StaticFusion.h"
    "Shaders/ComputePack.h"
    "Shaders/FeedbackBuffer.h"
    "Shaders/FillIn.h"
    "Shaders/Resize.h"
    "Shaders/Shaders.h"
    "Shaders/Uniform.h"
    "Shaders/Vertex.h"
    "Utils/GPUTexture.h"
    "Utils/GUI.h"
    "Utils/Img.h"
    "Utils/Intrinsics.h"
    "Utils/Parse.h"
    "Utils/Resolution.h"
)

foreach(HEADER ${PUBLIC_HEADERS})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${DIR})
endforeach()

# export library
install(EXPORT staticfusion
    DESTINATION share/${PROJECT_NAME}/cmake
    FILE ${PROJECT_NAME}Config.cmake
)
