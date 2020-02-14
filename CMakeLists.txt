PROJECT(static-fusion)

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

ADD_LIBRARY(sf_lib
   ${sf_srcs}
   ${sf_utils_srcs}
   ${sf_shader_srcs}
   ${sf_containers}	
)
	
TARGET_LINK_LIBRARIES(sf_lib
	${MRPT_LIBS}
	${OpenCV_LIBS}
   	${Pangolin_LIBRARIES}
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

