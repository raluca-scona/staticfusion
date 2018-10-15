PROJECT(static-fusion)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

# custom cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

MESSAGE(STATUS "CMAKE_MODULE_PATH:   " ${CMAKE_MODULE_PATH})

find_package(Pangolin 0.1 REQUIRED)

#Find dependencies (Eigen is included in MRPT)
find_package(MRPT REQUIRED base obs)
find_package(OpenCV REQUIRED)
find_package(OpenNI2 REQUIRED)

set(sfusion_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenNI2_INCLUDE_DIRS})

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB sf_utils_srcs Utils/*)
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
	${OpenNI2_LIBRARY}
	${OpenCV_LIBS}
   	${Pangolin_LIBRARIES}
)
		
#To run online with an RGB-D camera	
ADD_EXECUTABLE(StaticFusion-Camera 	StaticFusion-camera.cpp)	
TARGET_LINK_LIBRARIES(StaticFusion-Camera 	sf_lib)
		
#To test it with the TUM dataset		
ADD_EXECUTABLE(StaticFusion-Datasets 	StaticFusion-datasets.cpp)
TARGET_LINK_LIBRARIES(StaticFusion-Datasets 	sf_lib)	

#To test it with pre-recorded RGB-D sequences		
ADD_EXECUTABLE(StaticFusion-ImageSeqAssoc 	StaticFusion-imagesequenceassoc.cpp)
TARGET_LINK_LIBRARIES(StaticFusion-ImageSeqAssoc 	sf_lib)
		
set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -std=c++11 -DSHADER_DIR=${sfusion_SHADER_DIR}")		

# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_BUILD_TYPE "Release")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -mavx -std=c++11")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

