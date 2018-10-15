# StaticFusion #

This repository contains StaticFusion, a method for dense RGB-D SLAM in dynamic environments based on a strategy of simultaneous odometry and dynamic object segmentation.

It has been tested on Ubuntu 16.04.

# Publication

If you use StaticFusion in your own work, please cite our paper:

* **StaticFusion: Background Reconstruction for Dense RGB-D SLAM in Dynamic Environments**. Raluca Scona, Mariano Jaimez, Yvan R. Petillot, Maurice Fallon, Daniel Cremers. IEEE International Conference on Robotics and Automation (ICRA) 2018

# Configuration and Dependencies
StaticFusion is organised as a cmake project. The dependencies are:
- [OpenCV](http://opencv.org/)
- [MRPT](http://www.mrpt.org/)
- [OpenNI2](https://structure.io/openni)
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)

# Install Steps

## 1. Ubuntu 16.04

Given a clean installation of Ubuntu, the necessary steps for installing StaticFusion are:

0. Our approach requires a GPU to run, so make sure to install the necessary driver for your graphics card. We only tested the code on Nvidia GPUs so far.

1. Install the necessary libraries (and any required dependencies):

```sudo apt-get install cmake libmrpt-dev freeglut3-dev libglew-dev libopencv-dev libopenni2-dev git```

2. Download and compile [Pangolin](https://github.com/stevenlovegrove/Pangolin).

3. Compile the project, eg:

```
cd static-fusion 
mkdir build
cd build
cmake ..
make
```

## 2. Windows (using CMake and Visual Studio)

0. For the required dependencies: download pre-compiled binaries or the source code (and build it). Don't forget to add the dirs with the binaries to the PATH of Windows.

1. Please build Pangolin from source and disable the flag "MSVC_USE_STATIC_CRT" on CMake. This is required because Pangolin uses static linking by default on Windows and we use dynamic linking.

2. Generate the solution with CMake and compile / run the project you want to try.

Troubleshooting:
- The pangolin window looks completely white -> Resize it once to force it to show the content.
- Runtime error "OpenGL Error: XX (1282) In: ...\source\include\pangolin/gl/gl.hpp". We could solve it following https://github.com/stevenlovegrove/Pangolin/issues/149. Aparently some computers will decide to use the integrated GPU when they actually have a better one even if the integrated GPU does not support some of the functionalities used. If you only have an integrated GPU, I guess it could also happen that it does not support some of the operations implemented in our code...

# Running Experiments

There are three executables you can run:

**1) StaticFusion-Camera:** running off of live RGB-D camera feed, assuming an OpenNI compatible camera.

**2) StaticFusion-Datasets:** running using the TUM/Freiburg RGB-D datasets in Rawlog format. You can download these sequeces [here](https://www.mrpt.org/Collection_of_Kinect_RGBD_datasets_with_ground_truth_CVPR_TUM_2011). If you would like to perform quantitative evaluation, set ```bool save_results = true``` within the ```StaticFusion-datasets.cpp``` file to save the estimated trajectory to file. This trajectory can then be evaluated here: http://vision.in.tum.de/data/datasets/rgbd-dataset/online_evaluation

**2) StaticFusion-ImageSequenceAssoc:** running using images stored on disk. The expected format for images in this case is the same as listed [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools) if one was to use ```associate.py``` script to associate color and depth images.

The organisational structure of the dataset should be:

``` 
/dataset/rgb/   - folder containing all color images
/dataset/depth/ - folder containing all depth images
/dataset/rgbd_assoc.txt
```

Where ```rgbd_assoc.txt``` contain a list of items of the form:

```timestamp1 /rgb/timestamp1.png timestamp2 /depth/timestamp2.png```

Should you wish to modify this, the code can be found in the file ```FrontEnd.cpp```, in the methods ```loadAssoc, loadImageFromSequenceAssoc```.

The expected format of the images:

- color images - 8 bit in PNG. Resolution: VGA
- depth images - 16 bit monochrome in PNG, scaled by 1000. Resolution: VGA

**Example sequences** - coming soon
               
**All parameters are optimised for QVGA resolution.**

### ElasticFusion
We credit [ElasticFusion](https://github.com/mp3guy/ElasticFusion) as a significant basis for our work.
