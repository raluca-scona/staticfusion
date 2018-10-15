/***************************************************************************************
**StaticFusion: Background Reconstruction for Dense RGB-D SLAM in Dynamic Environments**
**				----------------------------------------					          **
**																			          **
**	Copyright(c) 2018, Raluca Scona, Edinburgh Centre for Robotics                    **
**	Copyright(c) 2015, Mariano Jaimez, University of Malaga & TU Munich		          **
**																			          **
**  This program is free software: you can redistribute it and/or modify	          **
**  it under the terms of the GNU General Public License (version 3) as		          **
**	published by the Free Software Foundation.								          **
**																			          **
**  This program is distributed in the hope that it will be useful, but		          **
**	WITHOUT ANY WARRANTY; without even the implied warranty of				          **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			          **
**  GNU General Public License for more details.							          **
**																			          **
**  You should have received a copy of the GNU General Public License		          **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>              **
**																			          **
***************************************************************************************/
/*********************************************************************************
**Fast Odometry and Scene Flow from RGB-D Cameras based on Geometric Clustering	**
**------------------------------------------------------------------------------**
**																				**
**	Copyright(c) 2017, Mariano Jaimez Tarifa, University of Malaga & TU Munich	**
**	Copyright(c) 2017, Christian Kerl, TU Munich								**
**	Copyright(c) 2017, MAPIR group, University of Malaga						**
**	Copyright(c) 2017, Computer Vision group, TU Munich							**
**																				**
**  This program is free software: you can redistribute it and/or modify		**
**  it under the terms of the GNU General Public License (version 3) as			**
**	published by the Free Software Foundation.									**
**																				**
**  This program is distributed in the hope that it will be useful, but			**
**	WITHOUT ANY WARRANTY; without even the implied warranty of					**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the				**
**  GNU General Public License for more details.								**
**																				**
**  You should have received a copy of the GNU General Public License			**
**  along with this program. If not, see <http://www.gnu.org/licenses/>.		**
**																				**
*********************************************************************************/
#include <Eigen/Core>
#include <OpenNI.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <iostream>
#include <fstream>



class RGBD_Camera {
public:

    RGBD_Camera(unsigned int res_factor);

	unsigned int cam_mode;
	float max_distance;

    openni::Status		rc;
    openni::Device		device;
    openni::VideoMode	options;
    openni::VideoStream rgb,dimage;

    bool openCamera();
    void closeCamera();
    void loadFrame(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &color_wf, cv::Mat & color_full, cv::Mat & depth_full);
    void disableAutoExposureAndWhiteBalance();
};




