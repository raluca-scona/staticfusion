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

#include <mrpt/math/utils.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/config/CConfigFileBase.h>
#include <mrpt/system/filesystem.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>


class Datasets {
public:

    Datasets(unsigned int res_factor);

	unsigned int rawlog_count;
	unsigned int last_gt_row;
	unsigned int downsample;
	float max_distance;

	mrpt::obs::CRawlog	dataset;
	std::ifstream		f_gt;
	std::ofstream		f_res;
	std::string			filename;

	Eigen::MatrixXd gt_matrix;
    Eigen::Matrix4f gt_current;
    Eigen::Matrix4f rotateByZ;
	double timestamp_obs;				//!< Timestamp of the last observation
	bool dataset_finished;

    void openRawlog();
    void loadFrameAndPoseFromDataset(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &intensity_wf, cv::Mat & depth_full, cv::Mat & color_full);
    void createResultsFile();
    void writeTrajectoryFile(Eigen::Matrix4f currPose, Eigen::MatrixXf &ddt);

};
