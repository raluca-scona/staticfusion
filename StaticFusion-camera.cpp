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

#include <StaticFusion.h>
#include <Utils/RGBD_Camera.h>

#include <iostream>
#include <fstream>

#include <opencv2/core/eigen.hpp>

int main()
{	
    unsigned int res_factor = 2;
    StaticFusion staticFusion;

    //Flags
    staticFusion.use_motion_filter = true;

    //Solver
    staticFusion.ctf_levels = log2(staticFusion.cols/40) + 2;
    staticFusion.max_iter_per_level = 3;
    staticFusion.previous_speed_const_weight = 0.1f;
    staticFusion.previous_speed_eig_weight = 2.f; //0.5f;

    staticFusion.k_photometric_res = 0.15f;
    staticFusion.irls_delta_threshold = 0.0015f;
    staticFusion.max_iter_irls = 6;
    staticFusion.lambda_reg = 0.35f; //0.4
    staticFusion.lambda_prior = 0.5f; //0.5
    staticFusion.kc_Cauchy = 0.5f; //0.5
    staticFusion.kb = 1.5f; //1.5
    staticFusion.kz = 1.5f;

    cv::Mat weightedImage = cv::Mat(Resolution::getInstance().width(), Resolution::getInstance().height(), CV_32F, 0.0);

    cv::Mat depth_full = cv::Mat(staticFusion.height, staticFusion.width,  CV_16U, 0.0);
    cv::Mat color_full = cv::Mat(staticFusion.height, staticFusion.width,  CV_8UC3,  cv::Scalar(0,0,0));

    bool denseModel = false;
    bool modelInitialised = false;

    int im_count = 0;

	RGBD_Camera camera(res_factor);

	//Initialize camera and method
    camera.openCamera();
    camera.disableAutoExposureAndWhiteBalance();
    camera.loadFrame(staticFusion.depthPrediction, staticFusion.intensityPrediction, color_full, depth_full);

    staticFusion.depthBuffer[im_count % staticFusion.bufferLength] = staticFusion.depthPrediction.replicate(1,1);
    staticFusion.intensityBuffer[im_count % staticFusion.bufferLength] = staticFusion.intensityPrediction.replicate(1,1);
    staticFusion.odomBuffer[im_count % staticFusion.bufferLength] = Eigen::Matrix4f::Identity();

    camera.loadFrame(staticFusion.depthCurrent, staticFusion.intensityCurrent, color_full, depth_full);

    staticFusion.createImagePyramid(true);

    staticFusion.kb = 1.05f;

    staticFusion.runSolver(true);

    staticFusion.buildSegmImage();

    im_count++;

    staticFusion.depthBuffer[im_count % staticFusion.bufferLength] = staticFusion.depthCurrent.replicate(1,1);
    staticFusion.intensityBuffer[im_count % staticFusion.bufferLength] = staticFusion.intensityCurrent.replicate(1,1);
    staticFusion.odomBuffer[im_count % staticFusion.bufferLength] = staticFusion.T_odometry;

    cv::eigen2cv(staticFusion.b_segm_perpixel, weightedImage);

    staticFusion.reconstruction->fuseFrame((unsigned char *) color_full.data, (unsigned short *) depth_full.data, (float *) weightedImage.data, im_count, &(staticFusion.T_odometry), 0, 1);
    staticFusion.reconstruction->uploadWeightAndClustersForVisualization((float *) weightedImage.data, staticFusion.clusterAllocation[0], (unsigned short *) depth_full.data);


    while(!pangolin::ShouldQuit()) {

        while (!pangolin::ShouldQuit() && !staticFusion.gui->pause->Get() ) {

            camera.loadFrame(staticFusion.depthCurrent, staticFusion.intensityCurrent, color_full, depth_full);

            im_count++;

            denseModel = staticFusion.reconstruction->checkIfDenseEnough();

            if (!denseModel && !modelInitialised) {

                staticFusion.kb = 1.05f; //1.25s
                modelInitialised = true;

            } else {

                staticFusion.kb = 1.5f;
                modelInitialised = true;
            }

            staticFusion.reconstruction->getPredictedImages(staticFusion.depthPrediction, staticFusion.intensityPrediction);
            staticFusion.reconstruction->getFilteredDepth(depth_full, staticFusion.depthCurrent);

            staticFusion.createImagePyramid(true);   //pyramid for the old model

            staticFusion.runSolver(true);

            if (im_count - staticFusion.bufferLength >= 0) {
                staticFusion.computeResidualsAgainstPreviousImage(im_count);
            }

            //Build segmentation image to use it for the data fusion
            staticFusion.buildSegmImage();

            staticFusion.depthBuffer[im_count % staticFusion.bufferLength] = staticFusion.depthCurrent.replicate(1,1);
            staticFusion.intensityBuffer[im_count % staticFusion.bufferLength] = staticFusion.intensityCurrent.replicate(1,1);
            staticFusion.odomBuffer[im_count % staticFusion.bufferLength] = staticFusion.T_odometry;

            cv::eigen2cv(staticFusion.b_segm_perpixel, weightedImage);

            staticFusion.reconstruction->fuseFrame((unsigned char *) color_full.data, (unsigned short *) depth_full.data,  (float *) weightedImage.data, im_count, &(staticFusion.T_odometry), 0, 1);

            staticFusion.reconstruction->uploadWeightAndClustersForVisualization((float *) weightedImage.data, staticFusion.clusterAllocation[0], (unsigned short *) depth_full.data);

            staticFusion.updateGUI();
        }

        staticFusion.updateGUI();
    }


    camera.closeCamera();
	return 0;
}

