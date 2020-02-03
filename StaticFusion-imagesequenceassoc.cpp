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

#include <string.h>
#include <StaticFusion.h>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
    if(argc<2) {
        throw std::runtime_error("missing log file");
    }

    const std::string dir = argv[1];

    const unsigned int res_factor = 2;
    StaticFusion staticFusion(res_factor);

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

    bool denseModel = false;
    bool modelInitialised = false;


    //Set first image to load, decimation factor and the sequence dir
    int im_count = 1;
    const unsigned int decimation = 1;

    std::vector<double> timestamps;
    std::vector<std::string> filesDepth;
    std::vector<std::string> filesColor;

    const std::string assocFile = "/rgbd_assoc.txt";

    staticFusion.loadAssoc(dir, assocFile, timestamps, filesDepth, filesColor);

    if (filesDepth.empty() || filesColor.empty()) {
        throw std::runtime_error("no image files");
    }

    cv::Mat weightedImage;

    bool stop = false;

    {
        stop = im_count >= filesDepth.size();

        staticFusion.loadImageFromSequenceAssoc(filesDepth[im_count], filesColor[im_count], res_factor);

        staticFusion.depthPrediction.swap(staticFusion.depthCurrent);
        staticFusion.intensityPrediction.swap(staticFusion.intensityCurrent);


        staticFusion.depthBuffer[im_count % staticFusion.bufferLength] = staticFusion.depthPrediction.replicate(1,1);
        staticFusion.intensityBuffer[im_count % staticFusion.bufferLength] = staticFusion.intensityPrediction.replicate(1,1);
        staticFusion.odomBuffer[im_count % staticFusion.bufferLength] = Eigen::Matrix4f::Identity();

        im_count += decimation;

        stop = im_count >= filesDepth.size();

        staticFusion.loadImageFromSequenceAssoc(filesDepth[im_count], filesColor[im_count], res_factor);

        staticFusion.createImagePyramid(true);   //pyramid for the old model

        staticFusion.kb = 1.05f; //0.8

        staticFusion.runSolver(true);

        staticFusion.buildSegmImage();

        staticFusion.depthBuffer[im_count % staticFusion.bufferLength] = staticFusion.depthCurrent.replicate(1,1);
        staticFusion.intensityBuffer[im_count % staticFusion.bufferLength] = staticFusion.intensityCurrent.replicate(1,1);
        staticFusion.odomBuffer[im_count % staticFusion.bufferLength] = staticFusion.T_odometry;

        cv::eigen2cv(staticFusion.b_segm_perpixel, weightedImage);

        staticFusion.reconstruction->fuseFrame((unsigned char *) staticFusion.color_full.data, (unsigned short *) staticFusion.depth_mm.data, (float *) weightedImage.data, im_count, &(staticFusion.T_odometry), 0, 1);
        staticFusion.reconstruction->uploadWeightAndClustersForVisualization((float *) weightedImage.data, staticFusion.clusterAllocation[0], (unsigned short *) staticFusion.depth_mm.data);

    }

    while (!pangolin::ShouldQuit())
	{	

	
        if (!( (im_count+decimation) >= filesDepth.size()) && !pangolin::ShouldQuit() && !staticFusion.gui->pause->Get() )
		{

            im_count += decimation;

            staticFusion.loadImageFromSequenceAssoc(filesDepth[im_count], filesColor[im_count], res_factor);

            denseModel = staticFusion.reconstruction->checkIfDenseEnough();

            if (!denseModel && !modelInitialised) {

                staticFusion.kb = 1.05f;
                modelInitialised = true;

            } else {

                staticFusion.kb = 1.5f;
                modelInitialised = true;
            }

            staticFusion.reconstruction->getPredictedImages(staticFusion.depthPrediction, staticFusion.intensityPrediction);
            staticFusion.reconstruction->getFilteredDepth(staticFusion.depth_mm, staticFusion.depthCurrent);

            staticFusion.createImagePyramid(true);

            staticFusion.runSolver(true);

            if (im_count - staticFusion.bufferLength >= 0) {
                staticFusion.computeResidualsAgainstPreviousImage(im_count);
            }

            staticFusion.buildSegmImage();

            staticFusion.depthBuffer[im_count % staticFusion.bufferLength] = staticFusion.depthCurrent.replicate(1,1);
            staticFusion.intensityBuffer[im_count % staticFusion.bufferLength] = staticFusion.intensityCurrent.replicate(1,1);
            staticFusion.odomBuffer[im_count % staticFusion.bufferLength] = staticFusion.T_odometry;

            cv::eigen2cv(staticFusion.b_segm_perpixel, weightedImage);

            staticFusion.reconstruction->fuseFrame((unsigned char *) staticFusion.color_full.data, (unsigned short *) staticFusion.depth_mm.data,  (float *) weightedImage.data, im_count, &(staticFusion.T_odometry), 0, 1);
            staticFusion.reconstruction->uploadWeightAndClustersForVisualization((float *) weightedImage.data, staticFusion.clusterAllocation[0], (unsigned short *) staticFusion.depth_mm.data);

            staticFusion.updateGUI();
		}

        staticFusion.updateGUI();

	}

	return 0;
}

