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

#ifndef STATICFUSION_H
#define STATICFUSION_H


#include <mrpt/poses/CPose3D.h>
#include <mrpt/math/utils.h>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Reconstruction.h"
#include "Shaders/Resize.h"
#include "Utils/GUI.h"
#include <Eigen/Core>

#define NUM_CLUSTERS 24

typedef Eigen::Matrix<float, 6, 1> Vector6f;


class StaticFusion {
public:

    //						EF
    //----------------------------------------------------------------
    cv::Mat depth_mm, color_full;

    Reconstruction * reconstruction;
    GUI * gui;

    float confidence;
    float depth_max;
    const std::string fileName;


	//						General
	//----------------------------------------------------------------
    std::vector<Eigen::MatrixXf> intensityPyr, intensityPredPyr, intensityInterPyr, intensityWarpedPyr;	//Intensity images
    std::vector<Eigen::MatrixXf> depthPyr, depthPredPyr, depthInterPyr, depthWarpedPyr;					//Depth images
    std::vector<Eigen::MatrixXf> xxPyr, xxInterPyr, xxPredPyr, xxWarpedPyr;								//x coordinates of points (proportional to the col index of the pixels)
    std::vector<Eigen::MatrixXf> yyPyr, yyInterPyr, yyPredPyr, yyWarpedPyr;								//y coordinates of points (proportional to the row index of the pixels)

    Eigen::MatrixXf depthCurrent, intensityCurrent;							//Original images read from the camera, dataset or file
    Eigen::MatrixXf depthPrediction, intensityPrediction;					//Images from model prediction

    //Buffering related variables
    Eigen::MatrixXf xxBuffer, yyBuffer, depthResiduals, intensityResiduals, cumulativeResiduals;
    Eigen::Array<float, NUM_CLUSTERS, 1> perClusterAverageResidual;
    std::vector<Eigen::MatrixXf> depthBuffer, intensityBuffer;
    std::vector<Eigen::Matrix4f> odomBuffer;
    int bufferLength = 5;

    Eigen::MatrixXf depthWarpedRefference;
    Eigen::MatrixXf intensityWarpedRefference;

    Eigen::MatrixXf dcu, dcv, dct;									//Gradients of the intensity images
    Eigen::MatrixXf ddu, ddv, ddt;									//Gradients of the depth images

    Eigen::MatrixXf weights_c, weights_d;							//Pre-weighting used in the solver
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> Null;		//Mask for pixels with null depth measurments
    Eigen::Array44f convMask;											//Convolutional kernel used to build the image pyramid

    //Velocities, transformations and poses
    mrpt::poses::CPose3D cam_pose, cam_oldpose; //Estimated camera poses (current and prev)
    Eigen::Matrix4f T_odometry;                                //Rigid transformation of the camera motion (odometry)
    Vector6f twist_odometry, twist_level_odometry, twist_odometry_old;      	//Twist encoding the odometry (accumulated and local for the pyramid level)
    Eigen::Matrix<float, 6, 6> est_cov;

	//Parameters
    float fovh, fovv;							//Field of view of the camera (intrinsic calibration)
    unsigned int rows, cols;					//Max resolution used for the solver (240 x 320 by default)
    unsigned int rows_i, cols_i;				//Aux variables
    unsigned int rows_km, cols_km;				//Aux variables
    unsigned int width, height;					//Resolution of the input images
	unsigned int ctf_levels;					//Number of coarse-to-fine levels
	unsigned int image_level, level;			//Aux variables
    unsigned int image_level_km;                //Aux variables


    StaticFusion(unsigned int width = 640 / 2, unsigned int height = 480 / 2, float fx =  527.34367917 / 2, float fy = 532.78024387 / 2, float cx = 320 / 2., float cy = 240 / 2.);
    void createImagePyramid(bool old_im);					//Create image pyramids (intensity and depth)
    void warpImagesAccurateInverse();
    void calculateCoord();						//Compute so-called "intermediate coordinates", related to a more precise linearization of optical and range flow
	void calculateDerivatives();				//Compute the image gradients
    void computeWeights();						//Compute pre-weighting functions for the solver
    void computeTransformationFromTwist(Vector6f &twist);	//Compute rigid transformation from twist
    void computeResidualsAgainstPreviousImage(int index);


    void runSolver(bool create_image_pyr);		//Main method to run whole algorithm

	//							Solver
	//--------------------------------------------------------------
    bool use_motion_filter;
    float previous_speed_const_weight;
    float previous_speed_eig_weight;
	unsigned int max_iter_irls;				//Max number of iterations for the IRLS solver
	unsigned int max_iter_per_level;		//Max number of complete iterations for every level of the pyramid
	float k_photometric_res;				//Weight of the photometric residuals (against geometric ones)
	float irls_delta_threshold;				//Convergence threshold for the IRLS solver (change in the solution)
    float kc_Cauchy, kb;
    std::vector<std::pair<int,int>> validPixels;     //Store indices of the pixels used for the solver

	//Estimate rigid motion for a set of pixels (given their indices)
    void solveOdometryAndSegmJoint();
    void filterEstimateAndComputeT(Vector6f &twist);

    //					Geometric clustering
    //--------------------------------------------------------------
    std::vector<Eigen::MatrixXi> clusterAllocation;											//Integer non-smooth scoring
    Eigen::Matrix<float, 3, NUM_CLUSTERS> kmeans;										//Centers of the KMeans clusters
    bool connectivity[NUM_CLUSTERS][NUM_CLUSTERS];										//Connectivity between the clusters

    void createClustersPyramidUsingKMeans();				//Create the label pyramid
	void initializeKMeans();							//Initialize KMeans by uniformly dividing the image plane
	void kMeans3DCoord();								//Segment the scene in clusters using the 3D coordinates of the points
    void computeRegionConnectivity();					//Compute connectivity graph (which cluster is contiguous to which)

	//						Static-Dynamic segmentation
	//--------------------------------------------------------------------------------
    Eigen::Matrix<float, NUM_CLUSTERS, 1> b_segm, b_prior, lambda_t_w;    //Exact b values of the segmentation (original and warped)
    Eigen::MatrixXf b_segm_perpixel;                                        //Per-pixel static-dynamic segmentation (value of b per pixel)
    Eigen::MatrixXf res_prev_image;                                     //Residuals after aligning against previous image
    Eigen::MatrixXf A_seg, AtA_seg;
    Eigen::VectorXf B_seg, AtB_seg;
    float lambda_reg, lambda_prior;
    float kz;

    void buildSystemSegm();
    void solveSegmIteration(const Eigen::Array<float, NUM_CLUSTERS, 1> &aver_res, float aver_res_overall, float kc_Cauchy);
    void computeSegPrior();
    void buildSegmImage();

    //					Input / Output
	//--------------------------------------------------------------
	bool loadImageFromSequence(std::string files_dir, unsigned int index, unsigned int res_factor);
    bool loadImageFromSequenceAssoc(const std::string &depthFile, const std::string &rgbFile, unsigned int res_factor);

    bool loadAssoc(const std::string & dir, const std::string &assocFile, std::vector<double> &timestamps,
                      std::vector<std::string> &filesDepth, std::vector<std::string> &filesColor) const;

    void updateGUI();

};

#endif



