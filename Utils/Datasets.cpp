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

#include <Utils/Datasets.h>

using namespace mrpt;
using namespace mrpt::obs;
using namespace std;


Datasets::Datasets(unsigned int res_factor)
{
    downsample = res_factor; // (1 - 640 x 480, 2 - 320 x 240)
    max_distance = 4.5f;
	dataset_finished = false;
    rawlog_count = 0; // 200 for walking_halfsphere

    rotateByZ = Eigen::Matrix4f::Identity();
    Eigen::AngleAxisf rotateByZAngle =  Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ());
    rotateByZ.topLeftCorner(3, 3) = rotateByZAngle.toRotationMatrix();
    gt_current = Eigen::Matrix4f::Identity();
}

void Datasets::openRawlog()
{

	//						Open Rawlog File
	//==================================================================
	if (!dataset.loadFromRawLogFile(filename))
		throw std::runtime_error("\nCouldn't open rawlog dataset file for input...");

	// Set external images directory:
	const string imgsPath = CRawlog::detectImagesDirectory(filename);
	utils::CImage::IMAGES_PATH_BASE = imgsPath;


	//					Load ground-truth
	//=========================================================
	filename = system::extractFileDirectory(filename);
	filename.append("/groundtruth.txt");
	f_gt.open(filename.c_str());
	if (f_gt.fail())
		throw std::runtime_error("\nError finding the groundtruth file: it should be contained in the same folder than the rawlog file");

	//Count number of lines (of the file)
	unsigned int number_of_lines = 0;
    std::string line;
    while (std::getline(f_gt, line))
        ++number_of_lines;

	gt_matrix.resize(number_of_lines-3, 8);
    f_gt.clear();
	f_gt.seekg(0, ios::beg);

	//Store the gt data in a matrix
	char aux[100];
	f_gt.getline(aux, 100);
	f_gt.getline(aux, 100);
	f_gt.getline(aux, 100);
	for (unsigned int k=0; k<number_of_lines-3; k++)
	{
		f_gt >> gt_matrix(k,0); f_gt >> gt_matrix(k,1); f_gt >> gt_matrix(k,2); f_gt >> gt_matrix(k,3);
		f_gt >> gt_matrix(k,4); f_gt >> gt_matrix(k,5); f_gt >> gt_matrix(k,6); f_gt >> gt_matrix(k,7);
		f_gt.ignore(10,'\n');	
	}

	f_gt.close();
	last_gt_row = 0;
}

void Datasets::loadFrameAndPoseFromDataset(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &intensity_wf, cv::Mat & depth_full, cv::Mat & color_full)
{
	if (dataset_finished)
	{
		printf("End of the dataset reached. Stop estimating motion!\n");
		return;
	}

	//Read images
	//-------------------------------------------------------
	CObservationPtr alfa = dataset.getAsObservation(rawlog_count);

	while (!IS_CLASS(alfa, CObservation3DRangeScan))
	{
       rawlog_count+=1;
		if (dataset.size() <= rawlog_count)
		{
			dataset_finished = true;
			return;
		}
		alfa = dataset.getAsObservation(rawlog_count);
	}

	CObservation3DRangeScanPtr obs3D = CObservation3DRangeScanPtr(alfa);
	obs3D->load();
	const Eigen::MatrixXf range = obs3D->rangeImage;
	const utils::CImage int_image =  obs3D->intensityImage;
	const unsigned int height = range.getRowCount();
	const unsigned int width = range.getColCount();
	const unsigned int cols = width/downsample, rows = height/downsample;

    math::CMatrixFloat intensity, r, g, b;
    intensity.resize(height, width);
	r.resize(height, width); g.resize(height, width); b.resize(height, width);
	int_image.getAsMatrix(intensity);
	int_image.getAsRGBMatrices(r, g, b);

// Reading images in this way makes the algorithm perform worse.. why?

//	for (unsigned int j = 0; j<cols; j++)
//		for (unsigned int i = 0; i<rows; i++)
//		{
//            const float z = range(downsample*i, downsample*j);
//            if (z < max_distance) {
//                depth_wf(i,j) = int (z * 1000.0) / 1000.0;
//                depth_full.at<unsigned short>(i,j) = z * 1000.0;
//            } else {
//                depth_wf(i,j) = 0.f;
//                depth_full.at<unsigned short>(i,j) = 0.f;
//            }

//			//Color image, just for the visualization
//            float rCol = b(downsample*i, downsample*j);
//            float gCol = g(downsample*i, downsample*j);
//            float bCol = r(downsample*i, downsample*j);

//            intensity_wf(i,j) = 0.299f* rCol + 0.587f*gCol + 0.114f*bCol;
//            color_full.at<cv::Vec3b>(i,j) = cv::Vec3b( rCol * 255, gCol* 255, bCol * 255);
//        }




    for (unsigned int j = 0; j<cols; j++)
        for (unsigned int i = 0; i<rows; i++)
        {
            const float z = range(height - downsample*i -1, width - downsample*j -1);
            if (z < max_distance) {
                depth_wf(i,j) = int (z * 1000.0) / 1000.0;
                depth_full.at<unsigned short>(i,j) = z * 1000.0;
            } else {
                depth_wf(i,j) = 0.f;
                depth_full.at<unsigned short>(i,j) = 0.f;
            }

            //Color image, just for the visualization
            float rCol = b(height - downsample*i -1, width - downsample*j -1);
            float gCol = g(height - downsample*i -1, width - downsample*j -1);
            float bCol = r(height - downsample*i -1, width - downsample*j -1);

            intensity_wf(i,j) = 0.299f* rCol + 0.587f*gCol + 0.114f*bCol;
            color_full.at<cv::Vec3b>(i,j) = cv::Vec3b( rCol * 255, gCol* 255, bCol * 255);
        }

	timestamp_obs = mrpt::system::timestampTotime_t(obs3D->timestamp);

	obs3D->unload();
    rawlog_count+=1;

	if (dataset.size() <= rawlog_count)
		dataset_finished = true;

	//Groundtruth
	//--------------------------------------------------

	while (abs(gt_matrix(last_gt_row,0) - timestamp_obs) > abs(gt_matrix(last_gt_row+1,0) - timestamp_obs))
	{
		last_gt_row++;
		if (last_gt_row >= gt_matrix.rows())
		{
			dataset_finished = true;
			return;		
		}
	}

	//Get the pose of the closest ground truth
	double x,y,z,qx,qy,qz,w;
	x = gt_matrix(last_gt_row,1); y = gt_matrix(last_gt_row,2); z = gt_matrix(last_gt_row,3);
	qx = gt_matrix(last_gt_row,4); qy = gt_matrix(last_gt_row,5); qz = gt_matrix(last_gt_row,6);
	w = gt_matrix(last_gt_row,7);

    Eigen::Quaternionf currentQuat = Eigen::Quaternionf(float(w), float(qx), float(qy), float(qz));
    gt_current.topLeftCorner(3,3) = currentQuat.normalized().toRotationMatrix();
    gt_current(0, 3) = float(x);
    gt_current(1, 3) = float(y);
    gt_current(2, 3) = float(z);
    gt_current = gt_current * rotateByZ;
}


void Datasets::createResultsFile()
{
	//Create file with the first free file-name.
	char	aux[100];
	int     nFile = 0;
	bool    free_name = false;

	system::createDirectory("./odometry_results");

	while (!free_name)
	{
		nFile++;
		sprintf(aux, "./odometry_results/experiment_%03u.txt", nFile );
		free_name = !system::fileExists(aux);
	}

	// Open log file:
	f_res.open(aux);
	printf(" Saving results to file: %s \n", aux);
}

void Datasets::writeTrajectoryFile(Eigen::Matrix4f currPose, Eigen::MatrixXf &ddt)
{	
	//Don't take into account those iterations with consecutive equal depth images
	if (abs(ddt.sumAll()) > 0)
	{		
        Eigen::Matrix4f convertedPose = currPose * rotateByZ;
        Eigen::Matrix3f rotationMat = convertedPose.topLeftCorner(3, 3);
        Eigen::Quaternionf currQuat = Eigen::Quaternionf(rotationMat);
	
		char aux[24];
		sprintf(aux,"%.04f", timestamp_obs);
        f_res << aux << " " << convertedPose(0, 3) << " " << convertedPose(1, 3) << " " << convertedPose(2, 3) << " ";
        f_res << currQuat.x() << " " << currQuat.y() << " " << currQuat.z() << " " << currQuat.w() << endl;
	}
}
