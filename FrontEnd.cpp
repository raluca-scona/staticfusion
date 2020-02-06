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

using namespace mrpt;
using namespace mrpt::utils;
using namespace std;
using namespace Eigen;

//A strange size for "ws..." due to the fact that some pixels are used twice for odometry and scene flow (hence the 3/2 safety factor)
StaticFusion::StaticFusion(unsigned int res_factor)
{
    //Resolutions and levels
    rows = 480/res_factor;
    cols = 640/res_factor;
    fovh = M_PI*62.5/180.0;
    fovv = M_PI*48.5/180.0;
    width = 640/res_factor;
    height = 480/res_factor;
    ctf_levels = log2(cols/40) + 2;
    float fx = 0.5 * cols / tan( fovh * 0.5);
    float fy = 0.5 * rows / tan( fovv * 0.5);

    //Solver
    k_photometric_res = 0.15f;
    irls_delta_threshold = 1e-6f;
    max_iter_irls = 10;
    max_iter_per_level = 2;
    previous_speed_const_weight = 0.05f;
    previous_speed_eig_weight = 0.5f;
    kc_Cauchy = 0.5f;
    kb = 1.25f;
    kz = 1.5f;

    use_motion_filter = false;

    //CamPose
    cam_pose.setFromValues(0,0,0,0,0,0);
    cam_oldpose = cam_pose;
    twist_odometry_old.fill(0.f);

    //Resize matrices which are not in a "pyramid"
    depthCurrent.setSize(height,width);
    depthPrediction.setSize(height,width);
    intensityCurrent.setSize(height,width);
    intensityPrediction.setSize(height,width);

    dct.resize(rows,cols); ddt.resize(rows,cols);
    dcu.resize(rows,cols); ddu.resize(rows,cols);
    dcv.resize(rows,cols); ddv.resize(rows,cols);
    Null.resize(rows,cols);
    weights_c.setSize(rows,cols);
    weights_d.setSize(rows,cols);

    intensityBuffer.resize(bufferLength);
    depthBuffer.resize(bufferLength);
    odomBuffer.resize(bufferLength);

    for (int i=0; i<bufferLength; i++) {
        intensityBuffer[i].resize(rows, cols);
        depthBuffer[i].resize(rows, cols);
    }

    perClusterAverageResidual.fill(std::numeric_limits<float>::quiet_NaN());

    //Resize matrices in a "pyramid"
    const unsigned int pyr_levels = round(log2(width/cols)) + ctf_levels;
    intensityPyr.resize(pyr_levels); intensityPredPyr.resize(pyr_levels); intensityInterPyr.resize(pyr_levels);
    depthPyr.resize(pyr_levels); depthPredPyr.resize(pyr_levels); depthInterPyr.resize(pyr_levels);
    xxPyr.resize(pyr_levels); xxInterPyr.resize(pyr_levels); xxPredPyr.resize(pyr_levels);
    yyPyr.resize(pyr_levels); yyInterPyr.resize(pyr_levels); yyPredPyr.resize(pyr_levels);

    intensityWarpedPyr.resize(pyr_levels);
    depthWarpedPyr.resize(pyr_levels);
    xxWarpedPyr.resize(pyr_levels);
    yyWarpedPyr.resize(pyr_levels);
    clusterAllocation.resize(pyr_levels);
    xxBuffer.setSize(height, width); yyBuffer.setSize(height, width);
    xxBuffer.assign(0.f); yyBuffer.assign(0.f);

    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        const unsigned int s = pow(2.f,int(i));
        cols_i = width/s; rows_i = height/s;
        intensityPyr[i].resize(rows_i, cols_i); intensityPredPyr[i].resize(rows_i, cols_i); intensityInterPyr[i].resize(rows_i, cols_i);
        depthPyr[i].resize(rows_i, cols_i); depthInterPyr[i].resize(rows_i, cols_i); depthPredPyr[i].resize(rows_i, cols_i);
        depthPyr[i].assign(0.f); depthPredPyr[i].assign(0.f);
        xxPyr[i].resize(rows_i, cols_i); xxInterPyr[i].resize(rows_i, cols_i); xxPredPyr[i].resize(rows_i, cols_i);
        xxPyr[i].assign(0.f); xxPredPyr[i].assign(0.f);
        yyPyr[i].resize(rows_i, cols_i); yyInterPyr[i].resize(rows_i, cols_i); yyPredPyr[i].resize(rows_i, cols_i);
        yyPyr[i].assign(0.f); yyPredPyr[i].assign(0.f);

        if (cols_i <= cols)
        {
            intensityWarpedPyr[i].resize(rows_i,cols_i);
            depthWarpedPyr[i].resize(rows_i,cols_i);
            xxWarpedPyr[i].resize(rows_i,cols_i);
            yyWarpedPyr[i].resize(rows_i,cols_i);
            clusterAllocation[i].resize(rows_i, cols_i);
            clusterAllocation[i].assign(0);
        }
    }

    //Compute gaussian and "fast-symmetric" mask
    const Vector4f v_mask(1.f, 2.f, 2.f, 1.f);
    for (unsigned int i=0; i<4; i++)
        for (unsigned int j=0; j<4; j++)
            convMask(i,j) = v_mask(i)*v_mask(j)/36.f;


    //              Labels (Initialized to uncertain)
    //=========================================================
    b_segm_perpixel.setSize(rows,cols);
    b_segm_perpixel.fill(0.5f);
    b_segm.fill(0.5f);

    depthWarpedRefference = Eigen::MatrixXf::Zero(rows, cols);
    intensityWarpedRefference = Eigen::MatrixXf::Zero(rows, cols);

    depth_mm = cv::Mat(height, width, CV_16U, 0.0);
    color_full = cv::Mat(height, width, CV_8UC3, cv::Scalar(0,0,0));

    Resolution::getInstance(cols, rows);
    Intrinsics::getInstance(fx, fy, cols/2, rows/2);

    confidence = 0.25f;
    depth_max = 4.5f;
    std::string fileName = "sf-mesh";

    gui = new GUI(fileName.length() == 0, false);

    gui->confidenceThreshold->Ref().Set(confidence);
    gui->depthCutoff->Ref().Set(depth_max);

    reconstruction = new Reconstruction(  std::numeric_limits<int>::max() ,
                                  confidence,
                                  depth_max,
                                  fileName,
                                  NUM_CLUSTERS);
}

bool StaticFusion::loadAssoc(const std::string & dir, const std::string &assocFile, std::vector<double> &timestamps,
               std::vector<std::string> &filesDepth, std::vector<std::string> &filesColor) const
{
    std::string assocPath = dir + assocFile;

    if (assocPath.empty())
        return false;

    std::ifstream assocIn;
    assocIn.open(assocPath.c_str());
    if (!assocIn.is_open())
        return false;

    std::string line;
    while (std::getline(assocIn, line))
    {
        if (line.empty() || line.compare(0, 1, "#") == 0)
            continue;
        std::istringstream iss(line);
        double timestampDepth, timestampColor;
        std::string fileDepth, fileColor;
        if (!(iss >> timestampColor >> fileColor >> timestampDepth >> fileDepth))
            break;

        timestamps.push_back(timestampDepth);
        filesDepth.push_back(dir + fileDepth);
        filesColor.push_back(dir + fileColor);
    }
    assocIn.close();

    return true;
}

bool StaticFusion::loadImageFromSequenceAssoc(const std::string &depthFile, const std::string &rgbFile, unsigned int res_factor)
{
    const float norm_factor = 1.f/255.f;

    cv::Mat color = cv::imread(rgbFile.c_str(), CV_LOAD_IMAGE_COLOR);

    if (color.data == NULL)
    {
        printf("End of sequence (or color image not found...)\n");
        return true;
    }

    for (unsigned int v=0; v<height; v++)
        for (unsigned int u=0; u<width; u++)
        {
            cv::Vec3b color_here = color.at<cv::Vec3b>(height * res_factor - res_factor*v -1, res_factor*u);
            float r = norm_factor*color_here[0];
            float g = norm_factor*color_here[1];
            float b = norm_factor*color_here[2];

            intensityCurrent(v,u) = 0.299f* r + 0.587f* g+ 0.114f* b;
            color_full.at<cv::Vec3b>(v,u) = cv::Vec3b( r * 255, g* 255, b * 255);
        }

    cv::Mat depth = cv::imread(depthFile.c_str(), -1);
    cv::Mat depth_float;
    cv::Mat depth_float_mm;
    depth.convertTo(depth_float, CV_32FC1, 1.0 / 1000.0);
    depth.convertTo(depth_float_mm, CV_16U, 1000.0 / 1000.0);


    for (unsigned int v=0; v<height; v++)
        for (unsigned int u=0; u<width; u++) {
            depthCurrent(v,u) = depth_float.at<float>(height * res_factor - res_factor*v -1 , res_factor*u);
            depth_mm.at<unsigned short>(v,u) = depth_float_mm.at<unsigned short>(height * res_factor - res_factor*v -1, res_factor*u);
        }

    return false;
}

void StaticFusion::createImagePyramid(bool old_im)
{
    //Threshold to use (or not) neighbours in the filter
    const float max_depth_dif = 0.1f;

    //The number of levels of the pyramid does not match the number of levels used
    //in the odometry computation (because we sometimes want to finish with lower resolutions)
    unsigned int pyr_levels = round(log2(width/cols)) + ctf_levels;


    //Generate levels
    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        unsigned int s = pow(2.f,int(i));
        cols_i = width/s;
        rows_i = height/s;
        const unsigned int i_1 = i-1;

        MatrixXf &depth_here = old_im ? depthPredPyr[i] : depthPyr[i];    //saying which one to modify
        MatrixXf &intensity_here = old_im ? intensityPredPyr[i] : intensityPyr[i];
        MatrixXf &xx_here = old_im ? xxPredPyr[i] : xxPyr[i];
        MatrixXf &yy_here = old_im ? yyPredPyr[i] : yyPyr[i];
        const MatrixXf &depth_prev = old_im ? depthPredPyr[i_1] : depthPyr[i_1];
        const MatrixXf &intensity_prev = old_im ? intensityPredPyr[i_1] : intensityPyr[i_1];


        if (i == 0 && !old_im)
        {
            depth_here = depthCurrent.replicate(1,1);
            intensity_here = intensityCurrent.replicate(1, 1);

        } else if (i==0 && old_im){

            depth_here = depthPrediction.replicate(1,1);
            intensity_here = intensityPrediction.replicate(1, 1);

        }

        //                              Downsampling
        //-----------------------------------------------------------------------------
        else
        {
            for (unsigned int u = 0; u < cols_i; u++)
                for (unsigned int v = 0; v < rows_i; v++)
                {
                    const int u2 = 2*u;
                    const int v2 = 2*v;

                    //Inner pixels
                    if ((v>0)&&(v<rows_i-1)&&(u>0)&&(u<cols_i-1))
                    {

                        const Matrix4f depth_block = depth_prev.block<4,4>(v2-1,u2-1);
                        const Matrix4f intensity_block = intensity_prev.block<4,4>(v2-1,u2-1);

                        float depths[4] = {depth_block(5), depth_block(6), depth_block(9), depth_block(10)};


                        //Find the "second maximum" value of the central block
                        if (depths[1] < depths[0]) {std::swap(depths[1], depths[0]);}
                        if (depths[3] < depths[2]) {std::swap(depths[3], depths[2]);}
                        const float dcenter = (depths[3] < depths[1]) ? max(depths[3], depths[0]) : max(depths[1], depths[2]);

                        if (dcenter != 0.f)
                        {
                            float sum_d = 0.f, sum_c = 0.f, weight = 0.f;

                            for (unsigned char k=0; k<16; k++)
                            {
                                const float abs_dif = abs(depth_block(k)-dcenter);
                                if (abs_dif < max_depth_dif)
                                {
                                    const float aux_w = convMask(k)*(max_depth_dif - abs_dif);
                                    weight += aux_w;
                                    sum_d += aux_w*depth_block(k);
                                    sum_c += aux_w*intensity_block(k);
                                }
                            }

                            depth_here(v,u) = sum_d/weight;
                            intensity_here(v,u) = sum_c/weight;

                        }
                        else
                        {
                            intensity_here(v,u) = (convMask*intensity_block.array()).sum();
                            depth_here(v,u) = 0.f;
                        }
                    }

                    //Boundary
                    else
                    {

                        const Matrix2f depth_block = depth_prev.block<2,2>(v2,u2);
                        const Matrix2f intensity_block = intensity_prev.block<2,2>(v2,u2);

                        intensity_here(v,u) = 0.25f*intensity_block.sumAll();

                        float new_d = 0.f;
                        unsigned int cont = 0;
                        for (unsigned int k=0; k<4;k++)
                            if (depth_block(k) != 0.f)
                            {
                                new_d += depth_block(k);
                                cont++;
                            }

                        if (cont != 0) {

                            depth_here(v,u) = new_d/float(cont);


                        } else {
                            depth_here(v,u) = 0.f;
                        }

                    }
                }
        }

        //Calculate coordinates "xy" of the points
        const float inv_f_i = 2.f*tan(0.5f*fovh)/float(cols_i);
        const float disp_u_i = 0.5f*(cols_i-1);
        const float disp_v_i = 0.5f*(rows_i-1);

        const ArrayXf v_col = ArrayXf::LinSpaced(rows_i, 0.f, float(rows_i-1));
        for (unsigned int u = 0; u != cols_i; u++)
        {
            yy_here.col(u) = (inv_f_i*(v_col - disp_v_i)*depth_here.col(u).array()).matrix();
            xx_here.col(u) = inv_f_i*(float(u) - disp_u_i)*depth_here.col(u);

        }

    }
}

void StaticFusion::calculateCoord()
{
    validPixels.clear();
    validPixels.reserve(rows_i * cols_i);
    Null.fill(false);

    //Refs
    MatrixXf &depth_inter_ref = depthInterPyr[image_level];
    MatrixXf &xx_inter_ref = xxInterPyr[image_level];
    MatrixXf &yy_inter_ref = yyInterPyr[image_level];
    MatrixXf &intensity_inter_ref = intensityInterPyr[image_level];
    const MatrixXf &depth_ref = depthPyr[image_level];
    const MatrixXf &depth_warped_ref = depthWarpedPyr[image_level];


    for (unsigned int u = 0; u != cols_i; u++)
        for (unsigned int v = 0; v != rows_i; v++)
        {
            if ((depth_ref(v,u) != 0.f) && (depth_warped_ref(v,u) != 0.f))
            {
                depth_inter_ref(v,u) = 0.5f*(depth_ref(v,u) + depth_warped_ref(v,u));
                xx_inter_ref(v,u) = 0.5f*(xxPyr[image_level](v,u) + xxWarpedPyr[image_level](v,u));
                yy_inter_ref(v,u) = 0.5f*(yyPyr[image_level](v,u) + yyWarpedPyr[image_level](v,u));

                if ((u!=0)&&(v!=0)&&(u!=cols_i-1)&&(v!=rows_i-1))
                    validPixels.push_back(make_pair(v,u));
            }
            else
            {
                Null(v,u) = true;
                depth_inter_ref(v,u) = 0.f;
                xx_inter_ref(v,u) = 0.f;
                yy_inter_ref(v,u) = 0.f;
            }

            intensity_inter_ref(v,u) = 0.5f*(intensityPyr[image_level](v,u) + intensityWarpedPyr[image_level](v,u));
        }
}

void StaticFusion::calculateDerivatives()
{
    //Compute weights for the gradients
    MatrixXf rx(rows_i,cols_i), ry(rows_i,cols_i);
    rx.fill(1.f); ry.fill(1.f);

    MatrixXf rx_intensity(rows_i,cols_i), ry_intensity(rows_i, cols_i);
    rx_intensity.fill(1.f); ry_intensity.fill(1.f);

    const MatrixXf &depth_ref = depthInterPyr[image_level];
    const MatrixXf &intensity_ref = intensityInterPyr[image_level];


    const float epsilon_intensity = 1e-6f;
    const float epsilon_depth = 0.005f;

    for (unsigned int u = 0; u < cols_i-1; u++)
        for (unsigned int v = 0; v < rows_i; v++)
            if (Null(v,u) == false)
            {
                rx(v,u) = abs(depth_ref(v,u+1) - depth_ref(v,u)) + epsilon_depth;
                rx_intensity(v,u) = abs(intensity_ref(v,u+1) - intensity_ref(v,u)) + epsilon_intensity;
            }

    for (unsigned int u = 0; u < cols_i; u++)
        for (unsigned int v = 0; v < rows_i-1; v++)
            if (Null(v,u) == false)
            {
                ry(v,u) = abs(depth_ref(v+1,u) - depth_ref(v,u)) + epsilon_depth;
                ry_intensity(v,u) = abs(intensity_ref(v+1,u) - intensity_ref(v,u)) + epsilon_intensity;
            }


    //Spatial derivatives
    for (unsigned int v = 1; v < rows_i-1; v++)
        for (unsigned int u = 1; u < cols_i-1; u++)
            if (Null(v,u) == false)
            {
                dcu(v,u) = (rx_intensity(v,u-1)*(intensity_ref(v,u+1)-intensity_ref(v,u)) + rx_intensity(v,u)*(intensity_ref(v,u) - intensity_ref(v,u-1)))/(rx_intensity(v,u)+rx_intensity(v,u-1));
                ddu(v,u) = (rx(v,u-1)*(depth_ref(v,u+1)-depth_ref(v,u)) + rx(v,u)*(depth_ref(v,u) - depth_ref(v,u-1)))/(rx(v,u)+rx(v,u-1));
                dcv(v,u) = (ry_intensity(v-1,u)*(intensity_ref(v+1,u)-intensity_ref(v,u)) + ry_intensity(v,u)*(intensity_ref(v,u) - intensity_ref(v-1,u)))/(ry_intensity(v,u)+ry_intensity(v-1,u));
                ddv(v,u) = (ry(v-1,u)*(depth_ref(v+1,u)-depth_ref(v,u)) + ry(v,u)*(depth_ref(v,u) - depth_ref(v-1,u)))/(ry(v,u)+ry(v-1,u));
            }

    //Temporal derivative
    dct = intensityPyr[image_level] - intensityWarpedPyr[image_level];
    ddt = depthPyr[image_level] - depthWarpedPyr[image_level];
}

void StaticFusion::computeWeights()
{
    weights_c.assign(0.f);
    weights_d.assign(0.f);

    //Parameters for error_linearization
    const float kduvt_c = 10.f;
    const float kduvt_d = 200.f;

    //Set measurement error
    const float error_m_c = 1.f;
    const float error_m_d = 0.01f;

    for (auto i : validPixels)
    {
        const int &v = i.first;
        const int &u = i.second;
        //Approximate linearization error
        const float error_l_c = kduvt_c*(abs(dct(v,u)) + abs(dcu(v,u)) + abs(dcv(v,u)));
        const float error_l_d = kduvt_d*(abs(ddt(v,u)) + abs(ddu(v,u)) + abs(ddv(v,u)));
        weights_c(v,u) = sqrtf(1.f/(error_m_c + error_l_c));
        weights_d(v,u) = sqrtf(1.f/(error_m_d + error_l_d));
    }

    const float inv_max_c = 1.f/weights_c.maximum();
    weights_c = inv_max_c*weights_c;

    const float inv_max_d = 1.f/weights_d.maximum();
    weights_d = inv_max_d*weights_d;
}


void StaticFusion::solveOdometryAndSegmJoint()
{
    //Prepare segmentation solver
    buildSystemSegm();


    //References and matrices
    const MatrixXi &labels_ref = clusterAllocation[image_level];
    const MatrixXf &depth_inter_ref = depthInterPyr[image_level];
    const MatrixXf &xx_inter_ref = xxInterPyr[image_level];
    const MatrixXf &yy_inter_ref = yyInterPyr[image_level];

    Matrix<float, Dynamic, Dynamic, ColMajor> A(2*validPixels.size(),6);
    Matrix<float, Dynamic, Dynamic, ColMajor> B(2*validPixels.size(),1);
    A.assign(0.f);
    Matrix<float, Dynamic, Dynamic, ColMajor> Aw(2*validPixels.size(),6);
    Matrix<float, Dynamic, Dynamic, ColMajor> Bw(2*validPixels.size(),1);
    Aw.assign(0.f);
    Vector6f Var;
    Var.assign(0.f);


    //Initialize jacobians
    unsigned int cont = 0;
    const float f_inv = float(cols_i)/(2.f*tan(0.5f*fovh));

    for (auto i: validPixels)
    {
        const int &v = i.first;
        const int &u = i.second;

        // Precomputed expressions
        const float d = depth_inter_ref(v,u);
        const float inv_d = 1.f/d;
        const float x = xx_inter_ref(v,u);
        const float y = yy_inter_ref(v,u);

        //                                          Color
        //------------------------------------------------------------------------------------------------
        const float dycomp_c = dcu(v,u)*f_inv*inv_d;
        const float dzcomp_c = dcv(v,u)*f_inv*inv_d;
        const float twc = weights_c(v,u)*k_photometric_res;

        //Fill the matrix A
        A(cont,0) = twc*(-dycomp_c);
        A(cont,1) = twc*(-dzcomp_c);
        A(cont,2) = twc*(dycomp_c*x*inv_d + dzcomp_c*y*inv_d);

        A(cont,3) = twc*(dycomp_c*inv_d*y*x + dzcomp_c*(y*y*inv_d + d));
        A(cont,4) = twc*(-dycomp_c*(x*x*inv_d + d) - dzcomp_c*inv_d*y*x);
        A(cont,5) = twc*(dycomp_c*y - dzcomp_c*x);

        B(cont) = twc*(-dct(v,u));
        cont++;

        //                                          Geometry
        //------------------------------------------------------------------------------------------------
        const float dycomp_d = ddu(v,u)*f_inv*inv_d;
        const float dzcomp_d = ddv(v,u)*f_inv*inv_d;
        const float twd = weights_d(v,u);

        //Fill the matrix A
        A(cont,0) = twd*(-dycomp_d);
        A(cont,1) = twd*(-dzcomp_d);
        A(cont,2) = twd*(1.f + dycomp_d*x*inv_d + dzcomp_d*y*inv_d);


        A(cont,3) = twd*(y + dycomp_d*inv_d*y*x + dzcomp_d*(y*y*inv_d + d));
        A(cont,4) = twd*(-x - dycomp_d*(x*x*inv_d + d) - dzcomp_d*inv_d*y*x);
        A(cont,5) = twd*(dycomp_d*y - dzcomp_d*x);

        B(cont) = twd*(-ddt(v,u));
        cont++;
    }

    MatrixXf AtA, AtB;
    VectorXf res = -B;
    float aver_res = res.cwiseAbs().sumAll() / res.size();


    //Solve iterative reweighted least squares
    //===================================================================
    Vector6f prev_sol = Var;

    //initialize bs:
    //---------------------------------------------------------------------
    //- Always with the same initial value for all the coarse-to-fine levels
    //b_segm = b_prior;

    //- Using the last solution from the prev coarse-to-fine level
    if (level == 0)
        b_segm = b_prior;

    //- Everything static
    // b_segm.fill(1.f);

    //Iterative solver
    //---------------------------------------------------------------------
    for (unsigned int k=1; k<=max_iter_irls; k++)
    {
        //Update the Cauchy parameter
        //const float aver_res = res.cwiseAbs().sumAll() / res.size();
        const float inv_c_Cauchy = 1.f/(kc_Cauchy*aver_res);

        //Compute the new weights (Cauchy and segmentation)
        cont = 0;
        for (auto i:validPixels)
        {
            //static/dynamic weighting
            const int &v = i.first;
            const int &u = i.second;
            const float b_weight = max(0.f, min(1.f, b_segm[labels_ref(v,u)]));

            //Fill the matrix Aw (Color)
            const float res_weight_color = b_weight*sqrtf(1.f/(1.f + square(res(cont)*inv_c_Cauchy)));
            Aw.row(cont) = res_weight_color*A.row(cont);
            Bw(cont) = res_weight_color*B(cont);
            cont++;

            //Fill the matrix Aw (Depth)
            const float res_weight_depth = b_weight*sqrtf(1.f/(1.f + square(res(cont)*inv_c_Cauchy)));
            Aw.row(cont) = res_weight_depth*A.row(cont);
            Bw(cont) = res_weight_depth*B(cont);
            cont++;
        }

        //Solve the linear system of equations using a minimum least squares method
        AtA.multiply_AtA(Aw);
        AtB.multiply_AtB(Aw,Bw);
        Var = AtA.ldlt().solve(AtB);
        //res = A*Var - B;
        res = -B;
        for (unsigned int k = 0; k<6; k++)
            res += Var(k)*A.col(k);

        //Compute residuals (overall and cluster-wise)
        //----------------------------------------------------
        Array<float, NUM_CLUSTERS, 1> aver_res_label; aver_res_label.fill(0.f);
        Array<int, NUM_CLUSTERS, 1> num_pix_label; num_pix_label.fill(1); //To avoid division by zero
        const float aver_res_old = aver_res;

        //Accumulate residual per label and compute average
        for (size_t i = 0; i < validPixels.size(); ++i)
        {
            const pair<int, int> &vu = validPixels[i];
            const int &v = vu.first;
            const int &u = vu.second;
            const float ress_here = abs(res(2*i)) + abs(res(2*i+1));
            aver_res_label(labels_ref(v,u)) += ress_here;

            num_pix_label(labels_ref(v,u))++;
        }

        aver_res = aver_res_label.matrix().sumAll()/float(2*validPixels.size());
        aver_res_label /= (2*num_pix_label).cast<float>();


        //Update segmentation
        //----------------------------------------------------------------------
        solveSegmIteration(aver_res_label, aver_res_old, kc_Cauchy);


        //Check convergence - It is using the old residuals to check convergence, not the very last one.
        const float delta_sol_max = (prev_sol - Var).lpNorm<Infinity>();
        prev_sol = Var;

        if ((delta_sol_max < irls_delta_threshold)||(k == max_iter_irls))
        {
            //printf("Level = %d: last mean_static_res = %f\n", level, static_res / count_static);
            break;
        }
    }

    //                  Save the solution
    //---------------------------------------------------------------
    //est_cov = (1.f/float(2*num_points-6))*AtA.inverse()*res.squaredNorm();
    est_cov = AtA.inverse()*res.squaredNorm();
    filterEstimateAndComputeT(Var);

}

void StaticFusion::computeTransformationFromTwist(Vector6f &twist)
{
    Matrix4f local_mat = Matrix4f::Zero();

    //Compute the rigid transformation associated to the twist
    local_mat(0,1) = -twist(5); local_mat(1,0) = twist(5);
    local_mat(0,2) = twist(4); local_mat(2,0) = -twist(4);
    local_mat(1,2) = -twist(3); local_mat(2,1) = twist(3);
    local_mat(0,3) = twist(0); local_mat(1,3) = twist(1); local_mat(2,3) = twist(2);

    twist_level_odometry = twist;
    T_odometry = local_mat.exp()*T_odometry;

    Matrix4f log_trans = T_odometry.log();
    twist_odometry(0) = log_trans(0,3); twist_odometry(1) = log_trans(1,3); twist_odometry(2) = log_trans(2,3);
    twist_odometry(3) = -log_trans(1,2); twist_odometry(4) = log_trans(0,2); twist_odometry(5) = -log_trans(0,1);
}


void StaticFusion::filterEstimateAndComputeT(Vector6f &twist)
{
    if (use_motion_filter)
    {
        //		Calculate Eigenvalues and Eigenvectors
        //----------------------------------------------------------
        SelfAdjointEigenSolver<MatrixXf> eigensolver(est_cov);
        if (eigensolver.info() != Success)
        {
            printf("Eigensolver couldn't find a solution. Pose is not updated\n");
            return;
        }

        //First, we have to describe both the new linear and angular velocities in the "eigenvector" basis
        //-------------------------------------------------------------------------------------------------
        const Matrix<float,6,6> Bii = eigensolver.eigenvectors();
        const Vector6f kai_b = Bii.colPivHouseholderQr().solve(twist);

        //Second, we have to describe both the old linear and angular velocities in the "eigenvector" basis
        //-------------------------------------------------------------------------------------------------
        Vector6f kai_loc_sub = twist_odometry_old;

        //Important: we have to substract the previous levels' solutions from the old velocity.
        Matrix4f log_trans = T_odometry.log();
        kai_loc_sub(0) -= log_trans(0,3); kai_loc_sub(1) -= log_trans(1,3); kai_loc_sub(2) -= log_trans(2,3);
        kai_loc_sub(3) += log_trans(1,2); kai_loc_sub(4) -= log_trans(0,2); kai_loc_sub(5) += log_trans(0,1);

        //Transform that local representation to the "eigenvector" basis
        const Vector6f kai_b_old = Bii.colPivHouseholderQr().solve(kai_loc_sub);

        //									Filter velocity
        //--------------------------------------------------------------------------------
        const float cf = previous_speed_eig_weight*expf(-int(level)), df = previous_speed_const_weight*expf(-int(level));
        Vector6f kai_b_fil;
        //float max_weight = 0.f;
        for (unsigned int i=0; i<6; i++)
        {
            kai_b_fil(i) = (kai_b(i) + (cf*eigensolver.eigenvalues()(i) + df)*kai_b_old(i))/(1.f + cf*eigensolver.eigenvalues()(i) + df);
//            if (cf*eigensolver.eigenvalues()(i,0) > max_weight)
//                max_weight = cf*eigensolver.eigenvalues()(i,0);
        }

        twist = Bii.inverse().colPivHouseholderQr().solve(kai_b_fil);
    }

    //Compute the rigid transformation associated to the twist
    Matrix4f local_mat = Matrix4f::Zero();
    local_mat(0,1) = -twist(5); local_mat(1,0) = twist(5);
    local_mat(0,2) = twist(4); local_mat(2,0) = -twist(4);
    local_mat(1,2) = -twist(3); local_mat(2,1) = twist(3);
    local_mat(0,3) = twist(0); local_mat(1,3) = twist(1); local_mat(2,3) = twist(2);

    twist_level_odometry = twist;
    T_odometry = local_mat.exp()*T_odometry;


    Matrix4f log_trans = T_odometry.log();
    twist_odometry(0) = log_trans(0,3); twist_odometry(1) = log_trans(1,3); twist_odometry(2) = log_trans(2,3);
    twist_odometry(3) = -log_trans(1,2); twist_odometry(4) = log_trans(0,2); twist_odometry(5) = -log_trans(0,1);
}


void StaticFusion::warpImagesAccurateInverse()
{
    //Camera parameters (which also depend on the level resolution)
    const float f = float(cols_i)/(2.f*tan(0.5f*fovh));
    const float disp_u_i = 0.5f*float(cols_i-1);
    const float disp_v_i = 0.5f*float(rows_i-1);

    //Refs
    MatrixXf &depth_warped_ref = depthWarpedPyr[image_level];
    MatrixXf &intensity_warped_ref = intensityWarpedPyr[image_level];
    MatrixXf &xx_warped_ref = xxWarpedPyr[image_level];
    MatrixXf &yy_warped_ref = yyWarpedPyr[image_level];
    const MatrixXf &depth_ref = depthPredPyr[image_level];
    const MatrixXf &intensity_ref = intensityPredPyr[image_level];
    const MatrixXf &xx_ref = xxPredPyr[image_level];
    const MatrixXf &yy_ref = yyPredPyr[image_level];
    depth_warped_ref.assign(0.f);
    intensity_warped_ref.assign(0.f);

    //Aux variables
    MatrixXf wacu(rows_i,cols_i); wacu.assign(0.f);
    const int cols_lim = 100*(cols_i-1);
    const int rows_lim = 100*(rows_i-1);

    //Inverse warping here (bringing the old one towards the new one)
    const Matrix4f T = T_odometry.inverse();

    //						Warping loop
    //---------------------------------------------------------
    for (unsigned int j = 0; j<cols_i; j++)
        for (unsigned int i = 0; i<rows_i; i++)
        {
            const float z = depth_ref(i,j);

            if (z != 0.f)
            {
                //Transform point to the warped reference frame
                const float intensity_w = intensity_ref(i,j);

                const float x_w = T(0,0)*xx_ref(i,j)+ T(0,1)*yy_ref(i,j) + T(0,2)*z + T(0,3);
                const float y_w = T(1,0)*xx_ref(i,j)+ T(1,1)*yy_ref(i,j) + T(1,2)*z + T(1,3);
                const float depth_w = T(2,0)*xx_ref(i,j)+ T(2,1)*yy_ref(i,j) + T(2,2)*z + T(2,3);

                //Calculate warping
                const int uwarp = int(100.f*(f*x_w/depth_w + disp_u_i));
                const int vwarp = int(100.f*(f*y_w/depth_w + disp_v_i));

                //The projection after transforming is not integer in general and, hence, the pixel contributes to all the surrounding ones
                if (( uwarp >= 0)&&( uwarp < cols_lim)&&( vwarp >= 0)&&( vwarp < rows_lim))
                {
                    const int uwarp_l = uwarp - uwarp % 100;
                    const int uwarp_r = uwarp_l + 100;
                    const int vwarp_d = vwarp - vwarp % 100;
                    const int vwarp_u = vwarp_d + 100;
                    const int delta_r = uwarp_r - uwarp;
                    const int delta_l = 100 - delta_r; //uwarp - float(uwarp_l);
                    const int delta_u = vwarp_u - vwarp;
                    const int delta_d =  100 - delta_u; //vwarp - float(vwarp_d);

                    //Warped pixel very close to an integer value
                    if (min(delta_r, delta_l) + min(delta_u, delta_d) < 5)
                    {
                        const int ind_u = delta_r > delta_l ? uwarp_l/100 : uwarp_r/100;
                        const int ind_v = delta_u > delta_d ? vwarp_d/100 : vwarp_u/100;

                        depth_warped_ref(ind_v,ind_u) += 200.f*depth_w;
                        intensity_warped_ref(ind_v,ind_u) += 200.f*intensity_w;
                        wacu(ind_v,ind_u) += 200;
                    }
                    else
                    {
                        const int v_d = vwarp_d/100, u_l = uwarp_l/100;
                        const int v_u = v_d + 1, u_r = u_l + 1;

                        const int w_ur = delta_l + delta_d; 	//const float w_ur = square(delta_l) + square(delta_d);
                        depth_warped_ref(v_u,u_r) += w_ur*depth_w;
                        intensity_warped_ref(v_u,u_r) += w_ur*intensity_w;
                        wacu(v_u,u_r) += w_ur;

                        const int w_ul = delta_r + delta_d; //const float w_ul = square(delta_r) + square(delta_d);
                        depth_warped_ref(v_u,u_l) += w_ul*depth_w;
                        intensity_warped_ref(v_u,u_l) += w_ul*intensity_w;
                        wacu(v_u,u_l) += w_ul;

                        const int w_dr = delta_l + delta_u; //const float w_dr = square(delta_l) + square(delta_u);
                        depth_warped_ref(v_d,u_r) += w_dr*depth_w;
                        intensity_warped_ref(v_d,u_r) += w_dr*intensity_w;
                        wacu(v_d,u_r) += w_dr;

                        const int w_dl = delta_r + delta_u; //const float w_dr = square(delta_l) + square(delta_u);
                        depth_warped_ref(v_d,u_l) += w_dl*depth_w;
                        intensity_warped_ref(v_d,u_l) += w_dl*intensity_w;
                        wacu(v_d,u_l) += w_dl;
                    }
                }
            }
        }

    //Scale the averaged depth and compute spatial coordinates
    const float inv_f_i = 1.f/f;
    for (unsigned int u = 0; u<cols_i; u++)
        for (unsigned int v = 0; v<rows_i; v++)
        {
            if (wacu(v,u) != 0)
            {
                intensity_warped_ref(v,u) /= float(wacu(v,u));
                depth_warped_ref(v,u) /= float(wacu(v,u));

                xx_warped_ref(v,u) = (u - disp_u_i)*depth_warped_ref(v,u)*inv_f_i;
                yy_warped_ref(v,u) = (v - disp_v_i)*depth_warped_ref(v,u)*inv_f_i;
            }
            else
            {
                xx_warped_ref(v,u) = 0.f;
                yy_warped_ref(v,u) = 0.f;
            }
        }
}



void StaticFusion::computeResidualsAgainstPreviousImage(int index) {

    int idx_to_warp = (index - bufferLength) % bufferLength;
    int trans_start = (index - bufferLength + 1);

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

    for (int i=trans_start; i <index; i++) {
        int idx = i % bufferLength;
        T = T * odomBuffer[idx];
    }

    T = T * T_odometry;
    T = T.inverse().eval();

    int valid_pixels = 0;

    //Inverse warping here (bringing the old one towards the new one)
   // const Matrix4f T = T_odometry.inverse();

    //first need to compute xx, yy
    const float inv_f_i = 2.f*tan(0.5f*fovh)/float(cols);
    const float disp_u_i = 0.5f*(cols-1);
    const float disp_v_i = 0.5f*(rows-1);

    const ArrayXf v_col = ArrayXf::LinSpaced(rows, 0.f, float(rows-1));

    for (unsigned int u = 0; u != cols; u++)
    {
        yyBuffer.col(u) = (inv_f_i*(v_col - disp_v_i)*depthBuffer[idx_to_warp].col(u).array()).matrix();
        xxBuffer.col(u) = inv_f_i*(float(u) - disp_u_i)*depthBuffer[idx_to_warp].col(u);
    }

    //now, warp previous image towards new
    //Camera parameters (which also depend on the level resolution)
    const float f = float(cols)/(2.f*tan(0.5f*fovh));

    //Refs
    depthWarpedRefference = Eigen::MatrixXf::Zero(rows, cols);
    intensityWarpedRefference = Eigen::MatrixXf::Zero(rows, cols);

    MatrixXf intensity_diff = intensityCurrent.replicate(1,1);
    const MatrixXi labels_ref = clusterAllocation[0].replicate(1, 1);

    //Aux variables
    MatrixXf wacu(rows,cols); wacu.assign(0.f);
    const int cols_lim = 100*(cols-1);
    const int rows_lim = 100*(rows-1);



//    //						Warping loop
//    //---------------------------------------------------------
    for (unsigned int j = 0; j<cols; j++)
        for (unsigned int i = 0; i<rows; i++)
        {
            const float z = depthBuffer[idx_to_warp](i,j);

            if (z != 0.f && depthCurrent(i, j) != 0.f)
            {
                valid_pixels++;

                //Transform point to the warped reference frame
                const float intensity_w = intensityBuffer[idx_to_warp](i,j);

                const float x_w = T(0,0)*xxBuffer(i,j) + T(0,1)*yyBuffer(i,j) + T(0,2)*z + T(0,3);
                const float y_w = T(1,0)*xxBuffer(i,j) + T(1,1)*yyBuffer(i,j) + T(1,2)*z + T(1,3);
                const float depth_w = T(2,0)*xxBuffer(i,j) + T(2,1)*yyBuffer(i,j) + T(2,2)*z + T(2,3);

                //Calculate warping
                const int uwarp = int(100.f*(f*x_w/depth_w + disp_u_i));
                const int vwarp = int(100.f*(f*y_w/depth_w + disp_v_i));

                //The projection after transforming is not integer in general and, hence, the pixel contributes to all the surrounding ones
                if (( uwarp >= 0)&&( uwarp < cols_lim)&&( vwarp >= 0)&&( vwarp < rows_lim))
                {
                    const int uwarp_l = uwarp - uwarp % 100;
                    const int uwarp_r = uwarp_l + 100;
                    const int vwarp_d = vwarp - vwarp % 100;
                    const int vwarp_u = vwarp_d + 100;
                    const int delta_r = uwarp_r - uwarp;
                    const int delta_l = 100 - delta_r; //uwarp - float(uwarp_l);
                    const int delta_u = vwarp_u - vwarp;
                    const int delta_d =  100 - delta_u; //vwarp - float(vwarp_d);

                    //Warped pixel very close to an integer value
                    if (min(delta_r, delta_l) + min(delta_u, delta_d) < 5)
                    {
                        const int ind_u = delta_r > delta_l ? uwarp_l/100 : uwarp_r/100;
                        const int ind_v = delta_u > delta_d ? vwarp_d/100 : vwarp_u/100;

                        depthWarpedRefference(ind_v,ind_u) += 200.f*depth_w;
                        intensityWarpedRefference(ind_v,ind_u) += 200.f*intensity_w;


                        wacu(ind_v,ind_u) += 200;

                    }
                    else
                    {
                        const int v_d = vwarp_d/100, u_l = uwarp_l/100;
                        const int v_u = v_d + 1, u_r = u_l + 1;

                        const int w_ur = delta_l + delta_d; 	//const float w_ur = square(delta_l) + square(delta_d);
                        depthWarpedRefference(v_u,u_r) += w_ur*depth_w;
                        intensityWarpedRefference(v_u,u_r) += w_ur*intensity_w;
                        wacu(v_u,u_r) += w_ur;

                        const int w_ul = delta_r + delta_d; //const float w_ul = square(delta_r) + square(delta_d);
                        depthWarpedRefference(v_u,u_l) += w_ul*depth_w;
                        intensityWarpedRefference(v_u,u_l) += w_ul*intensity_w;
                        wacu(v_u,u_l) += w_ul;

                        const int w_dr = delta_l + delta_u; //const float w_dr = square(delta_l) + square(delta_u);
                        depthWarpedRefference(v_d,u_r) += w_dr*depth_w;
                        intensityWarpedRefference(v_d,u_r) += w_dr*intensity_w;
                        wacu(v_d,u_r) += w_dr;

                        const int w_dl = delta_r + delta_u; //const float w_dr = square(delta_l) + square(delta_u);
                        depthWarpedRefference(v_d,u_l) += w_dl*depth_w;
                        intensityWarpedRefference(v_d,u_l) += w_dl*intensity_w;
                        wacu(v_d,u_l) += w_dl;
                    }

                }

            } else  {
                intensity_diff(i, j) = 0;
            }
        }

    //Scale the averaged depth
    for (unsigned int u = 0; u<cols; u++)
        for (unsigned int v = 0; v<rows; v++)
        {
            if (wacu(v,u) != 0)
            {

                intensityWarpedRefference(v,u) /= float(wacu(v,u));
                depthWarpedRefference(v,u) /= float(wacu(v,u));
            }
        }

    //Compute residuals (overall and cluster-wise)
    //----------------------------------------------------
    depthResiduals = depthCurrent - depthWarpedRefference;
    intensityResiduals = intensity_diff - intensityWarpedRefference;
    cumulativeResiduals = depthResiduals.cwiseAbs() + k_photometric_res * intensityResiduals.cwiseAbs();

    perClusterAverageResidual.fill(std::numeric_limits<float>::quiet_NaN());
    Array<int, NUM_CLUSTERS, 1> num_pix_label; num_pix_label.fill(1); //To avoid division by zero

    for (unsigned int j = 0; j<cols; j++)
        for (unsigned int i = 0; i<rows; i++)
        {

            if (depthWarpedRefference(i, j) != 0 && depthCurrent(i, j) !=0) {

                if (std::isnan(perClusterAverageResidual(labels_ref(i, j))) ) {

                    perClusterAverageResidual(labels_ref(i, j)) = cumulativeResiduals(i, j);

                } else {
                    perClusterAverageResidual(labels_ref(i, j)) += cumulativeResiduals(i, j);
                }

                num_pix_label(labels_ref(i, j))++;
            }


        }

    perClusterAverageResidual /= (2*num_pix_label).cast<float>();
}

void StaticFusion::runSolver(bool create_image_pyr)
{
    CTicTac clock; clock.Tic();

    //Create the image pyramid if it has not been computed yet (now pyramids for depth, colour and cofidence)
    //--------------------------------------------------------------------------------------------------------
    if (create_image_pyr)
        createImagePyramid(false);

    //Create labels
    //----------------------------------------------------------------------------------
    //Kmeans
    kMeans3DCoord();

    //Create the pyramid for the labels
    createClustersPyramidUsingKMeans();

    //Solve a robust odometry problem to segment the background (coarse-to-fine)
    //---------------------------------------------------------------------------------
    //Initialize the overall transformations to 0
    T_odometry.setIdentity();

    //Coarse-to-fine
    for (unsigned int i=0; i<ctf_levels; i++)
        for (unsigned int k=0; k<max_iter_per_level; k++)
        {
            level = i;
            unsigned int s = pow(2.f,int(ctf_levels-(i+1)));
            cols_i = cols/s; rows_i = rows/s;
            image_level = ctf_levels - i + round(log2(width/cols)) - 1;

            //1. Perform warping
            if ((i == 0)&&(k == 0))

            {
                depthWarpedPyr[image_level] = depthPredPyr[image_level];
                intensityWarpedPyr[image_level] = intensityPredPyr[image_level];
                xxWarpedPyr[image_level] = xxPredPyr[image_level];
                yyWarpedPyr[image_level] = yyPredPyr[image_level];
            }
            else
                warpImagesAccurateInverse();

            //2. Compute inter coords (better linearization of the range and optical flow constraints)
            calculateCoord();

            //3. Compute derivatives
            calculateDerivatives();

            //4. Compute weights
            computeWeights();

            //5 Compute warped b_segmentation (necessary for the robust estimation, in principle they will initialize the bs)
            computeSegPrior();

            //6. Solve odometry
            solveOdometryAndSegmJoint();

            //Check convergence of nonlinear iterations
            if (twist_level_odometry.norm() < 0.04f)
                break;
        }

    cam_oldpose = cam_pose;
    math::CMatrixDouble44 aux_acu = T_odometry;
    poses::CPose3D pose_aux(aux_acu); //mrpt representation of local odometry
    cam_pose = cam_pose + pose_aux; //global pose of the camera

    //Transform the local velocity to the new reference frame after motion
    //---------------------------------------------------------------------
    math::CMatrixDouble33 inv_trans;
    pose_aux.getRotationMatrix(inv_trans); //incremental rotation
    twist_odometry_old.topRows<3>() = inv_trans.inverse().cast<float>()*twist_odometry.topRows(3);
    twist_odometry_old.bottomRows<3>() = inv_trans.inverse().cast<float>()*twist_odometry.bottomRows(3);

}

void StaticFusion::updateGUI() {

    if(gui->followPose->Get())
    {
        pangolin::OpenGlMatrix mv;

        Eigen::Matrix4f currPose = reconstruction->getCurrPose();
        Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

        Eigen::Quaternionf currQuat(currRot);
        Eigen::Vector3f forwardVector(0, 0, 1);
        Eigen::Vector3f upVector(0, 1, 0);

        Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
        Eigen::Vector3f up = (currQuat * upVector).normalized();

        Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

        eye -= forward;

        Eigen::Vector3f at = eye + forward;

        Eigen::Vector3f z = (eye - at).normalized();  // Forward
        Eigen::Vector3f x = -1 * up.cross(z).normalized(); // Right
        Eigen::Vector3f y = -1 * z.cross(x);

        Eigen::Matrix4d m;
        m << x(0),  x(1),  x(2),  -(x.dot(eye)),
             y(0),  y(1),  y(2),  -(y.dot(eye)),
             z(0),  z(1),  z(2),  -(z.dot(eye)),
                0,     0,     0,              1;

        memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

        gui->s_cam.SetModelViewMatrix(mv);
    }

    gui->preCall();

    Eigen::Matrix4f pose = reconstruction->getCurrPose();

    if(gui->drawRawCloud->Get() || gui->drawFilteredCloud->Get())
    {
        reconstruction->computeFeedbackBuffers();
    }


    if(gui->drawRawCloud->Get())
    {
        reconstruction->getFeedbackBuffers().at(FeedbackBuffer::RAW)->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
    }


    if(gui->drawFilteredCloud->Get())
    {
        reconstruction->getFeedbackBuffers().at(FeedbackBuffer::FILTERED)->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
    }


    if(gui->drawGlobalModel->Get())
    {
        glFinish();


        reconstruction->getGlobalModel().renderPointCloud(gui->s_cam.GetProjectionModelViewMatrix(),
                                                   reconstruction->getConfidenceThreshold(),
                                                   gui->drawUnstable->Get(),
                                                   gui->drawNormals->Get(),
                                                   gui->drawColors->Get(),
                                                   gui->drawPoints->Get(),
                                                   false,
                                                   false,
                                                   true,
                                                   reconstruction->getTick(),
                                                   reconstruction->getTimeDelta());

        glFinish();
    }


    std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > poseGraph = reconstruction->getPoseGraph();
    std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > gtPoseGraph = reconstruction->getGTPoseGraph();

    glColor3f(1, 0, 0);

    gui->drawFrustum(pose);

    if (gtPoseGraph.size()>0) {
        glColor3f(0, 0, 0);

        gui->drawFrustum(gtPoseGraph.at(gtPoseGraph.size() -1 ).second);
    }

    //drawing trajectory
    if(gui->drawTrajectories->Get() && !poseGraph.empty())
    {

    for (int i = 0; i < poseGraph.size()-1; i++) {

        glColor3f(1, 0, 0);

        pangolin::glDrawLine(poseGraph.at(i).second(0, 3), poseGraph.at(i).second(1, 3), poseGraph.at(i).second(2, 3),
                           poseGraph.at(i+1).second(0, 3), poseGraph.at(i+1).second(1, 3), poseGraph.at(i+1).second(2, 3));

        glColor3f(0, 0, 0);


        if (i+1 < gtPoseGraph.size()) {
            pangolin::glDrawLine(gtPoseGraph.at(i).second(0, 3), gtPoseGraph.at(i).second(1, 3), gtPoseGraph.at(i).second(2, 3),
                               gtPoseGraph.at(i+1).second(0, 3), gtPoseGraph.at(i+1).second(1, 3), gtPoseGraph.at(i+1).second(2, 3));
        }

        }
    }


    glColor3f(1, 1, 1);

    reconstruction->normaliseDepth(0.3f, gui->depthCutoff->Get());

    for(std::map<std::string, GPUTexture*>::const_iterator it = reconstruction->getTextures().begin(); it != reconstruction->getTextures().end(); ++it)
    {
        if(it->second->draw)
        {
            gui->displayImg(it->first, it->second);
        }
    }

    reconstruction->getIndexMap().renderDepth(gui->depthCutoff->Get());

    gui->displayImg("ModelImg", reconstruction->getIndexMap().imageTexHighConf());

    gui->displayImg("Model", reconstruction->getIndexMap().drawTex());


    gui->postCall();

    reconstruction->setConfidenceThreshold(gui->confidenceThreshold->Get());
    reconstruction->setDepthCutoff(gui->depthCutoff->Get());

    if(pangolin::Pushed(*gui->save))
    {
        reconstruction->savePly();
    }
}


