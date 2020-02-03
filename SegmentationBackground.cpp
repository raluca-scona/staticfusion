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



void StaticFusion::computeSegPrior()
{
    Eigen::Matrix<int, NUM_CLUSTERS, 1> cluster_size, cluster_nonnull;
    const MatrixXi &labels_ref = clusterAllocation[image_level];
    const MatrixXf &depth_ref = depthPyr[image_level];
    const MatrixXf &depth_warped_ref = depthWarpedPyr[image_level];

    //Initialize counts, priors and temp_reg
    b_prior.fill(0.f);
    cluster_size.fill(0); cluster_nonnull.fill(0);
    lambda_t_w.fill(0.f);

    for (unsigned int u=0; u<cols_i; u++)
        for (unsigned int v=0; v<rows_i; v++)
        {
            const int l = labels_ref(v,u);
            if ( l != NUM_CLUSTERS )
            {
                //B_prior
                if (Null(v,u) == false)
                {
                    cluster_nonnull[l]++;
                    b_prior[l] += 1.f - kz*abs(ddt(v,u));
                }

                //Update size
                cluster_size[labels_ref(v,u)]++;
            }
        }


    for (unsigned int l=0; l<NUM_CLUSTERS; l++)
    {
        if (cluster_size[l] != 0)
        {
            //B_prior
            const float ratio = float(cluster_nonnull[l])/float(cluster_size[l]);

            if (ratio < 0.1f)
            {
                lambda_t_w[l] = 0.1f;
                b_prior[l] = -1.f; //Changed to test
            }
            else
            {
                lambda_t_w[l] = ratio;
                b_prior[l] = max(-1.f, min(2.f, b_prior[l]/cluster_nonnull[l]));
            }
        }
    }
}

void StaticFusion::buildSystemSegm()
{
    //Find the number of connections between clusters (for the reg term)
    unsigned int num_connections = 0;
    for (unsigned int l=0; l<NUM_CLUSTERS; l++)
        for (unsigned int lc=l+1; lc<NUM_CLUSTERS; lc++)
            if (connectivity[l][lc])
                num_connections++;

    A_seg.resize(NUM_CLUSTERS + num_connections, NUM_CLUSTERS);
    B_seg.resize(NUM_CLUSTERS + num_connections);
    A_seg.fill(0.f); B_seg.fill(0.f);


    //Spatial regularization
    unsigned int cont_reg = 0;
    for (unsigned int l=0; l<NUM_CLUSTERS; l++)
        for (unsigned int lc=l+1; lc<NUM_CLUSTERS; lc++)
            if (connectivity[l][lc] == true)
            {
                const float weight_reg = 2.f*lambda_reg;
                A_seg(NUM_CLUSTERS + cont_reg, l) = weight_reg;
                A_seg(NUM_CLUSTERS + cont_reg, lc) = -weight_reg;
                cont_reg++;
            }
}


void StaticFusion::solveSegmIteration(const Array<float, NUM_CLUSTERS, 1> &aver_res, float aver_res_overall, float kc_Cauchy)
{

    //Truncate aver res to avoid getting dynamic parts always
    const float repr_res = max(0.001f, aver_res_overall);


    //Change A and B (A is actually constant...)
    //----------------------------------------------------------------
    //Data term + temporal regularization
    const float fixed_term = log(1.f + square(kb*repr_res/(kc_Cauchy*aver_res_overall)));
    const float mult_res = 1.f/(kc_Cauchy*aver_res_overall);
    for (unsigned int l=0; l<NUM_CLUSTERS; l++)
    {
        //If this clusters have enough useful residuals (more than 10% of the overall pix in the cluster)
        if (lambda_t_w[l] > 0.1f)
        {
            const float dataterm = fixed_term - log(1.f + square(aver_res[l]*mult_res));  //c^2/2 ignored
            A_seg(l,l) = 2.f*lambda_t_w[l]*lambda_prior;
            B_seg(l) = dataterm + 2.f*lambda_prior*lambda_t_w[l]*b_prior[l];
        }
        //Otherwise impose trivial (and soft) constrain, just to avoid changing the size of the system
        else
        {
            A_seg(l,l) = 2.f*lambda_t_w[l];
            B_seg(l) = 2.f*lambda_t_w[l]*b_prior[l]; //I set bT to -1.f (moving)
        }
    }


    //Build AtA and AtB
    AtA_seg.multiply_AtA(A_seg);
    AtB_seg.multiply_AtB(A_seg,B_seg);

    //Solve and constrain to [-1,2] (I permit more than [0,1] to help regularizers have a stronger effect)
    b_segm = AtA_seg.ldlt().solve(AtB_seg);


    for (unsigned int l=0; l<NUM_CLUSTERS; l++)
        b_segm[l] = max(-1.f, min(2.f, b_segm[l]));

}

void StaticFusion::buildSegmImage()
{

    const MatrixXi &labels_maxres = clusterAllocation[0];
    for (unsigned int u=0; u<cols; u++)
        for (unsigned int v=0; v<rows; v++) {
            if (labels_maxres(v,u)==NUM_CLUSTERS) {
                // assume static for invalid cluster
                b_segm_perpixel(v,u) = 1;
                continue;
            }

            b_segm_perpixel(v,u) = max(0.f, min(1.f, b_segm[labels_maxres(v,u)]));

            if ( perClusterAverageResidual(labels_maxres(v, u)) < 0.017) {

                b_segm_perpixel(v,u) = max(b_segm_perpixel(v,u), 1.0f - b_segm_perpixel(v,u));

            }

        }
}


