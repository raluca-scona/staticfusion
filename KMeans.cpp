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


struct IndexAndDistance
{
    int idx;
    float distance;

    bool operator<(const IndexAndDistance &o) const
    {
        return distance < o.distance;
    }
};

void StaticFusion::initializeKMeans()
{
	//Initialization: kmeans are computed at one resolution lower than the max (to speed the process up)
    rows_km = rows/2; cols_km = cols/2;
    image_level_km = round(log2(width/cols_km));
    const MatrixXf &depth_ref = depthPyr[image_level_km];
    MatrixXi &labels_ref = clusterAllocation[image_level_km];
	labels_ref.assign(NUM_CLUSTERS);


	//Initialize from scratch at every iteration
	//-------------------------------------------------------------
	//Create seeds for the k-means by dividing the image domain
	unsigned int u_label[NUM_CLUSTERS], v_label[NUM_CLUSTERS];
	const unsigned int vert_div = ceil(sqrt(NUM_CLUSTERS));
    const float u_div = float(cols_km)/float(NUM_CLUSTERS+1);
    const float v_div = float(rows_km)/float(vert_div+1);
	for (unsigned int i=0; i<NUM_CLUSTERS; i++)
	{
		u_label[i] = round((i + 1)*u_div);
		v_label[i] = round((i%vert_div + 1)*v_div);
	}

	//Compute the coordinates associated to the initial seeds
    for (unsigned int u=0; u<cols_km; u++)
        for (unsigned int v=0; v<rows_km; v++)
			if (depth_ref(v,u) != 0.f)
			{
				unsigned int min_dist = 1000000.f, quad_dist;
				unsigned int ini_label = NUM_CLUSTERS;
				for (unsigned int l=0; l<NUM_CLUSTERS; l++)
					if ((quad_dist = square(v - v_label[l]) + square(u - u_label[l])) < min_dist)
					{
						ini_label = l;
						min_dist = quad_dist;
					}

				labels_ref(v,u) = ini_label;
			}

	//Compute the "center of mass" for each region
	std::vector<float> depth_sorted[NUM_CLUSTERS];

    for (unsigned int u=0; u<cols_km; u++)
        for (unsigned int v=0; v<rows_km; v++)
			if (depth_ref(v,u) != 0.f)
					depth_sorted[labels_ref(v,u)].push_back(depth_ref(v,u));


	//Compute the first KMeans values (using median to avoid getting a floating point between two regions)
    const float inv_f_i = 2.f*tan(0.5f*fovh)/float(cols_km);
    const float disp_u_i = 0.5f*(cols_km-1);
    const float disp_v_i = 0.5f*(rows_km-1);
	for (unsigned int l=0; l<NUM_CLUSTERS; l++)
	{
		const unsigned int size_label = depth_sorted[l].size();
		const unsigned int med_pos = size_label/2;

		if (size_label > 0)
		{
			std::nth_element(depth_sorted[l].begin(), depth_sorted[l].begin() + med_pos, depth_sorted[l].end());
					
			kmeans(0,l) = depth_sorted[l].at(med_pos);
			kmeans(1,l) = (u_label[l]-disp_u_i)*kmeans(0,l)*inv_f_i;
			kmeans(2,l) = (v_label[l]-disp_v_i)*kmeans(0,l)*inv_f_i;
		}
		else
		{
			kmeans.col(l).fill(0.f);
			//printf("label %d is empty from the beginning\n", l);
		}
	}
}

void StaticFusion::kMeans3DCoord()
{
	//Kmeans are computed at one resolution lower than the max (to speed the process up)
	const unsigned int max_level = round(log2(width/cols));
    const unsigned int lower_level = max_level+1;
	const unsigned int iter_kmeans = 10;

	//Refs
    const MatrixXf &depth_ref = depthPyr[lower_level];
    const MatrixXf &xx_ref = xxPyr[lower_level];
    const MatrixXf &yy_ref = yyPyr[lower_level];
	MatrixXi &labels_lowres = clusterAllocation[lower_level];

	//Initialization
	initializeKMeans();


    //                                      Iterate 
    //=======================================================================================
    vector<vector<IndexAndDistance> > cluster_distances(NUM_CLUSTERS, vector<IndexAndDistance>(NUM_CLUSTERS));

    MatrixXf centers_a(3,NUM_CLUSTERS), centers_b(3,NUM_CLUSTERS);
	int count[NUM_CLUSTERS];

	//Fill centers_a (I need to do it in this way to get maximum speed, I don't know why...)
	//centers_a.swap(kmeans);
	for (unsigned int c=0; c<NUM_CLUSTERS; c++)
		for (unsigned int r=0; r<3; r++)
			centers_a(r,c) = kmeans(r,c);

    for (unsigned int i=0; i<iter_kmeans-1; i++)
    {
        centers_b.setZero();

		//Compute and sort distances between the kmeans
        for (unsigned int l=0; l<NUM_CLUSTERS; l++)
        {
            count[l] = 0;
			vector<IndexAndDistance> &distances = cluster_distances.at(l);
            for (unsigned int li=0; li<NUM_CLUSTERS; li++)
            {
                IndexAndDistance &idx_and_distance = distances.at(li);
                idx_and_distance.idx = li;
                idx_and_distance.distance = (centers_a.col(l) - centers_a.col(li)).squaredNorm();
            }
            std::sort(distances.begin(), distances.end());
        }


        //Compute belonging to each label
        for (unsigned int u=0; u<cols_km; u++)
            for (unsigned int v=0; v<rows_km; v++)
                if (depth_ref(v,u) != 0.f)
                {
                    //Initialize
					const int last_label = labels_lowres(v,u);
					int best_label = last_label;
                    vector<IndexAndDistance> &distances = cluster_distances.at(last_label);

                    const Vector3f p(depth_ref(v,u), xx_ref(v,u), yy_ref(v,u));
                    const float distance_to_last_label = (centers_a.col(last_label) - p).squaredNorm();
                    float best_distance = distance_to_last_label;

                    for (size_t li = 1; li < distances.size(); ++li)
                    {
                        const IndexAndDistance &idx_and_distance = distances.at(li);
                        if(idx_and_distance.distance > 4.f*distance_to_last_label) break;

                        const float distance_to_label = (centers_a.col(idx_and_distance.idx) - p).squaredNorm();

                        if(distance_to_label < best_distance)
                        {
                            best_distance = distance_to_label;
                            best_label = idx_and_distance.idx;
                        }
                    }

                    labels_lowres(v,u) = best_label;
                    centers_b.col(best_label) += p;
                    count[best_label] += 1;
                }

        for (unsigned int l=0; l<NUM_CLUSTERS; l++)
            if (count[l] > 0)
				centers_b.col(l) /= count[l];

		//Checking convergence
        const float max_diff = (centers_a - centers_b).lpNorm<Infinity>();
        centers_a.swap(centers_b);

        if (max_diff < 1e-2f) break;
    }

	//Copy solution
	//kmeans.swap(centers_a);
	for (unsigned int c=0; c<NUM_CLUSTERS; c++)
		for (unsigned int r=0; r<3; r++)
			kmeans(r,c) = centers_a(r,c);



    //      Compute the labelling functions at the max resolution (rows,cols)
    //------------------------------------------------------------------------------------
    const MatrixXf &depth_highres = depthPyr[max_level];
    const MatrixXf &xx_highres = xxPyr[max_level];
    const MatrixXf &yy_highres = yyPyr[max_level];
	MatrixXi &labels_ref = clusterAllocation[max_level];

	//Initialize labels
	labels_ref.assign(NUM_CLUSTERS);

    //Update distances between the labels
    for (unsigned int l=0; l<NUM_CLUSTERS; l++)
    {
        vector<IndexAndDistance> &distances = cluster_distances.at(l);
        for (unsigned int li=0; li<NUM_CLUSTERS; li++)
        {
            IndexAndDistance &idx_and_distance = distances.at(li);
            idx_and_distance.idx = li;
            idx_and_distance.distance = (centers_a.col(l) - centers_a.col(li)).squaredNorm();
        }
        std::sort(distances.begin(), distances.end());
    }


    //Find the closest kmean and set the corresponding label to 1
    for (unsigned int u=0; u<cols; u++)
        for (unsigned int v=0; v<rows; v++)
            if (depth_highres(v,u) != 0.f)
            {
                const int label_lowres_here = labels_lowres(v/2,u/2);
				const int last_label = (label_lowres_here == NUM_CLUSTERS) ? 0 : label_lowres_here; //If it was invalid in the low res level initialize it randomly (at 0)

                int best_label = last_label;
                vector<IndexAndDistance> &distances = cluster_distances.at(last_label);
                const Vector3f p(depth_highres(v,u), xx_highres(v,u), yy_highres(v,u));

                const float distance_to_last_label = (centers_a.col(last_label) - p).squaredNorm();
                float best_distance = distance_to_last_label;

                for(size_t li = 1; li < distances.size(); ++li)
                {
                    const IndexAndDistance &idx_and_distance = distances.at(li);
                    if(idx_and_distance.distance > 4.f*distance_to_last_label) break;

                    const float distance_to_label = (centers_a.col(idx_and_distance.idx) - p).squaredNorm();

                    if(distance_to_label < best_distance)
                    {
                        best_distance = distance_to_label;
                        best_label = idx_and_distance.idx;
                    }
                }
                labels_ref(v,u) = best_label;
            }

    //Compute connectivity
    computeRegionConnectivity();
}

void StaticFusion::computeRegionConnectivity()
{
    const unsigned int max_level = round(log2(width/cols));
    const float dist2_threshold = square(0.03f*120.f/float(rows));

	//Refs
	const MatrixXi &labels_ref = clusterAllocation[max_level];
    const MatrixXf &depth_ref = depthPyr[max_level];
    const MatrixXf &xx_ref = xxPyr[max_level];
    const MatrixXf &yy_ref = yyPyr[max_level];

    for (unsigned int i=0; i<NUM_CLUSTERS; i++)
        for (unsigned int j=0; j<NUM_CLUSTERS; j++)
		{
			if (i == j) connectivity[i][j] = true;
			else		connectivity[i][j] = false;
		}

    for (unsigned int u=0; u<cols-1; u++)
        for (unsigned int v=0; v<rows-1; v++)					
            if (depth_ref(v,u) != 0.f)
            {
                //Detect change in the labelling (v+1,u)
                if ((labels_ref(v,u) != labels_ref(v+1,u))&&(labels_ref(v+1,u) != NUM_CLUSTERS))
                {
                    const float disty = square(depth_ref(v,u) - depth_ref(v+1,u)) + square(yy_ref(v,u) - yy_ref(v+1,u));
                    if (disty < dist2_threshold)
                    {
                        connectivity[labels_ref(v,u)][labels_ref(v+1,u)] = true;
                        connectivity[labels_ref(v+1,u)][labels_ref(v,u)] = true;
                    }
                }

                //Detect change in the labelling (v,u+1)
                if ((labels_ref(v,u) != labels_ref(v,u+1))&&(labels_ref(v,u+1) != NUM_CLUSTERS))
                {
                    const float distx = square(depth_ref(v,u) - depth_ref(v,u+1)) + square(xx_ref(v,u) - xx_ref(v,u+1));
                    if (distx < dist2_threshold)
                    {
                        connectivity[labels_ref(v,u)][labels_ref(v,u+1)] = true;
                        connectivity[labels_ref(v,u+1)][labels_ref(v,u)] = true;
                    }
                }
            }
}

void StaticFusion::createClustersPyramidUsingKMeans()
{

	//Compute distance between the kmeans (to improve runtime of the next phase)
	Matrix<float, NUM_CLUSTERS, NUM_CLUSTERS> kmeans_dist;
	for (unsigned int la=0; la<NUM_CLUSTERS; la++)
		for (unsigned int lb=la+1; lb<NUM_CLUSTERS; lb++)
			kmeans_dist(la,lb) = (kmeans.col(la) - kmeans.col(lb)).squaredNorm();
	
	//Generate levels
    for (unsigned int i = 2; i<ctf_levels; i++)
    {
        unsigned int s = pow(2.f,int(i));
        cols_km = cols/s; rows_km = rows/s;
        image_level_km = i + round(log2(width/cols));

		//Refs
        MatrixXi &labels_ref = clusterAllocation[image_level_km];
        const MatrixXf &depth_old_ref = depthPyr[image_level_km];
        const MatrixXf &xx_old_ref = xxPyr[image_level_km];
        const MatrixXf &yy_old_ref = yyPyr[image_level_km];

		labels_ref.assign(NUM_CLUSTERS);
	
		//Compute belonging to each label
        for (unsigned int u=0; u<cols_km; u++)
            for (unsigned int v=0; v<rows_km; v++)
				if (depth_old_ref(v,u) != 0.f)
				{			
					unsigned int label = 0;
					const Vector3f p(depth_old_ref(v,u), xx_old_ref(v,u), yy_old_ref(v,u));
					float min_dist = (kmeans.col(0) - p).squaredNorm();
					float dist_here;

					for (unsigned int l=1; l<NUM_CLUSTERS; l++)
					{
						if (kmeans_dist(label,l) > 4.f*min_dist) continue;

						else if ((dist_here = (kmeans.col(l)-p).squaredNorm()) < min_dist)
						{
							label = l;
							min_dist = dist_here;
						}
					}

					labels_ref(v,u) = label;
				}
	}
}




