/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#ifndef RECONSTRUCTION_H_
#define RECONSTRUCTION_H_

#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"
#include "Utils/GPUTexture.h"
#include "Shaders/Shaders.h"
#include "Shaders/ComputePack.h"
#include "Shaders/FeedbackBuffer.h"
#include "Shaders/FillIn.h"
#include "Shaders/Resize.h"

#include "GlobalModel.h"
#include "IndexMap.h"

#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <vector>


class Reconstruction
{
    public:
        Reconstruction(const int timeDelta = 200,
                      const float confidence = 10,
                      const float depthCut = 3,
                      const std::string fileName = "",
                      const int clusters = 24);

        virtual ~Reconstruction();

        /**
         * Process an rgb/depth map pair
         * @param rgb unsigned char row major order
         * @param depth unsigned short z-depth in millimeters, invalid depths are 0
         * @param weightedImage image of probabilities of dynamic objects
         * @param timestamp nanoseconds (actually only used for the output poses, not important otherwise)
         * @param weightMultiplier optional full frame fusion weight
         * @param bootstrap if true, use inPose as a pose guess rather than replacement
         */
        void fuseFrame(const unsigned char * rgb,
                          const unsigned short * depth,
                          const float * weightedImage,
                          const int64_t & timestamp,
                          const Eigen::Matrix4f * inPose = 0,
                          const Eigen::Matrix4f * gtPose = 0,
                          const float weightMultiplier = 1.f);

        /**
         * Predicts the current view of the scene, updates the [vertex/normal/image]Tex() members
         * of the indexMap class
         */
        void predict(float confidenceThreshold, int whichTexture);

        /**
         * This class contains all of the predicted renders
         * @return reference
         */
        IndexMap & getIndexMap();

        /**
         * This class contains the surfel map
         * @return
         */
        GlobalModel & getGlobalModel();

        /**
         * This class contains the fern keyframe database
         * @return
         */

        /**
         * This is the map of raw input textures (you can display these)
         * @return
         */
        std::map<std::string, GPUTexture*> & getTextures();

        /**
         * The point fusion confidence threshold
         * @return
         */
        const float & getConfidenceThreshold();

        /**
         * Raw data fusion confidence threshold
         * @param val default value is 10, but you can play around with this
         */
        void setConfidenceThreshold(const float & val);

        /**
         * Cut raw depth input off at this point
         * @param val default is 3 meters
         */
        void setDepthCutoff(const float & val);

        /**
         * Get the internal clock value of the fusion process
         * @return monotonically increasing integer value (not real-world time)
         */
        const int & getTick();

        /**
         * Get the time window length for model matching
         * @return
         */
        const int & getTimeDelta();

        /**
         * Cheat the clock, only useful for multisession/log fast forwarding
         * @param val control time itself!
         */
        void setTick(const int & val);

        /**
         * Internal maximum depth processed, this is defaulted to 20 (for rescaling depth buffers)
         * @return
         */
        const float & getMaxDepthProcessed();

        /**
         * The current global camera pose estimate
         * @return SE3 pose
         */
        const Eigen::Matrix4f & getCurrPose();

        /**
         * These are the vertex buffers computed from the raw input data
         * @return can be rendered
         */
        std::map<std::string, FeedbackBuffer*> & getFeedbackBuffers();

        /**
         * Calculate the above for the current frame (only done on the first frame normally)
         */
        void computeFeedbackBuffers();

        /**
         * Saves out a .ply mesh file of the current model
         */
        void savePly();

        void normaliseDepth(const float & minVal, const float & maxVal);

        void getCurrentImages(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &intensity_wf, Eigen::MatrixXf &im_r, Eigen::MatrixXf &im_g, Eigen::MatrixXf &im_b);

        void getFilteredDepth(cv::Mat depth, Eigen::MatrixXf & depthMat);

        void getPredictedImages(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &intensity_wf);

        void uploadWeightAndClustersForVisualization(const float * weightedImage, Eigen::MatrixXi labelledImage, const unsigned short * depth);
        bool checkIfDenseEnough();

        std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > getPoseGraph(); //poseGraph;
        std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > getGTPoseGraph(); //poseGraph;

        //Here be dragons
    private:
        IndexMap indexMap;
        GlobalModel globalModel;
        FillIn fillIn;

        const std::string saveFilename;
        std::map<std::string, GPUTexture*> textures;
        std::map<std::string, ComputePack*> computePacks;
        std::map<std::string, FeedbackBuffer*> feedbackBuffers;

        void createTextures();
        void createCompute();
        void createFeedbackBuffers();

        void filterDepth();
        void metriciseDepth();

        bool denseEnough(const Img<Eigen::Matrix<unsigned char, 3, 1>> & img);

        Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);

        Eigen::Matrix4f currPose;

        int tick;
        const int timeDelta;
        const int clusters;

        Resize resize;

        std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > poseGraph;
        std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > gtPoseGraph;

        std::vector<unsigned long long int> poseLogTimes;

        Img<Eigen::Matrix<unsigned char, 3, 1>> imageBuff;

        int trackingCount;
        const float maxDepthProcessed;

        float confidenceThreshold;

        float depthCutoff;

        cv::Mat outputFilteredDepth, colourImage, vertexPredict;
};

#endif /* RECONSTRUCTION_H_ */
