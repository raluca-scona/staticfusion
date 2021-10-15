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
 
#include "Reconstruction.h"

Reconstruction::Reconstruction(const int timeDelta,
                             const float confidence,
                             const float depthCut,
                             const std::string fileName,
                             const int clusters)
 : timeDelta(timeDelta),
   confidenceThreshold(confidence),
   depthCutoff(depthCut),
   saveFilename(fileName),
   clusters(clusters),
   currPose(Eigen::Matrix4f::Identity()),
   tick(1),
   resize(Resolution::getInstance().width() / 40,
          Resolution::getInstance().height() / 40),
   imageBuff(Resolution::getInstance().rows() / 40, Resolution::getInstance().cols() / 40),
   maxDepthProcessed(20.0f)
{
    poseGraph =  std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> >();
    gtPoseGraph =  std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> >();

    outputFilteredDepth = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC1, 0.0);
    vertexPredict = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC4, 0.0);
    colourImage = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3,  cv::Scalar(0,0,0));

    createTextures();
    createCompute();
    createFeedbackBuffers();

    std::string filename = fileName;
    filename.append(".freiburg");
}

Reconstruction::~Reconstruction()
{

    //Output deformed pose graph
    std::string fname = saveFilename;
    fname.append(".freiburg");

    std::ofstream f;
    f.open(fname.c_str(), std::fstream::out);

    for(size_t i = 0; i < poseGraph.size(); i++)
    {
        std::stringstream strs;


            strs << std::setprecision(6) << std::fixed << (double)poseLogTimes.at(i) / 1000000.0 << " ";


        Eigen::Vector3f trans = poseGraph.at(i).second.topRightCorner(3, 1);
        Eigen::Matrix3f rot = poseGraph.at(i).second.topLeftCorner(3, 3);

        f << strs.str() << trans(0) << " " << trans(1) << " " << trans(2) << " ";

        Eigen::Quaternionf currentCameraRotation(rot);

        f << currentCameraRotation.x() << " " << currentCameraRotation.y() << " " << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";
    }

    f.close();

    for(std::map<std::string, GPUTexture*>::iterator it = textures.begin(); it != textures.end(); ++it)
    {
        delete it->second;
    }

    textures.clear();

    for(std::map<std::string, ComputePack*>::iterator it = computePacks.begin(); it != computePacks.end(); ++it)
    {
        delete it->second;
    }

    computePacks.clear();

    for(std::map<std::string, FeedbackBuffer*>::iterator it = feedbackBuffers.begin(); it != feedbackBuffers.end(); ++it)
    {
        delete it->second;
    }

    feedbackBuffers.clear();
}

void Reconstruction::createTextures()
{
    textures[GPUTexture::RGB] = new GPUTexture(Resolution::getInstance().width(),
                                               Resolution::getInstance().height(),
                                               GL_RGBA,
                                               GL_RGB,
                                               GL_UNSIGNED_BYTE,
                                               true);

    textures[GPUTexture::WEIGHT_VIS] = new GPUTexture(Resolution::getInstance().width(),
                                               Resolution::getInstance().height(),
                                               GL_RGBA,
                                               GL_RGB,
                                               GL_UNSIGNED_BYTE,
                                                  true);

    textures[GPUTexture::LABELS] = new GPUTexture(Resolution::getInstance().width(),
                                               Resolution::getInstance().height(),
                                               GL_RGBA,
                                               GL_RGB,
                                               GL_UNSIGNED_BYTE,
                                                  true);

    textures[GPUTexture::DEPTH_RAW] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_LUMINANCE16UI_EXT,
                                                     GL_LUMINANCE_INTEGER_EXT,
                                                     GL_UNSIGNED_SHORT);


    textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
                                                          Resolution::getInstance().height(),
                                                          GL_LUMINANCE16UI_EXT,
                                                          GL_LUMINANCE_INTEGER_EXT,
                                                          GL_UNSIGNED_SHORT,
                                                          false);

    textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(Resolution::getInstance().width(),
                                                        Resolution::getInstance().height(),
                                                        GL_LUMINANCE32F_ARB,
                                                        GL_LUMINANCE,
                                                        GL_FLOAT);

    textures[GPUTexture::WEIGHT] = new GPUTexture(Resolution::getInstance().width(),
                                                        Resolution::getInstance().height(),
                                                        GL_LUMINANCE32F_ARB,
                                                        GL_LUMINANCE,
                                                        GL_FLOAT);

    textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
                                                                 Resolution::getInstance().height(),
                                                                 GL_LUMINANCE32F_ARB,
                                                                 GL_LUMINANCE,
                                                                 GL_FLOAT);

    textures[GPUTexture::DEPTH_NORM] = new GPUTexture(Resolution::getInstance().width(),
                                                      Resolution::getInstance().height(),
                                                      GL_LUMINANCE,
                                                      GL_LUMINANCE,
                                                      GL_FLOAT,
                                                      true);

   textures[GPUTexture::DEPTH_PRED] =  new GPUTexture(Resolution::getInstance().width(),
                                                       Resolution::getInstance().height(),
                                                       GL_LUMINANCE,
                                                       GL_LUMINANCE,
                                                       GL_FLOAT,
                                                       false);

   textures[GPUTexture::WEIGHT_PRED] =  new GPUTexture(Resolution::getInstance().width(),
                                                        Resolution::getInstance().height(),
                                                        GL_RGBA,
                                                        GL_RGB,
                                                        GL_UNSIGNED_BYTE,
                                                        false);

}

void Reconstruction::createCompute()
{
    computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"),
                                                      textures[GPUTexture::DEPTH_NORM]->texture);

    computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom"),
                                                        textures[GPUTexture::DEPTH_FILTERED]->texture);

    computePacks[ComputePack::METRIC] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
                                                        textures[GPUTexture::DEPTH_METRIC]->texture);

    computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
                                                                 textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);
}

void Reconstruction::createFeedbackBuffers()
{
    feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
    feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));

}

void Reconstruction::computeFeedbackBuffers()
{
    feedbackBuffers[FeedbackBuffer::RAW]->compute(textures[GPUTexture::RGB]->texture,
                                                  textures[GPUTexture::DEPTH_METRIC]->texture,
                                                  tick,
                                                  maxDepthProcessed);

    feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textures[GPUTexture::WEIGHT]->texture,   //apply weighted texture onto filtered input cloud instead of rgb
                                                       textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture,
                                                       tick,
                                                       maxDepthProcessed);
}

bool Reconstruction::denseEnough(const Img<Eigen::Matrix<unsigned char, 3, 1>> & img)
{
    int sum = 0;

    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            sum += img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(0) > 0 &&
                   img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(1) > 0 &&
                   img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(2) > 0;
        }
    }

    return float(sum) / float(img.rows * img.cols) > 0.25f;
}

void Reconstruction::fuseFrame(const unsigned char * rgb,
                                 const unsigned short * depth,
                                 const float * weightedImage,
                                 const int64_t & timestamp,
                                 const Eigen::Matrix4f * inPose,
                                 const Eigen::Matrix4f * gtPose,
                                 const float weightMultiplier)
{
    textures[GPUTexture::RGB]->texture->Upload(rgb, GL_RGB, GL_UNSIGNED_BYTE);
    textures[GPUTexture::WEIGHT]->texture->Upload(weightedImage, GL_LUMINANCE, GL_FLOAT);

    textures[GPUTexture::DEPTH_RAW]->texture->Upload(depth, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);

    filterDepth();
    metriciseDepth();

    computeFeedbackBuffers();

    if(tick == 1)
    {
        if (inPose) {
            currPose = currPose * (*inPose);
        }

        globalModel.initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED], textures[GPUTexture::WEIGHT], currPose);

    } else {

        Eigen::Matrix4f lastPose = currPose;

        currPose = currPose * ( * inPose );

        Eigen::Matrix4f diff = currPose.inverse() * lastPose;

        Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
        Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

        //Weight by velocity
        float weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());
        float largest = 0.15f; //0.01f
        float minWeight = 0.5f;

        if(weighting > largest)
        {
            weighting = largest;
        }

        weighting = std::max(1.0f - (weighting / largest), minWeight) * weightMultiplier;

        indexMap.predictIndices(currPose, tick, globalModel.model(), maxDepthProcessed, timeDelta);

        globalModel.fuse(currPose,
                         tick,
                         textures[GPUTexture::RGB],
                         textures[GPUTexture::DEPTH_METRIC],
                         textures[GPUTexture::DEPTH_METRIC_FILTERED],
                         textures[GPUTexture::WEIGHT],
                         indexMap.indexTex(),
                         indexMap.vertConfTex(),
                         indexMap.colorTimeTex(),
                         indexMap.normalRadTex(),
                         maxDepthProcessed,
                         confidenceThreshold,
                         weighting);

        indexMap.predictIndices(currPose, tick, globalModel.model(), maxDepthProcessed, timeDelta);

        globalModel.clean(currPose,
                          tick,
                          indexMap.indexTex(),
                          indexMap.vertConfTex(),
                          indexMap.colorTimeTex(),
                          indexMap.normalRadTex(),
                          indexMap.depthTex(),
                          confidenceThreshold,
                          timeDelta,
                          maxDepthProcessed);

    }

    poseGraph.push_back(std::pair<unsigned long long int, Eigen::Matrix4f>(tick, currPose));

    if (gtPose) {
        gtPoseGraph.push_back(std::pair<unsigned long long int, Eigen::Matrix4f>(tick, *(gtPose)));
    }

    poseLogTimes.push_back(timestamp);

    tick++;

}

void Reconstruction::metriciseDepth()
{
    std::vector<Uniform> uniforms;

    uniforms.push_back(Uniform("maxD", depthCutoff));

    computePacks[ComputePack::METRIC]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
    computePacks[ComputePack::METRIC_FILTERED]->compute(textures[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);
}

void Reconstruction::filterDepth()
{
    std::vector<Uniform> uniforms;

    uniforms.push_back(Uniform("cols", (float)Resolution::getInstance().cols()));
    uniforms.push_back(Uniform("rows", (float)Resolution::getInstance().rows()));
    uniforms.push_back(Uniform("maxD", depthCutoff));

    computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void Reconstruction::normaliseDepth(const float & minVal, const float & maxVal)
{
    std::vector<Uniform> uniforms;

    uniforms.push_back(Uniform("maxVal", maxVal * 1000.f));
    uniforms.push_back(Uniform("minVal", minVal * 1000.f));

    computePacks[ComputePack::NORM]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void Reconstruction::savePly()
{
    std::string filename = saveFilename;
    filename.append(".ply");

    // Open file
    std::ofstream fs;
    fs.open (filename.c_str ());

    Eigen::Vector4f * mapData = globalModel.downloadMap();

    int validCount = 0;

    for(unsigned int i = 0; i < globalModel.lastCount(); i++)
    {
        Eigen::Vector4f pos = mapData[(i * 3) + 0];

        if(pos[3] > confidenceThreshold)
        {
            validCount++;
        }
    }

    // Write header
    fs << "ply";
    fs << "\nformat " << "binary_little_endian" << " 1.0";

    // Vertices
    fs << "\nelement vertex "<< validCount;
    fs << "\nproperty float x"
          "\nproperty float y"
          "\nproperty float z";

    fs << "\nproperty uchar red"
          "\nproperty uchar green"
          "\nproperty uchar blue";

    fs << "\nproperty float nx"
          "\nproperty float ny"
          "\nproperty float nz";

    fs << "\nproperty float radius";

    fs << "\nend_header\n";

    // Close the file
    fs.close ();

    // Open file in binary appendable
    std::ofstream fpout (filename.c_str (), std::ios::app | std::ios::binary);

    for(unsigned int i = 0; i < globalModel.lastCount(); i++)
    {
        Eigen::Vector4f pos = mapData[(i * 3) + 0];

        if(pos[3] > confidenceThreshold)
        {
            Eigen::Vector4f col = mapData[(i * 3) + 1];
            Eigen::Vector4f nor = mapData[(i * 3) + 2];

            nor[0] *= -1;
            nor[1] *= -1;
            nor[2] *= -1;

            float value;
            memcpy (&value, &pos[0], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &pos[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &pos[2], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            unsigned char r = int(col[0]) >> 16 & 0xFF;
            unsigned char g = int(col[0]) >> 8 & 0xFF;
            unsigned char b = int(col[0]) & 0xFF;

            fpout.write (reinterpret_cast<const char*> (&r), sizeof (unsigned char));
            fpout.write (reinterpret_cast<const char*> (&g), sizeof (unsigned char));
            fpout.write (reinterpret_cast<const char*> (&b), sizeof (unsigned char));

            memcpy (&value, &nor[0], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[2], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[3], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));
        }
    }

    // Close file
    fs.close ();

    delete [] mapData;

    //Output pose graph
    std::string fname = saveFilename;
    fname.append(".freiburg");

    std::ofstream f;
    f.open(fname.c_str(), std::fstream::out);

    for(size_t i = 0; i < poseGraph.size(); i++)
    {
        std::stringstream strs;


        strs << std::setprecision(6) << std::fixed << (double)poseLogTimes.at(i) / 1000000.0 << " ";


        Eigen::Vector3f trans = poseGraph.at(i).second.topRightCorner(3, 1);
        Eigen::Matrix3f rot = poseGraph.at(i).second.topLeftCorner(3, 3);

        f << strs.str() << trans(0) << " " << trans(1) << " " << trans(2) << " ";

        Eigen::Quaternionf currentCameraRotation(rot);

        f << currentCameraRotation.x() << " " << currentCameraRotation.y() << " " << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";
    }

    f.close();

    // save poses
    {
      // convert from left-hand (RDF) to right-hand (RUF) camera coordinate system
      const Eigen::DiagonalMatrix<float, 4> P(Eigen::Vector4f(1, -1, 1, 1));

      // export poses as (time, px, py, pz, qx, qy, qz, qw)
      const std::string pose_filename = saveFilename + ".txt";
      std::ofstream fs;
      fs.open(pose_filename.c_str());
      for (const auto &[i, se3] : poseGraph) {
        const Eigen::Isometry3f T(P.inverse() * se3.matrix() * P);
        fs << poseLogTimes.at(i-1) << " ";
        fs << T.translation().transpose() << " ";
        fs << Eigen::Quaternionf(T.rotation()).coeffs().transpose();
        fs << std::endl;
      }
      fs.close();
    }
}

Eigen::Vector3f Reconstruction::rodrigues2(const Eigen::Matrix3f& matrix)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    if( s < 1e-5 )
    {
        double t;

        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

//Sad times ahead
IndexMap & Reconstruction::getIndexMap()
{
    return indexMap;
}

GlobalModel & Reconstruction::getGlobalModel()
{
    return globalModel;
}

std::map<std::string, GPUTexture*> & Reconstruction::getTextures()
{
    return textures;
}

const float & Reconstruction::getConfidenceThreshold()
{
    return confidenceThreshold;
}

void Reconstruction::setConfidenceThreshold(const float & val)
{
    confidenceThreshold = val;
}

void Reconstruction::setDepthCutoff(const float & val)
{
    depthCutoff = val;
}

const int & Reconstruction::getTick()
{
    return tick;
}

const int & Reconstruction::getTimeDelta()
{
    return timeDelta;
}

void Reconstruction::setTick(const int & val)
{
    tick = val;
}

const float & Reconstruction::getMaxDepthProcessed()
{
    return maxDepthProcessed;
}

const Eigen::Matrix4f & Reconstruction::getCurrPose()
{
    return currPose;
}

std::map<std::string, FeedbackBuffer*> & Reconstruction::getFeedbackBuffers()
{
    return feedbackBuffers;
}

std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > Reconstruction::getPoseGraph() {
    return poseGraph;
}

std::vector<std::pair<unsigned long long int, Eigen::Matrix4f> > Reconstruction::getGTPoseGraph() {
    return gtPoseGraph;
}


void Reconstruction::getCurrentImages(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &intensity_wf, Eigen::MatrixXf &im_r, Eigen::MatrixXf &im_g, Eigen::MatrixXf &im_b) {
    outputFilteredDepth = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC1, 0.0);
    colourImage = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_8UC3,  cv::Scalar(0,0,0));

    const float norm_factor = 1.f/255.f;


    textures[GPUTexture::RGB]->texture->Download(colourImage.data, GL_RGB, GL_UNSIGNED_BYTE);
    textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture->Download( (float *) outputFilteredDepth.data, GL_LUMINANCE, GL_FLOAT);

    cv::cv2eigen(outputFilteredDepth, depth_wf);
    std::vector<cv::Mat> colourPredChannels(3);
    cv::split(colourImage, colourPredChannels);
    cv::cv2eigen(colourPredChannels[0], im_r);
    cv::cv2eigen(colourPredChannels[1], im_g);
    cv::cv2eigen(colourPredChannels[2], im_b);

    im_r = im_r * norm_factor;
    im_g = im_g * norm_factor;
    im_b = im_b * norm_factor;

    intensity_wf = 0.299f* im_r+ 0.587f*im_g+ 0.114f*im_b;
}

void Reconstruction::getPredictedImages(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &intensity_wf) {

    float lowConf = 0.13;

    float highConf = confidenceThreshold;
    Eigen::MatrixXf im_r = Eigen::MatrixXf::Zero(depth_wf.rows(), depth_wf.cols());
    Eigen::MatrixXf im_g = Eigen::MatrixXf::Zero(depth_wf.rows(), depth_wf.cols());
    Eigen::MatrixXf im_b = Eigen::MatrixXf::Zero(depth_wf.rows(), depth_wf.cols());

    //populating indexMap.vertexTexLowConf, imageTexLowConf, etc
    indexMap.combinedPredict(currPose,
                             globalModel.model(),
                             maxDepthProcessed,
                             lowConf,
                             tick,
                             tick,
                             timeDelta,
                             IndexMap::LOW_CONF);

    //populating indexMap.vertexTexHighConf, imageTexHighConf, etc
    indexMap.combinedPredict(currPose,
                             globalModel.model(),
                             maxDepthProcessed,
                             highConf,
                             tick,
                             tick,
                             timeDelta,
                             IndexMap::HIGH_CONF);


    resize.image(indexMap.imageTexLowConf(), imageBuff);
    bool shouldFillInLowConfTex = !denseEnough(imageBuff);
    const float norm_factor = 1.f/255.f;


    if (shouldFillInLowConfTex) {

        fillIn.vertexFirstPass(indexMap.vertexTexLowConf(), textures[GPUTexture::DEPTH_FILTERED], textures[GPUTexture::WEIGHT], false);  //this fills in fillIn.vertextextureFirstPass
        fillIn.vertexSecondPass(indexMap.vertexTexHighConf(), &fillIn.vertexTextureFirstPass, textures[GPUTexture::WEIGHT], false);  //this fills in fillIn.vertexTextureSecondPass

        fillIn.imageFirstPass(indexMap.imageTexLowConf(), textures[GPUTexture::RGB], false);
        fillIn.imageSecondPass(indexMap.imageTexHighConf(), &fillIn.imageTextureFirstPass, false);

        fillIn.imageTextureSecondPass.texture->Download(colourImage.data, GL_RGB, GL_UNSIGNED_BYTE);

        fillIn.extractDepthFromPrediction();

        cv::Mat depthImage1 =  cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC1, 0.0);
        fillIn.depthTexture.texture->Download(depthImage1.data,  GL_LUMINANCE, GL_FLOAT);

        cv::cv2eigen(depthImage1, depth_wf);

        std::vector<cv::Mat> colourPredChannels(3);
        cv::split(colourImage, colourPredChannels);
        cv::cv2eigen(colourPredChannels[0], im_r);
        cv::cv2eigen(colourPredChannels[1], im_g);
        cv::cv2eigen(colourPredChannels[2], im_b);

        im_r = im_r * norm_factor;
        im_g = im_g * norm_factor;
        im_b = im_b * norm_factor;

        intensity_wf = 0.299f* im_r+ 0.587f*im_g+ 0.114f*im_b;

    } else {

        fillIn.vertexSecondPass(indexMap.vertexTexHighConf(), indexMap.vertexTexLowConf(), textures[GPUTexture::WEIGHT], false);  //this fills in fillIn.vertexTextureSecondPass

        fillIn.imageFirstPass(indexMap.imageTexHighConf(), indexMap.imageTexLowConf(), false);

        fillIn.imageTextureFirstPass.texture->Download(colourImage.data, GL_RGB, GL_UNSIGNED_BYTE);

        fillIn.extractDepthFromPrediction();

        cv::Mat depthImage1 =  cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_32FC1, 0.0);
        fillIn.depthTexture.texture->Download(depthImage1.data,  GL_LUMINANCE, GL_FLOAT);

        cv::cv2eigen(depthImage1, depth_wf);

        std::vector<cv::Mat> colourPredChannels(3);
        cv::split(colourImage, colourPredChannels);
        cv::cv2eigen(colourPredChannels[0], im_r);
        cv::cv2eigen(colourPredChannels[1], im_g);
        cv::cv2eigen(colourPredChannels[2], im_b);

        im_r = im_r * norm_factor;
        im_g = im_g * norm_factor;
        im_b = im_b * norm_factor;

        intensity_wf = 0.299f* im_r+ 0.587f*im_g+ 0.114f*im_b;
    }

}

void Reconstruction::getFilteredDepth(cv::Mat depth, Eigen::MatrixXf & depthMat) {
    textures[GPUTexture::DEPTH_RAW]->texture->Upload((unsigned short *) depth.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);

    filterDepth();
    metriciseDepth();

    textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture->Download( (float *) outputFilteredDepth.data, GL_LUMINANCE, GL_FLOAT);

    cv::cv2eigen(outputFilteredDepth, depthMat);

}

void Reconstruction::uploadWeightAndClustersForVisualization(const float * weightedImage, Eigen::MatrixXi labelledImage, const unsigned short * depth){
    std::vector<unsigned char> weightedImageTexture (Resolution::getInstance().width() * Resolution::getInstance().height() * 3, 0.0);

    std::vector<unsigned char> labelledImageTexture (Resolution::getInstance().width() * Resolution::getInstance().height() * 3, 0.0);


    for (int i=0; i< Resolution::getInstance().numPixels(); i++) {
        float weight = weightedImage[i];

            weightedImageTexture[i*3 + 0] = depth[i] ? (unsigned char) ( 255 * (1.0 - weight) ) : 0;
            weightedImageTexture[i*3 + 1] = depth[i] ? (unsigned char) ( 255 * 0) : 0;
            weightedImageTexture[i*3 + 2] = depth[i] ? (unsigned char) ( 255 * weight) : 0;


        int row = i/Resolution::getInstance().width() ;
        int col = i - row*Resolution::getInstance().width();

        labelledImageTexture[i*3 + 0] = (unsigned char) (255 * labelledImage(row, col) / clusters);
        labelledImageTexture[i*3 + 1] = (unsigned char) (255 * labelledImage(row, col) / clusters);
        labelledImageTexture[i*3 + 2] = (unsigned char) (255 * labelledImage(row, col) / clusters);

    }

    textures[GPUTexture::WEIGHT_VIS]->texture->Upload(weightedImageTexture.data(), GL_RGB, GL_UNSIGNED_BYTE);
    textures[GPUTexture::LABELS]->texture->Upload(labelledImageTexture.data(), GL_RGB, GL_UNSIGNED_BYTE);

}

bool Reconstruction::checkIfDenseEnough() {
    indexMap.combinedPredict(currPose,
                             globalModel.model(),
                             maxDepthProcessed,
                             confidenceThreshold,
                             tick,
                             tick,
                             timeDelta,
                             IndexMap::HIGH_CONF);


    resize.image(indexMap.imageTexLowConf(), imageBuff);
    return denseEnough(imageBuff);

}


