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

#ifndef FILLIN_H_
#define FILLIN_H_

#include "Shaders.h"
#include "Uniform.h"
#include "../Utils/Resolution.h"
#include "../Utils/Intrinsics.h"
#include "../Utils/GPUTexture.h"

class FillIn
{
    public:
        FillIn();
        virtual ~FillIn();

        void imageFirstPass(GPUTexture * existingRgb, GPUTexture * rawRgb, bool passthrough);
        void vertexFirstPass(GPUTexture * existingVertex, GPUTexture * rawDepth, GPUTexture * weights, bool passthrough);

        void vertexSecondPass(GPUTexture * existingVertex, GPUTexture * rawDepth, GPUTexture * weights, bool passthrough);
        void imageSecondPass(GPUTexture * existingRgb, GPUTexture * rawRgb, bool passthrough);

        void extractDepthFromPrediction();

        GPUTexture imageTextureFirstPass;
        GPUTexture imageTextureSecondPass;
        GPUTexture vertexTextureFirstPass;
        GPUTexture vertexTextureSecondPass;

        std::shared_ptr<Shader> imageProgramFirstPass;
        pangolin::GlRenderBuffer imageRenderBufferFirstPass;
        pangolin::GlFramebuffer imageFrameBufferFirstPass;

        pangolin::GlRenderBuffer imageRenderBufferSecondPass;
        pangolin::GlFramebuffer imageFrameBufferSecondPass;

        std::shared_ptr<Shader> vertexProgramFirstPass;
        pangolin::GlRenderBuffer vertexRenderBufferFirstPass;
        pangolin::GlFramebuffer vertexFrameBufferFirstPass;

        std::shared_ptr<Shader> vertexProgramSecondPass;
        pangolin::GlRenderBuffer vertexRenderBufferSecondPass;
        pangolin::GlFramebuffer vertexFrameBufferSecondPass;

        std::shared_ptr<Shader> depthProgram;
        pangolin::GlFramebuffer depthFrameBuffer;
        pangolin::GlRenderBuffer depthRenderBuffer;
        GPUTexture depthTexture;
};

#endif /* FILLIN_H_ */

