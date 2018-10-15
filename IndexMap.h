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

#ifndef INDEXMAP_H_
#define INDEXMAP_H_

#include "Shaders/Shaders.h"
#include "Shaders/Uniform.h"
#include "Shaders/Vertex.h"
#include "Utils/GPUTexture.h"
#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"
#include <pangolin/gl/gl.h>
#include <Eigen/LU>

class IndexMap
{
    public:
        IndexMap();
        virtual ~IndexMap();

        //predicts 4x4 vertex map for data association in fusion
        void predictIndices(const Eigen::Matrix4f & pose,
                            const int & time,
                            const std::pair<GLuint, GLuint> & model,
                            const float depthCutoff,
                            const int timeDelta);

        void renderDepth(const float depthCutoff);

        enum Texture
        {
            LOW_CONF,
            HIGH_CONF
        };

        //surface splatting used for frame-to-model alignment
        //whichTexture: 0 means default, 1 means low conf, 2 means high conf;
        void combinedPredict(const Eigen::Matrix4f & pose,
                             const std::pair<GLuint, GLuint> & model,
                             const float depthCutoff,
                             const float confThreshold,
                             const int time,
                             const int maxTime,
                             const int timeDelta,
                             IndexMap::Texture whichTexture);

        GPUTexture * indexTex()
        {
            return &indexTexture;
        }

        GPUTexture * vertConfTex()
        {
            return &vertConfTexture;
        }

        GPUTexture * colorTimeTex()
        {
            return &colorTimeTexture;
        }

        GPUTexture * normalRadTex()
        {
            return &normalRadTexture;
        }

        GPUTexture * drawTex()
        {
            return &drawTexture;
        }

        GPUTexture * depthTex()
        {
            return &depthTexture;
        }

        GPUTexture * imageTexLowConf()
        {
            return &imageTextureLowConf;
        }

        GPUTexture * imageTexHighConf()
        {
            return &imageTextureHighConf;
        }

        GPUTexture * vertexTexLowConf()
        {
            return &vertexTextureLowConf;
        }

        GPUTexture * vertexTexHighConf()
        {
            return &vertexTextureHighConf;
        }

        GPUTexture * normalTexLowConf()
        {
            return &normalTextureLowConf;
        }

        GPUTexture * normalTexHighConf()
        {
            return &normalTextureHighConf;
        }

        static const int FACTOR;

    private:
        std::shared_ptr<Shader> indexProgram;
        pangolin::GlFramebuffer indexFrameBuffer;
        pangolin::GlRenderBuffer indexRenderBuffer;
        GPUTexture indexTexture;
        GPUTexture vertConfTexture;
        GPUTexture colorTimeTexture;
        GPUTexture normalRadTexture;

        std::shared_ptr<Shader> drawDepthProgram;
        pangolin::GlFramebuffer drawFrameBuffer;
        pangolin::GlRenderBuffer drawRenderBuffer;
        GPUTexture drawTexture;

        std::shared_ptr<Shader> depthProgram;
        pangolin::GlFramebuffer depthFrameBuffer;
        pangolin::GlRenderBuffer depthRenderBuffer;
        GPUTexture depthTexture;

        std::shared_ptr<Shader> combinedProgram;

        pangolin::GlFramebuffer combinedFrameBufferLowConf;
        pangolin::GlRenderBuffer combinedRenderBufferLowConf;
        GPUTexture imageTextureLowConf;
        GPUTexture vertexTextureLowConf;
        GPUTexture normalTextureLowConf;
        GPUTexture timeTextureLowConf;

        pangolin::GlFramebuffer combinedFrameBufferHighConf;
        pangolin::GlRenderBuffer combinedRenderBufferHighConf;
        GPUTexture imageTextureHighConf;
        GPUTexture vertexTextureHighConf;
        GPUTexture normalTextureHighConf;
        GPUTexture timeTextureHighConf;

};

#endif /* INDEXMAP_H_ */
