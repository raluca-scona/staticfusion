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

#include "FillIn.h"

FillIn::FillIn()
 : imageTextureFirstPass(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                GL_RGBA,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                false),
   imageTextureSecondPass(Resolution::getInstance().width(),
                   Resolution::getInstance().height(),
                   GL_RGBA,
                   GL_RGB,
                   GL_UNSIGNED_BYTE,
                   false),
   vertexTextureFirstPass(Resolution::getInstance().width(),
                 Resolution::getInstance().height(),
                 GL_RGBA32F,
                 GL_LUMINANCE,
                 GL_FLOAT,
                 false),
   vertexTextureSecondPass(Resolution::getInstance().width(),
                 Resolution::getInstance().height(),
                 GL_RGBA32F,
                 GL_LUMINANCE,
                 GL_FLOAT,
                 false),
   depthTexture(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                GL_LUMINANCE32F_ARB,
                GL_LUMINANCE,
                GL_FLOAT,
                false),
   imageProgramFirstPass(loadProgramFromFile("empty.vert", "fill_rgb.frag", "quad.geom")),
   imageRenderBufferFirstPass(Resolution::getInstance().width(), Resolution::getInstance().height()),
   imageRenderBufferSecondPass(Resolution::getInstance().width(), Resolution::getInstance().height()),
   vertexProgramFirstPass(loadProgramFromFile("empty.vert", "fill_vertex.frag", "quad.geom")),
   vertexProgramSecondPass(loadProgramFromFile("empty.vert", "fill_vertex_from_texture.frag", "quad.geom")),
   vertexRenderBufferFirstPass(Resolution::getInstance().width(), Resolution::getInstance().height()),
   vertexRenderBufferSecondPass(Resolution::getInstance().width(), Resolution::getInstance().height()),
   depthRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
   depthProgram(loadProgramFromFile("empty.vert", "extract_depth.frag", "quad.geom"))
{
    imageFrameBufferFirstPass.AttachColour(*imageTextureFirstPass.texture);
    imageFrameBufferFirstPass.AttachDepth(imageRenderBufferFirstPass);

    imageFrameBufferSecondPass.AttachColour(*imageTextureSecondPass.texture);
    imageFrameBufferSecondPass.AttachDepth(imageRenderBufferSecondPass);

    vertexFrameBufferFirstPass.AttachColour(*vertexTextureFirstPass.texture);
    vertexFrameBufferFirstPass.AttachDepth(vertexRenderBufferFirstPass);

    vertexFrameBufferSecondPass.AttachColour(*vertexTextureSecondPass.texture);
    vertexFrameBufferSecondPass.AttachDepth(vertexRenderBufferSecondPass);

    depthFrameBuffer.AttachColour(*depthTexture.texture);
    depthFrameBuffer.AttachDepth(depthRenderBuffer);

}

FillIn::~FillIn()
{

}

void FillIn::imageFirstPass(GPUTexture * existingRgb, GPUTexture * rawRgb, bool passthrough)
{
    imageFrameBufferFirstPass.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, imageRenderBufferFirstPass.width, imageRenderBufferFirstPass.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    imageProgramFirstPass->Bind();

    imageProgramFirstPass->setUniform(Uniform("eSampler", 0));
    imageProgramFirstPass->setUniform(Uniform("rSampler", 1));
    imageProgramFirstPass->setUniform(Uniform("passthrough", (int)passthrough));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, existingRgb->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, rawRgb->texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    imageFrameBufferFirstPass.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    imageProgramFirstPass->Unbind();

    glPopAttrib();

    glFinish();
}


void FillIn::imageSecondPass(GPUTexture * existingRgb, GPUTexture * rawRgb, bool passthrough)
{
    imageFrameBufferSecondPass.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, imageRenderBufferSecondPass.width, imageRenderBufferSecondPass.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    imageProgramFirstPass->Bind();

    imageProgramFirstPass->setUniform(Uniform("eSampler", 0));
    imageProgramFirstPass->setUniform(Uniform("rSampler", 1));
    imageProgramFirstPass->setUniform(Uniform("passthrough", (int)passthrough));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, existingRgb->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, rawRgb->texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    imageFrameBufferSecondPass.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    imageProgramFirstPass->Unbind();

    glPopAttrib();

    glFinish();
}

void FillIn::vertexFirstPass(GPUTexture * existingVertex, GPUTexture * rawDepth, GPUTexture * weightedImage, bool passthrough)
{
    vertexFrameBufferFirstPass.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, vertexRenderBufferFirstPass.width, vertexRenderBufferFirstPass.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vertexProgramFirstPass->Bind();

    vertexProgramFirstPass->setUniform(Uniform("eSampler", 0));
    vertexProgramFirstPass->setUniform(Uniform("rSampler", 1));
    vertexProgramFirstPass->setUniform(Uniform("probIsStaticSamp", 2));
    vertexProgramFirstPass->setUniform(Uniform("passthrough", (int)passthrough));

    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                  Intrinsics::getInstance().cy(),
                  1.0f / Intrinsics::getInstance().fx(),
                  1.0f / Intrinsics::getInstance().fy());

    vertexProgramFirstPass->setUniform(Uniform("cam", cam));
    vertexProgramFirstPass->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    vertexProgramFirstPass->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, existingVertex->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, rawDepth->texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, weightedImage->texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    vertexFrameBufferFirstPass.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    vertexProgramFirstPass->Unbind();

    glPopAttrib();

    glFinish();
}

void FillIn::vertexSecondPass(GPUTexture * existingVertex, GPUTexture * rawDepth, GPUTexture * weightedImage, bool passthrough)
{
    vertexFrameBufferSecondPass.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, vertexRenderBufferSecondPass.width, vertexRenderBufferSecondPass.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vertexProgramSecondPass->Bind();

    vertexProgramSecondPass->setUniform(Uniform("eSampler", 0));
    vertexProgramSecondPass->setUniform(Uniform("rSampler", 1));
    vertexProgramSecondPass->setUniform(Uniform("probIsStaticSamp", 2));
    vertexProgramSecondPass->setUniform(Uniform("passthrough", (int)passthrough));

    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                  Intrinsics::getInstance().cy(),
                  1.0f / Intrinsics::getInstance().fx(),
                  1.0f / Intrinsics::getInstance().fy());

    vertexProgramSecondPass->setUniform(Uniform("cam", cam));
    vertexProgramSecondPass->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    vertexProgramSecondPass->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, existingVertex->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, rawDepth->texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, weightedImage->texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    vertexFrameBufferSecondPass.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    vertexProgramSecondPass->Unbind();

    glPopAttrib();

    glFinish();
}

void FillIn::extractDepthFromPrediction() {

       depthFrameBuffer.Bind();

       glPushAttrib(GL_VIEWPORT_BIT);

       glViewport(0, 0, depthRenderBuffer.width, depthRenderBuffer.height);

       glClearColor(0, 0, 0, 0);

       glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

       depthProgram->Bind();

       depthProgram->setUniform(Uniform("maxDepth", 4.5f));

       glActiveTexture(GL_TEXTURE0);
       glBindTexture(GL_TEXTURE_2D, vertexTextureSecondPass.texture->tid);

       depthProgram->setUniform(Uniform("texVerts", 0));

       glDrawArrays(GL_POINTS, 0, 1);

       depthFrameBuffer.Unbind();

       depthProgram->Unbind();

       glBindTexture(GL_TEXTURE_2D, 0);

       glPopAttrib();

       glFinish();
}
