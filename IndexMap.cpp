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

#include "IndexMap.h"

const int IndexMap::FACTOR = 4;

IndexMap::IndexMap()
: indexProgram(loadProgramFromFile("index_map.vert", "index_map.frag")),
  indexRenderBuffer(Resolution::getInstance().width() * IndexMap::FACTOR, Resolution::getInstance().height() * IndexMap::FACTOR),
  indexTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
               Resolution::getInstance().height() * IndexMap::FACTOR,
               GL_LUMINANCE32UI_EXT,
               GL_LUMINANCE_INTEGER_EXT,
               GL_UNSIGNED_INT),
  vertConfTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
                  Resolution::getInstance().height() * IndexMap::FACTOR,
                  GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
  colorTimeTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
                   Resolution::getInstance().height() * IndexMap::FACTOR,
                   GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
  normalRadTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
                   Resolution::getInstance().height() * IndexMap::FACTOR,
                   GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
  drawDepthProgram(loadProgramFromFile("empty.vert", "visualise_textures.frag", "quad.geom")),
  drawRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
  drawTexture(Resolution::getInstance().width(),
              Resolution::getInstance().height(),
              GL_RGBA,
              GL_RGB,
              GL_UNSIGNED_BYTE,
              false),
  depthProgram(loadProgramFromFile("splat.vert", "depth_splat.frag")),
  depthRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
  depthTexture(Resolution::getInstance().width(),
               Resolution::getInstance().height(),
               GL_LUMINANCE32F_ARB,
               GL_LUMINANCE,
               GL_FLOAT,
               false),
  combinedProgram(loadProgramFromFile("splat.vert", "combo_splat.frag")),
  combinedRenderBufferLowConf(Resolution::getInstance().width(), Resolution::getInstance().height()),
  combinedRenderBufferHighConf(Resolution::getInstance().width(), Resolution::getInstance().height()),
  imageTextureLowConf(Resolution::getInstance().width(),
               Resolution::getInstance().height(),
               GL_RGBA,
               GL_RGB,
               GL_UNSIGNED_BYTE,
               false),
  imageTextureHighConf(Resolution::getInstance().width(),
               Resolution::getInstance().height(),
               GL_RGBA,
               GL_RGB,
               GL_UNSIGNED_BYTE,
               false),
  vertexTextureLowConf(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false),
  vertexTextureHighConf(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false),
  normalTextureLowConf(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false),
  normalTextureHighConf(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false),
  timeTextureLowConf(Resolution::getInstance().width(),
              Resolution::getInstance().height(),
              GL_LUMINANCE16UI_EXT,
              GL_LUMINANCE_INTEGER_EXT,
              GL_UNSIGNED_SHORT,
              false),
  timeTextureHighConf(Resolution::getInstance().width(),
              Resolution::getInstance().height(),
              GL_LUMINANCE16UI_EXT,
              GL_LUMINANCE_INTEGER_EXT,
              GL_UNSIGNED_SHORT,
              false)

{
   indexFrameBuffer.AttachColour(*indexTexture.texture);
   indexFrameBuffer.AttachColour(*vertConfTexture.texture);
   indexFrameBuffer.AttachColour(*colorTimeTexture.texture);
   indexFrameBuffer.AttachColour(*normalRadTexture.texture);
   indexFrameBuffer.AttachDepth(indexRenderBuffer);

   drawFrameBuffer.AttachColour(*drawTexture.texture);
   drawFrameBuffer.AttachDepth(drawRenderBuffer);

   depthFrameBuffer.AttachColour(*depthTexture.texture);
   depthFrameBuffer.AttachDepth(depthRenderBuffer);

   combinedFrameBufferLowConf.AttachColour(*imageTextureLowConf.texture);
   combinedFrameBufferLowConf.AttachColour(*vertexTextureLowConf.texture);
   combinedFrameBufferLowConf.AttachColour(*normalTextureLowConf.texture);
   combinedFrameBufferLowConf.AttachColour(*timeTextureLowConf.texture);
   combinedFrameBufferLowConf.AttachDepth(combinedRenderBufferLowConf);

   combinedFrameBufferHighConf.AttachColour(*imageTextureHighConf.texture);
   combinedFrameBufferHighConf.AttachColour(*vertexTextureHighConf.texture);
   combinedFrameBufferHighConf.AttachColour(*normalTextureHighConf.texture);
   combinedFrameBufferHighConf.AttachColour(*timeTextureHighConf.texture);
   combinedFrameBufferHighConf.AttachDepth(combinedRenderBufferHighConf);

}

IndexMap::~IndexMap()
{
}

void IndexMap::predictIndices(const Eigen::Matrix4f & pose,
                              const int & time,
                              const std::pair<GLuint, GLuint> & model,
                              const float depthCutoff,
                              const int timeDelta)
{
    indexFrameBuffer.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, indexRenderBuffer.width, indexRenderBuffer.height);

    glClearColor(0, 0, 0, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    indexProgram->Bind();

    Eigen::Matrix4f t_inv = pose.inverse();

    Eigen::Vector4f cam(Intrinsics::getInstance().cx() * IndexMap::FACTOR,
                  Intrinsics::getInstance().cy() * IndexMap::FACTOR,
                  Intrinsics::getInstance().fx() * IndexMap::FACTOR,
                  Intrinsics::getInstance().fy() * IndexMap::FACTOR);

    indexProgram->setUniform(Uniform("t_inv", t_inv));
    indexProgram->setUniform(Uniform("cam", cam));
    indexProgram->setUniform(Uniform("maxDepth", depthCutoff));
    indexProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols() * IndexMap::FACTOR));
    indexProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows() * IndexMap::FACTOR));
    indexProgram->setUniform(Uniform("time", time));
    indexProgram->setUniform(Uniform("timeDelta", timeDelta));

    glBindBuffer(GL_ARRAY_BUFFER, model.first);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

    glDrawTransformFeedback(GL_POINTS, model.second);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    indexFrameBuffer.Unbind();

    indexProgram->Unbind();

    glPopAttrib();

    glFinish();
}

void IndexMap::renderDepth(const float depthCutoff)
{
    drawFrameBuffer.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, drawRenderBuffer.width, drawRenderBuffer.height);

    glClearColor(0, 0, 0, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawDepthProgram->Bind();

    drawDepthProgram->setUniform(Uniform("maxDepth", depthCutoff));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, vertexTextureHighConf.texture->tid);

    drawDepthProgram->setUniform(Uniform("texVerts", 0));

    glDrawArrays(GL_POINTS, 0, 1);

    drawFrameBuffer.Unbind();

    drawDepthProgram->Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glPopAttrib();

    glFinish();
}

void IndexMap::combinedPredict(const Eigen::Matrix4f & pose,
                               const std::pair<GLuint, GLuint> & model,
                               const float depthCutoff,
                               const float confThreshold,
                               const int time,
                               const int maxTime,
                               const int timeDelta,
                               IndexMap::Texture whichTexture)
{
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    if (whichTexture == IndexMap::LOW_CONF) {
        combinedFrameBufferLowConf.Bind();
    } else {
        combinedFrameBufferHighConf.Bind();
    }

    glPushAttrib(GL_VIEWPORT_BIT);

    if (whichTexture == IndexMap::LOW_CONF) {
        glViewport(0, 0, combinedRenderBufferLowConf.width, combinedRenderBufferLowConf.height);
    } else {
        glViewport(0, 0, combinedRenderBufferHighConf.width, combinedRenderBufferHighConf.height);

    }

    glClearColor(0, 0, 0, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    combinedProgram->Bind();

    Eigen::Matrix4f t_inv = pose.inverse();

    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                  Intrinsics::getInstance().cy(),
                  Intrinsics::getInstance().fx(),
                  Intrinsics::getInstance().fy());

    combinedProgram->setUniform(Uniform("t_inv", t_inv));
    combinedProgram->setUniform(Uniform("cam", cam));
    combinedProgram->setUniform(Uniform("maxDepth", depthCutoff));
    combinedProgram->setUniform(Uniform("confThreshold", confThreshold));
    combinedProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    combinedProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
    combinedProgram->setUniform(Uniform("time", time));
    combinedProgram->setUniform(Uniform("maxTime", maxTime));
    combinedProgram->setUniform(Uniform("timeDelta", timeDelta));

    glBindBuffer(GL_ARRAY_BUFFER, model.first);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

    glDrawTransformFeedback(GL_POINTS, model.second);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (whichTexture == IndexMap::LOW_CONF) {
        combinedFrameBufferLowConf.Unbind();
    } else {
        combinedFrameBufferHighConf.Unbind();
    }

    combinedProgram->Unbind();

    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_POINT_SPRITE);

    glPopAttrib();

    glFinish();
}


