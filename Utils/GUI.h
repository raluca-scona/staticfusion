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

#ifndef GUI_H_
#define GUI_H_

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>

#include <pangolin/display/display.h>
#include <map>
#include <Utils/GPUTexture.h>
#include <Utils/Intrinsics.h>
#include <Shaders/Shaders.h>


#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

class GUI
{
    public:
        GUI(bool liveCap, bool showcaseMode)
         : showcaseMode(showcaseMode)
        {
            width = 1280;
            height = 980;
            panel = 205;

            pangolin::Params windowParams;

            windowParams.Set("SAMPLE_BUFFERS", 0);
            windowParams.Set("SAMPLES", 0);

            pangolin::CreateWindowAndBind("StaticFusion", width, height, windowParams);

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);

            //Internally render at 3840x2160
            renderBuffer = new pangolin::GlRenderBuffer(3840, 2160),
            colorTexture = new GPUTexture(renderBuffer->width, renderBuffer->height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true);

            colorFrameBuffer = new pangolin::GlFramebuffer;
            colorFrameBuffer->AttachColour(*colorTexture->texture);
            colorFrameBuffer->AttachDepth(*renderBuffer);

            colorProgram = std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface_phong.frag", "draw_global_surface.geom"));

            pangolin::SetFullscreen(showcaseMode);

            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);

            s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640/2, 480/2, 420/2, 420/2, 320/2, 240/2, 0.1, 1000),
                                                pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));

            pangolin::Display("cam").SetBounds(0, 1.0f, 0, 1.0f, -640 / 480.0)
                                    .SetHandler(new pangolin::Handler3D(s_cam));

            pangolin::Display(GPUTexture::RGB).SetAspect(640.0f / 480.0f);
            pangolin::Display(GPUTexture::DEPTH_NORM).SetAspect(640.0f / 480.0f);
            pangolin::Display(GPUTexture::WEIGHT_VIS).SetAspect(640.0f / 480.0f);
            pangolin::Display(GPUTexture::LABELS).SetAspect(640.0f / 480.0f);


            pangolin::Display("ModelImg").SetAspect(640.0f / 480.0f);
            pangolin::Display("Model").SetAspect(640.0f / 480.0f);


            if(!showcaseMode)
            {
                pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panel));
                pangolin::Display("multi").SetBounds(pangolin::Attach::Pix(0), 1 / 4.0f, showcaseMode ? 0 : pangolin::Attach::Pix(180), 1.0)
                                          .SetLayout(pangolin::LayoutEqualHorizontal)
                                          .AddDisplay(pangolin::Display(GPUTexture::RGB))
                                          .AddDisplay(pangolin::Display(GPUTexture::DEPTH_NORM))
                                          .AddDisplay(pangolin::Display("ModelImg"))
                                          .AddDisplay(pangolin::Display("Model"))
                                          .AddDisplay(pangolin::Display(GPUTexture::WEIGHT_VIS))
                                          .AddDisplay(pangolin::Display(GPUTexture::LABELS))
                                          ;
            }

            pause = new pangolin::Var<bool>("ui.Stop", false, false);
            save = new pangolin::Var<bool>("ui.Save", false, false);

            if(liveCap)
            {
                autoSettings = new pangolin::Var<bool>("ui.Auto Settings", true, true);
            }
            else
            {
                autoSettings = 0;
            }

            confidenceThreshold = new pangolin::Var<float>("ui.Confidence threshold", 0.0, 0.0, 1.0);
            depthCutoff = new pangolin::Var<float>("ui.Depth cutoff", 3.0, 0.0, 12.0);

            followPose = new pangolin::Var<bool>("ui.Follow pose", true, true);
            drawRawCloud = new pangolin::Var<bool>("ui.Draw raw", false, true);
            drawFilteredCloud = new pangolin::Var<bool>("ui.Draw probability image", false, true);
            drawGlobalModel = new pangolin::Var<bool>("ui.Draw global model", true, true);
            drawUnstable = new pangolin::Var<bool>("ui.Draw unstable points", false, true);
            drawConf = new pangolin::Var<bool>("ui.Draw confidence", false, true);
            drawPoints = new pangolin::Var<bool>("ui.Draw points", false, true);
            drawColors = new pangolin::Var<bool>("ui.Draw colors", true, true);
            drawNormals = new pangolin::Var<bool>("ui.Draw normals", false, true);
            drawTrajectories = new pangolin::Var<bool>("ui.Draw trajectories", true, true);
        }

        virtual ~GUI()
        {
            if(autoSettings)
            {
                delete autoSettings;

            }
            delete save;
            delete pause;
            delete confidenceThreshold;
            delete depthCutoff;
            delete followPose;
            delete drawRawCloud;
            delete drawFilteredCloud;
            delete drawNormals;
            delete drawTrajectories;
            delete drawColors;
            delete drawGlobalModel;
            delete drawUnstable;
            delete drawConf;
            delete drawPoints;

            delete renderBuffer;
            delete colorFrameBuffer;
            delete colorTexture;
        }

        void preCall()
        {
            glClearColor(1 * !showcaseMode, 1 * !showcaseMode, 1 * !showcaseMode, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            width = pangolin::DisplayBase().v.w;
            height = pangolin::DisplayBase().v.h;

            pangolin::Display("cam").Activate(s_cam);
        }

        inline void drawFrustum(const Eigen::Matrix4f & pose)
        {
            if(showcaseMode)
                return;

            Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
            K(0, 0) = Intrinsics::getInstance().fx();
            K(1, 1) = Intrinsics::getInstance().fy();
            K(0, 2) = Intrinsics::getInstance().cx();
            K(1, 2) = Intrinsics::getInstance().cy();

            Eigen::Matrix3f Kinv = K.inverse();

            GLfloat scale = 0.1f;

            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glMultMatrixf( pose.data() );

            pangolin::glDrawFrustum((GLfloat)Kinv(0,2), (GLfloat)Kinv(1,2), (GLfloat)Kinv(0,0), (GLfloat)Kinv(1,1),
                           Resolution::getInstance().width(), Resolution::getInstance().height(), scale);

            glPopMatrix();

        }

        void displayImg(const std::string & id, GPUTexture * img)
        {
            if(showcaseMode)
                return;

            glDisable(GL_DEPTH_TEST);

            pangolin::Display(id).Activate();
            img->texture->RenderToViewport();

            glEnable(GL_DEPTH_TEST);
        }

        void postCall()
        {
            GLint cur_avail_mem_kb = 0;
            glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

            pangolin::FinishFrame();

            glFinish();
        }

        bool showcaseMode;
        int width;
        int height;
        int panel;

        pangolin::Var<bool> * save,
                            * followPose,
                            * drawRawCloud,
                            * drawFilteredCloud,
                            * drawNormals,
                            * drawTrajectories,
                            * autoSettings,
                            * drawColors,
                            * drawGlobalModel,
                            * drawUnstable,
                            * drawConf,
                            * drawPoints,
                            * pause;

        pangolin::Var<float> * confidenceThreshold,
                             * depthCutoff;

        pangolin::OpenGlRenderState s_cam;

        pangolin::GlRenderBuffer * renderBuffer;
        pangolin::GlFramebuffer * colorFrameBuffer;
        GPUTexture * colorTexture;
        std::shared_ptr<Shader> colorProgram;
};


#endif /* GUI_H_ */
