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

#version 330 core

layout (location = 0) in vec4 vPos;
layout (location = 1) in vec4 vCol;
layout (location = 2) in vec4 vNormR;

out vec4 vPosition;
out vec4 vColor;
out vec4 vNormRad;
flat out int test;

uniform int time;
uniform float scale;
uniform mat4 t_inv;
uniform vec4 cam; //cx, cy, fx, fy
uniform float cols;
uniform float rows;
uniform float confThreshold;
uniform usampler2D indexSampler;
uniform sampler2D vertConfSampler;
uniform sampler2D colorTimeSampler;
uniform sampler2D normRadSampler;
uniform sampler2D depthSampler;
uniform float maxDepth;
uniform int timeDelta;

void main()
{
    vPosition = vPos;
    vColor = vCol;
    vNormRad = vNormR;
    
    test = 1;

    vec3 localPos = (t_inv * vec4(vPosition.xyz, 1.0f)).xyz;
    
    float x = ((cam.z * localPos.x) / localPos.z) + cam.x;
    float y = ((cam.w * localPos.y) / localPos.z) + cam.y;
    
    vec3 localNorm = normalize(mat3(t_inv) * vNormRad.xyz);

    float indexXStep = (1.0f / (cols * scale)) * 0.5f;
    float indexYStep = (1.0f / (rows * scale)) * 0.5f;

    float windowMultiplier = 2;

    int count = 0;
    int zCount = 0;

    if(time - vColor.w < timeDelta && localPos.z > 0 && x > 0 && y > 0 && x < cols && y < rows)  //if it has been updated recently and it is a valid point
    {
        for(float i = x / cols - (scale * indexXStep * windowMultiplier); i < x / cols + (scale * indexXStep * windowMultiplier); i += indexXStep)
        {
            for(float j = y / rows - (scale * indexYStep * windowMultiplier); j < y / rows + (scale * indexYStep * windowMultiplier); j += indexYStep)
            {
               uint current = uint(textureLod(indexSampler, vec2(i, j), 0));
               
               if(current > 0U)
               {
                   vec4 vertConf = textureLod(vertConfSampler, vec2(i, j), 0);
                   vec4 colorTime = textureLod(colorTimeSampler, vec2(i, j), 0);

                   if(colorTime.z < vColor.z && 
                      vertConf.w > confThreshold &&  
                      vertConf.z > localPos.z &&
                      vertConf.z - localPos.z < 0.01 &&
                      sqrt(dot(vertConf.xy - localPos.xy, vertConf.xy - localPos.xy)) < vNormRad.w * 1.4)  
                   {
                       count++; //removing very similar points
                   }
                   
                   if(colorTime.w == time &&
                      vertConf.w > 0.4 * confThreshold  &&
                      vertConf.z > localPos.z &&
                      vertConf.z - localPos.z > 0.01)
                   {
                       zCount++;
                   }
               }
            }
        }
    }
    
    if(count > 6 || zCount > 5)
    {
        test = 0;
    }
    
    //New unstable point
    if(vColor.w == -2)
    {
        vColor.w = time;
    }


    //Degenerate case or too unstable
    if( (vColor.w == -1 || ((time - vColor.w) > 10 && vPosition.w < 0.5)) || (vPosition.w == 0.0) )
    {
        test = 0;
    }
    
    if(vColor.w > 0 && time - vColor.w > timeDelta)
    {
        test = 1;
    }
    
}

