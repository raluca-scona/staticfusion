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

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vColor;
layout (location = 2) in vec4 vNormRad;
layout (location = 3) in vec4 initProb;

out vec4 vPosition0;
out vec4 vColor0;
out vec4 vNormRad0;

uniform mat4 pose;

#include "color.glsl"

void main()
{
    //vPosition0 = vPosition;
    vPosition0 = pose * vec4(vPosition.xyz, 1);   //point from current raw 3d depth stream expressed in world coords

    vec3 decodedProb = decodeColor(initProb.x);
   
    vPosition0.w = decodedProb.x; //* 0.3; 

    vColor0 = vColor;
    vColor0.y = 1.0; //hist weight
    vColor0.z = 1; //This sets the vertex's initialisation time

    vNormRad0 = vec4(mat3(pose) * vNormRad.xyz, vNormRad.w);
}
