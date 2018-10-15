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

out vec4 vPosition0;
out vec4 vColor0;
out vec4 vNormRad0;
uniform float texDim;
uniform int time;
uniform float stableThreshold;

uniform sampler2D vertSamp;
uniform sampler2D colorSamp;
uniform sampler2D normSamp;

#include "color.glsl"

void main()
{
    int intY = gl_VertexID / int(texDim);
    int intX = gl_VertexID - (intY * int(texDim));

    float halfPixel = 0.5 * (1.0f / texDim);
    float y = (float(intY) / texDim) + halfPixel;
    float x = (float(intX) / texDim) + halfPixel;
    
    vec4 newColor = textureLod(colorSamp, vec2(x, y), 0);

    //Do averaging here -> If I found a point to merge with
    if(newColor.w == -1)
    {
        vec4 newPos = textureLod(vertSamp, vec2(x, y), 0);
        vec4 newNorm = textureLod(normSamp, vec2(x, y), 0);
        
        float c_k = vPosition.w;  //probability of existing point to fuse with
        vec3 v_k = vPosition.xyz;
        
        float a = newPos.w;   //this should contain the fused prob of all prob values - velocity, dynamic, radial dist
        vec3 v_g = newPos.xyz;

        float hist = vColor.y; //number of times I have seen this surfel

        //truncating
        const float max_val = 0.99f;
        const float min_val = 0.01f;
        a = max(min_val, min(0.53, 2*a*a));
        c_k = max(min_val, min(c_k, max_val));

        float ltm = log (1.0 / (1.0 - c_k) - 1.0);   //get the log odds from the probability
        ltm = ltm + log( a / (1.0 - a) ); //update log odds probability 
        float c_k1 = 1.0 - ( 1.0 / (1.0 + exp(ltm) )  ); //convert back to prob -> updated confidence

        if(newNorm.w < (1.0 + 0.5) * vNormRad.w)
        {
				vPosition0 = vec4(((hist * c_k * v_k) + (a * v_g)) / (hist * c_k + a), c_k1);

				vec3 oldCol = decodeColor(vColor.x);
				vec3 newCol = decodeColor(newColor.x);   

				vec3 avgColor = ((hist * c_k * oldCol.xyz) + (a * newCol.xyz)) / (hist * c_k + a);

				vColor0 = vec4(encodeColor(avgColor), hist + 1.0, vColor.z, time);

				vNormRad0 = ((hist * c_k * vNormRad) + (a * newNorm)) / ( hist * c_k + a);

			 	vNormRad0.xyz = normalize(vNormRad0.xyz);
        }
        else
        {
		//I found something to merge with, but it would make the reconstruction less fine, therefore I maintain the point.
		//if it is a static point and i merge it with a dynamic one -> could be something changed in the meantime

            vPosition0 = vPosition;
            vPosition0.w = c_k1;
            vColor0 = vColor;
            vColor0.y = hist + 1.0;
	    
            vNormRad0 = vNormRad;
            
            vColor0.w = time;
        }
    }
    else
    {
        //This point isn't being updated, so just transfer it
	    vPosition0 = vPosition;
		 vColor0 = vColor;
	    vNormRad0 = vNormRad;
    }
}
