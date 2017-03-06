// Miguel Ascanio Gómez
// GPU - Práctica 1
#pragma once

#include "const.hh"

namespace GPU {

void initConstantMemory();
void canny(const float *im, float *image_out, float level, int height, int width);
void one(const float *im, float *image_out, float level, int height, int width);

void noiseReduction(const float* im, float* NR, int height, int width);
void phi_G(const float* NR, float* phi, float* G, int height, int width);
void edge(const float* phi, const float* G, int* pedge, int height, int width);
void threshold(const float* G, const int* pedge, float* image_out, float level, int height, int width);
}

namespace CPU {
void canny(float *im, float *image_out, float level, int height, int width);

void noiseReduction(const float* im, float* NR, int height, int width);
void phi_G(float* NR, float* phi, float* G, float* Gx, float* Gy, int height, int width);
void edge(float* phi, float* G, int* pedge, int height, int width);
void threshold(float* G, int* pedge, float* image_out, float level, int height, int width);
}

namespace ACC {
void canny(float *im, float *image_out, float level, int height, int width);

void noiseReduction(const float* im, float* NR, int height, int width);
void phi_G(float* NR, float* phi, float* G, float* Gx, float* Gy, int height, int width);
void edge(float* phi, float* G, int* pedge, int height, int width);
void threshold(float* G, int* pedge, float* image_out, float level, int height, int width);
}
