// Miguel Ascanio Gómez
// GPU - Práctica 1
#include "imageProcess.hh"
#include "benchUtils.hh"

#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <iostream>
using namespace std;


namespace CPU {

void canny(float *im, float *image_out,
	float level,
	int height, int width)
{
	int size = sizeof(float) * width * height;

	float *NR;
	float *G;
	float *phi;
	float *Gx;
	float *Gy;
	int *pedge;

	// Aux. memory
	NR    = (float *) malloc(size);
	G     = (float *) malloc(size);
	phi   = (float *) malloc(size);
	Gx    = (float *) malloc(size);
	Gy    = (float *) malloc(size);
	pedge = (int   *) malloc(sizeof(int) * width * height);

	// Noise reduction
	noiseReduction(im, NR, height, width);

	// Compute Phi and G
	phi_G(NR, phi, G, Gx, Gy, height, width);

	// Edge
	edge(phi, G, pedge, height, width);

	// Hysteresis Thresholding
	threshold(G, pedge, image_out, level, height, width);

	free(NR   );
	free(G    );
	free(phi  );
	free(Gx   );
	free(Gy   );
	free(pedge);
}

void noiseReduction(const float* im, float* NR, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// Noise reduction
			if (i >= 2 && j >= 2 && i < (height - 2) && j < (width - 2)) {
				NR[i*width+j] =
				 (2.0f*im[(i-2)*width+(j-2)] +  4.0f*im[(i-2)*width+(j-1)] +  5.0f*im[(i-2)*width+(j)] +  4.0f*im[(i-2)*width+(j+1)] + 2.0f*im[(i-2)*width+(j+2)]
				+ 4.0f*im[(i-1)*width+(j-2)] +  9.0f*im[(i-1)*width+(j-1)] + 12.0f*im[(i-1)*width+(j)] +  9.0f*im[(i-1)*width+(j+1)] + 4.0f*im[(i-1)*width+(j+2)]
				+ 5.0f*im[(i  )*width+(j-2)] + 12.0f*im[(i  )*width+(j-1)] + 15.0f*im[(i  )*width+(j)] + 12.0f*im[(i  )*width+(j+1)] + 5.0f*im[(i  )*width+(j+2)]
				+ 4.0f*im[(i+1)*width+(j-2)] +  9.0f*im[(i+1)*width+(j-1)] + 12.0f*im[(i+1)*width+(j)] +  9.0f*im[(i+1)*width+(j+1)] + 4.0f*im[(i+1)*width+(j+2)]
				+ 2.0f*im[(i+2)*width+(j-2)] +  4.0f*im[(i+2)*width+(j-1)] +  5.0f*im[(i+2)*width+(j)] +  4.0f*im[(i+2)*width+(j+1)] + 2.0f*im[(i+2)*width+(j+2)])
				/159.0f;
			} else {
				NR[i*width+j] = 0;
			}
		}
	}
}

void phi_G(float* NR, float* phi, float* G, float* Gx, float* Gy, int height, int width) {
	for(int i=0; i<height; i++) {
		for(int j=0; j<width; j++) {
			if (i >= 2 && j >= 2 && i < (height - 2) && j < (width - 2)) {

				// Intensity gradient of the image
				Gx[i*width+j] =
					 (1.0f*NR[(i-2)*width+(j-2)] +  2.0f*NR[(i-2)*width+(j-1)] +  (-2.0f)*NR[(i-2)*width+(j+1)] + (-1.0f)*NR[(i-2)*width+(j+2)]
					+ 4.0f*NR[(i-1)*width+(j-2)] +  8.0f*NR[(i-1)*width+(j-1)] +  (-8.0f)*NR[(i-1)*width+(j+1)] + (-4.0f)*NR[(i-1)*width+(j+2)]
					+ 6.0f*NR[(i  )*width+(j-2)] + 12.0f*NR[(i  )*width+(j-1)] + (-12.0f)*NR[(i  )*width+(j+1)] + (-6.0f)*NR[(i  )*width+(j+2)]
					+ 4.0f*NR[(i+1)*width+(j-2)] +  8.0f*NR[(i+1)*width+(j-1)] +  (-8.0f)*NR[(i+1)*width+(j+1)] + (-4.0f)*NR[(i+1)*width+(j+2)]
					+ 1.0f*NR[(i+2)*width+(j-2)] +  2.0f*NR[(i+2)*width+(j-1)] +  (-2.0f)*NR[(i+2)*width+(j+1)] + (-1.0f)*NR[(i+2)*width+(j+2)]);


				Gy[i*width+j] =
					 ((-1.0f)*NR[(i-2)*width+(j-2)] + (-4.0f)*NR[(i-2)*width+(j-1)] +  (-6.0f)*NR[(i-2)*width+(j)] + (-4.0f)*NR[(i-2)*width+(j+1)] + (-1.0f)*NR[(i-2)*width+(j+2)]
					+ (-2.0f)*NR[(i-1)*width+(j-2)] + (-8.0f)*NR[(i-1)*width+(j-1)] + (-12.0f)*NR[(i-1)*width+(j)] + (-8.0f)*NR[(i-1)*width+(j+1)] + (-2.0f)*NR[(i-1)*width+(j+2)]
					+    2.0f*NR[(i+1)*width+(j-2)] +    8.0f*NR[(i+1)*width+(j-1)] +    12.0f*NR[(i+1)*width+(j)] +    8.0f*NR[(i+1)*width+(j+1)] +    2.0f*NR[(i+1)*width+(j+2)]
					+    1.0f*NR[(i+2)*width+(j-2)] +    4.0f*NR[(i+2)*width+(j-1)] +     6.0f*NR[(i+2)*width+(j)] +    4.0f*NR[(i+2)*width+(j+1)] +    1.0f*NR[(i+2)*width+(j+2)]);

				G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
				phi[i*width+j] = atan2f(fabsf(Gy[i*width+j]),fabsf(Gx[i*width+j]));

				if (fabs(phi[i * width + j]) <= C_PIO8)
					phi[i * width + j] = 0;
				else if (fabs(phi[i * width + j]) <= C_3PIO8)
					phi[i * width + j] = 45;
				else if (fabs(phi[i * width + j]) <= C_5PIO8)
					phi[i * width + j] = 90;
				else if (fabs(phi[i * width + j]) <= C_7PIO8)
					phi[i * width + j] = 135;
				else
					phi[i * width + j] = 0;
			} else {
				Gx[i*width+j] = 0;
				Gy[i*width+j] = 0;
				G [i*width+j] = 0;
				phi[i*width+j] = 0;
			}
		}
	}
}

void edge(float* phi, float* G, int* pedge, int height, int width) {
	for (int i = 0; i < height - 0; i++) {
		for (int j = 0; j < width - 0; j++) {
			pedge[i * width + j] = 0;
			if (i >= 3 && j >= 3 && i < (height - 3) && j < (width - 3)) {
				if (phi[i * width + j] == 0) {
					if (G[i * width + j] > G[i * width + j + 1] && G[i * width + j] > G[i * width + j - 1])
						//edge is in N-S
						pedge[i * width + j] = 1;
				} else if (phi[i * width + j] == 45) {
					if (G[i * width + j] > G[(i + 1) * width + j + 1] && G[i * width + j] > G[(i - 1) * width + j - 1])
						// edge is in NW-SE
						pedge[i * width + j] = 1;

				} else if (phi[i * width + j] == 90) {
					if (G[i * width + j] > G[(i + 1) * width + j] && G[i * width + j] > G[(i - 1) * width + j])
						//edge is in E-W
						pedge[i * width + j] = 1;

				} else if (phi[i * width + j] == 135) {
					if (G[i * width + j] > G[(i + 1) * width + j - 1] && G[i * width + j] > G[(i - 1) * width + j + 1])
						// edge is in NE-SW
						pedge[i * width + j] = 1;
				}
			}
		}
	}
}

void threshold(float* G, int* pedge, float* image_out, float level, int height, int width) {
	// Hysteresis Thresholding
	float lowthres, hithres;
	lowthres = level / 2;
	hithres = 2 * (level);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (i >= 3 && j >= 3 && i < (height - 3) && j < (width - 3)) {
				if (G[i * width + j] > hithres && pedge[i * width + j]) {
					image_out[i * width + j] = 255;
				} else if (pedge[i * width + j] && G[i * width + j] >= lowthres && G[i * width + j] < hithres) {
					// check neighbours 3x3
					for (int ii = -1; ii <= 1; ii++) {
						for (int jj = -1; jj <= 1; jj++) {
							if (G[(i + ii) * width + j + jj] > hithres) {
								image_out[i * width + j] = 255;
							}
						}
					}
				}
			} else {
				image_out[i * width + j] = 0;
			}
		}
	}
}
}
