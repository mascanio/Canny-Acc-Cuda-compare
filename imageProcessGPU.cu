// Miguel Ascanio Gómez
// GPU - Práctica 1
#include "imageProcess.hh"
#include <cuda.h>
#include <math_functions.h>
#include "benchUtils.hh"
#include <assert.h>

cudaError_t e;
#define TEST(x) e = x;\
	if (e != 0) { \
	cerr << "ERROR, line: " << __LINE__ << " File: " << __FILE__ << " Error: " <<  cudaGetErrorString(e) << endl;\
	exit(-1); }

namespace GPU {

const int DIMX = 16;
const int DIMY = 32;

const int DIM2X = DIMX + 4;
const int DIM2Y = DIMY + 4;

#define IM(i, j) im[(i) * width + (j)]

__constant__ float NGK[25];
__constant__ float GXK[20];
__constant__ float GYK[20];

//void initConstantMemory() {
////	cudaMemcpyToSymbol(NGK, gaussKernel,  25 * sizeof(float));
////	cudaMemcpyToSymbol(GXK, gxKernel,  20 * sizeof(float));
////	cudaMemcpyToSymbol(GYK, gyKernel,  20 * sizeof(float));
//}

void canny(const float *im, float *image_out, float level, int height, int width) {
	int size = height * width * sizeof(float);

	// ImageIn to GPU
	float* d_a;
	TEST(cudaMalloc(&d_a, size));
	TEST(cudaMemcpy(d_a, im, size, cudaMemcpyHostToDevice));

	float* d_b;
	TEST(cudaMalloc(&d_b, size));

	float* d_c;
	TEST(cudaMalloc(&d_c, size));

#ifdef DEBUG
	cudaMemset(d_b, 0, size);
	cudaMemset(d_c, 0, size);
#endif

	noiseReduction(d_a, d_b, height, width);

	phi_G(d_b, d_c, d_a, height, width);
	int* d_pedge = (int *) d_b;

	edge(d_c, d_a, d_pedge, height, width);

	threshold(d_a, d_pedge, d_c, level, height, width);

	TEST(cudaMemcpy(image_out, d_c, size, cudaMemcpyDeviceToHost));
	TEST(cudaFree(d_a));
	TEST(cudaFree(d_b));
	TEST(cudaFree(d_c));
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<int DIMX, int DIMY, int DIM2X, int DIM2Y>
__device__
void loadShared(const float* im, float s[DIM2X][DIM2Y], int tx, int ty, int bi, int bj, int height, int width) {

	int imi = bi + tx - 2, imj = bj + ty - 2;

	if (imi >= 0 && imi < height) {
		if ((imj >= 0) && (imj < width))
			s[tx][ty] = IM(imi, imj);
		imj = imj + DIMY;
		if ((imj >= 0) && (imj < width) && ((ty + DIMY) < DIM2Y))
			s[tx][ty + DIMY] = IM(imi, imj);
	}

	imi = imi + DIMX;
	tx  = tx  + DIMX;
	imj = bj + ty - 2;

	if (imi >= 0 && imi < height && tx < DIM2X) {
		if ((imj >= 0) && (imj < width))
			s[tx][ty] = IM(imi, imj);
		imj = imj + DIMY;
		if ((imj >= 0) && (imj < width) && ((ty + DIMY) < DIM2Y))
			s[tx][ty + DIMY] = IM(imi, imj);
	}
}

template<int DIMX, int DIMY, int DIM2X, int DIM2Y>
__device__
void processNRK(const float s[DIM2X][DIM2Y], float* NR, int i,
		int j, int tImi, int tImj, int height, int width) {

	float r = 0;
	if (i < (height) && j < (width)) {
		if (i >= 2 && j >= 2 && i < (height - 2) && j < (width - 2)) {
	//		NR[i*width+j] =
	//		float a = NGK[0] *s[(tImi-2)][(tImj-2)] + NGK[1] *s[(tImi-2)][(tImj-1)] + NGK[2] *s[(tImi-2)][(tImj)] + NGK[3] *s[(tImi-2)][(tImj+1)] + NGK[4] *s[(tImi-2)][(tImj+2)];
	//		float b = NGK[5] *s[(tImi-1)][(tImj-2)] + NGK[6] *s[(tImi-1)][(tImj-1)] + NGK[7] *s[(tImi-1)][(tImj)] + NGK[8] *s[(tImi-1)][(tImj+1)] + NGK[9] *s[(tImi-1)][(tImj+2)];
	//		float c = NGK[10]*s[(tImi)  ][(tImj-2)] + NGK[11]*s[(tImi)  ][(tImj-1)] + NGK[12]*s[(tImi)  ][(tImj)] + NGK[13]*s[(tImi)  ][(tImj+1)] + NGK[14]*s[(tImi)  ][(tImj+2)];
	//		float d = NGK[15]*s[(tImi+1)][(tImj-2)] + NGK[16]*s[(tImi+1)][(tImj-1)] + NGK[17]*s[(tImi+1)][(tImj)] + NGK[18]*s[(tImi+1)][(tImj+1)] + NGK[19]*s[(tImi+1)][(tImj+2)];
	//		float e = NGK[20]*s[(tImi+2)][(tImj-2)] + NGK[21]*s[(tImi+2)][(tImj-1)] + NGK[22]*s[(tImi+2)][(tImj)] + NGK[23]*s[(tImi+2)][(tImj+1)] + NGK[24]*s[(tImi+2)][(tImj+2)];
	//		NR[i*width+j] = (a+b+c+d+e)/159.0;

			r = fdividef(fmaf(
					 2.0f, s[(tImi-2)][(tImj-2)], fmaf(4.0f,  s[(tImi-2)][(tImj-1)], fmaf(5.0f,  s[(tImi-2)][(tImj)], fmaf(4.0f,  s[(tImi-2)][(tImj+1)], fmaf(2.0f, s[(tImi-2)][(tImj+2)],
				fmaf(4.0f, s[(tImi-1)][(tImj-2)], fmaf(9.0f,  s[(tImi-1)][(tImj-1)], fmaf(12.0f, s[(tImi-1)][(tImj)], fmaf(9.0f,  s[(tImi-1)][(tImj+1)], fmaf(4.0f, s[(tImi-1)][(tImj+2)],
				fmaf(5.0f, s[(tImi)  ][(tImj-2)], fmaf(12.0f, s[(tImi)  ][(tImj-1)], fmaf(15.0f, s[(tImi)  ][(tImj)], fmaf(12.0f, s[(tImi)  ][(tImj+1)], fmaf(5.0f, s[(tImi)  ][(tImj+2)],
				fmaf(4.0f, s[(tImi+1)][(tImj-2)], fmaf(9.0f,  s[(tImi+1)][(tImj-1)], fmaf(12.0f, s[(tImi+1)][(tImj)], fmaf(9.0f,  s[(tImi+1)][(tImj+1)], fmaf(4.0f, s[(tImi+1)][(tImj+2)],
				fmaf(2.0f, s[(tImi+2)][(tImj-2)], fmaf(4.0f,  s[(tImi+2)][(tImj-1)], fmaf(5.0f,  s[(tImi+2)][(tImj)], fmaf(4.0f,  s[(tImi+2)][(tImj+1)], 2.0f * s[(tImi+2)][(tImj+2)])))))))))))))))))))))))), 159.0f);

		}
		NR[i*width+j] = r;
	}
}

template<int DIMX, int DIMY, int DIM2X, int DIM2Y>
__device__
void processGX(const float s[DIM2X][DIM2Y], float &X, int i,
		int j, int tImi, int tImj, int height, int width) {
	if (i >= 2 && j >= 2 && i < (height - 2) && j < (width - 2)) {
//		X = fmaf(1.0f, s[(tImi-2)][(tImj-2)], fmaf( 2.0f, s[(tImi-2)][(tImj-1)], fmaf( (-2.0f), s[(tImi-2)][(tImj+1)], fmaf((-1.0f), s[(tImi-2)][(tImj+2)],
//			fmaf(4.0f, s[(tImi-1)][(tImj-2)], fmaf( 8.0f, s[(tImi-1)][(tImj-1)], fmaf( (-8.0f), s[(tImi-1)][(tImj+1)], fmaf((-4.0f), s[(tImi-1)][(tImj+2)],
//			fmaf(6.0f, s[(tImi)  ][(tImj-2)], fmaf(12.0f, s[(tImi)  ][(tImj-1)], fmaf((-12.0f), s[(tImi)  ][(tImj+1)], fmaf((-6.0f), s[(tImi)  ][(tImj+2)],
//			fmaf(4.0f, s[(tImi+1)][(tImj-2)], fmaf( 8.0f, s[(tImi+1)][(tImj-1)], fmaf( (-8.0f), s[(tImi+1)][(tImj+1)], fmaf((-4.0f), s[(tImi+1)][(tImj+2)],
//			fmaf(1.0f, s[(tImi+2)][(tImj-2)], fmaf( 2.0f, s[(tImi+2)][(tImj-1)], fmaf( (-2.0f), s[(tImi+2)][(tImj+1)], (-1.0f) * s[(tImi+2)][(tImj+2)])))))))))))))))))));


		X =	 (1.0f*s[(tImi-2)][(tImj-2)] +  2.0f*s[(tImi-2)][(tImj-1)] +  (-2.0f)*s[(tImi-2)][(tImj+1)] + (-1.0f)*s[(tImi-2)][(tImj+2)]
			+ 4.0f*s[(tImi-1)][(tImj-2)] +  8.0f*s[(tImi-1)][(tImj-1)] +  (-8.0f)*s[(tImi-1)][(tImj+1)] + (-4.0f)*s[(tImi-1)][(tImj+2)]
			+ 6.0f*s[(tImi  )][(tImj-2)] + 12.0f*s[(tImi  )][(tImj-1)] + (-12.0f)*s[(tImi  )][(tImj+1)] + (-6.0f)*s[(tImi  )][(tImj+2)]
			+ 4.0f*s[(tImi+1)][(tImj-2)] +  8.0f*s[(tImi+1)][(tImj-1)] +  (-8.0f)*s[(tImi+1)][(tImj+1)] + (-4.0f)*s[(tImi+1)][(tImj+2)]
			+ 1.0f*s[(tImi+2)][(tImj-2)] +  2.0f*s[(tImi+2)][(tImj-1)] +  (-2.0f)*s[(tImi+2)][(tImj+1)] + (-1.0f)*s[(tImi+2)][(tImj+2)]);
	}
}

template<int DIMX, int DIMY, int DIM2X, int DIM2Y>
__device__
void processGY(const float s[DIM2X][DIM2Y], float &Y, int i,
		int j, int tImi, int tImj, int height, int width) {
	if (i >= 2 && j >= 2 && i < (height - 2) && j < (width - 2)) {
		Y =	 ((-1.0f)*s[(tImi-2)][(tImj-2)] + (-4.0f)*s[(tImi-2)][(tImj-1)] +  (-6.0f)*s[(tImi-2)][(tImj)] + (-4.0f)*s[(tImi-2)][(tImj+1)] + (-1.0f)*s[(tImi-2)][(tImj+2)]
			+ (-2.0f)*s[(tImi-1)][(tImj-2)] + (-8.0f)*s[(tImi-1)][(tImj-1)] + (-12.0f)*s[(tImi-1)][(tImj)] + (-8.0f)*s[(tImi-1)][(tImj+1)] + (-2.0f)*s[(tImi-1)][(tImj+2)]
			+    2.0f*s[(tImi+1)][(tImj-2)] +    8.0f*s[(tImi+1)][(tImj-1)] +    12.0f*s[(tImi+1)][(tImj)] +    8.0f*s[(tImi+1)][(tImj+1)] +    2.0f*s[(tImi+1)][(tImj+2)]
			+    1.0f*s[(tImi+2)][(tImj-2)] +    4.0f*s[(tImi+2)][(tImj-1)] +     6.0f*s[(tImi+2)][(tImj)] +    4.0f*s[(tImi+2)][(tImj+1)] +    1.0f*s[(tImi+2)][(tImj+2)]);
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <int DIMX, int DIMY, int DIM2X, int DIM2Y>
__global__
void kNR_shared(const float* im, float* NR, int height, int width) {

	__shared__ float s[DIM2X][DIM2Y];

	int i, j, tx, ty, tImi, tImj, bi, bj;
	tx = threadIdx.x;
	ty = threadIdx.y;

	tImi = tx+2;
	tImj = ty+2;

	i = blockIdx.x * blockDim.x + tx;
	j = blockIdx.y * blockDim.y + ty;

	bi = blockIdx.x * blockDim.x;
	bj = blockIdx.y * blockDim.y;

	loadShared<DIMX, DIMY, DIM2X, DIM2Y>(im, s, tx, ty, bi, bj, height, width);

	__syncthreads();

	processNRK<DIMX, DIMY, DIM2X, DIM2Y>(s, NR, i, j, tImi, tImj, height, width);
}

template <int DIMX, int DIMY, int DIM2X, int DIM2Y>
__global__
void phi_shared(const float* im, float* phi, float* G, int height, int width) {

	__shared__ float s[DIM2X][DIM2Y];

	int i, j, tx, ty, tImi, tImj, bi, bj;
	tx = threadIdx.x;
	ty = threadIdx.y;

	tImi = tx+2;
	tImj = ty+2;

	i = blockIdx.x * blockDim.x + tx;
	j = blockIdx.y * blockDim.y + ty;

	bi = blockIdx.x * blockDim.x;
	bj = blockIdx.y * blockDim.y;

	loadShared<DIMX, DIMY, DIM2X, DIM2Y>(im, s, tx, ty, bi, bj, height, width);
	__syncthreads();

	if (i < (height) && j < (width)) {
		float pphi, GX, GY, phi_r = 0, G_r = 0;
		if (i >= 2 && j >= 2 && i < (height - 2) && j < (width - 2)) {

			processGX<DIMX, DIMY, DIM2X, DIM2Y>(s, GX, i, j, tImi, tImj, height, width);
			processGY<DIMX, DIMY, DIM2X, DIM2Y>(s, GY, i, j, tImi, tImj, height, width);

			G_r = sqrtf((GX * GX) + (GY * GY));	//G = √Gx²+Gy²
			pphi = fabs(atan2f(fabsf(GY), fabsf(GX)));

			if (pphi <= C_PIO8)
				phi_r = 0;
			else if (pphi <= C_3PIO8)
				phi_r = 45;
			else if (pphi <= C_5PIO8)
				phi_r = 90;
			else if (pphi <= C_7PIO8)
				phi_r = 135;
//			else
//				phi_r = 0;
		}
		G[i * width + j] = G_r;
		phi[i * width + j] = phi_r;
	}

}

template <int DIMX, int DIMY, int DIM2X, int DIM2Y>
__global__
void edge_shared(const float* phi, const float* G, int* pedge, int height, int width) {
	__shared__ float s[DIM2X][DIM2Y];

	int i, j, tx, ty, tImi, tImj, bi, bj;
	tx = threadIdx.x;
	ty = threadIdx.y;

	tImi = tx+2;
	tImj = ty+2;

	i = blockIdx.x * blockDim.x + tx;
	j = blockIdx.y * blockDim.y + ty;

	bi = blockIdx.x * blockDim.x;
	bj = blockIdx.y * blockDim.y;

	loadShared<DIMX, DIMY, DIM2X, DIM2Y>(G, s, tx, ty, bi, bj, height, width);

	__syncthreads();

	if (i < (height) && j < (width)) {
		float r = 0, p;
		if (i >= 3 && j >= 3 && i < (height - 3) && j < (width - 3)) {
			p = phi[i * width + j];
			if (p == 0) {
				//edge is in N-S
				r = (s[tImi][tImj] > s[tImi][tImj + 1] && s[tImi][tImj] > s[tImi][tImj - 1]);
			} else if (p == 45) {
				// edge is in NW-SE
				r = (s[tImi][tImj] > s[tImi + 1][tImj + 1] && s[tImi][tImj] > s[tImi - 1][tImj - 1]);
			} else if (p == 90) {
				//edge is in E-W
				r = (s[tImi][tImj] > s[tImi + 1][tImj] && s[tImi][tImj] > s[tImi - 1][tImj]);
			} else {
				// edge is in NE-SW
				r = (s[tImi][tImj] > s[tImi + 1][tImj - 1] && s[tImi][tImj] > s[tImi - 1][tImj + 1]);
			}
		}
		pedge[i * width + j] = r;
	}
}

template <int DIMX, int DIMY, int DIM2X, int DIM2Y>
__global__
void threshold_shared(const float* G, const int* pedge, float* image_out, float lowthres, float hithres, int height, int width) {
	__shared__ float s[DIM2X][DIM2Y];

	int i, j, tx, ty, tImi, tImj, bi, bj;
	tx = threadIdx.x;
	ty = threadIdx.y;

	tImi = tx+2;
	tImj = ty+2;

	i = blockIdx.x * blockDim.x + tx;
	j = blockIdx.y * blockDim.y + ty;

	bi = blockIdx.x * blockDim.x;
	bj = blockIdx.y * blockDim.y;

	loadShared<DIMX, DIMY, DIM2X, DIM2Y>(G, s, tx, ty, bi, bj, height, width);

	__syncthreads();

	if (i < (height) && j < (width)) {
		float r = 0;
		if (i >= 3 && j >= 3 && i < (height - 3) && j < (width - 3)) {
			if (s[tImi][tImj] > hithres && pedge[i * width + j]) {
				r = 255;
			} else if (pedge[i * width + j] && s[tImi][tImj] >= lowthres && s[tImi][tImj] < hithres) {
				// check neighbours 3x3
				r = 255 *
					(s[tImi-1][tImj-1] > hithres   || s[tImi-1][tImj] > hithres   || s[tImi-1][tImj+1] > hithres ||
					 s[tImi  ][tImj-1] > hithres /*|| s[tImi  ][tImj] > hithres*/ || s[tImi  ][tImj+1] > hithres ||
					 s[tImi+1][tImj-1] > hithres   || s[tImi+1][tImj] > hithres   || s[tImi+1][tImj+1] > hithres);
//				for (int ii = -1; ii <= 1; ii++) {
//					for (int jj = -1; jj <= 1; jj++) {
//						if (G[(i + ii) * width + j + jj] > hithres) {
//							image_out[i * width + j] = 255;
//						}
//					}
//				}
			} // else r = 0
		}
		image_out[i * width + j] = r;
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void noiseReduction(const float* in, float* NR, int height, int width) {
	dim3 dimBlock(DIMX, DIMY, 1);
	int blocksi = height / DIMX;
	int blocksj = width / DIMY;
	if (height % DIMX > 0)
		blocksi++;
	if (width % DIMY > 0)
		blocksj++;
	dim3 dimGrid(blocksi, blocksj, 1);

	kNR_shared<DIMX, DIMY, DIM2X, DIM2Y><<<dimGrid, dimBlock>>>(in, NR, height, width);
	TEST(cudaDeviceSynchronize());
}

void phi_G(const float* in, float* phi, float* G, int height, int width) {
	dim3 dimBlock(DIMX, DIMY, 1);
	int blocksi = height / DIMX;
	int blocksj = width / DIMY;
	if (height % DIMX > 0)
		blocksi++;
	if (width % DIMY > 0)
		blocksj++;
	dim3 dimGrid(blocksi, blocksj, 1);

	phi_shared<DIMX, DIMY, DIM2X, DIM2Y><<<dimGrid, dimBlock>>>(in, phi, G, height, width);
	TEST(cudaDeviceSynchronize());
}

void edge(const float* phi, const float* G, int* pedge, int height, int width) {
	dim3 dimBlock(DIMX, DIMY, 1);
	int blocksi = height / DIMX;
	int blocksj = width / DIMY;
	if (height % DIMX > 0)
		blocksi++;
	if (width % DIMY > 0)
		blocksj++;
	dim3 dimGrid(blocksi, blocksj, 1);

	edge_shared<DIMX, DIMY, DIM2X, DIM2Y><<<dimGrid, dimBlock>>>(phi, G, pedge, height, width);
	TEST(cudaDeviceSynchronize());
}

void threshold(const float* G, const int* pedge, float* image_out, float level, int height, int width) {
	dim3 dimBlock(DIMX, DIMY, 1);
	int blocksi = height / DIMX;
	int blocksj = width / DIMY;
	if (height % DIMX > 0)
		blocksi++;
	if (width % DIMY > 0)
		blocksj++;
	dim3 dimGrid(blocksi, blocksj, 1);

	float lowthres, hithres;
	lowthres = level / 2;
	hithres = 2 * (level);

	threshold_shared<DIMX, DIMY, DIM2X, DIM2Y><<<dimGrid, dimBlock>>>(G, pedge, image_out, lowthres, hithres, height, width);
	TEST(cudaDeviceSynchronize());
}

}
