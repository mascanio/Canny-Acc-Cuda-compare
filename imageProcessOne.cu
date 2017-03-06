// Miguel Ascanio Gómez
// GPU - Práctica 1
#include "imageProcess.hh"
#include <cuda.h>
#include <math_functions.h>
#include "benchUtils.hh"
#include <assert.h>

cudaError_t e2;
#define TEST(x) e2 = x;\
	if (e2 != 0) { \
	cerr << "ERROR, line: " << __LINE__ << " File: " << __FILE__ << " Error: " <<  cudaGetErrorString(e2) << endl;\
	exit(-1); }

namespace GPU {

// Valores óptimos sobre GTX660, pruebas entre [12,32] para ambos
const int DIMX = 16;
const int DIMY = 28;

const int BX = 5;
const int BY = 5;

const int DIM2X = DIMX + 2 * BX;
const int DIM2Y = DIMY + 2 * BY;

#define IM(i, j) im[(i) * width + (j)]

__constant__ float NGK[25];
__constant__ float GXK[20];
__constant__ float GYK[20];

void initConstantMemory() {
	cudaMemcpyToSymbol(NGK, gaussKernel, 25 * sizeof(float));
	cudaMemcpyToSymbol(GXK, gxKernel,    20 * sizeof(float));
	cudaMemcpyToSymbol(GYK, gyKernel,    20 * sizeof(float));
}

template <int BX, int BY, int DIMX, int DIMY, int DIM2X, int DIM2Y>
__global__
void one_kernel(const float* im, float* out, int lowthres, int hithres, int height, int width);

void one(const float *im, float *image_out, float level, int height, int width) {
		int size = height * width * sizeof(float);

		// ImageIn to GPU
		float* d_in;
		TEST(cudaMalloc(&d_in, size));
		TEST(cudaMemcpy(d_in, im, size, cudaMemcpyHostToDevice));

		float* d_out;
		TEST(cudaMalloc(&d_out, size));

#ifdef DEBUG
		cudaMemset(d_out, 0, size);
#endif

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

		assert(2 * BX < DIMX && 2 * BY < DIMY);

		one_kernel<BX, BY, DIMX, DIMY, DIM2X, DIM2Y><<<dimGrid, dimBlock>>>(d_in, d_out, lowthres, hithres, height, width);
		TEST(cudaDeviceSynchronize());

		TEST(cudaMemcpy(image_out, d_out, size, cudaMemcpyDeviceToHost));
		TEST(cudaFree(d_in));
		TEST(cudaFree(d_out));
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<int DIM2X, int DIM2Y>
__device__
void getNrPoint(const float s[DIM2X][DIM2Y], float NR[DIM2X][DIM2Y], int px, int py) {
	float rr = 0;

	if (px >= 2 && py >= 2 && px < (DIM2X - 2) && py < (DIM2Y - 2)) {
		rr = (
			 NGK[0] *s[(px-2)][(py-2)] + NGK[1] *s[(px-2)][(py-1)] + NGK[2] *s[(px-2)][(py)] + NGK[3] *s[(px-2)][(py+1)] + NGK[4] *s[(px-2)][(py+2)]
		   + NGK[5] *s[(px-1)][(py-2)] + NGK[6] *s[(px-1)][(py-1)] + NGK[7] *s[(px-1)][(py)] + NGK[8] *s[(px-1)][(py+1)] + NGK[9] *s[(px-1)][(py+2)]
		   + NGK[10]*s[(px)  ][(py-2)] + NGK[11]*s[(px)  ][(py-1)] + NGK[12]*s[(px)  ][(py)] + NGK[13]*s[(px)  ][(py+1)] + NGK[14]*s[(px)  ][(py+2)]
		   + NGK[15]*s[(px+1)][(py-2)] + NGK[16]*s[(px+1)][(py-1)] + NGK[17]*s[(px+1)][(py)] + NGK[18]*s[(px+1)][(py+1)] + NGK[19]*s[(px+1)][(py+2)]
		   + NGK[20]*s[(px+2)][(py-2)] + NGK[21]*s[(px+2)][(py-1)] + NGK[22]*s[(px+2)][(py)] + NGK[23]*s[(px+2)][(py+1)] + NGK[24]*s[(px+2)][(py+2)]
           ) / 159.0f;
	}
	NR[px][py] = rr;
}

template<int DIM2X, int DIM2Y>
__device__
void processGX(const float NR[DIM2X][DIM2Y], float &X, int px, int py) {
	X =	 (GXK[0] *NR[(px-2)][(py-2)] + GXK[1] *NR[(px-2)][(py-1)] + GXK[2] *NR[(px-2)][(py+1)] + GXK[3] *NR[(px-2)][(py+2)]
		+ GXK[4] *NR[(px-1)][(py-2)] + GXK[5] *NR[(px-1)][(py-1)] + GXK[6] *NR[(px-1)][(py+1)] + GXK[7] *NR[(px-1)][(py+2)]
		+ GXK[8] *NR[(px  )][(py-2)] + GXK[9] *NR[(px  )][(py-1)] + GXK[10]*NR[(px  )][(py+1)] + GXK[11]*NR[(px  )][(py+2)]
		+ GXK[12]*NR[(px+1)][(py-2)] + GXK[13]*NR[(px+1)][(py-1)] + GXK[14]*NR[(px+1)][(py+1)] + GXK[15]*NR[(px+1)][(py+2)]
		+ GXK[16]*NR[(px+2)][(py-2)] + GXK[17]*NR[(px+2)][(py-1)] + GXK[18]*NR[(px+2)][(py+1)] + GXK[19]*NR[(px+2)][(py+2)]);
}

template<int DIM2X, int DIM2Y>
__device__
void processGY(const float NR[DIM2X][DIM2Y], float &Y, int px, int py) {
	Y =	  GYK[0] *NR[(px-2)][(py-2)] + GYK[1] *NR[(px-2)][(py-1)] + GYK[2] *NR[(px-2)][(py)] + GYK[3] *NR[(px-2)][(py+1)] + GYK[4] *NR[(px-2)][(py+2)]
		+ GYK[5] *NR[(px-1)][(py-2)] + GYK[6] *NR[(px-1)][(py-1)] + GYK[7] *NR[(px-1)][(py)] + GYK[8] *NR[(px-1)][(py+1)] + GYK[9] *NR[(px-1)][(py+2)]
		+ GYK[10]*NR[(px+1)][(py-2)] + GYK[11]*NR[(px+1)][(py-1)] + GYK[12]*NR[(px+1)][(py)] + GYK[13]*NR[(px+1)][(py+1)] + GYK[14]*NR[(px+1)][(py+2)]
		+ GYK[15]*NR[(px+2)][(py-2)] + GYK[16]*NR[(px+2)][(py-1)] + GYK[17]*NR[(px+2)][(py)] + GYK[18]*NR[(px+2)][(py+1)] + GYK[19]*NR[(px+2)][(py+2)];
}

template<int DIM2X, int DIM2Y>
__device__
void getPhiGPoint(const float NR[DIM2X][DIM2Y], float PHI[DIM2X][DIM2Y], float G[DIM2X][DIM2Y], int px, int py) {
	float pphi, GX, GY, phi_r = 0, G_r = 0;
	if (px >= 4 && py >= 4 && px < (DIM2X - 4) && py < (DIM2Y - 4)) {
		processGX<DIM2X, DIM2Y>(NR, GX, px, py);
		processGY<DIM2X, DIM2Y>(NR, GY, px, py);

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
	PHI[px][py] = phi_r;
	  G[px][py] = G_r;
}


template<int DIM2X, int DIM2Y>
__device__
void getPedgePoint(const float PHI[DIM2X][DIM2Y], const float G[DIM2X][DIM2Y], float pedge[DIM2X][DIM2Y], int px, int py) {
	float r = 0, p;
	if (px >= 5 && py >= 5 && px < (DIM2X - 5) && py < (DIM2Y - 5)) {
		p = PHI[px][py];
		if (p == 0) {
			//edge is in N-S
			r = (G[px][py] > G[px][py + 1] && G[px][py] > G[px][py - 1]);
		} else if (p == 45) {
			// edge is in NW-SE
			r = (G[px][py] > G[px + 1][py + 1] && G[px][py] > G[px - 1][py - 1]);
		} else if (p == 90) {
			//edge is in E-W
			r = (G[px][py] > G[px + 1][py] && G[px][py] > G[px - 1][py]);
		} else {
			// edge is in NE-SW
			r = (G[px][py] > G[px + 1][py - 1] && G[px][py] > G[px - 1][py + 1]);
		}
	}
	pedge[px][py] = r;
}


template<int DIM2X, int DIM2Y>
__device__
void getThresholdPoint(const float G[DIM2X][DIM2Y], const float pedge[DIM2X][DIM2Y], float out[DIM2X][DIM2Y], float lowthres, float hithres, int px, int py) {
	float r = 0;
	if (px >= 5 && py >= 5 && px < (DIM2X - 5) && py < (DIM2Y - 5)) {
		if (G[px][py] > hithres && pedge[px][py]) {
			r = 255;
		} else if (pedge[px][py] && G[px][py] >= lowthres && G[px][py] < hithres) {
			// check neighbours 3x3
			r = 255 *
				(G[px-1][py-1] > hithres   || G[px-1][py] > hithres   || G[px-1][py+1] > hithres ||
				 G[px  ][py-1] > hithres /*|| G[px  ][py] > hithres*/ || G[px  ][py+1] > hithres ||
				 G[px+1][py-1] > hithres   || G[px+1][py] > hithres   || G[px+1][py+1] > hithres);
		} // else r = 0
	}
	out[px][py] = r;
}


template<int BX, int BY, int DIMX, int DIMY, int DIM2X, int DIM2Y>
__device__
void loadShared(const float* im, float s[DIM2X][DIM2Y], int tx, int ty, int bi, int bj, int height, int width) {

	int imi = bi + tx - BX, imj = bj + ty - BY;

	if (imi >= 0 && imi < height) {
		if ((imj >= 0) && (imj < width)) {
			s[tx][ty] = IM(imi, imj);
		} else {
			s[tx][ty] = 0;
		}

		imj = imj + DIMY;
		if ((imj >= 0) && (imj < width) && ((ty + DIMY) < DIM2Y)) {
			s[tx][ty + DIMY] = IM(imi, imj);
		} else if (ty + DIMY < DIM2Y) {
			s[tx][ty + DIMY] = 0;
		}
	} else {
		s[tx][ty] = 0;
		if (ty + DIMY < DIM2Y) {
			s[tx][ty + DIMY] = 0;
		}
	}

	imi = imi + DIMX;
	tx  = tx  + DIMX;
	imj = bj + ty - BY;

	if (imi >= 0 && imi < height && tx < DIM2X) {
		if ((imj >= 0) && (imj < width)) {
			s[tx][ty] = IM(imi, imj);
		} else {
			s[tx][ty] = 0;
		}

		imj = imj + DIMY;
		if ((imj >= 0) && (imj < width) && ((ty + DIMY) < DIM2Y)) {
			s[tx][ty + DIMY] = IM(imi, imj);
		} else if ((ty + DIMY) < DIM2Y) {
			s[tx][ty + DIMY] = 0;
		}
	} else if (tx < DIM2X) {
		s[tx][ty] = 0;
		if (ty + DIMY < DIM2Y) {
			s[tx][ty + DIMY] = 0;
		}
	}
}

template <int BX, int BY, int DIMX, int DIMY, int DIM2X, int DIM2Y>
__global__
void one_kernel(const float* im, float* out, int lowthres, int hithres, int height, int width) {
	__shared__ float A[DIM2X][DIM2Y];
	__shared__ float B[DIM2X][DIM2Y];
	__shared__ float C[DIM2X][DIM2Y];

	int i, j, tx, ty, bi, bj;
	tx = threadIdx.x;
	ty = threadIdx.y;

	i = blockIdx.x * blockDim.x + tx;
	j = blockIdx.y * blockDim.y + ty;

	bi = blockIdx.x * blockDim.x;
	bj = blockIdx.y * blockDim.y;

	loadShared<BX, BY, DIMX, DIMY, DIM2X, DIM2Y>(im, A, tx, ty, bi, bj, height, width);

	__syncthreads();

	// Noise reduction

	int px = tx, py = ty;
	if (px >= 0 && px < DIM2X) {
		if (py >= 0 && py < DIM2Y)
			getNrPoint<DIM2X, DIM2Y>(A, B, px, py);
		py = py + DIMY;
		if (py >= 0 && py < DIM2Y)
			getNrPoint<DIM2X, DIM2Y>(A, B, px, py);
	}
	px = px + DIMX;
	py = ty;

	if (px >= 0 && px < DIM2X) {
		if (py >= 0 && py < DIM2Y)
			getNrPoint<DIM2X, DIM2Y>(A, B, px, py);
		py = py + DIMY;
		if (py >= 0 && py < DIM2Y)
			getNrPoint<DIM2X, DIM2Y>(A, B, px, py);
	}

	__syncthreads();

	// Phi_G

	px = tx, py = ty;
	if (px >= 0 && px < DIM2X) {
		if (py >= 0 && py < DIM2Y)
			getPhiGPoint<DIM2X, DIM2Y>(B, C, A, px, py);
		py = py + DIMY;
		if (py >= 0 && py < DIM2Y)
			getPhiGPoint<DIM2X, DIM2Y>(B, C, A, px, py);
	}
	px = px + DIMX;
	py = ty;

	if (px >= 0 && px < DIM2X) {
		if (py >= 0 && py < DIM2Y)
			getPhiGPoint<DIM2X, DIM2Y>(B, C, A, px, py);
		py = py + DIMY;
		if (py >= 0 && py < DIM2Y)
			getPhiGPoint<DIM2X, DIM2Y>(B, C, A, px, py);
	}

	__syncthreads();

	// Pedge

	px = tx, py = ty;
	if (px >= 0 && px < DIM2X) {
		if (py >= 0 && py < DIM2Y)
			getPedgePoint<DIM2X, DIM2Y>(C, A, B, px, py);
		py = py + DIMY;
		if (py >= 0 && py < DIM2Y)
			getPedgePoint<DIM2X, DIM2Y>(C, A, B, px, py);
	}
	px = px + DIMX;
	py = ty;

	if (px >= 0 && px < DIM2X) {
		if (py >= 0 && py < DIM2Y)
			getPedgePoint<DIM2X, DIM2Y>(C, A, B, px, py);
		py = py + DIMY;
		if (py >= 0 && py < DIM2Y)
			getPedgePoint<DIM2X, DIM2Y>(C, A, B, px, py);
	}

	__syncthreads();

	// Threshold

	px = tx, py = ty;
	if (px >= 0 && px < DIM2X) {
		if (py >= 0 && py < DIM2Y)
			getThresholdPoint<DIM2X, DIM2Y>(A, B, C, lowthres, hithres, px, py);
		py = py + DIMY;
		if (py >= 0 && py < DIM2Y)
			getThresholdPoint<DIM2X, DIM2Y>(A, B, C, lowthres, hithres, px, py);
	}
	px = px + DIMX;
	py = ty;

	if (px >= 0 && px < DIM2X) {
		if (py >= 0 && py < DIM2Y)
			getThresholdPoint<DIM2X, DIM2Y>(A, B, C, lowthres, hithres, px, py);
		py = py + DIMY;
		if (py >= 0 && py < DIM2Y)
			getThresholdPoint<DIM2X, DIM2Y>(A, B, C, lowthres, hithres, px, py);
	}

	__syncthreads();

	if (i >= 5 && j >= 5 && (i < (height - 5)) && (j < (width -5)))
		out[i*width+j] = C[tx + BX][ty + BY];
	else if (i < height && j < width)
		out[i*width+j] = 0;

}

}
