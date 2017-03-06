// Miguel Ascanio Gómez
// GPU - Práctica 1
#pragma once

#include <math_constants.h>

const float gaussKernel[] = {
	2.0f,  4.0f,  5.0f,  4.0f,  2.0f,
	4.0f,  9.0f, 12.0f,  9.0f,  4.0f,
	5.0f, 12.0f, 15.0f, 12.0f,  5.0f,
	4.0f,  9.0f, 12.0f,  9.0f,  4.0f,
	2.0f,  4.0f,  5.0f,  4.0f,  2.0f
};

const float gxKernel[] = {
	1.0f,  2.0f,  -2.0f, -1.0f,
	4.0f,  8.0f,  -8.0f, -4.0f,
	6.0f, 12.0f, -12.0f, -6.0f,
	4.0f,  8.0f,  -8.0f, -4.0f,
	1.0f,  2.0f,  -2.0f, -1.0f
};

const float gyKernel[] = {
	-1.0f, -4.0f,  -6.0f, -4.0f, -1.0f,
	-2.0f, -8.0f, -12.0f, -8.0f, -2.0f,
	 2.0f,  8.0f,  12.0f,  8.0f,  2.0f,
	 1.0f,  4.0f,   6.0f,  4.0f,  1.0f
};

const float C_PIO8 = CUDART_PI_F / 8.0f;
const float C_3PIO8 = 3 * C_PIO8;
const float C_5PIO8 = 5 * C_PIO8;
const float C_7PIO8 = 7 * C_PIO8;
