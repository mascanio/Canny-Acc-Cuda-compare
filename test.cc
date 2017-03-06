// Miguel Ascanio Gómez
// GPU - Práctica 1
#include "test.hh"
#include <iostream>
using namespace std;

#ifdef DEBUG
#include <strings.h>
#endif

#include "imageProcess.hh"
#include "imageUtils.hh"
#include "benchUtils.hh"

#include <math.h>

const float TOLERANCIA = 1e-2;

bool equals(const float* C_cpu, const float* C_gpu, int height, int width) {
	int c = 0;

	for (int i = 5; i < height-5; i++) {
		for (int j = 5; j < width-5; j++) {
			if (fabsf(C_cpu[i * width + j] - C_gpu[i * width + j]) > TOLERANCIA) {
				cout << "i: " << i << " j: " << j << " " << setprecision(51) << C_cpu[i * width + j] << " "
						<< C_gpu[i * width + j] << endl;
				c++;
			}
		}
	}
		cout << "Errores: " << c << endl;

	return c == 0;
}
