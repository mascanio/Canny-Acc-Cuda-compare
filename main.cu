// Miguel Ascanio Gómez
// GPU - Práctica 1

#include <cuda.h>

#include <functional>
#include <stdlib.h>
#include <iostream>
using namespace std;

#include <strings.h>

#include <limits.h>

#include "imageProcess.hh"
#include "benchUtils.hh"
#include "imageUtils.hh"
#include "test.hh"

cudaError_t e1;
#define TEST(x) e1 = x;\
	if (e1 != 0) { \
	cerr << "ERROR, line: " << __LINE__ << " File: " << __FILE__ << " Error: " <<  cudaGetErrorString(e1) << endl;\
	exit(-1); }

void printHelp() {

	cerr << "Uso incorrecto de los parametros. exe  input.bmp output.bmp <selection>" << endl;
	cerr << "selection: " << endl;
	cerr << "\t[cpu|CPU|c] \t\t Ejecuta en la CPU" << endl;
	cerr << "\t[cuda|CUDA] \t\t Ejecuta el \"One kernel\" implementado en  CUDA" << endl;
	cerr << "\t[cuda basic] \t\t Ejecuta el \"Kernel básico\" implementado en  CUDA" << endl;
	cerr << "\t[acc|ACC|a] \t\t Ejecuta la implementación OpenACC" << endl;
	cerr << "\t[bench|benchmark|b] \t\t Ejecuta todas las implementaciones" << endl;
	cerr << "\t[check|test|t] \t\t Comprueba las ejecuciones paralelas con el resultado de la ejecución secuencial en al CPU" << endl;
	cerr << "\t[average|time] \t\t Ejecuta 10 veces" << endl;
}

void average(std::function<void ()> f, const string& name) {
	///////////////////////
	// AVERAGE TEST VARS //
	///////////////////////
	const int N = 100;
	int i = 0;
	long r = 0, min = LONG_MAX, max = 0;
	string s;
	std::stringstream ss;

	TicToc t;
	for (i = 0; i < N; i++) {
		t.tic();
		f();
		t.toc();
		cout << name << ": " << t << endl;
		if (t.nsec < min)
			min = t.nsec;
		if (t.nsec > max)
			max = t.nsec;
		r += t.nsec;
	}
	r = (r - min - max) / (double) (N - 2);

	ss << setw(9) << setfill('0') << r;
	s = ss.str();
	for (i = s.size() - 3; i > 0; i -= 3) {
		s.insert(s.begin() + i, ',');
	}

	cout << "Average " << name << ": " << setw(11) << s << endl;
}

int main(int argc, char **argv) {

	int width, height;
	unsigned char *imageUCHAR;
	float *imageBW;
	float *imageOUT, *imageOUT2;

	char header[54];

	//Tener menos de 3 argumentos es incorrecto
	if (argc < 4) {
		cerr
				<< "Uso incorrecto de los parametros. exe  input.bmp output.bmp [cgmbfa]"
				<< endl;
		exit(EXIT_FAILURE);
	}

	cout << "Image: " << argv[1] << endl;

	// READ IMAGE & Convert image
	imageUCHAR = readBMP(argv[1], header, &width, &height);
	// imageBW allocated with mallocHost
	imageBW = RGB2BW(imageUCHAR, width, height);

	int size = sizeof(float) * width * height;

	TEST(cudaMallocHost(&imageOUT, size));

	bzero(imageOUT, size);

	TicToc t;
	GPU::initConstantMemory();
	string testSel = argv[3];

	cout << "Running test " << testSel << endl;
	if (testSel == "cpu" || testSel == "CPU" || testSel == "c") {
		t.tic();
		CPU::canny(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "CPU: " << t << endl;
	} else if (testSel == "cuda" || testSel == "CUDA") {
		t.tic();
		GPU::one(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "CUDA ONE KERNEL: " << t << endl;
	} else if (testSel == "cuda basic") {
		t.tic();
		GPU::canny(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "CUDA BASIC: " << t << endl;
	} else if (testSel == "acc" || testSel == "ACC" || testSel == "a") {
		t.tic();
		ACC::canny(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "ACC: " << t << endl;
	} else if (testSel == "bench" || testSel == "benchmark" || testSel == "b") {

		t.tic();
		CPU::canny(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "CPU:             " << t << endl;

		t.tic();
		GPU::canny(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "CUDA BASIC:      " << t << endl;

		t.tic();
		GPU::one(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "CUDA ONE KERNEL: " << t << endl;

		t.tic();
		ACC::canny(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "ACC:              " << t << endl;
	} else if (testSel == "check" || testSel == "test" || testSel == "t") {

		TEST(cudaMallocHost(&imageOUT2, size));
		t.tic();
		CPU::canny(imageBW, imageOUT, 1000.0, height, width);
		t.toc();

		cout << "CPU: " << t << endl;
		cout << "--------------------------------------------------------------------------------" << endl;

		// ACC
		bzero(imageOUT2, size);
		t.tic();
		ACC::canny(imageBW, imageOUT2, 1000.0, height, width);
		t.toc();

		cout << "ACC: " << t << endl;
		equals(imageOUT, imageOUT2, height, width);
		cout << "--------------------------------------------------------------------------------" << endl;

		// CUDA BASIC
		bzero(imageOUT2, size);
		t.tic();
		GPU::canny(imageBW, imageOUT2, 1000.0, height, width);
		t.toc();

		cout << "CUDA BASIC: " << t << endl;

		equals(imageOUT, imageOUT2, height, width);
		cout << "--------------------------------------------------------------------------------" << endl;

		t.tic();

		// CUDA ONE
		bzero(imageOUT2, size);
		GPU::one(imageBW, imageOUT2, 1000.0, height, width);
		t.toc();

		cout << "CUDA ONE: " << t << endl;

		equals(imageOUT, imageOUT2, height, width);
		cout << "--------------------------------------------------------------------------------" << endl;

		TEST(cudaFreeHost(imageOUT2));
	} else if (testSel == "average" || testSel == "time") {

		average([&]() {
			ACC::canny(imageBW, imageOUT, 1000.0, height, width);
		}, "ACC");
		cout << endl;

		average([&]() {
			GPU::one(imageBW, imageOUT, 1000.0, height, width);
		}, "Cuda one");
		cout << endl;

		average([&]() {

			GPU::canny(imageBW, imageOUT, 1000.0, height, width);
		}, "GPU normal");
	} else {
		cout << "Not Implemented yet!!" << endl;
	}

	// WRITE IMAGE
	writeBMP(imageOUT, argv[2], header, width, height);

	TEST(cudaFreeHost(imageBW));
	TEST(cudaFreeHost(imageOUT));

	return EXIT_SUCCESS;
}

