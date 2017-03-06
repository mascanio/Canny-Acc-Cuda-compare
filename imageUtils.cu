// Miguel Ascanio Gómez
// GPU - Práctica 1
#include "imageUtils.hh"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

unsigned char *readBMP(char *file_name, char header[54], int *w, int *h) {
	//Se abre el fichero en modo binario para lectura
	FILE *f = fopen(file_name, "rb");
	if (!f) {
		perror(file_name);
		exit(1);
	}

	// Cabecera archivo imagen
	//***********************************

	//Devuelve cantidad de bytes leidos
	int n = fread(header, 1, 54, f);

	//Si no lee 54 bytes es que la imagen de entrada es demasiado pequeña
	if (n != 54)
		fprintf(stderr, "Entrada muy pequeña (%d bytes)\n", n), exit(1);

	//Si los dos primeros bytes no corresponden con los caracteres BM no es un fichero BMP
	if (header[0] != 'B' || header[1] != 'M')
		fprintf(stderr, "No BMP\n"), exit(1);

	//El tamaño de la imagen es el valor de la posicion 2 de la cabecera menos 54 bytes que ocupa esa cabecera
	int imagesize = *(int*) (header + 2) - 54;

	//Si la cabecera no tiene el tamaño de 54 o el numero de bits por pixel es distinto de 24 la imagen se rechaza
	if (*(int*) (header + 10) != 54 || *(short*) (header + 28) != 24)
		fprintf(stderr, "No color 24-bit\n"), exit(1);

	//Cuando la posicion 30 del header no es 0, es que existe compresion por lo que la imagen no es valida
	if (*(int*) (header + 30) != 0)
		fprintf(stderr, "Compresion no suportada\n"), exit(1);

	//Se recupera la altura y anchura de la cabecera
	int width = *(int*) (header + 18);
	int height = *(int*) (header + 22);

	//**************************************
	// Lectura de la imagen
	//*************************************

	//Se reservan "imagesize+256+width*6" bytes y se devuelve un puntero a estos datos
	unsigned char *image = (unsigned char*) malloc(imagesize + 256 + width * 6);

	image += 128 + width * 3;
	if ((n = fread(image, 1, imagesize + 1, f)) != imagesize)
		fprintf(stderr, "File size incorrect: %d bytes read insted of %d\n", n,
				imagesize), exit(1);

	fclose(f);
	printf("Image read correctly (width=%i height=%i, imagesize=%i).\n", width,
			height, imagesize);

	/* Output variables */
	*w = width;
	*h = height;

	return (image);
}

void writeBMP(float *imageFLOAT, char *file_name, char header[54], int width, int height)
{

	FILE *f;
	int i, n;

	int imagesize=*(int*)(header+2)-54;

	unsigned char *image = (unsigned char*)malloc(3*sizeof(unsigned char)*width*height);

	for (i=0;i<width*height;i++){
		image[3*i]   = imageFLOAT[i]; //R
		image[3*i+1] = imageFLOAT[i]; //G
		image[3*i+2] = imageFLOAT[i]; //B
	}


	f=fopen(file_name, "wb");		//Se abre el fichero en modo binario de escritura
	if (!f){
		perror(file_name);
		exit(1);
	}

	n=fwrite(header, 1, 54, f);		//Primeramente se escribe la cabecera de la imagen
	n+=fwrite(image, 1, imagesize, f);	//Y despues se escribe el resto de la imagen
	if (n!=54+imagesize)			//Si se han escrito diferente cantidad de bytes que la suma de la cabecera y el tamaño de la imagen. Ha habido error
		fprintf(stderr, "Escritos %d de %d bytes\n", n, imagesize+54);
	fclose(f);

	free(image);

}

float *RGB2BW(unsigned char *imageUCHAR, int width, int height)
{
	int i, j;
	//float *imageBW = (float *)malloc(sizeof(float)*width*height);
	float *imageBW;
	cudaError_t e;
	if((e = cudaMallocHost(&imageBW, sizeof(float)*width*height)) != 0) {
		cerr << "ERROR, line: " << __LINE__ << " File: " << __FILE__ << " Error: " <<  cudaGetErrorString(e) << endl;
		exit(-1);
	}

	unsigned char R, B, G;

	for (i=0; i<height; i++)
		for (j=0; j<width; j++)
		{
			R = imageUCHAR[3*(i*width+j)];
			G = imageUCHAR[3*(i*width+j)+1];
			B = imageUCHAR[3*(i*width+j)+2];

			imageBW[i*width+j] = 0.2989 * R + 0.5870 * G + 0.1140 * B;
		}

	return(imageBW);
}
