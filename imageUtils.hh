// Miguel Ascanio Gómez
// GPU - Práctica 1
#pragma once

unsigned char *readBMP(char *file_name, char header[54], int *w, int *h);
void writeBMP(float *imageFLOAT, char *file_name, char header[54], int width, int height);
float *RGB2BW(unsigned char *imageUCHAR, int width, int height);
