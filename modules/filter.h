#ifndef FILTER_H
#define FILTER_H

#include "modules/matrix.h"
#include <math.h>
#include <sndfile.h>
#include "cuda/matrixmultiply.kernel.h"
#include "cuda/blockdiagmatrixmultiply.kernel.h"
#include <cuda_runtime.h>
#include <cutil.h>

typedef struct {
	float l;
	float Ts;
	float rho;
	float A;
	float E;
	float I;
	float d1;
	float d3;
	float xa;
} String;

typedef struct {
	int T;
	int seconds;
	int samples;
	int filters;
	int blocksize;
} Synthesizer;

int filter(int, float, int, int, int);
void initializeCoefficients(float, int, int, int);
void createMatrices();
void createBlockprocessingMatrices();
void generateSignalCPU(float*, String, Synthesizer);
void generateSignalGPU(float*, String, Synthesizer);
void writeFile(const char*, float*, int, int);

#endif
