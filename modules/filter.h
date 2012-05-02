#ifndef FILTER_H
#define FILTER_H

#include "modules/matrix.h"
#include <math.h>
#include <sndfile.h>
#include "cuda/matrixmultiply.kernel.h"
#include "cuda/blockdiagmatrixmultiply.kernel.h"
#include <cuda_runtime.h>
#include <cutil.h>

int filter(float, int, int);
void initializeCoefficients(float, int, int);
void createMatrices();
void createBlockprocessingMatrices();
void generateSignal();

#endif
