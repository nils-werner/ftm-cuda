#ifndef FILTER_H
#define FILTER_H

#include "modules/matrix.h"
#include <math.h>
#include <sndfile.h>
#include "cuda/matrixmultiply.kernel.h"
#include "cuda/blockdiagmatrixmultiply.kernel.h"
#include <cuda_runtime.h>
#include <cutil.h>

int filter(float);
void initializeCoefficients(float);
void createMatrices();
void generateSignal();

#endif
