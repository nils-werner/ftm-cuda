#ifndef BLOCKDIAGMATRIXMULTIPLYKERNEL_H
#define BLOCKDIAGMATRIXMULTIPLYKERNEL_H

#include "../modules/matrix.h"

__global__ void BlockDiagMatrixMultiplyKernel(Matrix, Matrix, Matrix, int);

#endif
