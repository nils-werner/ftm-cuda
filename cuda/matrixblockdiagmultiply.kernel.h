#ifndef MATRIXBLOCKDIAGMULTIPLYKERNEL_H
#define MATRIXBLOCKDIAGMULTIPLYKERNEL_H

#include "../modules/matrix.h"

__global__ void MatrixBlockDiagMultiplyKernel(Matrix, Matrix, Matrix, int);

#endif
