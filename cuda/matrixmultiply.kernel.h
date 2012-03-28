#ifndef MATRIXMULTIPLYKERNEL_H
#define MATRIXMULTIPLYKERNEL_H

#include "../modules/matrix.h"

__global__ void MatrixMultiplyKernel(Matrix, Matrix, Matrix);

#endif
