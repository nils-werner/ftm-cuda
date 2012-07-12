#ifndef MATRIXMULTIPLYTRANSPOSEDKERNEL_H
#define MATRIXMULTIPLYTRANSPOSEDKERNEL_H

#include "../modules/matrix.h"

__global__ void MatrixMultiplyTransposedKernel(Matrix, Matrix, Matrix);

#endif
