#ifndef BLOCKDIAGBLOCKDIAGMATRIXMULTIPLYKERNEL_H
#define BLOCKDIAGBLOCKDIAGMATRIXMULTIPLYKERNEL_H

#include "../modules/matrix.h"

__global__ void BlockDiagBlockDiagMatrixMultiplyKernel(Matrix, Matrix, Matrix, int);

#endif
