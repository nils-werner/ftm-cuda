#ifndef CUDATEST_H
#define CUDATEST_H

#include "modules/matrix.h"
#include "cuda/matrixmultiply.kernel.h"
#include "cuda/blockdiagmatrixmultiply.kernel.h"
#include <cuda_runtime.h>
#include <cutil.h>

#define BLOCK_SIZE 10

int main();

#endif
