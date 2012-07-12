#include "matrixblockdiagmultiply.kernel.h"

__global__ void MatrixBlockDiagMultiplyKernel(Matrix A, Matrix B, Matrix C, int blocksize) {
	float sum = 0;
	int from, to;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	from = blocksize * (j / blocksize);
	to = from + blocksize;

	for (int k = from; k < to; ++k) {
		sum += A.elements[i * A.cols + k] * B.elements[k * B.cols + j];
	}
	C.elements[i * C.cols + j] = sum;
}
