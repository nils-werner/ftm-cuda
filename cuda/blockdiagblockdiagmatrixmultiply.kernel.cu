#include "blockdiagblockdiagmatrixmultiply.kernel.h"

__global__ void BlockDiagBlockDiagMatrixMultiplyKernel(Matrix A, Matrix B, Matrix C, int blocksize) {
	float sum = 0;
	int from, to;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;


	i = blocksize * (j / blocksize) + i;
	from = blocksize * (i / blocksize);
	to = from + blocksize;

	for (int k = from; k < to; ++k) {
		sum += A.elements[i * A.cols + k] * B.elements[k * B.cols + j];
	}
	C.elements[i * C.cols + j] = sum;
}
