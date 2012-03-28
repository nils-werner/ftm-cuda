#include "matrixmultiply.kernel.h"

__global__ void MatrixMultiplyKernel(Matrix A, Matrix B, Matrix C) {
	float sum = 0;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = 0; k < B.rows; ++k) {
		sum += A.elements[i * A.cols + k] * B.elements[k * B.cols + j];
	}
	C.elements[i * C.cols + j] = sum;
}
