#include "matrixmultiply.kernel.h"

__global__ void MatrixMultiplyTransposedKernel(Matrix A, Matrix B, Matrix C) {
	float sum = 0;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = 0; k < B.rows; ++k) {
		sum += A.elements[i * A.cols + k] * B.elements[j * B.cols + k];
	}
	C.elements[i * C.cols + j] = sum;
}
