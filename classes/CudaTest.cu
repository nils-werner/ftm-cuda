#include "CudaTest.h"

__global__ void CudaTestClassKernel() {
};

void CudaTest::invoke() {
	dim3 dimBlock(1, 1);
	dim3 dimGrid(1 / dimBlock.x, 1 / dimBlock.y);

	CudaTestClassKernel<<<dimGrid, dimBlock>>>();
	CUT_CHECK_ERROR("Kernel execution failed\n");
}
