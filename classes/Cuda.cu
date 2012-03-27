#include "Cuda.h"

void Cuda::copyToDevice(void** hostData, void** deviceData) {
	/*
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_A.elements, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));
	*/
}

void Cuda::copyToHost(void** deviceData, void** hostData) {
	/*
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_B.elements, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyDeviceToHost));
	*/
}

void Cuda::malloc(void** devicePtr, int devicesize) {
	/*
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_C.elements, size));
	*/
}

void Cuda::free(void** deviceData) {
	/*
	cudaFree(d_A.elements);
	*/
}

void Cuda::invoke() {
	dim3 dimBlock(blocksize, blocksize);
	dim3 dimGrid(1 / dimBlock.x, 1 / dimBlock.y);

	CudaClassKernel<<<dimGrid, dimBlock>>>();
	CUT_CHECK_ERROR("Kernel execution failed\n");
}
