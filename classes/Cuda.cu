#include "Cuda.h"

__global__ void CudaClassKernel() {
};

void Cuda::copyToDevice(void* hostData, void* deviceData, size_t size) {
	CUDA_SAFE_CALL(cudaMalloc((void**)&deviceData, size));
	CUDA_SAFE_CALL(cudaMemcpy(deviceData, hostData, size, cudaMemcpyHostToDevice));
}

void Cuda::copyToHost(void* deviceData, void* hostData, size_t size) {
	CUDA_SAFE_CALL(cudaMalloc((void**)&deviceData, size));
	CUDA_SAFE_CALL(cudaMemcpy(deviceData, hostData, size, cudaMemcpyDeviceToHost));
}

void Cuda::malloc(void* devicePtr, size_t size) {
	CUDA_SAFE_CALL(cudaMalloc((void**)&devicePtr, size));
}

void Cuda::free(void* deviceData) {
	cudaFree(deviceData);
}

void Cuda::invoke() {
	dim3 dimBlock(blocksize, blocksize);
	dim3 dimGrid(1 / dimBlock.x, 1 / dimBlock.y);

	CudaClassKernel<<<dimGrid, dimBlock>>>();
	CUT_CHECK_ERROR("Kernel execution failed\n");
}
