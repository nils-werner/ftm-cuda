#include "Cuda.h"

void Cuda::copyToDevice() {
	/*
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_A.elements, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));
	*/
}

void Cuda::copyToHost() {
	/*
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_B.elements, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyDeviceToHost));
	*/
}

void Cuda::malloc() {
	/*
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_C.elements, size));
	*/
}

void Cuda::free() {
	/*
	cudaFree(d_A.elements);
	*/
}

void Cuda::invoke() {
	/*
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	CUT_CHECK_ERROR("Kernel execution failed\n");
	*/
}
