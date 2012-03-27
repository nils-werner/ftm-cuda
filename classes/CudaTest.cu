#include "CudaTest.h"

__global__ void CudaTestClassKernel(float * res, float* arg, int N) {
	int i;
	for (i = threadIdx.x; i < N; i += CUDATEST_THREAD_CNT) {
		res[i] = __cosf(arg[i] );
	}
};

void CudaTest::invoke() {
	int N = 10000;
	float* cosRes = 0;
	float* cosArg = 0;
	float* arg = (float *) malloc(N*sizeof(float));
	float* res = (float *) malloc(N*sizeof(float));

	// cudaStat = cudaMalloc ((void **)&cosArg, N * sizeof(cosArg[0]));
	//mallocOnDevice(cosArg, N*sizeof(float));

	//cudaStat = cudaMalloc ((void **)&cosRes, N * sizeof(cosRes[0]));
	mallocOnDevice((void **) &cosRes, N*sizeof(float));

	//cudaStat = cudaMemcpy (cosArg, arg, N * sizeof(arg[0]), cudaMemcpyHostToDevice);
	copyToDevice(cosArg, arg, N*sizeof(float));

	CudaTestClassKernel<<<1, CUDATEST_THREAD_CNT>>>(res, arg, N);
	CUT_CHECK_ERROR("Kernel execution failed\n");
}
