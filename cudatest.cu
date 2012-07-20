#include "cudatest.h"

/* Just some tests to make sure matrix multiplications work on the GPU */

int main() {
	/*
	 *
	 * MATRIZEN ERZEUGEN
	 *
	 */

	int matsize = 100;

	Matrix a = m_new(matsize,matsize);
	m_filllimit(a,-3,3);
	m_stat(a);

	Matrix b = m_new(matsize,matsize);
	m_filllimit(b,-3,3);
	m_stat(b);


	
	/*
	 *
	 * CUDA
	 *
	 */

	printf("CUDAing\n");

	cudaSetDevice(0);

	cudaStream_t streams[3];
	Matrix da, db, dc, c;
	size_t size;

	for(int i = 0; i < 3; i++) {
		cudaStreamCreate(& streams[i]);
	}


	c = m_new(a.rows, b.cols);

	da.rows = a.rows;
	da.cols = a.cols;

	db.rows = b.rows;
	db.cols = b.cols;

	dc.rows = a.rows;
	dc.cols = b.cols;

	size = da.rows * da.cols * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void**) &da.elements, size));
	CUDA_SAFE_CALL(cudaMemcpy(da.elements,a.elements, size, cudaMemcpyHostToDevice));

	size = db.rows * db.cols * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void**) &db.elements, size));
	CUDA_SAFE_CALL(cudaMemcpy(db.elements,b.elements, size, cudaMemcpyHostToDevice));

	size = dc.rows * dc.cols * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void**) &dc.elements, size));

	for(int i = 0; i < 5; i++) {
		b = m_multiply(a,b);
	}

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(b.cols / dimBlock.x, a.rows / dimBlock.y);

	for(int i = 0; i < 5; i++) {
		size = db.rows * db.cols * sizeof(float);
		if(i % 2 == 0) {
			MatrixMultiplyKernel<<<dimGrid, dimBlock, 1, streams[0]>>>(da, db, dc);
			CUT_CHECK_ERROR("Kernel execution failed\n");

			cudaMemcpyAsync(c.elements, db.elements, size, cudaMemcpyDeviceToHost, streams[1]);
		}
		else {
			MatrixMultiplyKernel<<<dimGrid, dimBlock, 1, streams[0]>>>(da, dc, db);
			CUT_CHECK_ERROR("Kernel execution failed\n");

			cudaMemcpyAsync(c.elements, dc.elements, size, cudaMemcpyDeviceToHost, streams[1]);

		}

		cudaStreamSynchronize(streams[0]);
		cudaStreamSynchronize(streams[1]);
	}

	return 0;
}
