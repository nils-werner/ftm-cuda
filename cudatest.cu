#include "cudatest.h"

int main() {
	/*
	 *
	 * MATRIZEN ERZEUGEN
	 *
	 */

	Matrix a = m_new(10,10);
	m_filllimit(a,-3,3);
	m_stat(a);

	Matrix b = m_new(10,10);
	m_filllimit(b,-3,3);
	m_stat(b);


	
	/*
	 *
	 * CUDA
	 *
	 */

	printf("CUDAing\n");

	cudaSetDevice(0);

	Matrix da, db, dc, c;
	size_t size;

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
		m_print(b);
	}

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(b.cols / dimBlock.x, a.rows / dimBlock.y);

	for(int i = 0; i < 5; i++) {
		MatrixMultiplyKernel<<<dimGrid, dimBlock>>>(da, db, db);
		CUT_CHECK_ERROR("Kernel execution failed\n");

		size = db.rows * db.cols * sizeof(float);
		cudaMemcpy(c.elements, db.elements, size, cudaMemcpyDeviceToHost);

		m_print(c);
	}

	return 0;
}
