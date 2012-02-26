#include "BlockDiagMatrix.h"

BlockDiagMatrix::BlockDiagMatrix() : Matrix() {
}

BlockDiagMatrix::BlockDiagMatrix(int rows, int cols, int blocksize) : Matrix(rows, cols) {
	this->blocksize = blocksize;
}

BlockDiagMatrix::BlockDiagMatrix(Matrix& m) : Matrix(m) {
}

void BlockDiagMatrix::resize(int rows, int cols, int blocksize) {
	this->blocksize = blocksize;
	return Matrix::resize(rows, cols);
}

Matrix BlockDiagMatrix::multiply(Matrix& m) {
	int i, j, k;
	float sum = 0;

	assert(this->cols == m.getRows());

	Matrix result(this->rows, m.getCols());

	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < m.getCols(); j++) {
			for(k = 0; k < m.getRows(); k++) {
				sum = sum + (get(i,k) * m.get(k,j));
			}
			result.set(i,j, sum);
			sum = 0;
		}
	}
	return result;
}
