#include "Matrix.h"

Matrix::Matrix(int rows, int cols) {
	int i;

	this->rows = rows;
	this->cols = cols;

	matrix = (float**) malloc(rows * sizeof(float*));
	for (i = 0; i < rows; i++){
		matrix[i] = (float*) malloc(cols * sizeof(float));
	}

	return;
}


int Matrix::getRows() {
	return this->rows;
}

int Matrix::getCols() {
	return this->cols;
}

int* Matrix::getSize() {
	static int size[] = { this->rows, this->cols };
	return size;
}



Matrix Matrix::Multiply(Matrix otherMatrix) {
	return *this;
}