#include "Matrix.h"

Matrix::Matrix(int rows, int cols) {
	int i;

	matrix = (float**) malloc(rows * sizeof(float*));
	for (i = 0; i < rows; i++){
		matrix[i] = (float*) malloc(cols * sizeof(float));
	}

	return;
}