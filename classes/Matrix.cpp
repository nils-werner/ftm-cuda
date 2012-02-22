#include "Matrix.h"

Matrix::Matrix() {
	this->rows = 0;
	this->cols = 0;
}

Matrix::Matrix(int rows, int cols) {
	resize(rows, cols);
}

Matrix::Matrix(const Matrix& m) {
	int i, j;

	resize(m.rows, m.cols);

	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < this->cols; j++) {
			this->matrix[i][j] = m.matrix[i][j];
		}
	}
}

void Matrix::resize(int rows, int cols) {
	int i;

	if(this->rows > 0 && this->cols > 0) {
		for(i = 0; i < this->rows; i++) {
			free(this->matrix[i]);
		}
		free(this->matrix);

	}

	this->matrix = (float**) malloc(rows * sizeof(float*));
	for (i = 0; i < rows; i++){
		this->matrix[i] = (float*) malloc(cols * sizeof(float));
	}

	this->rows = rows;
	this->cols = cols;

	return;
}

void Matrix::set(int row, int col, float value) {
	this->matrix[row][col] = value;
}

float Matrix::get(int row, int col) {
	return matrix[row][col];
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



Matrix Matrix::multiply(Matrix m) {
	int i, j, k;
	float sum = 0;
	Matrix result(this->getRows(), m.getCols());

	for(i = 0; i < result.getRows(); i++) {
		for(j = 0; j < result.getCols(); j++) {
			for(k = 0; k < this->getCols(); k++) {
				sum = sum + this->get(i,k) * m.get(k,j);
			}
			result.set(i,j, sum);
			sum = 0;
		}
	}
}

Matrix Matrix::pow(int power) {
	int i;
	Matrix result(*this);

	for(i = 1; i < power; i++) {
		result = this->multiply(result);
	}

	return result;
}

void Matrix::fill() {
	int i, j;

	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < this->cols; j++) {
			this->matrix[i][j] = rand();
		}
	}
}
