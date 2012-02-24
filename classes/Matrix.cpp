#include "Matrix.h"
#include <string>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;

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
	int i, j;

	if(this->rows > 0 && this->cols > 0) {
		for(i = 0; i < this->rows; i++) {
			free(this->matrix[i]);
		}
		free(this->matrix);

	}

	this->matrix = (float**) malloc(rows * sizeof(float*));
	for (i = 0; i < rows; i++){
		this->matrix[i] = (float*) malloc(cols * sizeof(float));
		for(j = 0; j < cols; j++)
			this->matrix[i][j] = 0;
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

string Matrix::stat() {
	std::ostringstream val;

	val << this->rows << "x" << this->cols << " Matrix\n";
	return val.str();
}



Matrix Matrix::multiply(const Matrix& m) {
	int i, j, k;
	float sum = 0;

	if(this->cols != m.rows)
		throw 30;

	Matrix result(this->rows, m.cols);

	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < m.cols; j++) {
			for(k = 0; k < m.rows; k++) {
				sum = sum + (this->matrix[i][k] * m.matrix[k][j]);
			}
			result.set(i,j, sum);
			sum = 0;
		}
	}
	return result;
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

void Matrix::identity() {
	int i, j;

	if(this->rows != this->cols)
		throw 200;

	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < this->cols; j++) {
			if(i == j)
				this->matrix[i][j] = 1;
			else
				this->matrix[i][j] = 0;
		}
	}
}

string Matrix::toString() {
	std::ostringstream val;
	int i, j;
	string result("");

	val << "\n\n" << this->stat();
	result += val.str();


	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < this->cols; j++) {
			val.str("");
			val << this->matrix[i][j];
			result += val.str();
			result += "\t";
		}
		result += "\n";
	}

	return result;
}
