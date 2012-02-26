#include "Matrix.h"

using namespace std;

Matrix::Matrix() {
	matrix = new float[1];
	this->rows = 1;
	this->cols = 1;
}

Matrix::Matrix(int rows, int cols) {
	matrix = new float[1];
	this->rows = 1;
	this->cols = 1;
	resize(rows, cols);
}

Matrix::Matrix(Matrix& m) {
	int i, j;
	matrix = new float[1];
	this->rows = 1;
	this->cols = 1;
	resize(m.getRows(), m.getCols());

	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < this->cols; j++) {
			set(i, j, m.get(i, j));
		}
	}
}

Matrix::~Matrix() {
	//delete[] matrix;
}

void Matrix::resize(int rows, int cols) {
	int i, j;

	assert(rows > 0);
	assert(cols > 0);

	delete matrix;
	matrix = new float[rows * cols];

	this->rows = rows;
	this->cols = cols;

	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			set(i, j, 0);
		}
	}

	return;
}

int Matrix::getindex(int row, int col) {
	assert(row < this->rows);
	assert(col < this->cols);

	return row * this->cols + col;
}

void Matrix::set(int row, int col, float value) {
	this->matrix[getindex(row, col)] = value;
}

float Matrix::get(int row, int col) {
	return matrix[getindex(row, col)];
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



Matrix Matrix::multiply(Matrix& m) {
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

Matrix Matrix::pow(int power) {
	int i;
	Matrix result(*this);

	assert(power > 0);

	for(i = 1; i < power; i++) {
		result = this->multiply(result);
	}

	return result;
}

void Matrix::fill() {
	int i, j;

	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < this->cols; j++) {
			set(i, j, rand());
		}
	}
}

void Matrix::identity() {
	int i, j;

	assert(this->rows == this->cols);

	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < this->cols; j++) {
			if(i == j)
				set(i, j, 1);
			else
				set(i, j, 0);
		}
	}
}

string Matrix::toString() {
	std::ostringstream val;
	int i, j;
	string result("");

	val << setprecision(4) << noshowpoint;
	val << "\n\n" << this->stat();
	result += val.str();


	for(i = 0; i < this->rows; i++) {
		for(j = 0; j < this->cols; j++) {
			val.str("");
			val << get(i, j);
			result += val.str();
			result += "\t";
		}
		result += "\n";
	}

	return result;
}
