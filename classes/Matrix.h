/* Copyright (C) 2012 Nils Werner */

#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

class Matrix {

	private:
		float** matrix;
		int rows, cols;

	public:
		Matrix(int rows, int cols);
		Matrix(const Matrix& m);

		void set(int row, int col, float value);
		float get(int row, int col);
		int* getSize();
		int getRows();
		int getCols();

		Matrix multiply(Matrix otherMatrix);
		Matrix pow(int power);

};

#endif
