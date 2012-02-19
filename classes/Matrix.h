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

		bool set(int row, int col, float value);
		int* getSize();
		int getRows();
		int getCols();

		Matrix Multiply(Matrix otherMatrix);

};

#endif