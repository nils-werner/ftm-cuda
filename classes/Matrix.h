/* Copyright (C) 2012 Nils Werner */

#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string>

class Matrix {

	private:
		float** matrix;
		int rows, cols;

	public:
		Matrix();
		Matrix(int rows, int cols);
		Matrix(const Matrix& m);

		void resize(int rows, int cols);
		void set(int row, int col, float value);
		float get(int row, int col);
		int* getSize();
		int getRows();
		int getCols();

		Matrix multiply(const Matrix& m);
		Matrix pow(int power);

		void fill();
		std::string toString();

};

#endif
