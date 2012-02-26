/* Copyright (C) 2012 Nils Werner */

#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>

class Matrix {

	private:
		float* matrix;
		int rows, cols;
		int getindex(int row, int col);

	public:
		Matrix();
		Matrix(int rows, int cols);
		Matrix(Matrix& m);
		~Matrix();

		void resize(int rows, int cols);
		void set(int row, int col, float value);
		float get(int row, int col);
		int* getSize();
		int getRows();
		int getCols();

		Matrix multiply(Matrix& m);
		Matrix pow(int power);

		void fill();
		void identity();
		std::string toString();
		std::string stat();

};

#endif
