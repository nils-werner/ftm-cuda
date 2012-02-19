/* Copyright (C) 2012 Nils Werner */

#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

class Matrix {

	private:
		float** matrix;

	public:
		Matrix(int rows, int cols);

};

#endif