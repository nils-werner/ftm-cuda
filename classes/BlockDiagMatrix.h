
/* Copyright (C) 2012 Nils Werner */

#ifndef BLOCKDIAGMATRIX_H
#define BLOCKDIAGMATRIX_H

#include "Matrix.h"

class BlockDiagMatrix : public Matrix {
	public:
		BlockDiagMatrix() : Matrix() {}
		BlockDiagMatrix(int rows, int cols) : Matrix(rows, cols) {}
		BlockDiagMatrix(Matrix& m) : Matrix(m) {}
};

#endif
