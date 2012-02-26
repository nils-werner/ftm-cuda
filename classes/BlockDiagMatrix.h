
/* Copyright (C) 2012 Nils Werner */

#ifndef BLOCKDIAGMATRIX_H
#define BLOCKDIAGMATRIX_H

#include "Matrix.h"

class BlockDiagMatrix : public Matrix {
	protected:
		int blocksize;

	public:
		BlockDiagMatrix();
		BlockDiagMatrix(int rows, int cols, int blocksize);
		BlockDiagMatrix(Matrix& m);

		void resize(int rows, int cols, int blocksize);

		Matrix multiply(Matrix& m);
};

#endif
