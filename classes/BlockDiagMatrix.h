
/* Copyright (C) 2012 Nils Werner */

#ifndef BLOCKDIAGMATRIX_H
#define BLOCKDIAGMATRIX_H

#include "Matrix.h"

class BlockDiagMatrix : public Matrix {
	protected:
		int blocksize;

	public:
		BlockDiagMatrix();
		BlockDiagMatrix(int rows, int cols);
		BlockDiagMatrix(int rows, int cols, int blocksize);
		BlockDiagMatrix(Matrix& m);
		BlockDiagMatrix(Matrix& m, int blocksize);

		void resize(int rows, int cols, int blocksize);
		int getBlocksize();
		void setBlocksize(int blocksize);
		std::string toString();

		Matrix multiply(Matrix& m);
		void fill();
		BlockDiagMatrix multiply(BlockDiagMatrix& m);
};

#endif
