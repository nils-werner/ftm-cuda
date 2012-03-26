/* Copyright (C) 2012 Nils Werner */

#ifndef CUDAMATRIX_H
#define CUDAMATRIX_H

#include "Cuda.h"
#include "Matrix.h"

class CudaMatrix : public Matrix, public Cuda {
};

#endif
