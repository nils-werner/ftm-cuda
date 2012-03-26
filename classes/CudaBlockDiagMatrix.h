/* Copyright (C) 2012 Nils Werner */

#ifndef CUDABLOCKDIAGMATRIX_H
#define CUDABLOCKDIAGMATRIX_H

#include "Cuda.h"
#include "BlockDiagMatrix.h"

class CudaBlockDiagMatrix : public BlockDiagMatrix, public Cuda {
};

#endif
