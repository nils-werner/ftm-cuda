/* Copyright (C) 2012 Nils Werner */

#ifndef CUDA_H
#define CUDA_H

#include <stdio.h>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <cutil.h>

class Cuda {
	private:
		int blocksize, gridsize;

		void copyToDevice();
		void copyToHost();
		void malloc();
		void free();
		void invoke();
};

__global__ void CudaClassKernel() {
};

#endif
