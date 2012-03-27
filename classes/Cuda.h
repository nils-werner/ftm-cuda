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

		void copyToDevice(void**,void**);
		void copyToHost(void**,void**);
		void malloc(void**, int);
		void free(void**);
		void invoke();
};

__global__ void CudaClassKernel() {
};

#endif
