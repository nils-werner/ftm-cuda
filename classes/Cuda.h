/* Copyright (C) 2012 Nils Werner */

#ifndef CUDA_H
#define CUDA_H

#include <stdio.h>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <cutil.h>

class Cuda {
	protected:
		int blocksize, gridsize;

	public:
		Cuda();
		void setDevice(int);
		void copyToDevice(void*, void*, size_t);
		void copyToHost(void*, void*, size_t);
		void malloc(void*, size_t);
		void free(void*);
		void invoke();
};

#endif
