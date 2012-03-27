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
		void copyToDevice(void**,void**);
		void copyToHost(void**,void**);
		void malloc(void**, int);
		void free(void**);
		void invoke();
};

#endif
