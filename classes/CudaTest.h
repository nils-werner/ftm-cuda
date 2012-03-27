/* Copyright (C) 2012 Nils Werner */

#ifndef CUDATEST_H
#define CUDATEST_H

#include "Cuda.h"

#define CUDATEST_THREAD_CNT 200

class CudaTest : public Cuda {
	public:
		void invoke();
};

#endif
