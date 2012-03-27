CUDA_INSTALL_PATH := /usr/baetz/cuda
CUDA_SDK_PATH := /HOMES/werner/NVIDIA_GPU_Computing_SDK
CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc


# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS) -DDEBUG=$(DEBUG)
CFLAGS += $(COMMONFLAGS)

.PHONY: all cpu clean time preview

DEBUG = 0
LIBS := -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_SDK_PATH)/C/lib -lcudart -lcutil_x86_64 -lasound -lsndfile



### PHONY RULES ###

all: build/iirfilter build/matrixtest build/cudatest
	
clean:
	- rm -f build/*
	- rm -f classes/*.o
	- rm -f *.o

time: build/iirfilter
	time -p ./build/iirfilter

preview: time
	cvlc filter.wav vlc://quit




### ABHAENGIGKEITEN ###

build/iirfilter: iirfilter.c.o modules/matrix.c.o modules/utils.c.o
build/matrixtest: matrixtest.c.o modules/matrix.c.o modules/utils.c.o
build/cudatest: cudatest.cpp.o classes/Cuda.cu.o classes/CudaTest.cu.o





### DATEIENDUNGEN ###

build/%: 
	$(LINK) -o $@ $^ $(LIBS)

%.c.o: %.c %.h
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
