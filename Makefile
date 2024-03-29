CUDA_INSTALL_PATH := /usr/baetz/cuda
CUDA_SDK_PATH := /HOMES/werner/NVIDIA_GPU_Computing_SDK
CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc


# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc

# Common flags
COMMONFLAGS += $(INCLUDES) -DDEBUG=$(DEBUG)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
.PHONY: all cpu clean time preview

DEBUG = 0
LIBS := -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_SDK_PATH)/C/lib -lcudart -lcutil_x86_64 -lasound -lsndfile



### PHONY RULES ###

default: build/iirfilter
all: build/iirfilter build/cudatest
	
clean:
	- rm -f build/*
	- rm -f classes/*.o
	- rm -f modules/*.o
	- rm -f cuda/*.o
	- rm -f *.o
	- rm filter.wav

cpu: MODE=0
cpu: clean all

gpu: clean all

md5: time
	md5sum filter.wav

time: build/iirfilter
	./build/iirfilter gpu
	./build/iirfilter cpu

preview: time
	cvlc filter.wav vlc://quit




### ABHAENGIGKEITEN ###

build/iirfilter: iirfilter.cu.o modules/filter.cu.o modules/matrix.c.o modules/utils.c.o cuda/matrixmultiply.kernel.cu.o cuda/blockdiagmatrixmultiply.kernel.cu.o cuda/matrixblockdiagmultiply.kernel.cu.o cuda/blockdiagblockdiagmatrixmultiply.kernel.cu.o modules/settings.h
build/cudatest: cudatest.cu.o modules/matrix.c.o modules/utils.c.o cuda/matrixmultiply.kernel.cu.o cuda/blockdiagmatrixmultiply.kernel.cu.o






### DATEIENDUNGEN ###

build/%: 
	$(LINK) -o $@ $^ $(LIBS)

%.c.o: %.c %.h
	$(CXX) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
