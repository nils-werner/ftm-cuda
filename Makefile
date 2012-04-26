CUDA_INSTALL_PATH := /usr/baetz/cuda
CUDA_SDK_PATH := /HOMES/werner/NVIDIA_GPU_Computing_SDK
CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc


# Includes
INCLUDES = -I.

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS) -DDEBUG=$(DEBUG)
CFLAGS += $(COMMONFLAGS) -DDEBUG=$(DEBUG)

.PHONY: all cpu clean time preview

DEBUG = 0
LIBS := -lasound -lsndfile



### PHONY RULES ###

default: build/iirfilter
all: build/iirfilter build/matrixtest
	
clean:
	- rm -f build/*
	- rm -f classes/*.o
	- rm -f modules/*.o
	- rm -f *.o

md5: time
	md5sum filter.wav

time: build/iirfilter
	time -p ./build/iirfilter

preview: time
	cvlc filter.wav vlc://quit




### ABHAENGIGKEITEN ###

build/iirfilter: iirfilter.c.o modules/filter.c.o modules/matrix.c.o modules/utils.c.o
build/matrixtest: matrixtest.c.o modules/matrix.c.o modules/utils.c.o
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
