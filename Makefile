MODE = cuda

CUDA_INSTALL_PATH ?= /usr/baetz/cuda
CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc


# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include 

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS) -DDEBUG=$(DEBUG)
CFLAGS += $(COMMONFLAGS)

.PHONY: all clean preview

DEBUG = 0
LIBS := -L$(CUDA_INSTALL_PATH)/lib64 -lasound -lsndfile
ifeq ($(MODE),cuda)
	LIBS += -lcudart
endif

OBJS = classes/Filter.cpp.o classes/Matrix.cpp.o classes/BlockDiagMatrix.cpp.o classes/Buffer.cpp.o main.cpp.o

all: build/iirfilter build/matrixtest

clean:
	- rm -f build/*
	- rm -f classes/*.o
	- rm -f *.o

time: build/iirfilter
	time -p ./build/iirfilter

preview: time
	cvlc filter.wav vlc://quit

build/%: main.cpp.o
	$(LINK) -o $@ $< $(LIBS)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
