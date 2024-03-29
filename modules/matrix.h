#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.h"

typedef struct {
	int rows;
	int cols;
	float* elements;
} Matrix;

void m_new(Matrix*, int, int);
void m_free(Matrix*);
void m_set(Matrix*, int, int, float);
float m_get(Matrix*, int, int);
void m_prepare_multiply(Matrix*, Matrix*, Matrix*);
void m_multiply(Matrix*, Matrix*, Matrix*);
void m_blockdiagmultiply(Matrix*, Matrix*, Matrix*,  int);
void m_multiplyblockdiag(Matrix*, Matrix*, Matrix*,  int);
void m_multiplyblockdiagblockdiag(Matrix*, Matrix*, Matrix*,  int);
void m_identity(Matrix*);
void m_fill(Matrix*);
void m_filllimit(Matrix*, float, float);
void m_print(Matrix*);
void m_stat(Matrix*);
size_t m_size(Matrix*);
void m_swap(Matrix**, Matrix**);


// Matrix m_pow(Matrix, int);
// string m_toString();

#endif
