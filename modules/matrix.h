#ifndef MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct {
	int rows;
	int cols;
	float* elements;
} Matrix;

Matrix m_new(int, int);
void m_set(Matrix, int, int, float);
float m_get(Matrix, int, int);
Matrix m_multiply(Matrix, Matrix);
void m_identity(Matrix);
void m_fill(Matrix);


// Matrix m_pow(Matrix, int);
// string m_toString();

#endif
