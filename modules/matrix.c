#include "matrix.h"

Matrix m_new(int rows, int cols) {
	Matrix m;

	m.rows = rows;
	m.cols = cols;

	m.elements= (float*) malloc(sizeof(float) * rows * cols);

	return m;
}

void m_free(Matrix m) {
	free(m.elements);
}

void m_set(Matrix m, int row, int col, float value) {
	m.elements[row * m.cols + col] = value;
}

float m_get(Matrix m, int row, int col) {
	return m.elements[row * m.cols + col];
}

Matrix m_multiply(Matrix a, Matrix b) {
	int i, j, k;
	float sum = 0;

	assert(a.cols == b.rows);

	Matrix m = m_new(a.rows, b.cols);

	for(i = 0; i < a.rows; i++) {
		for(j = 0; j < b.cols; j++) {
			for(k = 0; k < b.rows; k++) {
				sum = sum + m_get(a,i,k) * m_get(b,k,j);
			}
			m_set(m, i, j, sum);
			sum = 0;
		}
	}
	return m;
}

void m_identity(Matrix m) {
	int i, j;

	assert(m.rows = m.cols);

	for(i = 0; i < m.rows; i++) {
		for(j = 0; j < m.cols; j++) {
			if(i == j)
				m_set(m, i, j, 1);
			else
				m_set(m, i, j, 0);
		}
	}
}

void m_fill(Matrix m) {
	m_filllimit(m, 0.0, 1.0);
}

void m_filllimit(Matrix m, float min, float max) {
	int i, j;

	for(i = 0; i < m.rows; i++) {
		for(j = 0; j < m.cols; j++) {
			m_set(m, i, j, (fl_rand() * (max-min)) + min);
		}
	}
}

void m_print(Matrix m) {
	int i, j;

	printf("Matrix %dx%d\n\n", m.rows, m.cols);

	for(i = 0; i < m.rows; i++) {
		for(j = 0; j < m.cols; j++) {
			printf("%5.2f ", m_get(m, i, j));
		}
		printf("\n");
	}
	printf("\n\n");
}
