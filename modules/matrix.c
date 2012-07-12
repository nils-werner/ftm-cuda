#include "matrix.h"

/**
 * Creates a new matrix by allocating space and pre-initializing size information
 *
 * @param int rows
 * @param int cols
 * @return void
 */

void m_new(Matrix* m, int rows, int cols) {
	m->rows = rows;
	m->cols = cols;

	m->elements= (float*) malloc(sizeof(float) * rows * cols);
}

/**
 * Frees the space allocated in a matrix
 *
 * @param Matrix m
 * @return void
 */

void m_free(Matrix* m) {
	free(m->elements);
}

/**
 * Sets a specific cell in a matrix
 *
 * @param Matrix m
 * @param int row
 * @param int col
 * @param float value
 * @return void
 */

void m_set(Matrix* m, int row, int col, float value) {
	assert(row < m->rows);
	assert(col < m->cols);

	m->elements[row * m->cols + col] = value;
}

/**
 * Fetches the value of a specific cell in a matrix
 *
 * @param Matrix m
 * @param int row
 * @param int col
 * @return float
 */

float m_get(Matrix* m, int row, int col) {
	assert(row < m->rows);
	assert(col < m->cols);

	return m->elements[row * m->cols + col];
}

/**
 * Frees and reinitializes an existing matrix to make room for the result of multiplication
 *
 * @param Matrix a
 * @param Matrix b
 * @param Matrix* c
 * @return void
 */

void m_prepare_multiply(Matrix* a, Matrix* b, Matrix* c) {
	m_new(c, a->rows, b->cols);
}
/**
 * Multiplies two generic matrices
 *
 * @param Matrix a
 * @param Matrix b
 * @return Matrix
 */

void m_multiply(Matrix* a, Matrix* b, Matrix* c) {
	int i, j, k;
	float sum = 0;

	assert(a->cols == b->rows);
	assert(c->rows == a->rows);
	assert(c->cols == b->cols);

	for(i = 0; i < a->rows; i++) {
		for(j = 0; j < b->cols; j++) {
			for(k = 0; k < b->rows; k++) {
				sum = sum + m_get(a,i,k) * m_get(b,k,j);
			}
			m_set(c, i, j, sum);
			sum = 0;
		}
	}
	return;
}

/**
 * Multiplies a matrix with another, blockdiagonal matrix in a more efficient manner.
 *
 * @param Matrix a
 * @param Matrix b
 * @param int blocksize
 * @return Matrix
 */

void m_blockdiagmultiply(Matrix* a, Matrix* b, Matrix* c, int blocksize) {
	int i, j, k, from, to;
	float sum = 0;

	assert(a->cols == b->rows);
	assert(c->rows == a->rows);
	assert(c->cols == b->cols);

	for(i = 0; i < a->rows; i++) {
		for(j = 0; j < b->cols; j++) {
			from = blocksize * (j / blocksize);
			to = from + blocksize;
			for(k = from; k < to; k++) {
				sum = sum + m_get(a,i,k) * m_get(b,k,j);
			}
			m_set(c, i, j, sum);
			sum = 0;
		}
	}
	return;
}
/**
 * Multiplies a matrix with another, blockdiagonal matrix in a more efficient manner.
 *
 * @param Matrix a
 * @param Matrix b
 * @param int blocksize
 * @return Matrix
 */

void m_multiplyblockdiag(Matrix* a, Matrix* b, Matrix* c, int blocksize) {
	int i, j, k, from, to;
	float sum = 0;

	assert(a->cols == b->rows);
	assert(c->rows == a->rows);
	assert(c->cols == b->cols);

	for(i = 0; i < a->rows; i++) {
		for(j = 0; j < b->cols; j++) {
			from = blocksize * (i / blocksize);
			to = from + blocksize;
			for(k = from; k < to; k++) {
				sum = sum + m_get(a,i,k) * m_get(b,k,j);
			}
			m_set(c, i, j, sum);
			sum = 0;
		}
	}
	return;
}
/**
 * Multiplies two blockdiagonal matrices in a more efficient manner.
 *
 * @param Matrix a
 * @param Matrix b
 * @param int blocksize
 * @return Matrix
 */

void m_multiplyblockdiagblockdiag(Matrix* a, Matrix* b, Matrix* c, int blocksize) {
	int i, j, k, from, to;
	float sum = 0;

	assert(a->cols == b->rows);
	assert(c->rows == a->rows);
	assert(c->cols == b->cols);

	for(i = 0; i < a->rows; i++) {
		from = blocksize * (i / blocksize);
		to = from + blocksize;
		for(j = from; j < to; j++) {
			for(k = from; k < to; k++) {
				sum = sum + m_get(a,i,k) * m_get(b,k,j);
			}
			m_set(c, i, j, sum);
			sum = 0;
		}
	}
	return;
}

/**
 * Fills a square matrix with 1 on the main diagonal, 0 elswhere
 *
 * @param Matrix m
 * @return void
 */

void m_identity(Matrix* m) {
	int i, j;

	assert(m->rows == m->cols);

	for(i = 0; i < m->rows; i++) {
		for(j = 0; j < m->cols; j++) {
			if(i == j)
				m_set(m, i, j, 1);
			else
				m_set(m, i, j, 0);
		}
	}
}

/**
 * Fills a matrix with random numbers from 0 to 1
 *
 * @param Matrix m
 * @return void
 */

void m_fill(Matrix* m) {
	m_filllimit(m, 0.0, 1.0);
}

/**
 * Fills a matrix with random numbers between the parameters min and max

 * @param Matrix m
 * @param float min
 * @param float max
 * @return void
 */

void m_filllimit(Matrix* m, float min, float max) {
	int i, j;

	for(i = 0; i < m->rows; i++) {
		for(j = 0; j < m->cols; j++) {
			m_set(m, i, j, (fl_rand() * (max-min)) + min);
		}
	}
}

/**
 * Prints an entire matrix to stdout
 *
 * @param Matrix m
 * @return void
 */

void m_print(Matrix* m) {
	int i, j;

	m_stat(m);
	printf("\n");

	for(i = 0; i < m->rows; i++) {
		for(j = 0; j < m->cols; j++) {
			printf("%5.2f ", m_get(m, i, j));
		}
		printf("\n");
	}
	printf("\n\n");
}

/**
 * Prints basic size information of a matrix to stdout
 *
 * @param Matrix m
 * @return void
 */

void m_stat(Matrix* m) {
	printf("Matrix %dx%d\n", m->rows, m->cols);
}

/**
 * Returns the size of an existing matrix to be used in memory allocation.
 *
 * @param Matrix m
 * @return size_t
 */

size_t m_size(Matrix* m) {
	return m->rows * m->cols * sizeof(float);
}

/**
 * Transposes a matrix
 *
 * @param *Matrix a
 * @return void
 */

void m_transpose(Matrix *m) {
	float tmp;
	int tmpint;
	for (int i = 0; i < m->cols; i++) {
		for (int j = i+1; j < m->rows; j++) {
			tmp = m->elements[i * m->cols + j];
			m->elements[i * m->cols + j] = m->elements[j * m->cols + i];
			m->elements[j * m->cols + i] = tmp;
		}
	}
	tmpint = m->rows;
	m->rows = m->cols;
	m->cols = tmpint;
}
/**
 * Swaps two matrix pointers, called by reference
 *
 * @param *Matrix a
 * @param *Matrix b
 * @return void
 */

void m_swap(Matrix **a, Matrix **b) {
	Matrix *tmp = *a;
	*a = *b;
	*b = tmp;
}
