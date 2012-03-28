#include "cudatest.h"

int main() {
	Matrix a = m_new(100,60);
	m_filllimit(a,-3,3);
	m_print(a);
	return 0;
}
