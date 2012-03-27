#include "matrixtest.h"

int main () {
	int x = 6;

	Matrix a = m_new(x,x);
	m_fill(a);
	m_print(a);


	Matrix b = m_new(x,x);
	m_identity(b);
	m_print(b);

	Matrix c = m_multiply(a,b);
	m_print(c);

	/*
	BlockDiagMatrix a;
	a.resize(x,x,2);

	a.identity();


	BlockDiagMatrix b(x,x,2);
	b.fill();

	cout << a.toString() << endl;
	cout << b.toString() << endl;
	cout << a.multiply(b).toString();
	*/

	return 0;
}
