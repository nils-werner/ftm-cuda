#include "matrixtest.h"

int main () {
	int x = 6;

	Matrix a = m_new(x,x);
	m_fill(a);

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
