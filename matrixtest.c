#include "matrixtest.h"

int main () {
	int x = 6;

	Matrix a = m_new(100,60);
	m_filllimit(a,-3,3);
	m_print(a);


	Matrix b = m_new(60,1);
	m_filllimit(b,-4,10);
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
