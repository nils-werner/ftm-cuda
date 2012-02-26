#include <iostream>
#include "classes/Filter.cpp"
#include "classes/Matrix.cpp"
#include "classes/BlockDiagMatrix.cpp"
#include "classes/Buffer.cpp"

using namespace std;

int main () {
	int x = 60;

	Matrix a(x,x);
	a.identity();

	cout << a.toString() << endl << endl;

	Matrix b(x,1);
	b.fill();

	Matrix c(1,x);
	c.fill();

	cout << c.multiply(a).toString();

	return 0;
}
