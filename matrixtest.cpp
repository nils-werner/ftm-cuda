#include <iostream>
#include "classes/Filter.cpp"
#include "classes/Matrix.cpp"
#include "classes/BlockDiagMatrix.cpp"
#include "classes/Buffer.cpp"

using namespace std;

int main () {
	int x = 6;

	BlockDiagMatrix a;
	a.resize(x,x,2);

	a.identity();


	BlockDiagMatrix b(x,x,2);
	b.fill();

	cout << a.toString() << endl;
	cout << b.toString() << endl;
	cout << a.multiply(b).toString();

	return 0;
}
