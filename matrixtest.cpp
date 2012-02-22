#include <iostream>
#include "classes/Filter.cpp"
#include "classes/Matrix.cpp"
#include "classes/Buffer.cpp"

using namespace std;

int main () {

	Matrix a(4,3);

	a.set(0,2,1);
	a.set(1,1,1);
	a.set(2,0,1);

	cout << a.toString() << endl << endl;

	Matrix b(3,4);

	b.set(2,0,1);
	b.set(1,0,1);
	b.set(0,2,1);
	b.set(0,1,1);
	b.set(0,0,2);

	cout << b.toString() << endl << endl;

	Matrix c = a.multiply(b);
	cout << c.toString();

	return 0;
}
