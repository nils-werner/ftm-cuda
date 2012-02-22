#include <iostream>
#include "classes/Filter.cpp"
#include "classes/Matrix.cpp"
#include "classes/Buffer.cpp"

using namespace std;

int main () {

	Matrix test;

	test.resize(20,10);

	cout << test.getRows() << " " << test.getCols() << endl;

	test.fill();
	Matrix b(test);

	cout << b.get(0,0) << endl;

	return 0;
}
