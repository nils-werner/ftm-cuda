#include <iostream>
#include "classes/Filter.cpp"
#include "classes/Matrix.cpp"
#include "classes/Buffer.cpp"

using namespace std;

int main () {

	Matrix test(20,10);
	Matrix b(10,20);

	cout << test.getRows() << " " << test.getCols();

	test.get(0,0);

	return 0;
}
