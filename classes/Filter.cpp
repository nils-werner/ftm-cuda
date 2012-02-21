#include "Filter.h"
#include <vector>

using namespace std;

Filter::Filter(float length = 0.65) {
	// Saiten-Koeffizienten
	this->l = length;
	this->Ts = 60.97;
	this->rho = 1140;
	this->A = 0.5188e-6;
	this->E = 5.4e9;
	this->I = 0.171e-12;
	this->d1 = 8e-5;
	this->d3 = -1.4e-5;

	// Abtastpunkt
	this->xa = 0.1;

	// Abtastrate und Samplelänge
	this->T = 44100;
	this->seconds = 10;
	this->samples = seconds*T;
	this->filters = 30;

	// Blockverarbeitungs-Länge
	this->blocksize = 100;

	// Ausgangssignal
	// y = zeros(1,samples);


	this->createMatrices();
		
	std::cout << "Hello World!";
}

void Filter::createMatrices() {
	int i;
	double gamma, sigma, omega;
	double a, b, c1, c0;

	Matrix C(1, 2 * this->filters);
	Matrix A(2 * this->filters, 2 * this->filters);
	Matrix state(2 * this->filters, 1);

	for(i = 0; i < this->filters; i++) {
		gamma = i * ( M_PI / this->l );
		sigma = (1 / (2 * this->rho * this->A) ) * (this->d3 * pow(gamma,2) - this->d1);
		omega = sqrt(
				  ( (this->E * this->I) / (this->rho * this->A) )
				- ( (pow(this->d3, 2) / pow(2 * this->rho * this->A, 2)) * pow(gamma, 4) )
				+ ( (this->Ts / (this->rho * this->A) ) * pow(gamma, 2) )
				+ ( pow(this->d1 / (2 * this->rho * this->A), 2) )
			);
		
		a = sin(i * M_PI * this->xa / this->l);

		b = this->T * sin(omega * this->l / this->T) / (omega * this->l / this->T);
		c1 = -2 * exp(sigma * this->l / this->T) * cos(omega * this->l / this->T);
		c0 = exp( 2 * sigma * this->l / T);

		C.set(0, i  , 0);
		C.set(0, i+1, a);

		A.set(i  , i  , 0);
		A.set(i  , i+1, -c0);
		A.set(i+1, i  , 1);
		A.set(i+1, i+1, -c1);

		state.set(i  , 0, 1);
		state.set(i+1, 0, 0);

	}
}
