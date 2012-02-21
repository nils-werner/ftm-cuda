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

	this->MC.resize(1, 2 * this->filters);
	this->MA.resize(2 * this->filters, 2 * this->filters);
	this->Mstate.resize(2 * this->filters, 1);

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

		MC.set(0, i  , 0);
		MC.set(0, i+1, a);

		MA.set(i  , i  , 0);
		MA.set(i  , i+1, -c0);
		MA.set(i+1, i  , 1);
		MA.set(i+1, i+1, -c1);

		Mstate.set(i  , 0, 1);
		Mstate.set(i+1, 0, 0);

	}
}
