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

	this->sigmas.resize(this->filters);
	this->omegas.resize(this->filters);

	this->createPoles();
		
	std::cout << "Hello World!";
}

void Filter::createPoles() {
	int i;
	double gamma, sigma, omega;

	for(i = 0; i < this->filters; i++) {
		gamma = i * ( M_PI / this->l );
		sigma = (1 / (2 * this->rho * this->A) ) * (this->d3 * pow(gamma,2) - this->d1);
		omega = sqrt(
				  ( (this->E * this->I) / (this->rho * this->A) )
				- ( (pow(this->d3, 2) / pow(2 * this->rho * this->A, 2)) * pow(gamma, 4) )
				+ ( (this->Ts / (this->rho * this->A) ) * pow(gamma, 2) )
				+ ( pow(this->d1 / (2 * this->rho * this->A), 2) )
			);
		
		this->sigmas.push_back(sigma);
		this->omegas.push_back(omega);
		
	}
}

void Filter::createMatrices() {
	int i;

	Matrix A(this->filters * 2, this->filters * 2);

	for(i = 0; i < this->filters; i++) {
	
	}

	A = A.pow(this->blocksize);
}
