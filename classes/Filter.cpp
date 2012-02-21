#include "Filter.h"

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

	this->createPoles();
		
	std::cout << "Hello World!";
}

void Filter::createPoles() {
	int i;
	vector<float> sigmas(this->filters);
	vector<float> omegas(this->filters);

	for(i = 0; i < this->filters; i++) {
		sigmas.push_back(1);
		omegas.push_back(2);
	}
}
