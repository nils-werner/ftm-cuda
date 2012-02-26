#include "Filter.h"

using namespace std;

Filter::Filter(float length = 0.65) {
	this->initializeCoefficients(length);
	this->createMatrices();
	this->generateSignal();
}

void Filter::initializeCoefficients(float length) {
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
	this->seconds = 1;
	this->samples = seconds*T;
	this->filters = 30;

	// Blockverarbeitungs-Länge
	this->blocksize = 100;
}

void Filter::createMatrices() {
	int i, mu;
	double gamma, sigma;
	double omega;
	double a, b, c1, c0;

	this->MC.resize(1, 2 * this->filters);
	this->MA.resize(2 * this->filters, 2 * this->filters, 2);
	this->Mstate.resize(2 * this->filters, 1);

	for(i = 0; i < this->filters; i++) {
		mu = i+1;
		gamma = mu * ( M_PI / this->l );
		sigma = (1 / (2 * this->rho * this->A) ) * (this->d3 * pow(gamma,2) - this->d1);
		omega = sqrt(
				  (
					(
						( (this->E * this->I) / (this->rho * this->A) )
						- ( pow(this->d3, 2) / pow(2 * this->rho * this->A, 2) )
					) * pow(gamma, 4) 
				  )
				+ ( (this->Ts / (this->rho * this->A) ) * pow(gamma, 2) )
				// + ( pow(this->d1 / (2 * this->rho * this->A), 2) )
			);

		a = sin(mu * M_PI * this->xa / this->l);

		b = this->T * sin(omega * 1 / this->T) / (omega * 1 / this->T);
		c1 = -2 * exp(sigma * 1 / this->T) * cos(omega * 1 / this->T);
		c0 = exp( 2 * sigma * 1 / this->T);

#if DEBUG == 1
		cout << i << " " << mu << " sigma " << sigma << endl;
		cout << "      omega " << omega << endl;
#endif

		this->MC.set(0, 2*i  , 0);
		this->MC.set(0, 2*i+1, a);

		this->MA.set(2*i  , 2*i  , 0);
		this->MA.set(2*i  , 2*i+1, -c0);
		this->MA.set(2*i+1, 2*i  , 1);
		this->MA.set(2*i+1, 2*i+1, -c1);

		this->Mstate.set(2*i  , 0, 1);
		this->Mstate.set(2*i+1, 0, 0);

	}
}

void Filter::generateSignal() {
	int i;
	float* sample;

	MCA = MC.multiply(MA);

#if DEBUG == 2
	cout << "MA";
	cout << MA.toString();
	cout << "MCA";
	cout << MCA.toString();
	cout << "MC";
	cout << MC.toString();
#endif

	SndfileHandle outfile("filter.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, this->T);
	assert(outfile);

	sample = new float[this->samples];

	for(i = 0; i < this->samples; i++) {
		sample[i] = this->MCA.multiply(this->Mstate).get(0,0)/128;
#if DEBUG == 10
		cout << sample[i] << ", ";
#endif
		this->Mstate = this->MA.multiply(this->Mstate);
	}

	outfile.write(&sample[0],this->samples);
}
