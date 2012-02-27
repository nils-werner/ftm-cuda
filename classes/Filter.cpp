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
	this->d3 = -0.1e-5;

	// Abtastpunkt
	this->xa = 0.1;

	// Abtastrate und Samplelänge
	this->T = 44100;
	this->seconds = 10;
	this->samples = seconds*T;
	this->filters = 30;

	// Blockverarbeitungs-Länge
	this->blocksize = 1000;

	assert(this->samples % this->blocksize == 0);
}

void Filter::createMatrices() {
	int i, mu;
	double gamma, sigma;
	double omega;
	double a, b, c1, c0;

	this->MatrixC.resize(1, 2 * this->filters);
	this->MatrixA.resize(2 * this->filters, 2 * this->filters, 2);
	this->state.resize(2 * this->filters, 1);

	for(i = 0; i < this->filters; i++) {
		mu = i+1;
		gamma = mu * ( M_PI / this->l );
		sigma = (1 / (2 * this->rho * this->A) ) * (this->d3 * pow(gamma,2) - this->d1);
		omega = sqrt(
				  (
					(
						(this->E * this->I)/(this->rho * this->A)
					      - pow(this->d3, 2)/pow(2 * this->rho * this->A, 2)
					) * pow(gamma, 4)
				  )
				+ (	(
						(this->Ts)/(this->rho * this->A) 
					      + (this->d1 + this->d3)/(2*pow(this->rho*this->A,2))
					) * pow(gamma, 2) )
				+ (
					pow((this->d1)/(2 * this->rho * this->A), 2)
				  )
			);

		a = sin(mu * M_PI * this->xa / this->l);

		b = this->T * sin(omega * 1 / this->T) / (omega * 1 / this->T);
		c1 = -2 * exp(sigma * 1 / this->T) * cos(omega * 1 / this->T);
		c0 = exp( 2 * sigma * 1 / this->T);

#if DEBUG == 1
		cout << i << " " << mu << " sigma " << sigma << endl;
		cout << "      omega " << omega << endl;
#endif

		this->MatrixC.set(0, 2*i  , 0);
		this->MatrixC.set(0, 2*i+1, a);

		this->MatrixA.set(2*i  , 2*i  , 0);
		this->MatrixA.set(2*i  , 2*i+1, -c0);
		this->MatrixA.set(2*i+1, 2*i  , 1);
		this->MatrixA.set(2*i+1, 2*i+1, -c1);

		this->state.set(2*i  , 0, 0);
		this->state.set(2*i+1, 0, 1);

	}
}

void Filter::generateSignal() {
	int i, j;
	float* sample;
	Matrix block_CA, block_CA_line, block_samples;
	BlockDiagMatrix MatrixA_pow;

	MatrixCA = MatrixC.multiply(MatrixA);

#if DEBUG == 2
	cout << "MatrixA";
	cout << MatrixA.toString();
	cout << "MatrixCA";
	cout << MatrixCA.toString();
	cout << "MatrixC";
	cout << MatrixC.toString();
#endif

	sample = new float[this->samples];
	this->blocksize = 100;

	block_CA.resize(blocksize, MatrixA.getCols());
	block_samples.resize(blocksize,1);
	MatrixA_pow.resize(MatrixA.getRows(), MatrixA.getCols(), 2);
	MatrixA_pow.identity();


	for(i = 1; i <= this->blocksize; i++) {
		block_CA_line = MatrixC.multiply(MatrixA_pow);
		for(j = 0; j < block_CA_line.getCols(); j++) {
			block_CA.set(i-1,j,block_CA_line.get(0,j));
		}
		MatrixA_pow = MatrixA_pow.multiply(MatrixA);
	}

	MatrixA_pow.setBlocksize(2);

#if DEBUG == 3
	cout << "MatrixA";
	cout << MatrixA.toString();
	cout << "block_CA";
	cout << block_CA.toString();
#endif

	for(i = 0; i < this->samples;) {
		block_samples = block_CA.multiply(this->state);

#if DEBUG == 4
		cout << MatrixA_pow.toString() << endl;
		cout << block_samples.toString() << endl;
		cout << this->state.toString() << endl;
#endif

		for(j = 0; j < blocksize; j++) {
			sample[i+j] = block_samples.get(j,0)/128;
#if DEBUG == 10
			cout << sample[i+j] << ", ";
#endif
		}
		this->state = MatrixA_pow.multiply(this->state);
		i = i + blocksize;
	}

	SndfileHandle outfile("filter.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, this->T);
	assert(outfile);
	outfile.write(&sample[0],this->samples);
}
