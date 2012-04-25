#include "iirfilter.h"

float l, Ts, rho, A, E, I, d1, d3, xa ;
int T, seconds, samples, filters, blocksize;
Matrix MatrixC,MatrixA,state;

int filter(float length) {
	initializeCoefficients(length);
	createMatrices();
	generateSignal();
	return 0;
}

void initializeCoefficients(float length) {
	// Saiten-Koeffizienten
	l = length;
	Ts = 60.97;
	rho = 1140;
	A = 0.5188e-6;
	E = 5.4e9;
	I = 0.171e-12;
	d1 = 8e-5;
	d3 = -1.4e-5;

	// Abtastpunkt
	xa = 0.1;

	// Abtastrate und Samplelänge
	T = 44100;
	seconds = 10;
	samples = seconds*T;
	filters = 30;

	// Blockverarbeitungs-Länge
	blocksize = 1000;

	assert(samples % blocksize == 0);
}

void createMatrices() {
	int i, mu;
	double gamma, sigma;
	double omega;
	double a, b, c1, c0;

	cudaSetDevice(0);

	MatrixC = m_new(1, 2 * filters);
	MatrixA = m_new(2 * filters, 2 * filters); //BLOCKMATRIX
	state = m_new(2 * filters, 1);

	for(i = 0; i < filters; i++) {
		mu = i+1;
		gamma = mu * ( M_PI / l );
		sigma = (1 / (2 * rho * A) ) * (d3 * pow(gamma,2) - d1);
		omega = sqrt(
				  (
					(
						(E * I)/(rho * A)
					      - pow(d3, 2)/pow(2 * rho * A, 2)
					) * pow(gamma, 4)
				  )
				+ (	(
						(Ts)/(rho * A) 
					      + (d1 + d3)/(2*pow(rho*A,2))
					) * pow(gamma, 2) )
				+ (
					pow((d1)/(2 * rho * A), 2)
				  )
			);

		a = sin(mu * M_PI * xa / l);

		b = T * sin(omega * 1 / T) / (omega * 1 / T);
		c1 = -2 * exp(sigma * 1 / T) * cos(omega * 1 / T);
		c0 = exp( 2 * sigma * 1 / T);

#if DEBUG == 1
		printf("%d %d sigma %f\n", i, mu, sigma);
		printf("      omega %f\n", omega);
#endif

		m_set(MatrixC, 0, 2*i  , 0);
		m_set(MatrixC, 0, 2*i+1, a);

		m_set(MatrixA, 2*i  , 2*i  , 0);
		m_set(MatrixA, 2*i  , 2*i+1, -c0);
		m_set(MatrixA, 2*i+1, 2*i  , 1);
		m_set(MatrixA, 2*i+1, 2*i+1, -c1);

		m_set(state, 2*i  , 0, 0);
		m_set(state ,2*i+1, 0, 1);

	}
}

void generateSignal() {
	int i, j;
	float* sample;
	Matrix block_CA, block_CA_line, block_samples;
	Matrix MatrixA_pow, MatrixA_powtmp;
	Matrix MatrixCA;
	Matrix statetmp;

	Matrix dMatrixA_pow, dblock_CA, dstate, dblock_samples;

	MatrixCA = m_multiply(MatrixC, MatrixA);

#if DEBUG == 2
	printf("MatrixA");
	m_print(MatrixA);
	printf("MatrixCA");
	m_print(MatrixCA);
	printf("MatrixC");
	m_print(MatrixC);
#endif

	sample = (float *) malloc(sizeof(float) * samples);
	blocksize = 100;

	block_CA = m_new(blocksize, MatrixA.cols);
	block_samples = m_new(blocksize,1);
	MatrixA_pow = m_new(MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_identity(MatrixA_pow);


	for(i = 1; i <= blocksize; i++) {
		block_CA_line = m_multiply(MatrixC, MatrixA_pow);
		for(j = 0; j < block_CA_line.cols; j++) {
			m_set(block_CA, i-1, j, m_get(block_CA_line, 0, j));
		}
		MatrixA_powtmp = m_multiplyblockdiag(MatrixA_pow, MatrixA, 2);
		MatrixA_pow = MatrixA_powtmp;
	}


	//MatrixA_pow.setBlocksize(2);

#if DEBUG == 3
	printf("MatrixA");
	m_print(MatrixA);
	printf("block_CA");
	m_print(block_CA);
	printf("state");
	m_print(state);
#endif

	CUDA_SAFE_CALL(cudaMalloc((void**) &dMatrixA_pow.elements, m_size(MatrixA_pow)));
	CUDA_SAFE_CALL(cudaMemcpy(dMatrixA_pow.elements,MatrixA_pow.elements, m_size(MatrixA_pow), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &dblock_CA.elements, m_size(block_CA)));
	CUDA_SAFE_CALL(cudaMemcpy(dblock_CA.elements,block_CA.elements, m_size(block_CA), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &dblock_samples.elements, m_size(block_samples)));
	CUDA_SAFE_CALL(cudaMemcpy(dblock_samples.elements,block_samples.elements, m_size(block_samples), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &dstate.elements, m_size(state)));
	CUDA_SAFE_CALL(cudaMemcpy(dstate.elements,state.elements, m_size(state), cudaMemcpyHostToDevice));

	dim3 dimBlockCA(1, 2);
	dim3 dimGridCA(block_CA.cols / dimBlockCA.x, block_CA.rows / dimBlockCA.y);

	dim3 dimBlockA(2, 2);
	dim3 dimGridA(MatrixA_pow.cols / dimBlockA.x, MatrixA_pow.rows / dimBlockA.y);

	for(i = 0; i < samples;) {
	//	block_samples = m_multiply(block_CA,state);
		MatrixMultiplyKernel<<<dimGridCA, dimBlockCA>>>(dblock_CA, dstate, dblock_samples);

#if DEBUG == 4
		m_print(MatrixA_pow);
		m_print(block_samples);
		m_print(state);
#endif
	
		cudaDeviceSynchronize();	
		cudaMemcpy(block_samples.elements, dblock_samples.elements, m_size(block_samples), cudaMemcpyDeviceToHost);

		for(j = 0; j < blocksize; j++) {
			sample[i+j] = m_get(block_samples,j,0)/128;
#if DEBUG == 10
			printf("%f, ", sample[i+j]);
#endif
		}
	//	statetmp = m_multiplyblockdiag(MatrixA_pow,state,2);
		MatrixMultiplyKernel<<<dimGridA, dimBlockA>>>(dMatrixA_pow, dstate, dstate);
	//	state = statetmp;
		i = i + blocksize;
	}

	SF_INFO info;
	info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	info.channels = 1;
	info.samplerate = T;

	SNDFILE *outfile = sf_open("filter.wav", SFM_WRITE, &info);
	assert(outfile);
	sf_writef_float(outfile, sample,samples);
}
