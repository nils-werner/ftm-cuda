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
	float* output;
	Matrix MatrixCA, MatrixCA_line, output_chunk;
	Matrix MatrixAp, tmp_MatrixAp;

	Matrix device_MatrixAp, device_MatrixCA, device_state1, device_state2, device_tmp_state, device_output_chunk;

	cudaSetDevice(0);

#if DEBUG == 2
	printf("MatrixA");
	m_print(MatrixA);
	printf("MatrixC");
	m_print(MatrixC);
#endif

	output = (float *) malloc(sizeof(float) * samples);
	blocksize = 100;

	device_MatrixCA = m_new(blocksize, MatrixA.cols);
	device_output_chunk = m_new(blocksize,1);
	device_MatrixAp = m_new(MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	device_state1 = m_new(2 * filters, 1);
	device_state2 = m_new(2 * filters, 1);

	MatrixCA = m_new(blocksize, MatrixA.cols);
	output_chunk = m_new(blocksize,1);
	MatrixAp = m_new(MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_identity(MatrixAp);


	for(i = 1; i <= blocksize; i++) {
		MatrixCA_line = m_multiply(MatrixC, MatrixAp);
		for(j = 0; j < MatrixCA_line.cols; j++) {
			m_set(MatrixCA, i-1, j, m_get(MatrixCA_line, 0, j));
		}
		m_free(MatrixCA_line);
		tmp_MatrixAp = m_multiplyblockdiag(MatrixAp, MatrixA, 2);

		m_free(MatrixAp);
		MatrixAp = tmp_MatrixAp;
	}


	//MatrixAp.setBlocksize(2);

#if DEBUG == 3
	printf("MatrixA");
	m_print(MatrixA);
	printf("MatrixCA");
	m_print(MatrixCA);
	printf("state");
	m_print(state);
#endif

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixAp.elements, m_size(MatrixAp)));
	CUDA_SAFE_CALL(cudaMemcpy(device_MatrixAp.elements, MatrixAp.elements, m_size(MatrixAp), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixCA.elements, m_size(MatrixCA)));
	CUDA_SAFE_CALL(cudaMemcpy(device_MatrixCA.elements, MatrixCA.elements, m_size(MatrixCA), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_output_chunk.elements, m_size(output_chunk)));
	CUDA_SAFE_CALL(cudaMemcpy(device_output_chunk.elements, output_chunk.elements, m_size(output_chunk), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_state1.elements, m_size(state)));
	CUDA_SAFE_CALL(cudaMemcpy(device_state1.elements, state.elements, m_size(state), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_state2.elements, m_size(state)));
	CUDA_SAFE_CALL(cudaMemcpy(device_state2.elements, state.elements, m_size(state), cudaMemcpyHostToDevice));

	dim3 dimBlockCA(1, 1);
	dim3 dimGridCA(state.cols / dimBlockCA.x, MatrixCA.rows / dimBlockCA.y);

	dim3 dimBlockA(1, 1);
	dim3 dimGridA(state.cols / dimBlockA.x, MatrixAp.rows / dimBlockA.y);


	for(i = 0; i < samples;) {
	//	output_chunk = m_multiply(MatrixCA,state);
		MatrixMultiplyKernel<<<dimGridCA, dimBlockCA>>>(device_MatrixCA, device_state1, device_output_chunk);

#if DEBUG == 4
		m_print(MatrixAp);
		m_print(output_chunk);
		m_print(state);
#endif
	
		cudaDeviceSynchronize();	
		cudaMemcpy(output_chunk.elements, device_output_chunk.elements, m_size(output_chunk), cudaMemcpyDeviceToHost);

		//printf("%d\n", i);

	//	if(i == 0)
	//		m_print(output_chunk);

		for(j = 0; j < blocksize; j++) {
			output[i+j] = m_get(output_chunk,j,0)/128;
#if DEBUG == 10
			printf("%f, ", output[i+j]);
#endif
		}
	//	tmp_state = m_multiplyblockdiag(MatrixAp,state,2);
		MatrixMultiplyKernel<<<dimGridA, dimBlockA>>>(device_MatrixAp, device_state1, device_state2);
		device_tmp_state = device_state1;
		device_state1 = device_state2;
		device_state2 = device_tmp_state;
	//	state = tmp_state;
		i = i + blocksize;
	}


//	m_print(MatrixAp);

	SF_INFO info;
	info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	info.channels = 1;
	info.samplerate = T;

	SNDFILE *outfile = sf_open("filter.wav", SFM_WRITE, &info);
	assert(outfile);
	sf_writef_float(outfile, output,samples);
}
