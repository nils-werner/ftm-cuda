#include "iirfilter.h"

float l, Ts, rho, A, E, I, d1, d3, xa ;
int T, seconds, samples, filters, blocksize;
Matrix MatrixC, MatrixA, state;
Matrix MatrixAp, MatrixCA;
Matrix output_chunk_read, output_chunk_write;
Matrix *pointer_output_chunk_read, *pointer_output_chunk_write;

/**
 * Wrapper for the methods required in the filter, just calls them in the correct order
 *
 * @param int length
 * @return int 0
 */

int filter(float par_length, int par_samples, int par_blocksize) {
	initializeCoefficients(par_length, par_blocksize, par_samples);
	createMatrices();
	createBlockprocessingMatrices();
	generateSignal();
	return 0;
}











/**
 * Initializes coefficients for a standard Nylon-b-String. The values are, with exception for
 * length (l) are hard-coded according to the values in \cite{rabenstein03}.
 *
 * @param float length
 * @return void
 */

void initializeCoefficients(float length, int par_blocksize, int par_samples) {
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
	samples = par_samples;
	filters = 30;

	blocksize = par_blocksize;

	assert(samples % blocksize == 0);
}















/**
 * Creates the required matrices by calculating the required number of poles using the equations
 * to be found in \cite{rabenstein03}. The matrices generated are not yet in blockprocessing form.
 *
 * @param void
 * @return void
 */

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
















/**
 * Generates matrices suitable for blockprocessing
 *
 * @param int blocksize
 * @return void
 */ 

void createBlockprocessingMatrices() {
	int i, j;
	Matrix MatrixCA_line, MatrixAp_tmp;

	MatrixCA = m_new(blocksize, MatrixA.cols);
	output_chunk_read = m_new(blocksize,1);
	output_chunk_write = m_new(blocksize,1);
	pointer_output_chunk_read = &output_chunk_read;
	pointer_output_chunk_write = &output_chunk_write;
	MatrixAp = m_new(MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_identity(MatrixAp);


	for(i = 1; i <= blocksize; i++) {
		MatrixCA_line = m_multiply(MatrixC, MatrixAp);
		for(j = 0; j < MatrixCA_line.cols; j++) {
			m_set(MatrixCA, i-1, j, m_get(MatrixCA_line, 0, j));
		}
		m_free(MatrixCA_line);
		MatrixAp_tmp = m_multiplyblockdiag(MatrixAp, MatrixA, 2);

		m_free(MatrixAp);
		MatrixAp = MatrixAp_tmp;
	}

#if DEBUG == 3
	printf("MatrixA");
	m_print(MatrixA);
	printf("MatrixCA");
	m_print(MatrixCA);
	printf("state");
	m_print(state);
#endif
}



















/**
 * Generates the signal using the matrices generated earlier.
 * The signal is generated in chunks the size of the first parameter. The space for the resulting signal is pre-
 * allocated earlier and being filled by the filter. These values are then passed on to the sndfile library and
 * written to the file `filter.wav`.
 *
 * @param void
 * @return void
 */

void generateSignal() {
	int i, j;
	float* output;


#if DEBUG == 2
	printf("MatrixA");
	m_print(MatrixA);
	printf("MatrixC");
	m_print(MatrixC);
#endif

	output = (float *) malloc(sizeof(float) * samples);

#if MODE == 1
	Matrix device_MatrixAp;
	Matrix device_MatrixCA;
	Matrix device_state_read, device_state_write;
	Matrix *pointer_device_state_read, *pointer_device_state_write;
	Matrix device_output_chunk_read, device_output_chunk_write;
	Matrix *pointer_device_output_chunk_read, *pointer_device_output_chunk_write;

	device_output_chunk_read = m_new(blocksize,1);
	device_output_chunk_write = m_new(blocksize,1);
	device_MatrixCA = m_new(blocksize, MatrixA.cols);
	device_MatrixAp = m_new(MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	device_state_read = m_new(2 * filters, 1);
	device_state_write = m_new(2 * filters, 1);

	pointer_device_state_read = &device_state_read;
	pointer_device_state_write = &device_state_write;
	pointer_device_output_chunk_read = &device_output_chunk_read;
	pointer_device_output_chunk_write = &device_output_chunk_write;

	cudaSetDevice(0);

	cudaStream_t streams[3];

	cudaEvent_t MatrixCA_start, MatrixCA_stop;
	cudaEvent_t MatrixAp_start, MatrixAp_stop;
	cudaEvent_t Memcpy_start, Memcpy_stop;

	cudaEventCreate(&MatrixCA_start);
	cudaEventCreate(&MatrixCA_stop);
	cudaEventCreate(&MatrixAp_start);
	cudaEventCreate(&MatrixAp_stop);
	cudaEventCreate(&Memcpy_start);
	cudaEventCreate(&Memcpy_stop);

	float MatrixCA_time, MatrixAp_time, Memcpy_time;


	for(int i = 0; i < 3; i++) {
		cudaStreamCreate(& streams[i]);
	}

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixAp.elements, m_size(MatrixAp)));
	CUDA_SAFE_CALL(cudaMemcpy(device_MatrixAp.elements, MatrixAp.elements, m_size(MatrixAp), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixCA.elements, m_size(MatrixCA)));
	CUDA_SAFE_CALL(cudaMemcpy(device_MatrixCA.elements, MatrixCA.elements, m_size(MatrixCA), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_output_chunk_read.elements, m_size(output_chunk_read)));
	CUDA_SAFE_CALL(cudaMemcpy(device_output_chunk_read.elements, output_chunk_read.elements, m_size(output_chunk_read), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_output_chunk_write.elements, m_size(output_chunk_write)));
	CUDA_SAFE_CALL(cudaMemcpy(device_output_chunk_write.elements, output_chunk_write.elements, m_size(output_chunk_write), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_state_read.elements, m_size(state)));
	CUDA_SAFE_CALL(cudaMemcpy(device_state_read.elements, state.elements, m_size(state), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_state_write.elements, m_size(state)));
	CUDA_SAFE_CALL(cudaMemcpy(device_state_write.elements, state.elements, m_size(state), cudaMemcpyHostToDevice));

	dim3 dimBlockCA(1, 1);
	dim3 dimGridCA(state.cols / dimBlockCA.x, MatrixCA.rows / dimBlockCA.y);

	dim3 dimBlockA(1, 1);
	dim3 dimGridA(state.cols / dimBlockA.x, MatrixAp.rows / dimBlockA.y);

	cudaEventRecord(MatrixCA_start, streams[0]);
	MatrixMultiplyKernel<<<dimGridCA, dimBlockCA, 1, streams[0]>>>(device_MatrixCA, *pointer_device_state_read, *pointer_device_output_chunk_write);
	cudaEventRecord(MatrixCA_stop, streams[0]);

	cudaEventRecord(MatrixAp_start, streams[1]);
	MatrixMultiplyKernel<<<dimGridA, dimBlockA, 1, streams[1]>>>(device_MatrixAp, *pointer_device_state_read, *pointer_device_state_write);
	cudaEventRecord(MatrixAp_stop, streams[1]);
#else
	Matrix state_tmp;
#endif

	for(i = 0; i < samples;) {
#if MODE == 1
		/*
	       	 * CUDA IMPLEMENTATION
		 */

		cudaThreadSynchronize();

		cudaEventElapsedTime(&MatrixCA_time, MatrixCA_start, MatrixCA_stop);
		cudaEventElapsedTime(&MatrixAp_time, MatrixAp_start, MatrixAp_stop);
		cudaEventElapsedTime(&Memcpy_time, Memcpy_start, Memcpy_stop);

		if(i == 5*blocksize) {
			printf("MatrixCA: %d\n", MatrixCA_time);
			printf("MatrixAp: %d\n", MatrixAp_time);
			printf("  Memcpy: %d\n", Memcpy_time);
		}

		m_swap(&pointer_device_state_read, &pointer_device_state_write);
		m_swap(&pointer_device_output_chunk_read, &pointer_device_output_chunk_write);
		m_swap(&pointer_output_chunk_read, &pointer_output_chunk_write);

		cudaEventRecord(MatrixCA_start, streams[0]);
		MatrixMultiplyKernel<<<dimGridCA, dimBlockCA, 1, streams[0]>>>(device_MatrixCA, *pointer_device_state_read, *pointer_device_output_chunk_write);
		cudaEventRecord(MatrixCA_stop, streams[0]);

		cudaEventRecord(MatrixAp_start, streams[1]);
		BlockDiagMatrixMultiplyKernel<<<dimGridA, dimBlockA, 1, streams[1]>>>(device_MatrixAp, *pointer_device_state_read, *pointer_device_state_write, 2);
		cudaEventRecord(MatrixAp_stop, streams[1]);

		cudaEventRecord(Memcpy_start, streams[2]);
		cudaMemcpyAsync(pointer_output_chunk_write->elements, pointer_device_output_chunk_read->elements, m_size(output_chunk_write), cudaMemcpyDeviceToHost, streams[2]);
		cudaEventRecord(Memcpy_stop, streams[2]);

		for(j = 0; j < blocksize; j++) {
			output[i+j] = m_get(*pointer_output_chunk_read,j,0)/128;
		}

#else
		/*
	       	 * CPU IMPLEMENTATION
		 */
		output_chunk_write = m_multiply(MatrixCA,state);

		for(j = 0; j < blocksize; j++) {
			output[i+j] = m_get(output_chunk_write,j,0)/128;
		}
		state_tmp = m_multiplyblockdiag(MatrixAp,state,2);
		m_free(state);
		state = state_tmp;
#endif

		i = i + blocksize;
	}


	SF_INFO info;
	info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	info.channels = 1;
	info.samplerate = T;

	SNDFILE *outfile = sf_open("filter.wav", SFM_WRITE, &info);
	assert(outfile);
	sf_writef_float(outfile, output,samples);
}
