#include "iirfilter.h"

String string;
Synthesizer synth;
Matrix MatrixC, MatrixA, state;
Matrix MatrixAp, MatrixCA;

/**
 * Wrapper for the methods required in the filter, just calls them in the correct order
 *
 * @param int length
 * @return int 0
 */

int filter(float length, int samples, int blocksize, int filters) {
	initializeCoefficients(length, blocksize, samples, filters);

	float * output = (float *) malloc(sizeof(float) * synth.samples);
	createMatrices();
	createBlockprocessingMatrices();

#if DEBUG == 2
	printf("MatrixA");
	m_print(MatrixA);
	printf("MatrixC");
	m_print(MatrixC);
#endif

	generateSignalGPU(output, string, synth);
	writeFile("filter.wav", output, synth.samples, synth.T);
	return 0;
}











/**
 * Initializes coefficients for a standard Nylon-b-String. The values are, with exception for
 * length (l) are hard-coded according to the values in \cite{rabenstein03}.
 *
 * @param float length
 * @return void
 */

void initializeCoefficients(float length, int blocksize, int samples, int filters) {
	// Saiten-Koeffizienten
	string.l = length;
	string.Ts = 60.97;
	string.rho = 1140;
	string.A = 0.5188e-6;
	string.E = 5.4e9;
	string.I = 0.171e-12;
	string.d1 = 8e-5;
	string.d3 = -1.4e-5;

	// Abtastpunkt
	string.xa = 0.1;

	// Abtastrate und Samplelänge
	synth.T = 44100;
	synth.seconds = 10;
	synth.samples = samples;
	synth.filters = filters;
	synth.blocksize = blocksize;

	assert(synth.samples % synth.blocksize == 0);
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

	m_new(&MatrixC, 1, 2 * synth.filters);
	m_new(&MatrixA, 2 * synth.filters, 2 * synth.filters); //BLOCKMATRIX
	m_new(&state, 2 * synth.filters, 1);

	for(i = 0; i < synth.filters; i++) {
		mu = i+1;
		gamma = mu * ( M_PI / string.l );
		sigma = (1 / (2 * string.rho * string.A) ) * (string.d3 * pow(gamma,2) - string.d1);
		omega = sqrt(
				  (
					(
						(string.E * string.I)/(string.rho * string.A)
					      - pow(string.d3, 2)/pow(2 * string.rho * string.A, 2)
					) * pow(gamma, 4)
				  )
				+ (	(
						(string.Ts)/(string.rho * string.A) 
					      + (string.d1 + string.d3)/(2*pow(string.rho*string.A,2))
					) * pow(gamma, 2) )
				+ (
					pow((string.d1)/(2 * string.rho * string.A), 2)
				  )
			);

		a = sin(mu * M_PI * string.xa / string.l);

		b = synth.T * sin(omega * 1 / synth.T) / (omega * 1 / synth.T);
		c1 = -2 * exp(sigma * 1 / synth.T) * cos(omega * 1 / synth.T);
		c0 = exp( 2 * sigma * 1 / synth.T);

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
	Matrix *pointer_MatrixAp, *pointer_MatrixAp_tmp;

	pointer_MatrixAp = &MatrixAp;
	pointer_MatrixAp_tmp = &MatrixAp_tmp;

	m_new(&MatrixCA, synth.blocksize, MatrixA.cols);
	m_new(&MatrixAp, MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_identity(MatrixAp);

	m_prepare_multiply(MatrixC, MatrixAp, &MatrixCA_line);
	m_prepare_multiply(MatrixAp, MatrixA, &MatrixAp_tmp);

	for(i = 1; i <= synth.blocksize; i++) {
		m_multiply(MatrixC, *pointer_MatrixAp, &MatrixCA_line);
		for(j = 0; j < MatrixCA_line.cols; j++) {
			m_set(MatrixCA, i-1, j, m_get(MatrixCA_line, 0, j));
		}
		m_multiplyblockdiag(*pointer_MatrixAp, MatrixA, pointer_MatrixAp_tmp, 2);

		m_swap(&pointer_MatrixAp_tmp, &pointer_MatrixAp);
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
 * Generates the signal on the CPU using the matrices generated earlier.
 * The signal is generated in chunks the size of synth.blocksize. The space for the resulting signal has been pre-
 * allocated earlier and is being filled by the filter.
 *
 * @param void
 * @return void
 */

void generateSignalCPU(float * output, String string, Synthesizer synth) {
	int i, j;
	Matrix state_tmp, output_chunk;

	Matrix *pointer_state_read, *pointer_state_write;

	m_new(&output_chunk, synth.blocksize,1);
	pointer_state_read = &state;
	pointer_state_write = &state_tmp;

	m_prepare_multiply(MatrixAp, state, &state_tmp);

	for(i = 0; i < synth.samples;) {
		m_multiply(MatrixCA, *pointer_state_read, &output_chunk);

		for(j = 0; j < synth.blocksize; j++) {
			output[i+j] = m_get(output_chunk,j,0)/128;
		}
		m_multiplyblockdiag(MatrixAp, *pointer_state_read, pointer_state_write, 2);
		m_swap(&pointer_state_read, &pointer_state_write);
		i = i + synth.blocksize;
	}
}



















/**
 * Generates the signal on the GPU using the matrices generated earlier.
 * The signal is generated in chunks the size of synth.blocksize. The space for the resulting signal has been pre-
 * allocated earlier and is being filled by the filter.
 *
 * @param void
 * @return void
 */

void generateSignalGPU(float * output, String string, Synthesizer synth) {
	int i, j;

	Matrix device_MatrixAp;
	Matrix device_MatrixCA;
	Matrix device_state_read, device_state_write;
	Matrix output_chunk_read, output_chunk_write;
	Matrix *pointer_output_chunk_read, *pointer_output_chunk_write;

	Matrix *pointer_device_state_read, *pointer_device_state_write;
	Matrix device_output_chunk_read, device_output_chunk_write;
	Matrix *pointer_device_output_chunk_read, *pointer_device_output_chunk_write;

	pointer_output_chunk_read = &output_chunk_read;
	pointer_output_chunk_write = &output_chunk_write;
	m_new(&output_chunk_read, synth.blocksize,1);
	m_new(&output_chunk_write, synth.blocksize,1);
	m_new(&device_output_chunk_read, synth.blocksize,1);
	m_new(&device_output_chunk_write, synth.blocksize,1);
	m_new(&device_MatrixCA, synth.blocksize, MatrixA.cols);
	m_new(&device_MatrixAp, MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_new(&device_state_read, 2 * synth.filters, 1);
	m_new(&device_state_write, 2 * synth.filters, 1);

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

	dim3 dimBlockCA(1, synth.blocksize/10); // @TODO Optimierungspotential
	dim3 dimGridCA(state.cols / dimBlockCA.x, MatrixCA.rows / dimBlockCA.y);

	dim3 dimBlockA(1, synth.blocksize/10); // @TODO Optimierungspotential
	dim3 dimGridA(state.cols / dimBlockA.x, MatrixAp.rows / dimBlockA.y);

	cudaEventRecord(MatrixCA_start, streams[0]);
	MatrixMultiplyKernel<<<dimGridCA, dimBlockCA, 1, streams[0]>>>(device_MatrixCA, *pointer_device_state_read, *pointer_device_output_chunk_write);
	cudaEventRecord(MatrixCA_stop, streams[0]);

	cudaEventRecord(MatrixAp_start, streams[1]);
	MatrixMultiplyKernel<<<dimGridA, dimBlockA, 1, streams[1]>>>(device_MatrixAp, *pointer_device_state_read, *pointer_device_state_write);
	cudaEventRecord(MatrixAp_stop, streams[1]);

	for(i = 0; i < synth.samples;) {
		/*
	       	 * CUDA IMPLEMENTATION
		 */

		cudaThreadSynchronize();

		cudaEventElapsedTime(&MatrixCA_time, MatrixCA_start, MatrixCA_stop);
		cudaEventElapsedTime(&MatrixAp_time, MatrixAp_start, MatrixAp_stop);
		cudaEventElapsedTime(&Memcpy_time, Memcpy_start, Memcpy_stop);

#if DEBUG == 10
		if(i == 5*synth.blocksize) {
			printf("MatrixCA: %d\n", MatrixCA_time);
			printf("MatrixAp: %d\n", MatrixAp_time);
			printf("  Memcpy: %d\n", Memcpy_time);
		}
#endif

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

		for(j = 0; j < synth.blocksize; j++) {
			output[i+j] = m_get(*pointer_output_chunk_read,j,0)/128;
		}

		i = i + synth.blocksize;
	}
}




















/**
 * The values from output are passed on to the sndfile library and
 * written to the file `filter.wav`.
 *
 * @param void
 * @return void
 */

void writeFile(const char * filename, float* input, int samples, int samplerate) {
	SF_INFO info;
	info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	info.channels = 1;
	info.samplerate = samplerate;

	SNDFILE *outfile = sf_open(filename, SFM_WRITE, &info);
	assert(outfile);
	sf_writef_float(outfile, input, samples);
}
