#include "filter.h"

/* Global variables */

// String settings (length etc.)
String string;

// Synthesizer settings (blocksize, filters etc.)
Synthesizer synth;

// Matrices required for the filter
Matrix MatrixC, MatrixA, state;
Matrix MatrixAp, MatrixCA;
Matrix device_MatrixAp, device_MatrixCA;

// Timers
Timer turnaround, overall;
int xmloutput;

/**
 * Wrapper for the methods required in the filter, just calls them in the correct order
 *
 * @param void
 * @return int 0
 */

int filter() {
	/* Start timers when filter starts */
	time_start(&overall);
	time_start(&turnaround);

	/* Initialize and set coefficients */
	initializeCoefficients();

	/* Allocate memory to save data in */
	float * output = (float *) malloc(sizeof(float) * synth.samples);

	/* Create Matrices from values generated in `initializeCoefficients()` */
	createMatrices();


	/* Create Blockprocessing matrices, either on CPU or on GPU */
	if(settings.matrixmode == 0) {
		createBlockprocessingMatrices();
	}
	else {
		initializeGPU();
		createBlockprocessingMatricesGPU();
	}

	/* Either copy matrices from GPU or to GPU, if only one part of the pipe has to be done on the CPU */
	if(settings.mode == 0 && settings.matrixmode == 1) {
		copyMatricesFromGPU();
	}
	if(settings.mode == 1 && settings.matrixmode == 0) {
		initializeGPU();
		copyMatricesToGPU();
	}

	/* Generate signal, either on CPU or GPU */
	if(settings.mode == 0) {
		generateSignalCPU(output, string, synth);
	}
	else {
		generateSignalGPU(output, string, synth);
	}

	/* Write output to disk */
	writeFile("filter.wav", output, synth.samples, synth.T);

	/* Stop timer overall, print its value */
	time_stop(&overall);
	time_print(&overall, "overall");
	return 0;
}











/**
 * Initializes coefficients for a standard Nylon-b-String. The values are, with exception for
 * length (l) are hard-coded according to the values in \cite{rabenstein03}.
 *
 * Settings are read and written to global structs.
 *
 * @param void
 * @return void
 */

void initializeCoefficients() {
	// String-Coefficients
	string.l = settings.length;
	string.Ts = 60.97;
	string.rho = 1140;
	string.A = 0.5188e-6;
	string.E = 5.4e9;
	string.I = 0.171e-12;
	string.d1 = 8e-5;
	string.d3 = -1.4e-5;

	// Sample point on the string
	string.xa = 0.1;

	// Samplingrate and number of Samples
	synth.T = 44100;
	synth.seconds = 10;
	synth.samples = settings.samples;
	synth.filters = settings.filters;
	synth.blocksize = settings.chunksize;

	// Make sure the samples fit neatly in the blocksize
	assert(synth.samples % synth.blocksize == 0);
}















/**
 * Creates the required matrices by calculating the required number of poles using the equations
 * to be found in \cite{rabenstein03}. The matrices generated are not yet in blockprocessing form.
 *
 * Settings are read from global structs and written into global matrices
 *
 * @param void
 * @return void
 */

void createMatrices() {
	int i, mu;
	double gamma, sigma;
	double omega;
	double a, b, c1, c0;


	/* Create Matrices */
	m_new(&MatrixC, 1, 2 * synth.filters);
	m_new(&MatrixA, 2 * synth.filters, 2 * synth.filters); //BLOCKMATRIX
	m_new(&state, 2 * synth.filters, 1);

	/* Create and start "createm" timer */
	Timer timer;
	time_start(&timer);

	/* Create system matrices and input vector for all filters */
	for(i = 0; i < synth.filters; i++) {

		/* Calculate sigma and omega, see script for formulas */
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

		/* Apply correct weighting so we can put it inside Matrix C as well
		 * This removes the need to do the weighting afterwards, in a separate step
		 * Value is divided by 128 to avoid accidental clipping */
		a = sin(mu * M_PI * string.xa / string.l)/128;

		/* Create discretized values from continuous pole information */
		b = synth.T * sin(omega * 1 / synth.T) / (omega * 1 / synth.T);
		c1 = -2 * exp(sigma * 1 / synth.T) * cos(omega * 1 / synth.T);
		c0 = exp( 2 * sigma * 1 / synth.T);

		/* Don't do anything with b but suppress annoying warnings */
		noop(&b);

		/* Set values in C */
		m_set(&MatrixC, 0, 2*i  , 0);
		m_set(&MatrixC, 0, 2*i+1, a);

		/* Set values in A */
		m_set(&MatrixA, 2*i  , 2*i  , 0);
		m_set(&MatrixA, 2*i  , 2*i+1, -c0);
		m_set(&MatrixA, 2*i+1, 2*i  , 1);
		m_set(&MatrixA, 2*i+1, 2*i+1, -c1);

		/* Set values in x */
		m_set(&state, 2*i  , 0, 0);
		m_set(&state ,2*i+1, 0, 1);

	}

	/* Done, stop timer and display its value */
	time_stop(&timer);
	time_print(&timer, "createm");
}
















/**
 * Generates matrices suitable for blockprocessing, on the GPU
 *
 * @param void
 * @return void
 */ 

void createBlockprocessingMatricesGPU() {
	int i;
	Timer timer;

	Matrix device_MatrixCA_line_read, device_MatrixCA_line_write;
	Matrix device_MatrixAp_read, device_MatrixAp_write;
	Matrix device_MatrixC, device_MatrixA;

	/* We need pointers to our matrices so we can swap then easily */
	Matrix *pointer_device_MatrixCA_line_read, *pointer_device_MatrixCA_line_write;
	Matrix *pointer_device_MatrixAp_read, *pointer_device_MatrixAp_write;

	/* We need three streams:
	 *  1. A*A
	 *  2. C*A
	 *  3. Data transfers */
	cudaStream_t streams[3];

	/* Assign pointers */
	pointer_device_MatrixAp_write = &device_MatrixAp_write;
	pointer_device_MatrixAp_read = &device_MatrixAp_read;
	pointer_device_MatrixCA_line_write = &device_MatrixCA_line_write;
	pointer_device_MatrixCA_line_read = &device_MatrixCA_line_read;

	/* okay, start timer */
	time_start(&timer);

	/* Create streams */
	for(int i = 0; i < 3; i++) {
		cudaStreamCreate(& streams[i]);
	}

	/* Create space for all matrices we will be getting out of this */
	m_new(&MatrixCA, synth.blocksize, MatrixA.cols);
	m_new(&MatrixAp, MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_new(&device_MatrixA, MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_new(&device_MatrixCA, synth.blocksize, MatrixA.cols);
	m_new(&device_MatrixAp, MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_new(&device_MatrixAp_write, MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX
	m_new(&device_MatrixAp_read, MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX

	m_new(&device_MatrixC, MatrixC.rows, MatrixC.cols); // BLOCKDIAGMATRIX

	/* Allocate space for results of Matrix-multiplications */
	m_prepare_multiply(&MatrixC, &MatrixAp, &device_MatrixCA_line_read);
	m_prepare_multiply(&MatrixC, &MatrixAp, &device_MatrixCA_line_write);
	m_prepare_multiply(&MatrixAp, &MatrixA, &device_MatrixAp_read);
	m_prepare_multiply(&MatrixAp, &MatrixA, &device_MatrixAp_write);

	/* Create an identity matrix to be fed into the system */
	m_identity(&MatrixAp);

	/* Allocate space and copy stuff to the GPU */
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixAp.elements, m_size(&MatrixAp)));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixAp_read.elements, m_size(&MatrixAp)));
	CUDA_SAFE_CALL(cudaMemcpyAsync(device_MatrixAp_read.elements, MatrixAp.elements, m_size(&MatrixAp), cudaMemcpyHostToDevice, streams[0]));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixAp_write.elements, m_size(&MatrixAp)));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixA.elements, m_size(&MatrixA)));
	CUDA_SAFE_CALL(cudaMemcpyAsync(device_MatrixA.elements, MatrixA.elements, m_size(&MatrixA), cudaMemcpyHostToDevice, streams[1]));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixCA.elements, m_size(&MatrixCA)));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixC.elements, m_size(&MatrixC)));
	CUDA_SAFE_CALL(cudaMemcpyAsync(device_MatrixC.elements, MatrixC.elements, m_size(&MatrixC), cudaMemcpyHostToDevice, streams[2]));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixCA_line_read.elements, m_size(&device_MatrixCA_line_read)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixCA_line_write.elements, m_size(&device_MatrixCA_line_write)));


	/* Initialize blocksize for C*A */
	dim3 dimBlockCA(1, min(settings.matrixblocksize, device_MatrixCA_line_read.rows));
	assert(MatrixAp.cols % dimBlockCA.x == 0);
	assert(MatrixC.rows % dimBlockCA.y == 0);
	dim3 dimGridCA(device_MatrixCA_line_read.cols / dimBlockCA.x, device_MatrixCA_line_read.rows / dimBlockCA.y);

	/* Initialize blocksize for A*A */
	dim3 dimBlockA(min(settings.matrixblocksize, MatrixAp.cols), 2);
	assert(MatrixAp.cols % dimBlockA.x == 0);
	assert(2 % dimBlockA.y == 0);
	dim3 dimGridA(MatrixAp.cols / dimBlockA.x, 2 / dimBlockA.y);

	/* Wait until all data transfers are done */
	cudaThreadSynchronize();

	for(i = 1; i <= synth.blocksize; i++) {
		/* C*A */	
		MatrixBlockDiagMultiplyKernel<<<dimGridCA, dimBlockCA, 1, streams[0]>>>(device_MatrixC, *pointer_device_MatrixAp_read, *pointer_device_MatrixCA_line_write, 2);
		// m_multiply(&MatrixC, pointer_MatrixAp, &MatrixCA_line);

		/* A*A */
		BlockDiagBlockDiagMatrixMultiplyKernel<<<dimGridA, dimBlockA, 1, streams[2]>>>(*pointer_device_MatrixAp_read, device_MatrixA, *pointer_device_MatrixAp_write, 2);
		// m_multiplyblockdiag(pointer_MatrixAp, &MatrixA, pointer_MatrixAp_tmp, 2);

		/* Wait until both are done */
		cudaThreadSynchronize();

		/* Swap pointers from result to input matrix and vice versa */
		m_swap(&pointer_device_MatrixAp_write, &pointer_device_MatrixAp_read);
		m_swap(&pointer_device_MatrixCA_line_write, &pointer_device_MatrixCA_line_read);

		/* Copy result into the correct matrix-row on the GPU */
		CUDA_SAFE_CALL(cudaMemcpyAsync(&device_MatrixCA.elements[(i-1) * MatrixCA.cols], pointer_device_MatrixCA_line_read->elements, m_size(&device_MatrixCA_line_read), cudaMemcpyDeviceToDevice, streams[1]));
	}

	/* Wait until all is done */
	cudaThreadSynchronize();

	/* Copy A^p into where we expect it to be in the filter */
	CUDA_SAFE_CALL(cudaMemcpyAsync(device_MatrixAp.elements, pointer_device_MatrixAp_read->elements, m_size(&MatrixAp), cudaMemcpyDeviceToDevice, streams[0]));

	/* Stop timer, display the value */
	time_stop(&timer);
	time_print(&timer, "blockprocm");
}
















/**
 * Generates matrices suitable for blockprocessing, on the CPU
 *
 * @param void
 * @return void
 */ 

void createBlockprocessingMatrices() {
	int i, j;
	Matrix MatrixCA_line, MatrixAp_tmp;
	Matrix *pointer_MatrixAp, *pointer_MatrixAp_tmp;

	/* We need pointers to swap them lateron */
	pointer_MatrixAp = &MatrixAp;
	pointer_MatrixAp_tmp = &MatrixAp_tmp;

	/* Initialize timer */
	Timer timer;
	time_start(&timer);

	/* Initialize matrices to write stuff to */
	m_new(&MatrixCA, synth.blocksize, MatrixA.cols);
	m_new(&MatrixAp, MatrixA.rows, MatrixA.cols); // BLOCKDIAGMATRIX

	/* Create identity matrix to feed into the system */
	m_identity(&MatrixAp);

	/* Preallocate memory for multiplications */
	m_prepare_multiply(&MatrixC, &MatrixAp, &MatrixCA_line);
	m_prepare_multiply(&MatrixAp, &MatrixA, &MatrixAp_tmp);

	for(i = 1; i <= synth.blocksize; i++) {
		/* C*A */
		m_blockdiagmultiply(&MatrixC, pointer_MatrixAp, &MatrixCA_line, 2);

		/* Copy result */
		for(j = 0; j < MatrixCA_line.cols; j++) {
			m_set(&MatrixCA, i-1, j, m_get(&MatrixCA_line, 0, j));
		}

		/* A*A */
		m_multiplyblockdiagblockdiag(pointer_MatrixAp, &MatrixA, pointer_MatrixAp_tmp, 2);

		/* Swap Matrices */
		m_swap(&pointer_MatrixAp_tmp, &pointer_MatrixAp);
	}

	/* Stop timer and display it's result */
	time_stop(&timer);
	time_print(&timer, "blockproc");
}



















/**
 * Initializes the GPU
 *
 * @param void
 * @return void
 */

void initializeGPU() {
	/* Oh well, select device 1, duh. */
	cudaSetDevice(0);
}



















/**
 * Copies the blockprocessing matrices to the GPU
 *
 * @param void
 * @return void
 */

void copyMatricesToGPU() {
	/* Allocate space and copy matrices */
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixAp.elements, m_size(&MatrixAp)));
	CUDA_SAFE_CALL(cudaMemcpy(device_MatrixAp.elements, MatrixAp.elements, m_size(&MatrixAp), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**) &device_MatrixCA.elements, m_size(&MatrixCA)));
	CUDA_SAFE_CALL(cudaMemcpy(device_MatrixCA.elements, MatrixCA.elements, m_size(&MatrixCA), cudaMemcpyHostToDevice));
}



















/**
 * Copies the blockprocessing matrices from the GPU
 *
 * @param void
 * @return void
 */

void copyMatricesFromGPU() {
	/* Copy data FROM GPU */
	CUDA_SAFE_CALL(cudaMemcpy(MatrixAp.elements, device_MatrixAp.elements, m_size(&MatrixAp), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(MatrixCA.elements, device_MatrixCA.elements, m_size(&MatrixCA), cudaMemcpyDeviceToHost));
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
	int i;
	Matrix state_tmp, output_chunk;
	Matrix *pointer_state_read, *pointer_state_write;
	Timer roundtrip;

	/* Allocate data for output chunk */
	m_new(&output_chunk, synth.blocksize,1);

	/* Initialize pointers so we can swap them later */
	pointer_state_read = &state;
	pointer_state_write = &state_tmp;

	/* Allocate temporary state vector */
	m_prepare_multiply(&MatrixAp, &state, &state_tmp);

	/* DOOOING, iterate! */
	for(i = 0; i < synth.samples;) {
		/* Start timer roundtrip */
		time_start(&roundtrip);

		/* C*x */
		m_multiply(&MatrixCA, pointer_state_read, &output_chunk);

		/* Copy result */
		memcpy(&output[i], output_chunk.elements, sizeof(float) * synth.blocksize);

		/* A*X */
		m_multiplyblockdiag(&MatrixAp, pointer_state_read, pointer_state_write, 2);

		/* Swap vectors. the reading one becomes the writing one and vice versa */
		m_swap(&pointer_state_read, &pointer_state_write);

		/* First chunk done? Turnaround finished, give data */
		if(i == 0) {
			time_stop(&turnaround);
			time_print(&turnaround, "turnaround");
		}

		/* Stop timer roundtrip since one round is done */
		time_stop(&roundtrip);

		/* Ok, we're 5 iterations into the game, display roundtrip data */
		if(i == 5*synth.blocksize) {
			time_print(&roundtrip, "roundtrip");
		}

		/* Increment index, obviously */
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
	int i;

	Matrix device_state_read, device_state_write;
	Matrix output_chunk_read, output_chunk_write;
	Matrix *pointer_output_chunk_read, *pointer_output_chunk_write;

	Matrix *pointer_device_state_read, *pointer_device_state_write;
	Matrix device_output_chunk_read, device_output_chunk_write;
	Matrix *pointer_device_output_chunk_read, *pointer_device_output_chunk_write;

	/* We need a timer, again */
	Timer roundtrip;

	/* Also, we need space to put stuff in */
	m_new(&output_chunk_read, synth.blocksize,1);
	m_new(&output_chunk_write, synth.blocksize,1);
	m_new(&device_output_chunk_read, synth.blocksize,1);
	m_new(&device_output_chunk_write, synth.blocksize,1);
	m_new(&device_state_read, 2 * synth.filters, 1);
	m_new(&device_state_write, 2 * synth.filters, 1);

	/* We need pointers for swapping */
	pointer_output_chunk_read = &output_chunk_read;
	pointer_output_chunk_write = &output_chunk_write;
	pointer_device_state_read = &device_state_read;
	pointer_device_state_write = &device_state_write;
	pointer_device_output_chunk_read = &device_output_chunk_read;
	pointer_device_output_chunk_write = &device_output_chunk_write;

	/* Again 3 streams
	 *  1. A*x
	 *  2. C*x
	 *  3. data transfers */
	cudaStream_t streams[3];

	/* I tried timing executions using CUDA events. It did not work */
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

	/* Initialize streams. This takes ages btw, maybe wise to
	 * do it at the beginning and leave them open for consecutive operations */
	for(int i = 0; i < 3; i++) {
		cudaStreamCreate(& streams[i]);
	}

	/* Malloc and copy matrices to GPU */
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_state_read.elements, m_size(&state)));
	CUDA_SAFE_CALL(cudaMemcpyAsync(device_state_read.elements, state.elements, m_size(&state), cudaMemcpyHostToDevice, streams[2]));
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_state_write.elements, m_size(&state)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_output_chunk_read.elements, m_size(&output_chunk_read)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &device_output_chunk_write.elements, m_size(&output_chunk_write)));


	/* Initialize Blocksizes for C*x */
	dim3 dimBlockCA(1, min(settings.blocksize, output_chunk_read.rows));
	assert(output_chunk_read.cols % dimBlockCA.x == 0);
	assert(output_chunk_read.rows % dimBlockCA.y == 0);
	dim3 dimGridCA(output_chunk_read.cols / dimBlockCA.x, output_chunk_read.rows / dimBlockCA.y);


	/* Initialize Blocksizes for A*x */
	dim3 dimBlockA(1, min(settings.matrixblocksize, state.rows)); 
	assert(state.cols % dimBlockA.x == 0);
	assert(state.rows % dimBlockA.y == 0);
	dim3 dimGridA(state.cols / dimBlockA.x, state.rows / dimBlockA.y);

	/* Wait until all data is available */
	cudaThreadSynchronize();

	/* This loop might seem backwards and a bit retarded but it has to be done this way because
	 * of the asynchronousity of CUDA calls.
	 * Everything has finished when `cudaThreadSyncronize` returns, so this marks the end of the loop.
	 * Everything after that actually runs before it */
	for(i = -synth.blocksize; i < synth.samples;) {
		/* Wait until all is done, from last iteration mainly */
		cudaThreadSynchronize();

		/* Stop timer, we're done */
		time_stop(&roundtrip);

		/* 5th iteration, time to print the roundtrip result */
		if(i == 5*synth.blocksize) {
			time_print(&roundtrip, "roundtrip");
		}

		/* Ready, set, go! CUDA running! */
		time_start(&roundtrip);

		/* These didnt work */
		cudaEventElapsedTime(&MatrixCA_time, MatrixCA_start, MatrixCA_stop);
		cudaEventElapsedTime(&MatrixAp_time, MatrixAp_start, MatrixAp_stop);
		cudaEventElapsedTime(&Memcpy_time, Memcpy_start, Memcpy_stop);

		/* Swap data, but only if this is not the very first iteration */
		if(i >= 0) {
			m_swap(&pointer_device_state_read, &pointer_device_state_write);
			m_swap(&pointer_device_output_chunk_read, &pointer_device_output_chunk_write);
			m_swap(&pointer_output_chunk_read, &pointer_output_chunk_write);
		}

		/* C*x */
		cudaEventRecord(MatrixCA_start, streams[0]);
		MatrixMultiplyKernel<<<dimGridCA, dimBlockCA, 1, streams[0]>>>(device_MatrixCA, *pointer_device_state_read, *pointer_device_output_chunk_write);
		cudaEventRecord(MatrixCA_stop, streams[0]);

		/* A*x */
		cudaEventRecord(MatrixAp_start, streams[1]);
		BlockDiagMatrixMultiplyKernel<<<dimGridA, dimBlockA, 1, streams[1]>>>(device_MatrixAp, *pointer_device_state_read, *pointer_device_state_write, 2);
		cudaEventRecord(MatrixAp_stop, streams[1]);

		if(i >= 0) {
			/* Copy result from last iteration from GPU to Host */
			cudaEventRecord(Memcpy_start, streams[2]);
			cudaMemcpyAsync(pointer_output_chunk_write->elements, pointer_device_output_chunk_read->elements, m_size(&output_chunk_write), cudaMemcpyDeviceToHost, streams[2]);
			cudaEventRecord(Memcpy_stop, streams[2]);

			/* Copy transferred data from last iteration into the final signal */
			memcpy(&output[i], pointer_output_chunk_read->elements, sizeof(float) * synth.blocksize);

			/* First iteration done, data ready. How long did it take? */
			if(i == 0) {
				time_stop(&turnaround);
				time_print(&turnaround, "turnaround");
			}
		}

		/* Increment index */
		i = i + synth.blocksize;
	}
}




















/**
 * The values from output are passed on to the sndfile library and
 * written to the file `filter.wav`.
 *
 * @param const char* filename
 * @param float* input signal
 * @param int samples
 * @param int samplerate
 * @return void
 */

void writeFile(const char * filename, float* input, int samples, int samplerate) {

	/* Create SF_INFO struct, put stuff in it */
	SF_INFO info;
	info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	info.channels = 1;
	info.samplerate = samplerate;

	/* Open sound file */
	SNDFILE *outfile = sf_open(filename, SFM_WRITE, &info);

	/* Make sure it's there */
	assert(outfile);

	/* Dump wave signal into file */
	sf_writef_float(outfile, input, samples);
}
