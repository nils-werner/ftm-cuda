#include "iirfilter.h"

/**
 * Main method required in all C programs. Simply calls filter with a predefined string length
 *
 * @param int
 * @param **char argv
 * @return int 0
 */

int main(int argc, char *argv[]) {
	int c;

	settings.xml = 0;
	settings.chunksize = 384;
	settings.blocksize = 6;
	settings.length = 0.65;
	settings.samples = 441000;
	settings.filters = 32;
	settings.mode = 0;
	settings.matrixmode = 0;
	settings.matrixblocksize = 128;


struct option longopts[] = {
	{ "synth-gpu",	no_argument,		NULL,		'g' },
	{ "matrix-gpu",	no_argument,		NULL,		'p' },
	{ "blocksize",	required_argument,	NULL,		'b' },
	{ "filters",	required_argument,	NULL,		'f' },
	{ "chunksize",	required_argument,	NULL,		'c' },
	{ "seconds",	required_argument,	NULL,		's' },
	{ "length",	required_argument,	NULL,		'l' },
	{ "xml",	no_argument,		NULL,		'x' },
	{ "help",	no_argument,		NULL,		'h' },
	{ "matrix-blocksize",	required_argument,	NULL,	'm' },
	{ 0, 0, 0, 0 }
};

	setlocale(LC_ALL, "");

	while ((c = getopt_long(argc, argv, "gpf:c:b:m:s:l:xh", longopts, NULL)) != -1) {
		switch (c) {
			case 'g':
				settings.mode = 1;
				break;
			case 'p':
				settings.matrixmode = 1;
				break;
			case 'f':
				settings.filters = atoi(optarg);
				break;
			case 'c':
				settings.chunksize = atoi(optarg);
				break;
			case 'b':
				settings.blocksize = atoi(optarg);
				break;
			case 'm':
				settings.matrixblocksize = atoi(optarg);
				break;
			case 's':
				settings.samples = atoi(optarg)*44100;
				break;
			case 'l':
				settings.length = atof(optarg);
				break;
			case 'x':
				settings.xml = 1;
				break;
			case 'h':
				printf("Available parameters:\n", argv[0]);
				printf("  -g         use GPU for signal generation\n");
				printf("  -p         use GPU for generating matrices\n");
				printf("  -b <int>   CUDA blocksize for synthesis\n");
				printf("  -m <int>   CUDA blocksize for matrix generation\n");
				printf("  -f <int>   number of filters\n");
				printf("  -c <int>   chunksize\n");
				printf("  -s <int>   length of signals in seconds\n");
				printf("  -l <float> length of string\n");
				printf("  -x         return benchmark data as XML\n");
				return 0;
			case '?':
				return 1;
			default:
				abort ();
		}
	}

	if(settings.xml != 1) {
		printf("GPGPU-Based recursive sound synthesis filter.\n\n");
		printf("Use option -h to see all available switches.\n\n");
	}

	if(settings.blocksize % 2 == 1) settings.blocksize++;
	if(settings.matrixblocksize % 2 == 1) settings.matrixblocksize++;

	settings.samples = settings.samples - (settings.samples % settings.chunksize);


	print_prefix();

	filter();

	print_suffix();

	return 0;
}


void print_prefix() {
	if(settings.xml == 1)
		printf("<run>\n<settings mode=\"%s\" matrixmode=\"%s\" blocksize=\"%d\" matrixblocksize=\"%d\" filters=\"%d\" chunksize=\"%d\" samples=\"%d\" />\n",
				(settings.mode == 0?"cpu":"gpu"),
				(settings.matrixmode == 0?"cpu":"gpu"),
				settings.blocksize,
				settings.matrixblocksize,
				settings.filters,
				settings.chunksize,
				settings.samples
			);
	else
		printf("Settings:\n  Mode %s\n  Matrixmode %s\n  Blocksize %d\n  Matrixblocksize %d\n  Filters %d\n  Chunksize %d\n  Samples %d\n\n",
				(settings.mode == 0?"CPU":"GPU"),
				(settings.matrixmode == 0?"CPU":"GPU"),
				settings.blocksize,
				settings.matrixblocksize,
				settings.filters,
				settings.chunksize,
				settings.samples
			);
}

void print_suffix() {
	if(settings.xml == 1)
		printf("</run>\n");
}
