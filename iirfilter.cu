#include "iirfilter.h"

/**
 * Main method required in all C programs. Simply calls filter with a predefined string length
 *
 * @param int
 * @param **char argv
 * @return int 0
 */

int main(int argc, char *argv[]) {
	int chunksize = 100;
	float length = 0.65;
	int samples = 441000;
	int filters = 30;
	int mode = 0;
	int c;

	setlocale(LC_ALL, "");

#ifndef BENCHMARK
	printf("GPGPU-Based recursive sound synthesis filter.\n\n");

	if(argc == 1) {
		printf("Available parameters:\n", argv[0]);
		printf("  -g         use GPU\n");
		printf("  -f <int>   number of filters\n");
		printf("  -c <int>   chunksize\n");
		printf("  -s <int>   length of signals in seconds\n");
		printf("  -l <float> length of string\n\n");
		return 1;
	}
#endif

	while ((c = getopt (argc, argv, "gf:b:c:s:")) != -1)
		switch (c) {
			case 'g':
				mode = 1;
				break;
			case 'f':
				filters = atoi(optarg);
				break;
			case 'c':
				chunksize = atoi(optarg);
				break;
			case 's':
				samples = atoi(optarg)*44100;
				break;
			case 'l':
				length = atof(optarg);
				break;
			case '?':
				if (optopt == 'f' || optopt == 'c' || optopt == 's' || optopt == 'l')
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				else if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				return 1;
			default:
				abort ();
		}

#ifdef BENCHMARK
		printf("<run>\n<settings mode=\"%s\" filters=\"%d\" chunksize=\"%d\" samples=\"%d\" />\n", (mode == 0?"cpu":"gpu"), filters, chunksize, samples);
#else
		printf("Settings:\n  Mode %s\n  Filters %d\n  Chunksize %d\n  Samples %d\n\n", (mode == 0?"CPU":"GPU"), filters, chunksize, samples);
#endif

	filter(mode, length, samples, chunksize, filters);

#ifdef BENCHMARK
		printf("</run>\n");
#endif

	return 0;
}
