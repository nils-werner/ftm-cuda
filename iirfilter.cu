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
	settings.chunksize = 100;
	settings.length = 0.65;
	settings.samples = 441000;
	settings.filters = 30;
	settings.mode = 0;

	setlocale(LC_ALL, "");

	while ((c = getopt (argc, argv, "gf:c:s:l:xh")) != -1) {
		switch (c) {
			case 'g':
				settings.mode = 1;
				break;
			case 'f':
				settings.filters = atoi(optarg);
				break;
			case 'c':
				settings.chunksize = atoi(optarg);
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
				printf("  -g         use GPU\n");
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

	settings.samples = settings.samples - (settings.samples % settings.chunksize);


	print_prefix();

	filter();

	print_suffix();

	return 0;
}


void print_prefix() {
	if(settings.xml == 1)
		printf("<run>\n<settings mode=\"%s\" filters=\"%d\" chunksize=\"%d\" samples=\"%d\" />\n", (settings.mode == 0?"cpu":"gpu"), settings.filters, settings.chunksize, settings.samples);
	else
		printf("Settings:\n  Mode %s\n  Filters %d\n  Chunksize %d\n  Samples %d\n\n", (settings.mode == 0?"CPU":"GPU"), settings.filters, settings.chunksize, settings.samples);
}

void print_suffix() {
	if(settings.xml == 1)
		printf("</run>\n");
}
