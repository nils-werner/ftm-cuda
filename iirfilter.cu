#include "iirfilter.h"

/**
 * Main method required in all C programs. Simply calls filter with a predefined string length
 *
 * @param int
 * @param **char argv
 * @return int 0
 */

int main(int argc, char *argv[]) {
	int blocksize = 100;
	float length = 0.65;
	int samples = 441000;
	int filters = 30;
	int xmloutput = 0;
	int mode = 1;

	setlocale(LC_ALL, "");

	if(argc == 1) {
		printf("Call Syntax: %s mode xml-outputmode filters chunksize seconds\n\n", argv[0]);
	}

	if(argc > 1) {
		if(0 == strcmp(argv[1], "CPU") || 0 == strcmp(argv[1], "cpu"))
			mode = 0;
		else
			mode = 1;
	}

	if(argc > 2) {
		if(0 == strcmp(argv[2], "XML") || 0 == strcmp(argv[2], "xml"))
			xmloutput = 1;
		else
			xmloutput = 0;
	}

	if(argc > 3)
		filters = atoi(argv[3]);

	if(argc > 4)
		blocksize = atoi(argv[4]);

	if(argc > 5)
		samples = atoi(argv[5])*44100;


	if(xmloutput == 0) {
		printf("GPGPU-Based recursive sound synthesis filter.\n\n");
		printf("Settings:\n  Mode %s\n  Filters %d\n  Chunksize %d\n  Samples %d\n\n", (mode == 0?"CPU":"GPU"), filters, blocksize, samples);
	}
	else
		printf("<run>\n<settings mode=\"%s\" filters=\"%d\" chunksize=\"%d\" samples=\"%d\" />\n", (mode == 0?"cpu":"gpu"), filters, blocksize, samples);

	filter(mode, xmloutput, length, samples, blocksize, filters);

	if(xmloutput == 1)
		printf("</run>\n");
	return 0;
}
