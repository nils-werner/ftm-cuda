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
	int mode = 1;

	if(argc == 1)
		printf("%s mode stringlength blocksize seconds filters\n", argv[0]);

	if(argc > 1) {
		if(0 == strcmp(argv[1], "cpu"))
			mode = 0;
		else
			mode = 1;
	}

	if(argc > 1)
		length = atof(argv[1]);

	if(argc > 2)
		blocksize = atoi(argv[2]);

	if(argc > 3)
		samples = atoi(argv[3])*44100;

	if(argc > 4)
		filters = atoi(argv[4]);

	printf("Running in mode %d, %d filters with length %fcm, %d samples in chunks of %d\n", mode, filters, length, samples, blocksize);	

	filter(mode, length, samples, blocksize, filters);
	return 0;
}
