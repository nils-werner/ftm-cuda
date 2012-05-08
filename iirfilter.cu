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

	if(argc == 1)
		printf("%s stringlength blocksize seconds filters\n", argv[0]);

	if(argc > 1)
		length = atof(argv[1]);

	if(argc > 2)
		blocksize = atoi(argv[2]);

	if(argc > 3)
		samples = atoi(argv[3])*44100;

	if(argc > 4)
		filters = atoi(argv[4]);

	printf("Running %d filters with length %fcm, %d samples in chunks of %d\n", filters, length, samples, blocksize);	

	filter(length, samples, blocksize, filters);
	return 0;
}
