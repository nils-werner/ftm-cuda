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

	if(argc > 1)
		length = atof(argv[1]);

	if(argc > 2)
		blocksize = atoi(argv[2]);

	printf("Running filter with length %fcm in chunks of %d\n", length, blocksize);	

	filter(length, blocksize);
	return 0;
}
