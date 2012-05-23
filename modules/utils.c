#include "utils.h"

/**
 * Returns a random number between 0 and 1
 *
 * @param void
 * @return float
 */

float fl_rand() {
	return (float) rand()/RAND_MAX;
}

void print_time(struct timeval* startTime, struct timeval* endTime, const char* string) {
	double tS = startTime->tv_sec*1000000 + (startTime->tv_usec);
	double tE = endTime->tv_sec*1000000  + (endTime->tv_usec);
	printf("Timer %s: %f\n", string, tE - tS);
}
