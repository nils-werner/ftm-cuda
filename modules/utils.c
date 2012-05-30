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

void time_stop(Timer* timer) {
	gettimeofday(&timer->stop, NULL);
}

void time_start(Timer* timer) {
	gettimeofday(&timer->start, NULL);
}

void time_print(Timer* timer, const char* string) {
	double tS = timer->start.tv_sec*1000000 + (timer->start.tv_usec);
	double tE = timer->stop.tv_sec*1000000  + (timer->stop.tv_usec);
	if(settings.xml == 1)
		printf("<timer name=\"%s\">%.0f</timer>\n", string, tE - tS);
	else
		printf("Timer %s: %'.0f usec\n", string, tE - tS);
}

void noop(void * nothing) {
}
