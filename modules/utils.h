#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <cutil.h>
#include <sys/time.h>

typedef struct {
	struct timeval start;
	struct timeval stop;
} Timer;

float fl_rand();
void time_start(Timer*);
void time_stop(Timer*);
void print_time(Timer*, const char*);

#endif
