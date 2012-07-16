#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <cutil.h>
#include <sys/time.h>
#include "modules/settings.h"

typedef struct {
	struct timeval start;
	struct timeval stop;
} Timer;

extern Settings settings;

float fl_rand();
void time_start(Timer*);
void time_stop(Timer*);
void time_print(Timer*, const char*);
int gcd(int, int);
int lcm(int, int);
void noop(void *);

#endif
