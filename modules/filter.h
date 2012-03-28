#ifndef FILTER_H
#define FILTER_H

#include "modules/matrix.h"
#include <math.h>
#include <sndfile.h>

int filter(float);
void initializeCoefficients(float);
void createMatrices();
void generateSignal();

#endif
