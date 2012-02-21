/* Copyright (C) 2012 Nils Werner */

#ifndef FILTER_H
#define FILTER_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "Matrix.h"

class Filter {

	private:
		// Saiten-Koeffizienten
		float l, Ts, rho, A, E, I, d1, d3;

		// Abtastpunkt
		float xa;

		// Abtastrate und Samplelänge
		int T, seconds, samples, filters;

		// Blockverarbeitungs-Länge
		int blocksize;

		// Matrizen
		Matrix MC, MA, Mstate;

		// Ausgangssignal
		float* y;

	public:
		Filter(float lenght);
	
	private:
		void createMatrices();

};

#endif
