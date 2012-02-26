/* Copyright (C) 2012 Nils Werner */

#ifndef FILTER_H
#define FILTER_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <cassert>
#include <sndfile.hh>
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
		Matrix MC, MA, Mstate, MCA;

		// Ausgangssignal
		float* y;

	public:
		Filter(float length);
	
	private:
		void initializeCoefficients(float length);
		void createMatrices();
		void generateSignal();

};

#endif
