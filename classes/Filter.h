/* Copyright (C) 2012 Nils Werner */

#ifndef FILTER_H
#define FILTER_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>

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



	public:
		Filter(float lenght);
	
	private:
		void createPoles();

};

#endif
