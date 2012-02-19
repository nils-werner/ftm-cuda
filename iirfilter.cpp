// my first program in C++

#include <iostream>
#include "classes/Matrix.cpp"

using namespace std;

int main ()
{

		int filters = 30;

		// Saiten-Koeffizienten
		float l = 0.65;
		float Ts = 60.97;
		float rho = 1140;
		float A = 0.5188e-6;
		float E = 5.4e9;
		float I = 0.171e-12;
		float d1 = 8e-5;
		float d3 = -1.4e-5;

		// Abtastpunkt
		float xa = 0.1;

		// Abtastrate und Samplelänge
		int T = 44100;
		int seconds = 10;
		int samples = seconds*T;

		// Blockverarbeitungs-Länge
		int blocksize = 100;

		// Ausgangssignal
		// y = zeros(1,samples);
		
		cout << "Hello World!";
		return 0;
}