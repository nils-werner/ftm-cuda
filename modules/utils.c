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

/**
 * Stops a timer by writing the current time into the .stop value
 *
 * @param Timer* timer
 * @return void
 */

void time_stop(Timer* timer) {
	gettimeofday(&timer->stop, NULL);
}

/**
 * Starts a timer by writing the current time into the .start value
 *
 * @param Timer* timer
 * @return void
 */

void time_start(Timer* timer) {
	gettimeofday(&timer->start, NULL);
}

/**
 * Prints out timer data by calculating the difference between .start and .stop.
 * Prints XML-data if XML output is enabled
 *
 * @param Timer* timer
 * @param const char* string
 * @return void
 */

void time_print(Timer* timer, const char* string) {
	double tS = timer->start.tv_sec*1000000 + (timer->start.tv_usec);
	double tE = timer->stop.tv_sec*1000000  + (timer->stop.tv_usec);
	if(settings.xml == 1)
		printf("<timer name=\"%s\">%.0f</timer>\n", string, tE - tS);
	else
		printf("Timer %s: %'.0f usec\n", string, tE - tS);
}

/**
 * Calculates greates common divisor. This method is never used.
 *
 * @param int x
 * @param int y
 * @return void
 */

int gcd(int x, int y) {  
	/*; 
	  a = qb + r,  0 <= r < b 

	  a => dividend, q => quotient, b => divisor, r => remainder 
	 */  
	if (x == y) {  
		return x /*or y*/;  
	}  

	int dividend = x, divisor = y, remainder = 0, quotient = 0;  

	do {  
		remainder = dividend % divisor;  
		quotient = dividend / divisor;  

		if(remainder) {  
			dividend = divisor;  
			divisor = remainder;  
		}  
	}  
	while(remainder);  

	return divisor;  
}  

/**
 * Calculates least common multiple. This method is never used.
 *
 * @param int x
 * @param int y
 * @return void
 */

int lcm(int x, int y) {  
	/* 
	   lcm(x,y) = (x * y) / gcd(x,y) 
	 */  
	return x == y ? x /*or y*/ : (x * y) / gcd(x,y);  
}  

/**
 * This method does nothing. You just pass it the address of some variable
 * so that the compiler won't complain "variable b is initialized but never used"
 *
 * @param void* nothing
 * @return void
 */

void noop(void * nothing) {
}
