// Miguel Ascanio Gómez
// GPU - Práctica 1
#pragma once

#include <sys/time.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
using namespace std;

class TicToc;
std::ostream& operator<<(std::ostream& out, const TicToc& f);

class TicToc {

	timespec start, end;

public:

	inline void tic() {
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	}

	inline void toc() {
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
		diff();
	}

	time_t sec;
	long nsec;

private:
	void diff() {
		if ((end.tv_nsec - start.tv_nsec) < 0) {
			sec = end.tv_sec - start.tv_sec - 1;
			nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
		} else {
			sec = end.tv_sec - start.tv_sec;
			nsec = end.tv_nsec - start.tv_nsec;
		}
	}
};
