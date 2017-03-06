// Miguel Ascanio Gómez
// GPU - Práctica 1
#include "benchUtils.hh"

std::ostream& operator<<(std::ostream& out, const TicToc& f) {
	string s;
	std::stringstream ss;
	ss << setw(9) << setfill('0') << f.nsec;
	s = ss.str();
	for (int i = s.size() - 3; i > 0; i -= 3) {
		s.insert(s.begin() + i, ',');
	}

	return out << "Secs: " << f.sec << " Nsecs: " << setw(11) << s;
}
