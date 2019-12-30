#include "timer.h"

namespace watrix {
	namespace algorithm {

		timer::timer()
		{

		}

		timer::~timer()
		{

		}

		void timer::tick()
		{
			pt1 = boost::posix_time::microsec_clock::local_time();
		}

		int64_t timer::toke()
		{
			pt2 = boost::posix_time::microsec_clock::local_time();
			int64_t delta = (pt2 - pt1).total_milliseconds();
			m_count++;

			m_totoal += delta;
			return delta;
		}

		int64_t timer::total()
		{
			return m_totoal;
		}

	}
}// end namespace