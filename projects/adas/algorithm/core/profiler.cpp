#include "profiler.h"

// glog
#include <glog/logging.h>

#include <iostream>

namespace watrix {
	namespace algorithm {

		profiler::profiler(const char* func_name)
		{
			pt1 = boost::posix_time::microsec_clock::local_time();
			m_func_name = func_name;
		}

		profiler::~profiler()
		{
			boost::posix_time::ptime pt2 = boost::posix_time::microsec_clock::local_time();
			int cost = (pt2 - pt1).total_milliseconds();
			//! post to some manager
			//test:
			LOG(INFO)<<"[API] [" << m_func_name << "] cost=" << cost << " ms" << std::endl;
		}

	}
}// end namespace