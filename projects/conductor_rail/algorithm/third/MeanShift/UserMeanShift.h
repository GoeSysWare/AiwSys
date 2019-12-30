#pragma once 

#include <vector>
#include "algorithm/algorithm_type.h"


namespace watrix {
	namespace algorithm {

        class UserMeanShift {
        public:
            UserMeanShift() { set_kernel(nullptr); }
            UserMeanShift(double (*_kernel_func)(double,double)) { set_kernel(kernel_func); }
            mean_shift_result_t cluster(const std::vector<dpoint_t>&, double, double);

        private:
            double (*kernel_func)(double,double);
            void set_kernel(double (*_kernel_func)(double,double));
            void shift_point(const dpoint_t&, const std::vector<dpoint_t>&, double, dpoint_t&);
            std::vector<dpoint_t> meanshift(const std::vector<dpoint_t>& points,double kernel_bandwidth);
            mean_shift_result_t cluster(const std::vector<dpoint_t>&, const std::vector<dpoint_t>&, double);
        };


	}
}// end namespace