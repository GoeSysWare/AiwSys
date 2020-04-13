
#include "projects/adas/algorithm/algorithm_type.h" 
namespace watrix {
	namespace algorithm {
        class LaneStatus{
            public:
                static double GetFirstDer(std::vector<double>& param_list, double x);
                static double GetSecondDer(std::vector<double>& param_list, double x);
                static double GetCurvature(std::vector<double>& param_list, double x);
                static double GetCurvatureR(std::vector<double>& param_list, double x);

                static std::vector<double> polyfit(std::vector<double>& x, std::vector<double>& y, int n);
                static double polyfit_predict(std::vector<double>& param_list, double x);

                static int GetLaneStatus(
                    dpoints_t& v_param_list, std::vector<dpoints_t>& v_src_dist_lane_points,
                    dpoints_t& curved_point_list, std::vector<double>& curved_r_list
                );
        };
    }
}//namespace
