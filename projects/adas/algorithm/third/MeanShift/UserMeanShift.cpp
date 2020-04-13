#include <cmath>

#include <iostream>
#include "UserMeanShift.h"

using namespace std;

//const double MIN_DISTANCE = 0.000001;

namespace watrix {
	namespace algorithm {


        double euclidean_distance(const vector<double> &point_a, const vector<double> &point_b){
            double total = 0;
            for(int i=0; i<point_a.size(); i++){
                const double temp = (point_a[i] - point_b[i]);
                total += temp*temp;
            }
            return sqrt(total);
        }

        double euclidean_distance_sqr(const vector<double> &point_a, const vector<double> &point_b){
            double total = 0;
            for(int i=0; i<point_a.size(); i++){
                const double temp = (point_a[i] - point_b[i]);
                total += temp*temp;
            }
            return (total);
        }

        double gaussian_kernel(double distance, double kernel_bandwidth){
            double temp =  exp(-1.0/2.0 * (distance*distance) / (kernel_bandwidth*kernel_bandwidth));
            return temp;
        }

        void UserMeanShift::set_kernel( double (*_kernel_func)(double,double) ) {
            if(!_kernel_func){
                kernel_func = gaussian_kernel;
            } else {
                kernel_func = _kernel_func;    
            }
        }

        void UserMeanShift::shift_point(const dpoint_t &point,
                                    const std::vector<dpoint_t> &points,
                                    double kernel_bandwidth,
                                    dpoint_t &shifted_point) {
            shifted_point.resize( point.size() ) ;
            for(int dim = 0; dim<shifted_point.size(); dim++){
                shifted_point[dim] = 0;
            }
            double total_weight = 0;
            for(int i=0; i<points.size(); i++){
                const dpoint_t& temp_point = points[i];
                double distance = euclidean_distance(point, temp_point);
                double weight = kernel_func(distance, kernel_bandwidth);
                for(int j=0; j<shifted_point.size(); j++){
                    shifted_point[j] += temp_point[j] * weight;
                }
                total_weight += weight;
            }

            const double total_weight_inv = 1.0/total_weight;
            for(int i=0; i<shifted_point.size(); i++){
                shifted_point[i] *= total_weight_inv;
            }
        }

        std::vector<dpoint_t> UserMeanShift::meanshift(
            const std::vector<dpoint_t> &points,
            double kernel_bandwidth
        )
        {
            double MIN_DISTANCE = 0.000001;

            vector<bool> stop_moving(points.size(), false);
            vector<dpoint_t> shifted_points = points;
            double max_shift_distance;
            dpoint_t point_new;
            do {
                max_shift_distance = 0;
                for(int i=0; i<points.size(); i++){
                    if (!stop_moving[i]) {
                        shift_point(shifted_points[i], points, kernel_bandwidth, point_new);
                        double shift_distance_sqr = euclidean_distance_sqr(point_new, shifted_points[i]);
                        if(shift_distance_sqr > max_shift_distance){
                            max_shift_distance = shift_distance_sqr;
                        }
                        if(shift_distance_sqr <= MIN_DISTANCE) {
                            stop_moving[i] = true;
                        }
                        shifted_points[i] = point_new;
                    }
                }
                //printf("max_shift_distance: %f\n", sqrt(max_shift_distance));
            } while (max_shift_distance > MIN_DISTANCE);
            return shifted_points;
        }

        mean_shift_result_t UserMeanShift::cluster(
            const std::vector<dpoint_t>& points,
            const std::vector<dpoint_t>& shifted_points,
            double cluster_epsilon
        )
        {
            std::vector<dpoint_t> cluster_centers;
            std::vector<int> cluster_ids;

            for (int i = 0; i < shifted_points.size(); i++) {

                int c = 0;
                for (; c < cluster_centers.size(); c++) {
                    double distance = euclidean_distance(shifted_points[i], cluster_centers[c]);
                    //std::cout<<" distance = "<< distance << std::endl;
                    if ( distance <= cluster_epsilon) {
                        break;
                    }
                }

                if (c == cluster_centers.size()) {
                    cluster_centers.push_back(shifted_points[i]);
                }
                cluster_ids.push_back(c); 
            }
            
            mean_shift_result_t result;
            result.original_points = points;
            result.cluster_centers = cluster_centers;
            result.cluster_ids = cluster_ids;
            return result;
        }


        // # case(1) demo:     kernel_bandwidth = 3.0, cluster_epsilon = 6
        // # case(2) laneseg:  kernel_bandwidth = 0.5, cluster_epsilon = 2
        mean_shift_result_t UserMeanShift::cluster(
            const std::vector<dpoint_t> &points, 
            double kernel_bandwidth,
            double cluster_epsilon
        )
        {
            vector<dpoint_t> shifted_points = meanshift(points, kernel_bandwidth);
            return cluster(points, shifted_points, cluster_epsilon);
        }

	}
}// end namespace
