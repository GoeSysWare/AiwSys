#include "cluster_util.h"

// std
#include <iostream>
using namespace std;

// user defined mean shift 
#include "projects/adas/algorithm/third/MeanShift/UserMeanShift.h"

/*
// compile error with pcl 20190829
// mlpack clustering
#include <mlpack/core.hpp>
#include <mlpack/methods/mean_shift/mean_shift.hpp>

#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/dbscan/random_point_selection.hpp>

//using namespace mlpack;
//using namespace mlpack::dbscan;
//using namespace mlpack::meanshift;
//using namespace mlpack::distribution;
*/

namespace watrix {
	namespace algorithm {
		
		mean_shift_result_t ClusterUtil::user_meanshift(
				const std::vector<dpoint_t>& points, 
				double kernel_bandwidth,
				double cluster_epsilon
		)
		{
			UserMeanShift ms_obj;
			mean_shift_result_t result = ms_obj.cluster(
				points, kernel_bandwidth, cluster_epsilon
				);
			return result;
		}

		/* 
		mean_shift_result_t ClusterUtil::mlpack_meanshift(
				const std::vector<dpoint_t>& points, 
				double radius,
				int max_iterations,
				double kernel_bandwidth
		)
		{
			int rows = points.size();
			assert(rows>=1);
			int cols = points[0].size();

			arma::mat meanShiftData(rows, cols); // 330*8
			for(int r=0; r< rows; r++){
				for(int c=0; c< cols ;c++){
					meanShiftData.row(r).col(c) = points[r][c];
				}
			}

			arma::mat meanShiftData_trans = (arma::mat) arma::trans(meanShiftData);

			// use gaussian kernel
			mlpack::kernel::GaussianKernel kernel(kernel_bandwidth);
			mlpack::meanshift::MeanShift<true> cluster_obj(radius, max_iterations, kernel);
			//mlpack::dbscan::DBSCAN<> cluster_obj(cluster_epsilon, 10);

			arma::Row<size_t> assignments;
			arma::mat centroids;
			cluster_obj.Cluster(meanShiftData_trans, assignments, centroids);

#ifdef DEBUG_INFO
			std::cout<<"rows = "<< rows << std::endl;
			std::cout<<"cols = "<< cols << std::endl; // 330*8
			//std::cout<<"assignments = "<< assignments << std::endl; // 0,1,2,3,4

			std::cout<<"centroids.n_rows = "<< centroids.n_rows << std::endl;
			std::cout<<"centroids.n_cols = "<< centroids.n_cols << std::endl;
			std::cout<<"centroids = \n"<< centroids << std::endl; // 8*5
#endif 

			// get centers and cluster ids
			std::vector<dpoint_t> cluster_centers;
			std::vector<int> cluster_ids(rows);

			for(int i=0;i<rows;i++){
				cluster_ids[i] = assignments(i);
			}

			// center by col
			for(int c=0; c< centroids.n_cols ;c++){
				dpoint_t center;
				for(int r=0; r< centroids.n_rows; r++){
					double value = centroids.row(r).col(c)[0];
					//std::cout<<"value = "<< value << std::endl;
					center.push_back(value);
				}
				cluster_centers.push_back(center);
			}

			// pass out mean shift result
			mean_shift_result_t result;
            result.original_points = points;
            result.cluster_centers = cluster_centers;
            result.cluster_ids = cluster_ids;
			return  result;
		}

		mean_shift_result_t ClusterUtil::mlpack_dbscan(
				const std::vector<dpoint_t>& points, 
				double cluster_epsilon,
				int min_pts
		)
		{
			int rows = points.size();
			assert(rows>=1);
			int cols = points[0].size();

			arma::mat meanShiftData(rows, cols); // 330*8
			for(int r=0; r< rows; r++){
				for(int c=0; c< cols ;c++){
					meanShiftData.row(r).col(c) = points[r][c];
				}
			}

			arma::mat meanShiftData_trans = (arma::mat) arma::trans(meanShiftData);

			//mlpack::meanshift::MeanShift<> cluster_obj(cluster_epsilon);
			mlpack::dbscan::DBSCAN<> cluster_obj(cluster_epsilon, min_pts);

			arma::Row<size_t> assignments;
			arma::mat centroids;
			cluster_obj.Cluster(meanShiftData_trans, assignments, centroids);

#ifdef DEBUG_INFO
			std::cout<<"rows = "<< rows << std::endl;
			std::cout<<"cols = "<< cols << std::endl; // 330*8
			//std::cout<<"assignments = "<< assignments << std::endl; // 0,1,2,3,4

			std::cout<<"centroids.n_rows = "<< centroids.n_rows << std::endl;
			std::cout<<"centroids.n_cols = "<< centroids.n_cols << std::endl;
			std::cout<<"centroids = \n"<< centroids << std::endl; // 8*5
#endif 

			// get centers and cluster ids
			std::vector<dpoint_t> cluster_centers;
			std::vector<int> cluster_ids(rows);

			for(int i=0;i<rows;i++){
				cluster_ids[i] = assignments(i);
			}

			// center by col
			for(int c=0; c< centroids.n_cols ;c++){
				dpoint_t center;
				for(int r=0; r< centroids.n_rows; r++){
					double value = centroids.row(r).col(c)[0];
					//std::cout<<"value = "<< value << std::endl;
					center.push_back(value);
				}
				cluster_centers.push_back(center);
			}

			// pass out mean shift result
			mean_shift_result_t result;
            result.original_points = points;
            result.cluster_centers = cluster_centers;
            result.cluster_ids = cluster_ids;
			return  result;
		}
		*/


		mean_shift_result_t ClusterUtil::cluster(
			const std::vector<dpoint_t>& points, 
			const LaneInvasionConfig& config
		)
		{
			mean_shift_result_t result;
			switch (config.cluster_type)
			{
			case CLUSTER_TYPE::USER_MEANSHIFT:
				// # case(1) demo:     kernel_bandwidth = 3.0, cluster_epsilon = 6
				// # case(2) laneseg:  kernel_bandwidth = 0.5, cluster_epsilon = 2
				result	= ClusterUtil::user_meanshift(
					points, 
					config.user_meanshift_kernel_bandwidth, 
					config.user_meanshift_cluster_epsilon
				);
				break;
			/* 
			case CLUSTER_TYPE::MLPACK_MEANSHIFT:
				result	= ClusterUtil::mlpack_meanshift(
					points, 
					config.mlpack_meanshift_radius,
					config.mlpack_meanshift_max_iterations,
					config.mlpack_meanshift_bandwidth
				);
				break;
			case CLUSTER_TYPE::MLPACK_DBSCAN:
				result	= ClusterUtil::mlpack_dbscan(
					points, 
					config.mlpack_dbscan_cluster_epsilon, 
					config.mlpack_dbscan_min_pts
				);
				break;
			*/
			default:
				break;
			}
			return result;
		}


	}
}// end namespace

/*
cluster_centers.size() = 4
cluster_ids.size() = 330
cluster 0, grid count = 77 
cluster 1, grid count = 101 
cluster 2, grid count = 73 
cluster 3, grid count = 79 

lane count = 4, id_left = 3,  id_right = 0 
[result-box] invasion_status = 1, distance = 0.309597 
[result-box] invasion_status = 1, distance = 1111.000000 



=========================================================
rows = 330
cols = 8
centroids.n_rows = 8
centroids.n_cols = 5
centroids = 
   0.0916   0.0032   0.6940   0.7440   0.0035
   0.8008   1.4091   0.9335        0   2.3600
        0        0        0        0        0
   0.2241   0.0079   1.7164   1.8591   0.0083
        0        0        0        0        0
        0   0.0028   1.6989   0.0016   0.5017
        0        0        0        0        0
   0.7391   2.3713   0.0040   1.6809   0.0067


cluster_centers.size() = 5
cluster_ids.size() = 330
cluster 0, grid count = 2 
cluster 1, grid count = 76 
cluster 2, grid count = 78 
cluster 3, grid count = 101 
cluster 4, grid count = 73 

*/