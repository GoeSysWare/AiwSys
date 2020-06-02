#include "laneseg_util.h"

#include "projects/adas/algorithm/core/util/filesystem_util.h"
#include "projects/adas/algorithm/core/util/display_util.h"
#include "projects/adas/algorithm/core/util/opencv_util.h"
#include "projects/adas/algorithm/core/util/numpy_util.h"
#include "projects/adas/algorithm/core/util/polyfiter.h"
#include "projects/adas/algorithm/core/util/lane_util.h"

// user defined mean shift 
//#include "algorithm/third/NumCpp.hpp"
//#include "algorithm/third/cluster_util.h"

#include "projects/adas/algorithm/autotrain/monocular_distance_api.h"
#include "projects/adas/algorithm/autotrain/ObjectAndTrainDetection.h"
#include "projects/adas/algorithm/autotrain/GetLaneStatus.h"

// std
#include <iostream>
#include <map>
#include <omp.h>
#include <fstream>
#include <string>

// glog
#include <glog/logging.h>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

namespace watrix {
	namespace algorithm {
		namespace internal {

			cv::Size lanesegutil::ORIGIN_SIZE(1920,1080);
			// cv::Size lanesegutil::UPPER_SIZE(1920,568);
			// cv::Size lanesegutil::CLIP_SIZE(1920,512);
			cv::Size lanesegutil::UPPER_SIZE(1920,440);
			cv::Size lanesegutil::CLIP_SIZE(1920,640);
			cv::Size lanesegutil::CAFFE_INPUT_SIZE(1024,256);
			// // pt_simple v2 v3 v4 resize 480 160
			// cv::Size lanesegutil::PT_SIMPLE_INPUT_SIZE(480,160);

			// pt_simple v5 resize 480 240
			// cv::Size lanesegutil::PT_SIMPLE_INPUT_SIZE(480,240);

			// pt_simple v7 sequence resize 480 272
			cv::Size lanesegutil::PT_SIMPLE_INPUT_SIZE(480,272);


#pragma region get_lane_full_binary_mask

			cv::Mat lanesegutil::get_lane_full_binary_mask(
				const cv::Mat& binary_mask
			)
			{
				// binary_mask 255: 256, 1024 value=0,1 (caffe V1)
				// binary_mask 255: 128, 480  value=0,1  (pytorch simple laneseg)
				cv::Mat binary_mask_255 = NumpyUtil::np_binary_mask_as255(binary_mask);
				/*
				binary_mask_255: 256, 1024  ===> resize to 512,1920
				binary_mask 255: 128, 480   ===> resize to 512,1920
				*/
				cv::Mat clip_binary_mask; // 512,1920
				cv::resize(binary_mask_255, clip_binary_mask, CLIP_SIZE);

				//==========================================
				cv::Mat upper_mask(UPPER_SIZE, CV_8UC1, Scalar(0)); // 568,1920
				cv::Mat full_binary;
				cv::vconcat(upper_mask, clip_binary_mask, full_binary);
				//==========================================
				return full_binary;
			}
#pragma endregion


#pragma region point cvpoint/dpoints 

			dpoints_t lanesegutil::cvpoints_to_dpoints(const cvpoints_t& cvpoints)
			{
				dpoints_t points; // n*2
				if (cvpoints.size() <= 0){
					return points;
				}
				for (auto& cvpoint: cvpoints)
				{
					dpoint_t point;
					point.push_back(cvpoint.x);
					point.push_back(cvpoint.y);
					// std::cout << cvpoint.x << "," << cvpoint.y << std::endl;
					points.push_back(point);
				}
				return points;
			}

			cvpoints_t lanesegutil::dpoints_to_cvpoints(const dpoints_t& points)
			{
				cvpoints_t cvpoints; // n*2
				if (points.size() <= 0){
					return cvpoints;
				}
				for (auto& point: points)
				{
					int x = int(round(point[0]));
					int y = int(round(point[1]));
					// std::cout << point[0] << "," << point[1] << "," << x << "," << y << std::endl;
					cvpoint_t cvpoint = cv::Point2i(x, y);
					cvpoints.push_back(cvpoint);
				}
				return cvpoints;
			}

			void lanesegutil::print_points(const std::string& name, const dpoints_t& points)
			{
				std::cout<<"=========================================="<<std::endl;
				std::cout<<name<<std::endl;
				std::cout<<"=========================================="<<std::endl;

				for(auto& point: points){
					for(auto& dim: point){
						std::cout<<" "<<dim;
					}
					std::cout<<std::endl;
				}
				std::cout<<std::endl;
			}

			void lanesegutil::print_points(const std::string& name, const cvpoints_t& cvpoints)
			{
				std::cout<<"=========================================="<<std::endl;
				std::cout<<name<<std::endl;
				std::cout<<"=========================================="<<std::endl;

				for(auto& point: cvpoints){
					std::cout<<" "<<point.x <<", "<<point.y<<std::endl;
				}
				std::cout<<std::endl;
			}
#pragma endregion


#pragma region transform points
			dpoints_t lanesegutil::get_tr33_from_tr34(const dpoints_t& tr34)
			{
				// get coloum 0,2,3 (skip column 1)
				dpoints_t tr33_inverse = NumpyUtil::np_delete(tr34, 1);
				dpoints_t tr33 = NumpyUtil::np_inverse(tr33_inverse);

#ifdef DEBUG_INFO 
			if (false)
			{
				print_points("tr34", tr34);
				print_points("tr33_inverse", tr33_inverse);
				print_points("tr33", tr33);
			}	
#endif

				return tr33;
			}

#pragma region transform v0
			dpoints_t lanesegutil::image_points_to_distance_points_v0(
				const dpoints_t& points, 
				const dpoints_t& tr33,
				double z_height
			)
			{
				//========================================================
				// for lane point in origin_image(1080,1920)
				//========================================================
				// image coord 1080,1920  ===> distance coord  x=[-5,5] y=[15,60] 
				dpoints_t coord_trans;
				if (points.size()==0){
					printf("[hit] image_points_to_distance_points_v0 ==0 \n");
					return coord_trans;
				}

				int n = points.size();
				dpoints_t points_dim3 = NumpyUtil::np_insert(points, 1); // n*3 (x,y,z=1)
				dpoints_t points_T = NumpyUtil::np_transpose(points_dim3); // 3*n

				// # 3*3, 3*n (x,y,z=1) ===> 3*n (x1,y1,z1) 
				coord_trans = NumpyUtil::np_matmul(tr33, points_T);// 3*n
				assert(coord_trans.size()==3);
				for(int i=0;i<n;i++)
				{
					coord_trans[0][i] = z_height * (coord_trans[0][i]/coord_trans[1][i]); //  1.8*x1/y1  ===> X
					coord_trans[2][i] = z_height * (coord_trans[2][i]/coord_trans[1][i]); //  1.8*z1/y1  ===> Y
				}
				return coord_trans; // 3*n (X,y1,Y)
			}

			cvpoints_t lanesegutil::distance_points_to_image_points_v0(
				const dpoints_t& coord_trans_, 
				const dpoints_t& tr33,
				double z_height
			)
			{
				/*
				 #  distance coord  x=[-5,5] y=[15,60]  ===> image coord 1080,1920
    			 #  coord_trans: 3*n (X,y1,Y) ===> coord: n*2 (x,y)
				*/
				cvpoints_t cvpoints; 

				dpoints_t coord_trans = coord_trans_;
				if (coord_trans.size()==0){
					printf("[hit] distance_points_to_image_points_v0 ==0 \n");
					return cvpoints;
				}

				// # (1)  3*n (X,y1,Y) ===> 3*n (x1,y1,z1)
				assert(coord_trans.size()==3);
				int n = coord_trans[0].size();
				for(int i=0;i<n;i++)
				{
					coord_trans[0][i] = (coord_trans[0][i] * coord_trans[1][i]) / z_height; //  x1 = X*y1/1.8
					coord_trans[2][i] = (coord_trans[2][i] * coord_trans[1][i]) / z_height; //  z1 = Y*y1/1.8
				}

				// # (2)  3*3_Inverse, 3*n (x1,y1,z1) ===> 3*n (x,y,1)
				dpoints_t inverse_tr33 = NumpyUtil::np_inverse(tr33); 
				/*
				6222.2 0 889.046
 				0 6273.33 453.648
 				0 0 1
				*/
				dpoints_t points_T = NumpyUtil::np_matmul(inverse_tr33, coord_trans);// 3*n
				dpoints_t points_dim3  = NumpyUtil::np_transpose(points_T); // n*3 (x,y,1)
				dpoints_t points = NumpyUtil::np_delete(points_dim3, 2); // n*3 (x,y,1) ===> n*2 (x,y)

				cvpoints = dpoints_to_cvpoints(points); 
				return cvpoints;
			}
#pragma endregion

#pragma region transform v1
			dpoints_t lanesegutil::nouse_image_points_to_distance_points_v1(
				const dpoints_t& points, 
				const dpoints_t& tr33,
				double z_height
			)
			{
				//========================================================
				// for lane point in origin_image(1080,1920)
				//========================================================
				// image coord 1080,1920  ===> distance coord  x=[-5,5] y=[15,60] 
				dpoints_t coord_trans;
				if (points.size()==0){
					printf("[hit] nouse_image_points_to_distance_points_v1 ==0 \n");
					return coord_trans;
				}

				int n = points.size();
				dpoints_t points_dim3 = NumpyUtil::np_insert(points, 1); // n*3 (x,y,z=1)
				//print_points("points_dim3", points_dim3);

				dpoints_t points_T = NumpyUtil::np_transpose(points_dim3); // 3*n
				//print_points("points_T", points_T);

				// # 3*3, 3*n (x,y,z=1) ===> 3*n (x1,y1,z1) 
				coord_trans = NumpyUtil::np_matmul(tr33, points_T);// 3*n
				//print_points("coord_trans", coord_trans);

				/*
				995,955

				0.0236071
				0.943424
				0.0766408

				X= 0.308023
				Y= 12.3097
				==========================================
				coord_trans
				==========================================
				0.308023
				0.0766408
				12.3097
				*/

				assert(coord_trans.size()==3);
				for(int i=0;i<n;i++)
				{
					double x1 = coord_trans[0][i];
					double y1 = coord_trans[1][i];
					double z1 = coord_trans[2][i];

					// z_height *
					double X =  (x1/z1); //  1.8*x1/z1  ===> X
					double Y =  (y1/z1);  // 1.8*y1/z1  ===> Y

					// keep order X,z1,Y
					coord_trans[0][i] = X;
					//coord_trans[1][i] = z1; // Key step
					coord_trans[1][i] = 0; // Key step
					coord_trans[2][i] = Y; 
				}
				return coord_trans; // 3*n (X,z1,Y)
			}


			dpoints_t lanesegutil::image_points_to_distance_points_v1(
				TABLE_TYPE table_type, // long_a, short_a
				const dpoints_t& points
			)
			{
				// image point [n,2] (x,y) ---> coord point (3,n) (X,z1,Y) 
				dpoints_t coord_trans; //  (X,z1,Y) (3,n)
				int n = points.size();
				// std::cout << "n:" << n << std::endl;
				int dim = 3; 
				coord_trans.resize(dim);
				// for(int d=0; d<dim; d++){
				// 	coord_trans[d].resize(n);
				// }

				if (n==0){
					printf("[hit] image_points_to_distance_points_v1 ==0 \n");
					return coord_trans;
				}

				for(int i=0;i<n;i++)
				{
					unsigned int X = int(points[i][0]);
					unsigned int Y = int(points[i][1]);


					float x,y;
					bool success = MonocularDistanceApi::get_distance(table_type, Y,X, x, y);
					//  (X,z1,Y) (3,n)
					if (success){
						// CAMERA_TYPE::CAMERA_LONG  1 
						if (table_type == 1){
							if (round(x) > -10 && round(x) < 10 && round(y) > 30 && round(y) < 350){
								// std::cout << success << "," << X << "," << Y << "," << x << "," << y << std::endl;
								//std::cout << success << ",long " << points[i][0] << "," << points[i][1] << "," << X << "," << Y << "," << x << "," << y << std::endl;
								coord_trans[0].push_back(x);
								coord_trans[1].push_back(0);
								coord_trans[2].push_back(y);
							}
						}
						// CAMERA_TYPE::CAMERA_SHORT
						else {
							if (round(x) > -20 && round(x) < 20 && round(y) > 0 && round(y) < 80){
							// if (round(x) > -20 && round(x) < 20 && round(y) > 0 && round(y) < 150){
								//std::cout << success << ",short  " << points[i][0] << "," << points[i][1] << "," << X << "," << Y << "," << x << "," << y << std::endl;
								coord_trans[0].push_back(x);
								coord_trans[1].push_back(0);
								coord_trans[2].push_back(y);
							}
						}
					}
				}
				return coord_trans;
			}


			cvpoints_t lanesegutil::distance_points_to_image_points_v1(
				const dpoints_t& coord_trans_, 
				const dpoints_t& tr34,
				double z_height
			)
			{
				/*
				 #  distance coord  x=[-5,5] y=[15,60]  ===> image coord 1080,1920
    			 #  coord_trans: 3*n (X,z1,Y) ===> coord: n*2 (x,y)
				*/
				cvpoints_t cvpoints; 

				dpoints_t coord_trans = coord_trans_;
				if (coord_trans.size()==0){
					printf("[hit] distance_points_to_image_points_v1 ==0 \n");
					return cvpoints;
				}

				// # (1)  3*n (X,z1,Y) ===> 4*n (X,0,Y,1)
				assert(coord_trans.size()==3);

				int n = coord_trans[0].size();
				//std::cout<<" n =" << n << std::endl;
				//std::cout<<"******************************\n";
				//print_points("coord_trans", coord_trans);

				if (n <= 0){
					return cvpoints;
				}

				int dim = 4; // 
				dpoints_t new_coord_trans; //  4*n (X,0,Y,1)
				new_coord_trans.resize(dim);
				for(int d=0; d<dim; d++){
					new_coord_trans[d].resize(n);
				}

				for(int i=0;i<n;i++)
				{
					double X = coord_trans[0][i];
					double Y = coord_trans[2][i];

					new_coord_trans[0][i] = X;
					new_coord_trans[1][i] = 0; // at ground 
					new_coord_trans[2][i] = Y;
					new_coord_trans[3][i] = 1;
				}
				//print_points("new_coord_trans", new_coord_trans);

				// # (2)  3*4, 4*n (X,0,Y,1) ===> 3*n (x1,y1,z1)
				dpoints_t points = NumpyUtil::np_matmul(tr34, new_coord_trans);// 3*n
				// print_points("points", points);

				for(int i=0;i<n;i++)
				{
					double x1 = points[0][i];
					double y1 = points[1][i];
					double z1 = points[2][i];

					double x = x1/z1;
					double y = y1/z1;

					// keep result (x,y,z1)
					points[0][i] = x;
					points[1][i] = y;
					// std::cout << new_coord_trans[0][i] << "," << new_coord_trans[2][i] << "," << x1 << "," << y1 << "," << x << "," << y << std::endl;
				}
				// std::cout << "\n" << std::endl;

				dpoints_t points_T = NumpyUtil::np_transpose(points); // 3*n ===>n*3

				// n*3 (x,y,z1) ===> n*2 (x,y)
				dpoints_t points_results = NumpyUtil::np_delete(points_T, 2); 

				// for (int idx=0; idx<points_T.size(); idx++){
				// 	std::cout << points_T[idx][0] << "," << points_T[idx][1] << std::endl;
				// }
				// std::cout << "\n" << std::endl;

				cvpoints = dpoints_to_cvpoints(points_results); 

				return cvpoints;
			}
			/*
			==========================================
			coord_trans
			==========================================
			0.308023
			0.0766408
			12.3097

			==========================================
			new_coord_trans
			==========================================
			0.308023
			0
			12.3097
			1

			==========================================
			points
			==========================================
			12982.6
			12460.7
			13.0479

			==========================================
			points_results
			==========================================
			995
			955
			13.0479
			*/
#pragma endregion

#pragma region transform vector
			dpoints_t lanesegutil::image_points_to_distance_points(
				CAMERA_TYPE camera_type, // long_a, short_a
				const LaneInvasionConfig& config,
				const dpoints_t& points
			)
			{
				TABLE_TYPE table_type = TABLE_LONG_A;
				if (camera_type == CAMERA_SHORT){
					table_type = TABLE_SHORT_A;
				}
				return image_points_to_distance_points_v1(table_type, points);
			}

			cvpoints_t lanesegutil::distance_points_to_image_points(
				CAMERA_TYPE camera_type, // long_b, short_b
				const LaneInvasionConfig& config,
				const dpoints_t& coord_trans
			)
			{
					dpoints_t tr34 = config.tr34_long_b;
					if (camera_type == CAMERA_SHORT){
						tr34 = config.tr34_short_b;
					} 
					return distance_points_to_image_points_v1(coord_trans, tr34, config.z_height);
			}
#pragma endregion

#pragma region transform sigle
			dpoint_t lanesegutil::image_point_to_dist_point(
				CAMERA_TYPE camera_type, // long_a, short_a
				const LaneInvasionConfig& config,
				cvpoint_t point
			){
				// image point (n,2) ---> coord point (X,z1,Y) (3,n)
				dpoint_t pt{(double)point.x, (double)point.y};
				dpoints_t points{pt};
				dpoints_t points_trans = image_points_to_distance_points(camera_type, config, points); 
				double x_dist = points_trans[0][0];
				double z1_dist = points_trans[1][0];
				double y_dist = points_trans[2][0];
				dpoint_t point_trans{x_dist, z1_dist, y_dist};
				return point_trans;
			}

			cvpoint_t lanesegutil::dist_point_to_image_point(
				CAMERA_TYPE camera_type, // long_b, short_b
				const LaneInvasionConfig& config,
				dpoint_t point_trans
			){
				// coord point (X,z1,Y) ---> image point
				dpoint_t point_x{point_trans[0]};
				dpoint_t point_z1{point_trans[1]};
				dpoint_t point_y{point_trans[2]};

				dpoints_t points_trans{point_x, point_z1, point_y};
		
				cvpoints_t points = distance_points_to_image_points(camera_type, config, points_trans);
				cvpoint_t point = points[0];
				return point;
			}

			void lanesegutil::bound_cvpoint(cvpoint_t& point, cv::Size size, int delta)
			{
				// (1965,904)  ===> (1919,904)
				if(point.x<0 ){
					point.x = delta;
				}
				if(point.x>= size.width){
					point.x = size.width - 1 - delta;
				}

				if(point.y<0 ){
					point.y = delta;
				}
				if(point.y>= size.height){
					point.y = size.height - 1 - delta;
				}

			}
#pragma endregion


#pragma endregion


#pragma region find x points by y
			/*
			y_mesh = (coord_trans[2,:]-y)*1000
			y_mesh = np.abs(y_mesh)
			y_mesh = y_mesh - np.min(y_mesh)
			y_id = np.where(y_mesh==0)
			*/
			std::vector<double> lanesegutil::get_lane_x_coords(const dpoints_t& lane_coord_trans, double y)
			{
				/*
				# for one single lane:  3*n
				# y_id = np.where(Y==y)  for integer array (not for float array)
				# find all x coords where Y==y for left/right lane
				# x_coords.size() MAY BE 0
				*/
				//printf(" coord_trans[2] Y.size = %d \n", coord_trans[2].size());
				std::vector<int> y_idx = NumpyUtil::np_argwhere_eq(lane_coord_trans[2], y);
				std::vector<double> x_coords;
				for(auto& index: y_idx){
					x_coords.push_back(lane_coord_trans[0][index]);
				};
				//printf(" x_coords.size = %d \n", x_coords.size());
				return x_coords;
			}

			std::vector<int> lanesegutil::get_lane_x_image_points(const cvpoints_t& lane_image_points, double y)
			{
				std::vector<int> lane_y_points;
				for(auto& point: lane_image_points){
					lane_y_points.push_back(point.y);
				}
				std::vector<int> y_idx = NumpyUtil::np_argwhere_eq(lane_y_points, y);
				std::vector<int> x_points;
				for(auto& index: y_idx){
					x_points.push_back(lane_image_points[index].x);
				};
				return x_points;
			}
#pragma endregion 


#pragma region get clustered lane points

			
			void lanesegutil::x_get_clustered_lane_points(
				int lane_model_type, // caffe, pt_simple, pt_complex
				const LaneInvasionConfig& config,
				const cv::Mat& binary_mask,  // [256,1024] v=[0,1]
				const channel_mat_t& instance_mask, // [8, 256,1024] 8-dim float feature map
				std::vector<dpoints_t>& v_src_lane_points
			)
			{
				switch (lane_model_type)
				{
				case LANE_MODEL_CAFFE:
					LaneUtil::get_clustered_lane_points_from_features(
						config, binary_mask, instance_mask, v_src_lane_points
					);
					break;
				case LANE_MODEL_PT_SIMPLE:
					LaneUtil::get_clustered_lane_points_from_left_right(
						config, binary_mask, instance_mask, v_src_lane_points
					);
					break;
				case LANE_MODEL_PT_COMPLEX:
					/* code */
					break;
				default:
					break;
				}
			}


#pragma endregion


#pragma region binary lanes to origin lanes		

			void lanesegutil::__transform_binary_lanes_to_origin_lanes(
				std::vector<dpoints_t>& v_lane_points,
				cv::Size input_size // (256,1024) / (128,480)
			)
			{
				// (256,1024) ===(512,1920)===>(1080,1920) 

				// // pt_simple v2 v3 v4 use CLIP_SIZE
				// float h_rad = CLIP_SIZE.height/(input_size.height*1.f);
				// float w_rad = CLIP_SIZE.width/(input_size.width*1.f);

				// pt_simple v5 use ORIGIN_SIZE
				float h_rad = ORIGIN_SIZE.height/(input_size.height*1.f);
				float w_rad = ORIGIN_SIZE.width/(input_size.width*1.f);

				//printf("h_rad %f, w_rad = %f \n", h_rad, w_rad); // h_rad 2.0, w_rad = 1.875 

				// process 4-lane points
				for(auto& one_lane_points: v_lane_points)
				{
					// (256,1024) ===(512,1920)===>(1080,1920) 
					for(auto& pointxy: one_lane_points)
					{
						// // pt_simple v2 v3 v4 use CLIP_SIZE use UPPER_SIZE.height
						// pointxy[0] = round(pointxy[0] * w_rad);
						// pointxy[1] = round(pointxy[1] * h_rad + UPPER_SIZE.height);
						// pt_simple v5 
						pointxy[0] = round(pointxy[0] * w_rad);
						pointxy[1] = round(pointxy[1] * h_rad);
					}
				}
			}

			void lanesegutil::x_transform_binary_lanes_to_origin_lanes(
				int lane_model_type, // caffe, pt_simple, pt_complex
				std::vector<dpoints_t>& v_lane_points
			)
			{
				cv::Size input_size = CAFFE_INPUT_SIZE;

				switch (lane_model_type)
				{
				case LANE_MODEL_CAFFE:
					input_size = CAFFE_INPUT_SIZE;
					break;
				case LANE_MODEL_PT_SIMPLE:
					input_size = PT_SIMPLE_INPUT_SIZE;
					break;
				case LANE_MODEL_PT_COMPLEX:
					/* code */
					break;
				default:
					break;
				}

				__transform_binary_lanes_to_origin_lanes(v_lane_points, input_size);
			}
#pragma endregion


#pragma region polyfit
		void lanesegutil::lane_polyfit(
				CAMERA_TYPE camera_type, // long, short
				const LaneInvasionConfig& config,
				int image_index,
				cv::Size origin_size,
				const std::vector<dpoints_t>& v_lane_points,
				std::vector<dpoints_t>& v_auto_range_lane_points,
				std::vector<dpoints_t>& v_user_range_lane_points,
				std::vector<cv::Mat>& v_left_right_polyfit_matk
			)
			{
				// polyfit for image point in (1080,1920) 
				std::vector<dpoints_t> v_fitted_lane_points;

#ifdef SAVE_POLYFIT_FUNC
				ofstream csv_file;
				csv_file.open("./seg.csv",ios::app);
				if(!csv_file){
					std::cout << "seg.csv can't open." << std::endl;
					abort();
				}
#endif

				// process 4-lane points
				int lane_id = 0;
				for(auto& one_lane_points: v_lane_points)
				{
					dpoints_t auto_range_lane_dpoints;
					
					if (one_lane_points.size() >= (config.polyfit_order+1)){
						cvpoints_t cvpoints = dpoints_to_cvpoints(one_lane_points);
						Polyfiter fiter(cvpoints, 
							config.polyfit_order, 
							config.reverse_xy,
							config.x_range_min,
							config.x_range_max,
							config.y_range_min,
							config.y_range_max
						); 
				
						TABLE_TYPE table_type = TABLE_LONG_A;
						if (camera_type == CAMERA_SHORT){
							table_type = TABLE_SHORT_A;
						}
						// CAMERA_TYPE::CAMERA_LONG  1 
						double thresh = 0.0;
						// if (table_type == 1){
						// 	thresh = 0.05;
						// }
						
						int thresh_num = int(one_lane_points.size() * thresh);
						int size_num = one_lane_points.size() - thresh_num;
						std::vector<dpoint_t> one_lane_points_new(one_lane_points.begin()+thresh_num, one_lane_points.end());
						// std::copy(one_lane_points.begin()+thresh_num, one_lane_points.begin()+size_num, one_lane_points_new);
						// std::cout << "one_lane_points size:" << one_lane_points.size() << " one_lane_points_new size:" << one_lane_points_new.size() << std::endl;
						cv::Mat mat_k = fiter.cal_mat_add_points(one_lane_points_new);
						v_left_right_polyfit_matk.push_back(mat_k);
#ifdef SAVE_POLYFIT_FUNC
						for (int j=5; j >= 0; j--){
							csv_file << "," << mat_k.at<double>(j, 0);
						}
#endif
						// change mat
						fiter.set_mat_k(mat_k);
						// std::cout << mat_k << std::endl;
						// cvpoints_t auto_range_lane_cvpoints = fiter.fit(true); 
						auto_range_lane_dpoints = fiter.fit_dpoint(camera_type, one_lane_points_new, true);
						// may be 0 sized
						//cvpoints_t user_range_lane_cvpoints = fiter.fit(false);
#ifdef DEBUG_INFO
						/*
						printf("[polyfit]  lane_id = %d, N (auto)= %d, N (full)= %d \n", 
							lane_id, (int)auto_range_lane_cvpoints.size(), (int)user_range_lane_cvpoints.size()
						); 
						*/

						// printf("[polyfit]  lane_id = %d, N (auto)= %d \n", 
						// 	lane_id, (int)auto_range_lane_cvpoints.size()
						// ); 
#endif

						if(config.b_save_temp_images){
							cv::Mat out = fiter.draw_image(origin_size);
							std::string filepath = config.output_dir + FilesystemUtil::str_pad(image_index) + std::to_string(lane_id)+"_c_fitted_lane.jpg";
							cv::imwrite(filepath, out);
						}

					}
					


					// if (auto_range_lane_cvpoints.size()>=config.min_lane_pts){
					// 	v_auto_range_lane_points.push_back(cvpoints_to_dpoints(auto_range_lane_cvpoints));
					// 	//v_user_range_lane_points.push_back(cvpoints_to_dpoints(full_range_lane_cvpoints));
					// }
					// lane_id++;

					// std::cout << "auto_range_lane_dpoints size:" << auto_range_lane_dpoints.size() << std::endl;
					v_auto_range_lane_points.push_back(auto_range_lane_dpoints);
					// fwc
					// if (auto_range_lane_dpoints.size()>=config.min_lane_pts){
					// 	v_auto_range_lane_points.push_back(auto_range_lane_dpoints);
					// 	//v_user_range_lane_points.push_back(cvpoints_to_dpoints(full_range_lane_cvpoints));
					// }
					lane_id++;
				}

#ifdef SAVE_POLYFIT_FUNC
				csv_file << "\n";
				csv_file.close();
#endif

#ifdef DEBUG_INFO
					printf("[polyfit]  v_auto_range_lane_points.size() = %d \n", 
						(int)v_auto_range_lane_points.size()
					); 
					printf("[polyfit]  v_user_range_lane_points.size() = %d \n", 
						(int)v_user_range_lane_points.size()
					); 
#endif

			}
#pragma endregion


#pragma region polyline intersect
			bool lanesegutil::polyline_intersect(
				const LaneInvasionConfig& config,
				const cvpoints_t& lane_cvpoints,
				const cvpoints_t& train_points
			)
			{
				// todo 
				// train_points may be empty 
				if (train_points.size()<2){
					return false;
				}

				return false;
			}
#pragma endregion


#pragma region get left/right lane 

			void lanesegutil::get_left_right_lane(
				CAMERA_TYPE camera_type, // long, short
				const LaneInvasionConfig& config,
				const std::vector<dpoints_t>& v_auto_range_lane_points,
				const std::vector<dpoints_t>& v_user_range_lane_points,
				std::vector<dpoints_t>& v_merged_lane_points, // merged lane points
				std::vector<LaneKeypoint>& v_lane_keypoint, // keypoints for lane
				int& lane_count,
				int& id_left, 
				int& id_right,
				double& x_left,
				double& x_right,
				dpoints_t& coord_left,
				dpoints_t& coord_right
				)
			{
				v_lane_keypoint.clear();
				//  get lane_trans, main left/right lane id
				id_left = -1;
				id_right = -1;
				x_left = -1000;
				x_right = 1000; 
				lane_count = v_auto_range_lane_points.size();

				// (1) find left/right lane id by auto range 
// 				for(int lane_id=0; lane_id< v_auto_range_lane_points.size(); lane_id++)
// 				{
// 					const dpoints_t& one_lane_points = v_auto_range_lane_points[lane_id]; // (1080,1920)
// 					int n = one_lane_points.size();
// 					// for lane point: image coord 1080,1920  ===> distance coord  x=[-5,5] y=[15,60]
// 					dpoints_t lane_trans = image_points_to_distance_points(camera_type, config, one_lane_points); // 3*n
// 					// X= lane_trans[0,:], Y= lane_trans[2,:]
// 					// top keypoint
// 					double top_lane_x = lane_trans[0][0]; 
// 					double top_lane_z1 = lane_trans[1][0];
// 					double top_lane_y = lane_trans[2][0];

// 					dpoint_t top_point_trans{top_lane_x, top_lane_z1, top_lane_y};
// 					cvpoint_t top_lane_cvpoint = dist_point_to_image_point(camera_type, config, top_point_trans);

// 					// bottom keypoint (find left/right lane based on lane x)
// 					double lane_x = lane_trans[0][n-1];
// 					double lane_z1 = lane_trans[1][n-1];
// 					double lane_y = lane_trans[2][n-1];
					
// 					dpoint_t bottom_point_trans{lane_x, lane_z1, lane_y};
// 					cvpoint_t bottom_lane_cvpoint = dist_point_to_image_point(camera_type, config, bottom_point_trans);
// 					bound_cvpoint(bottom_lane_cvpoint, ORIGIN_SIZE, 15); // for better display lane point

// 					// lane keypoints
// 					LaneKeypoint lane_keypoint;
// 					lane_keypoint.top = top_lane_cvpoint;
// 					lane_keypoint.bottom = bottom_lane_cvpoint;
// 					v_lane_keypoint.push_back(lane_keypoint);

// #ifdef DEBUG_INFO
// 					printf("land_id = %d, N= %d, lane_x = %f, lane_point = (%d,%d) \n", 
// 						lane_id, n, lane_x, bottom_lane_cvpoint.x,bottom_lane_cvpoint.y
// 					); 
// #endif

// 					if (lane_count == 2){ // only 2 lane
// 						if (lane_id == 0){ // lane 0
// 							id_left = lane_id;
// 							x_left = lane_x;
// 							coord_left = lane_trans;
// 						} else { // lane 1
// 							id_right = lane_id;
// 							x_right = lane_x;
// 							coord_right = lane_trans;
// 						}

// 						if (lane_id == 1){
// 							//printf("[HIT] lane 2 \n"); 
// 							if (x_left > x_right) { // swap left and right
// 								std::swap(id_left,id_right);
// 								std::swap(x_left,x_right);
// 								std::swap(coord_left,coord_right);
// 							}
// 						}
// 					} else {
// 						// other case
// 						if (lane_x >=0 && lane_x < x_right){
// 							id_right = lane_id;
// 							x_right = lane_x;
// 							coord_right = lane_trans;
// 						}	
// 						if (lane_x < 0 && lane_x > x_left){
// 							id_left = lane_id;
// 							x_left = lane_x;
// 							coord_left = lane_trans;
// 						}
// 					}					
// 				}

				// (2) based on left/right lane, find yrange: y_min,y_max  and re-polyfit left/right lane
// 				int y_min = UPPER_SIZE.height;// NEED TO UPDATED BY LANE y_min  568; ===> 440
// 				int y_max = ORIGIN_SIZE.height; // image height 1080

// 				cvpoints_t left_user_range_lane_cvpoints;
// 				cvpoints_t right_user_range_lane_cvpoints;

// 				if (id_left>=0 && id_right>=0){
// 					int left_y_min = v_lane_keypoint[id_left].top.y;
// 					int right_y_min = v_lane_keypoint[id_right].top.y;
// 					y_min = std::min(left_y_min, right_y_min); // update y_min by min value of left/right y_min

// #ifdef DEBUG_INFO
// 					printf("[re-polyfit] left_y_min = %d, right_y_min= %d, y_min = %d\n", 
// 						left_y_min, right_y_min, y_min
// 					); 
// #endif

// 					// (3) re-polyfit by yrange
// 					// (3.1) left
// 					cvpoints_t left_cvpoints = dpoints_to_cvpoints(v_auto_range_lane_points[id_left]);
// 					Polyfiter fiter(left_cvpoints, 
// 						config.polyfit_order, 
// 						config.reverse_xy,
// 						config.x_range_min,
// 						config.x_range_max,
// 						y_min,
// 						y_max
// 					); 
				
// 					// change mat
// 					cv::Mat mat_k = fiter.cal_mat_add_points(v_auto_range_lane_points[id_left]);
// 					fiter.set_mat_k(mat_k);
// 					// left_user_range_lane_cvpoints = fiter.fit(false);
// 					dpoints_t left_user_range_lane_dpoints = fiter.fit_dpoint(v_auto_range_lane_points[id_left], false);

// 					// (3.2) right
// 					cvpoints_t right_cvpoints = dpoints_to_cvpoints(v_auto_range_lane_points[id_right]);
// 					Polyfiter fiter2(right_cvpoints, 
// 						config.polyfit_order, 
// 						config.reverse_xy,
// 						config.x_range_min,
// 						config.x_range_max,
// 						y_min,
// 						y_max
// 					); 
// 					// change mat
// 					cv::Mat mat_k_2 = fiter2.cal_mat_add_points(v_auto_range_lane_points[id_right]);
// 					fiter2.set_mat_k(mat_k_2);
// 					// right_user_range_lane_cvpoints = fiter2.fit(false);
// 					dpoints_t right_user_range_lane_dpoints = fiter2.fit_dpoint(v_auto_range_lane_points[id_right], false);
					
// 				}
				
				// fwc
				// std::cout << "id_left:" << id_left << "," << "id_right:" << id_right << std::endl;
				// std::cout << v_auto_range_lane_points.size() << std::endl;
				
				dpoints_t left_user_range_lane_dpoints;
				dpoints_t right_user_range_lane_dpoints;

				id_left = -1;
				id_right = -1;
				// printf("%s %d  v_auto_range_lane_points %d \n",__FUNCTION__,__LINE__, v_auto_range_lane_points.size());
				// printf("%s %d  0= %d   1=%d \n",__FUNCTION__,__LINE__, v_auto_range_lane_points[0].size(), v_auto_range_lane_points[1].size());
				if (v_auto_range_lane_points[0].size() >= 2){
					id_left = 0;
					left_user_range_lane_dpoints = v_auto_range_lane_points[id_left];
				}
				if (v_auto_range_lane_points[1].size() >= 2){
					id_right = 1;
					right_user_range_lane_dpoints = v_auto_range_lane_points[id_right];
				}

				// (4) get merged lane points 
				for(int lane_id=0; lane_id< v_auto_range_lane_points.size(); lane_id++)
				{
					if (lane_id == id_left){
						// v_merged_lane_points.push_back(cvpoints_to_dpoints(left_user_range_lane_cvpoints));
						v_merged_lane_points.push_back(left_user_range_lane_dpoints);
					} else if(lane_id == id_right){
						// v_merged_lane_points.push_back(cvpoints_to_dpoints(right_user_range_lane_cvpoints));
						v_merged_lane_points.push_back(right_user_range_lane_dpoints);
					} else {
						v_merged_lane_points.push_back(v_auto_range_lane_points[lane_id]);
					}
				}
printf("%s %d \n",__FUNCTION__,__LINE__);
				// (5) force left/right lane coords by merge range
				coord_left.resize(3);
				coord_right.resize(3);
				if (id_left>=0){
					coord_left = image_points_to_distance_points(camera_type, config, v_merged_lane_points[id_left]); 
				}
				if (id_right>=0){
					coord_right = image_points_to_distance_points(camera_type, config, v_merged_lane_points[id_right]); 
				}

				if(coord_left[0].size()<=0){
					id_left = -1;
				}

				if(coord_right[0].size()<=0){
					id_right = -1;
				}

printf("%s %d \n",__FUNCTION__,__LINE__);
#ifdef DEBUG_INFO
				printf("lane_count = %d \n",lane_count);
				printf("id_left = %d \n",id_left);
				printf("id_right = %d \n",id_right);
				printf("x_left = %f \n",x_left);
				printf("x_right = %f \n",x_right);
#endif

			}
#pragma endregion


#pragma region detection boxs invasion detect
	

#pragma region case1 box
			bool lanesegutil::is_case1_box(
				const detection_box_t& box,
				double case1_x_threshold, 
				double case1_y_threshold
			)
			{
				if (box.valid_dist){
					if (std::abs(box.dist_x) <= case1_x_threshold && std::abs(box.dist_y) <= case1_y_threshold){
						printf("[CASE1] HIT \n");
						return true; 
					} else {
						return false;
					}
				} 
				return false;
			}

			box_invasion_result_t lanesegutil::do_case1_box_invasion_detect(const detection_box_t& box)
			{
				box_invasion_result_t invasion_result; // {YES_INVASION, 1111};
				invasion_result.invasion_status = YES_INVASION;
				invasion_result.invasion_distance = 1111;
				return invasion_result;
			}
#pragma endregion

/*
#pragma region train box
			bool lanesegutil::is_train_box(
				const detection_box_t& box
			)
			{
				return box.class_name == "train"; //
			}


			box_invasion_result_t lanesegutil::do_train_box_invasion_detect(
				CAMERA_TYPE camera_type, // long, short
				const LaneInvasionConfig& config,
				const cvpoints_t& left_expand_lane_cvpoints,
				const cvpoints_t& right_expand_lane_cvpoints,
				const cvpoints_t& train_points
			)
			{
				box_invasion_result_t invasion_result; 

				// polyfit 
				Polyfiter fiter(train_points, 
						config.polyfit_order, 
						config.reverse_xy,
						config.x_range_min,
						config.x_range_max,
						config.y_range_min,
						config.y_range_max
				); 
				cvpoints_t fitted_train_points = fiter.fit(true); 

				// check whether intersect with left/right lane
				bool left_flag = polyline_intersect(config, left_expand_lane_cvpoints, fitted_train_points);
				bool right_flag = polyline_intersect(config, right_expand_lane_cvpoints, fitted_train_points);
				
				if (left_flag || right_flag) {
					invasion_result.invasion_status = YES_INVASION;
					invasion_result.invasion_distance = 2222;
				} else{
					invasion_result.invasion_status = NO_INVASION;
					invasion_result.invasion_distance = 0;
				}
				
				return invasion_result;
			}
#pragma endregion
 */

#pragma region other/common box invasion detect

			box_invasion_result_t lanesegutil::__do_box_invasion_detect(
				const LaneInvasionConfig& config,
				const dpoints_t& coord_expand_left,
				const dpoints_t& coord_expand_right,
				double x1, double y1, 
				double x2, double y2
			)
			{
				// # return invasion_result
    			//# left point(x,y1)  right point(x2,y2)  y1==y2  (in meters)

				box_invasion_result_t box_invasion_result;
				//assert(y1==y2);
				double y = y1;

#ifdef DEBUG_INFO
				printf("============================= \n");
				printf("[box] left x= %f, right x = %f\n",x1,x2);
				printf("y = %f \n",y); 
#endif 		

				//printf("coord_left size = %d, coord_right size = %d\n",coord_expand_left.size(),coord_expand_right.size());

				// find all x coords where Y==y for left
				std::vector<double> x_left_coords = get_lane_x_coords(coord_expand_left, y);
				//  find all x coords where Y==y for right lane 
				std::vector<double> x_right_coords = get_lane_x_coords(coord_expand_right, y);

				if (x_left_coords.size()>0 && x_right_coords.size()>0){
					double x_min = MINV(x_left_coords);
					double x_max = MAXV(x_right_coords);

					//  # center of left/right lane 
					double center_x = (x_min+x_max)/2.0;

#ifdef DEBUG_INFO
					//printf("[lane] left xmin= %f, right xmax = %f, center_x = %f \n",x_min, x_max, center_x);
#endif 		

					// init invasion result
					box_invasion_result.invasion_status = NO_INVASION;
					box_invasion_result.invasion_distance = 0;

					double dist_y1 = 0;
					if (x1 > x_min && x1 < x_max) {
						//printf("111 \n");
						box_invasion_result.invasion_status = YES_INVASION;
						dist_y1 = std::abs(x1-center_x); // #the distance bettwen x1 and center at y1 position
					} else {
						//printf("222 \n");
						dist_y1 = std::min( std::abs(x1-x_min), std::abs(x1-x_max));
					}
					double dist_y2 = 0;
					if (x2 > x_min && x2 < x_max) {
						//printf("333 \n");
						box_invasion_result.invasion_status = YES_INVASION;
						dist_y2 = std::abs(x2-center_x); // #the distance bettwen x1 and center at y1 position
					} else {
						//printf("444 \n");
						dist_y2 = std::min( std::abs(x2-x_min), std::abs(x2-x_max));
					}

					// # minus limit 
					dist_y1 = config.railway_limit_width - dist_y1;
					dist_y2 = config.railway_limit_width - dist_y2;
					box_invasion_result.invasion_distance = min(dist_y1,dist_y2);

				} else { // error case
					printf("[WARNING-3] invaison detect failed(x_left_coords,x_right_coords may be empty) \n");

					// init invasion result
					box_invasion_result.invasion_status = UNKNOW;
					box_invasion_result.invasion_distance = 0;
				}
			#ifdef DEBUG_INFO				
				printf("[box] left x= %f, right x = %f\n",x1,x2);
			#endif 
				return box_invasion_result;
			}


			box_invasion_result_t lanesegutil::do_box_invasion_detect(
				CAMERA_TYPE camera_type, // long, short
				const LaneInvasionConfig& config,
				const dpoints_t& coord_expand_left,
				const dpoints_t& coord_expand_right,
				const detection_box_t& detection_box
			)
			{
			
				// (1) get box_trans 
				//========================================================
				// for box point in origin_image(1080,1920)
				//========================================================
				dpoints_t box_points; // n*2     # origin_size  1080,1920
				
				// # 1 box===>2 lower point (left + right)
				dpoint_t left_point; // # left point (xmin,ymax)
				left_point.push_back(detection_box.xmin);
				left_point.push_back(detection_box.ymax);

				dpoint_t right_point; // # right point (xmax,ymax)
				right_point.push_back(detection_box.xmax);
				right_point.push_back(detection_box.ymax);

				box_points.push_back(left_point);
				box_points.push_back(right_point);
				

				// for box point: image coord 1080,1920  ===> distance coord  x=[-5,5] y=[15,60]
				dpoints_t box_trans = image_points_to_distance_points(camera_type, config, box_points);
				// X= box_trans[0,:], Y= box_trans[2,:]
				
				box_invasion_result_t box_invasion_result;
				box_invasion_result.invasion_status = UNKNOW;
				box_invasion_result.invasion_distance = 0;	

				// classname == train
				if (detection_box.class_index == 3){
					if (box_trans[0].size() == 2){
						// (2) do real invasion detection with distance coords
						// left point
						int i=0;
						double y1 = box_trans[2][i*2];
						double x1 = box_trans[0][i*2];

						// right point
						double y2 = box_trans[2][i*2+1];
						double x2 = box_trans[0][i*2+1];

						// std::cout << detection_box.class_name << std::endl;
						// std::cout << box_points[0][0] << " " << box_points[0][1] << " " << box_points[1][0] << " " << box_points[1][1] << std::endl;
						// std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
						// std::cout << coord_expand_left.size() << std::endl;
						// std::cout << coord_expand_left[0][0] << " " << coord_expand_left[0][coord_expand_left.size()-1] << std::endl;
						// std::cout << coord_expand_left[2][0] << " " << coord_expand_left[2][coord_expand_left.size()-1] << std::endl;
						// std::cout << coord_expand_right[0][0] << " " << coord_expand_right[0][coord_expand_right.size()-1] << std::endl;
						// std::cout << coord_expand_right[2][0] << " " << coord_expand_right[2][coord_expand_right.size()-1] << std::endl;

						// train_box 的正中心在三维世界坐标系的坐标为 x，y，z；x要在两轨x范围之间，并且距离小于220m时则为侵界
						if (x1 > coord_expand_right[0][coord_expand_right[0].size()-1] || x2 < coord_expand_left[0][coord_expand_left[0].size()-1]){
							box_invasion_result.invasion_status = INVASION_STATUS::NO_INVASION;
						}
						else{
							double dist_train_box = (y1+y2) / 2.0;
							if (dist_train_box < 220){
								box_invasion_result.invasion_status = INVASION_STATUS::YES_INVASION;
								box_invasion_result.invasion_distance = dist_train_box;
							}
							else{
								box_invasion_result.invasion_status = INVASION_STATUS::NO_INVASION;
							}
						}
					}
				}
				// classname == traffic light
				else if (detection_box.class_index == 2){
					box_invasion_result.invasion_status = INVASION_STATUS::NO_INVASION;
				}
				else{
					if (box_trans[0].size() == 2){
						// (2) do real invasion detection with distance coords
						// left point
						int i=0;
						double y1 = box_trans[2][i*2];
						double x1 = box_trans[0][i*2];

						// right point
						double y2 = box_trans[2][i*2+1];
						double x2 = box_trans[0][i*2+1];

						box_invasion_result = __do_box_invasion_detect(
							config, coord_expand_left, coord_expand_right, x1, y1, x2, y2
						);
					}
				}
				

				return box_invasion_result;
			}
#pragma endregion

			box_invasion_results_t lanesegutil::box_image_points_invasion_detect(
				CAMERA_TYPE camera_type, // long, short
				const LaneInvasionConfig& config,
				const dpoints_t& coord_expand_left,
				const dpoints_t& coord_expand_right,
				const cvpoints_t& left_expand_lane_cvpoints,
				const cvpoints_t& right_expand_lane_cvpoints,
				const detection_boxs_t& detection_boxs,
				const std::vector<cvpoints_t>& trains_cvpoints
			)
			{
				box_invasion_results_t box_invasion_results; 

				for(int i=0; i< detection_boxs.size(); i++){
					const detection_box_t& detection_box = detection_boxs[i];

					box_invasion_result_t box_invasion_result; 
					if (is_case1_box(detection_box, config.case1_x_threshold, config.case1_y_threshold)){
						box_invasion_result = do_case1_box_invasion_detect(
							detection_box
						);
					} /*else if (is_train_box(detection_box)){
						box_invasion_result = do_train_box_invasion_detect(
							camera_type, config, left_expand_lane_cvpoints, right_expand_lane_cvpoints, trains_cvpoints[i]
						);
					} */
					else { // other cases
						box_invasion_result = do_box_invasion_detect(
							camera_type, config, coord_expand_left, coord_expand_right, detection_box
						);
					}
					box_invasion_results.push_back(box_invasion_result);
				}
				return box_invasion_results;
			}

#pragma endregion


#pragma region lidar invasion detect			
			INVASION_STATUS lanesegutil::do_point_invasion_detect(
				const LaneInvasionConfig& config,
				const dpoints_t& coord_expand_left,
				const dpoints_t& coord_expand_right,
				double x, double y 
			)
			{
				/*
				 # return invasion_status (UNKNOW_STATUS = -1, NO_INVASION = 0, YES_INVASION = 1)
    			 # x,y in distance coord (meters)
				*/
				INVASION_STATUS invasion_status = UNKNOW;

				// find all x coords where Y==y for left
				std::vector<double> x_left_coords = get_lane_x_coords(coord_expand_left, y);
				//  find all x coords where Y==y for right lane 
				std::vector<double> x_right_coords = get_lane_x_coords(coord_expand_right, y);

				if (x_left_coords.size()>0 && x_right_coords.size()>0){
					double x_min = MINV(x_left_coords);
					double x_max = MAXV(x_right_coords);

#ifdef DEBUG_INFO
					//printf("[lidar] (x,y) = %f,%f, (xmin,xmax) = %f,%f \n",x, y, x_min, x_max);
#endif 		
					if (x >= x_min && x <= x_max){
						invasion_status = YES_INVASION;
					} else {
						invasion_status = NO_INVASION;
					}
				}
				return invasion_status;
			}

			std::vector<int> lanesegutil::lidar_image_points_invasion_detect(
				CAMERA_TYPE camera_type, // long, short
				const LaneInvasionConfig& config,
				const dpoints_t& coord_expand_left,
				const dpoints_t& coord_expand_right,
				const cvpoints_t& cvpoints
			)
			{
				/*
				# points n*2 in image coord  1080*1920 
    			# get points in distance coord  3*n (X,y1,Y)
				*/

				int n = cvpoints.size(); // n*2
				std::vector<int> v_invasion_status;
				v_invasion_status.resize(n, UNKNOW);
				
				dpoints_t points = cvpoints_to_dpoints(cvpoints);
				// dpoints_t points_trans = image_points_to_distance_points(camera_type, config, points);  // 3*n
				
				// std::vector<int> yes_invasion_index_list;
				// std::vector<int> no_invasion_index_list;
				// std::vector<int> unknow_invasion_index_list;

				//omp_set_num_threads(8);
				// #pragma omp parallel for
				for(int i=0; i<n; i++){
					dpoints_t pts{points[i]};
					// std::cout << points[i][0] << " " << points[i][1] << std::endl;
					dpoints_t points_trans = image_points_to_distance_points(camera_type, config, pts); 
		
					if (points_trans[0].size() > 0){
						double x = points_trans[0][0];
						double y = points_trans[2][0];

						v_invasion_status[i] = do_point_invasion_detect(
							config, coord_expand_left, coord_expand_right, x, y
						);
					}
					else{
						v_invasion_status[i] = UNKNOW;
					}
					
				}

				return v_invasion_status;
			}
#pragma endregion

#pragma region lidar pointcloud small objects invasion detect
			void lanesegutil::lidar_pointcloud_smallobj_invasion_detect(
				// pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
				std::vector<cv::Point3f> cloud,
				InvasionData  invasion,
				cvpoints_t& input_l,
				cvpoints_t& input_r,
				std::vector<LidarBox>& obstacle_box,
				std::vector<lidar_invasion_cvbox>& cv_obstacle_box
			){
				// std::vector<LidarBox> obstacle_box;
				// std::vector<lidar_invasion_cvbox>& cv_obstacle_box;

				pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>);
				Eigen::Matrix4f rotation;
				int top_y = 0;
				// Mat t_mat(1, 1, CV_8UC1, Scalar::all(0));
				std::string image_file = "";
				// std::vector<cv::Point2f> input_l_f, input_r_f;
				// for (int idx=0; idx<input_l.size(); idx++){
				// 	cv::Point2f t_p;
				// 	t_p.x = input_l[idx].x;
				// 	t_p.y = input_l[idx].y;
				// 	input_l_f.push_back(t_p);
				// }
				// for (int idx=0; idx<input_r.size(); idx++){
				// 	cv::Point2f t_p;
				// 	t_p.x = input_r[idx].x;
				// 	t_p.y = input_r[idx].y;
				// 	input_r_f.push_back(t_p);
				// }

				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);
				for(int index=0;index<cloud.size();++index){

					pcl::PointXYZ pt;
					pt.x = cloud[index].x;
					pt.y = cloud[index].y;
					pt.z = cloud[index].z;
					cloud_pcl->points.push_back(pt);
				}

	
				points = LOD::getPointFrom2DAnd3D(cloud_pcl, LOD::getInvasionMap(input_l, input_r, top_y), top_y, invasion, rotation);
#ifdef SHOW_PCD
				pcl::io::savePCDFileBinary("output.pcd", *points);
				pcl::visualization::CloudViewer viewer("Show");
				viewer.showCloud(points);
				// system("pause");
				while (!viewer.wasStopped()){ };
#endif
				obstacle_box = LOD::object_detection(points, rotation, invasion, image_file);
				cv_obstacle_box = LOD::lidarboxTocvbox(obstacle_box);
			}
#pragma endregion

#pragma region get lane status
			int lanesegutil::get_lane_status(
				InvasionData  invasion,
				std::vector<dpoints_t>& v_src_dist_lane_points,
                dpoints_t& curved_point_list, // out
				std::vector<double>& curved_r_list // out
			){
				dpoints_t v_param_list;
				v_param_list.push_back(invasion.coeff_left);
				v_param_list.push_back(invasion.coeff_right);
				
				int status_type = LaneStatus::GetLaneStatus(v_param_list, v_src_dist_lane_points, curved_point_list, curved_r_list);
				return status_type;
			}
#pragma endregion


#pragma region draw lane with mask
			cvpoint_t lanesegutil::get_nearest_invasion_box_point(
				const detection_boxs_t& detection_boxs,
				const box_invasion_results_t& box_invasion_results
				)
			{
					cvpoint_t nearest_point;
					for(int i=0;i<detection_boxs.size();i++){
						// train 不计算nearest point
						if (detection_boxs[i].class_index != 3){
							if (box_invasion_results[i].invasion_status == INVASION_STATUS::YES_INVASION){
								const detection_box_t& box = detection_boxs[i];
								int cx = (box.xmin + box.xmax)/2;
								int cy = box.ymax;
								if (nearest_point.y<cy){ // find max y
									nearest_point.y = cy; 
									nearest_point.x = cx; 
								}
							}
						}
					}
					return nearest_point;
			}

#pragma region clip lane by nearest point
			cvpoints_t lanesegutil::clip_lane_by_nearest_box_point(
				const dpoints_t& lane,
				int near_x, int near_y
			){
					cvpoints_t clipped_lane;
					for(auto& point: lane){
						if (point[1]>=near_y){ // keep large y
							clipped_lane.push_back(cvpoint_t(point[0],point[1]));
						}
					}
					return clipped_lane;
			}

			cvpoints_t lanesegutil::clip_lane_by_nearest_box_point(
				const cvpoints_t& lane,
				int near_x, int near_y
			){
					cvpoints_t clipped_lane;
					for(auto& point: lane){
						if (point.y>=near_y){ // keep large y
							clipped_lane.push_back(point);
						}
					}
					return clipped_lane;
			}
#pragma endregion

#pragma region draw safe area
			
			cv::Mat lanesegutil::draw_lane_safe_area(
				const LaneInvasionConfig& config,
				const cv::Mat& image_,
				const cvpoint_t& nearest_point,
				const cvpoints_t& left_lane_cvpoints,
				const cvpoints_t& right_lane_cvpoints,
				int y_upper, int y_lower,
				lane_safe_area_corner_t& lane_safe_area_corner
			){
					cv::Mat image = image_.clone(); // key step

					cv::Scalar yellow(0, 255, 255);
					cv::Scalar red(0, 0, 255);
					int valid_y_count = 0;

					lane_safe_area_corner.valid = false;

					if (left_lane_cvpoints.size()>0 && right_lane_cvpoints.size()>0){
						lane_safe_area_corner.valid = true;
						
						lane_safe_area_corner.left_upper = left_lane_cvpoints[left_lane_cvpoints.size()-1];
						lane_safe_area_corner.right_upper = right_lane_cvpoints[right_lane_cvpoints.size()-1];

						lane_safe_area_corner.left_lower = left_lane_cvpoints[0];
						lane_safe_area_corner.right_lower = right_lane_cvpoints[0];
					}

					// for(int y=y_upper; y<=y_lower;y+= config.safe_area_y_step)
					// {
					// 	std::vector<int> x_left_points = get_lane_x_image_points(left_lane_cvpoints, y);
					// 	std::vector<int> x_right_points = get_lane_x_image_points(right_lane_cvpoints, y);

					// 	if (x_left_points.size()>0 && x_right_points.size()>0)
					// 	{
					// 		int left_x  = MAXV(x_left_points);
					// 		int right_x = MINV(x_right_points);

					// 		cv::Point2i left_point(left_x, y);
					// 		cv::Point2i right_point(right_x, y);

					// 		// cv::line(image, left_point, right_point, green, config.safe_area_y_step);

					// 		if (valid_y_count==0){ // upper
					// 			lane_safe_area_corner.valid = true;
					// 			lane_safe_area_corner.left_upper = left_point;
					// 			lane_safe_area_corner.right_upper = right_point;
					// 		} else {// lower
					// 			lane_safe_area_corner.left_lower = left_point;
					// 			lane_safe_area_corner.right_lower = right_point;
					// 		}

					// 		valid_y_count++;
					// 	}
					// }

					if (config.b_draw_safe_area_corner){
						// nearest invasion box point
						if (nearest_point.x >0 && nearest_point.y>0){
							DisplayUtil::draw_circle_point(image, nearest_point, red, 10);
						}
						
						int radius = 10;
						//lane_safe_area_corner.left_upper = cvpoint_t(800,568); // test
						DisplayUtil::draw_circle_point(image, lane_safe_area_corner.left_upper, red, radius);
						DisplayUtil::draw_circle_point(image, lane_safe_area_corner.right_upper, red, radius);
						DisplayUtil::draw_circle_point(image, lane_safe_area_corner.left_lower, yellow, radius);
						DisplayUtil::draw_circle_point(image, lane_safe_area_corner.right_lower, yellow, radius);
					}

					return image;
			}
#pragma endregion


#pragma region draw safe area new
			
			cv::Mat lanesegutil::draw_lane_safe_area(
				const LaneInvasionConfig& config,
				const cv::Mat& image_,
				const cvpoint_t& nearest_point,
				const cvpoints_t& left_lane_cvpoints,
				const cvpoints_t& right_lane_cvpoints
			){
					cv::Mat image = image_.clone(); // key step
					cv::Scalar green(0, 255, 0);
					int points_size = left_lane_cvpoints.size() + right_lane_cvpoints.size();
					if(points_size>3){
						cv::Point pt[1][points_size];
						int idx=0;
						for (int i=0; i < left_lane_cvpoints.size(); i++){
							pt[0][idx] = left_lane_cvpoints[i];
							idx++;
						}
						for (int i = right_lane_cvpoints.size()-1; i >= 0; i--){
							pt[0][idx] = right_lane_cvpoints[i];
							idx++;
						}
						const cv::Point* ppt[1] = {pt[0]};
						int npt[] = {points_size};
						cv::fillPoly(image, ppt, npt, 1, green, 0);
					}
					return image;
			}
#pragma endregion

#pragma region draw final lane 
			
			void lanesegutil::x_draw_lane_with_mask(
				int lane_model_type, // caffe, pt_simple, pt_complex
				const LaneInvasionConfig& config,
				const cv::Mat& origin_image,
				const cv::Mat& binary_mask, 
				const channel_mat_t& instance_mask,
				const std::vector<dpoints_t>& v_src_lane_points,
				const std::vector<dpoints_t>& v_fitted_lane_points,
				const std::vector<LaneKeypoint>& v_lane_keypoint,
				int id_left, 
				int id_right,
				const cvpoint_t& nearest_point,
				const cvpoints_t& left_expand_lane_cvpoints,
				const cvpoints_t& right_expand_lane_cvpoints,
				const detection_boxs_t& detection_boxs,
				const box_invasion_results_t& box_invasion_results, 
				const std::vector<cvpoints_t>& trains_cvpoints,
				cv::Mat& out,
				lane_safe_area_corner_t& lane_safe_area_corner
				)
			{
				int near_x = nearest_point.x ;
				int near_y = nearest_point.y;
				// bgr
				cv::Scalar blue(255, 0, 0), blue2(128, 0, 0);
				cv::Scalar red(0, 0, 255), red2(0, 0, 128);
				cv::Scalar yellow(0, 255, 255);
				cv::Scalar fitted_color(255, 255, 255);
				cv::Scalar surface_color(0, 0, 128);
				bool closed = false;
				int thick = 2;
				cv::Scalar color_map[] = {
					cv::Scalar(125, 125, 0),
					cv::Scalar(125, 0, 125),
					cv::Scalar(0, 125, 125),
					cv::Scalar(50, 50, 100),
					cv::Scalar(50, 100, 50),
					cv::Scalar(100, 50, 50),
					cv::Scalar(30, 125, 80),
					cv::Scalar(80, 125, 30)
				};
				int color_count = 8;

				out = origin_image.clone();

				// (0) draw surface 
				if (config.b_draw_lane_surface){
					if ( lane_model_type == LANE_MODEL_TYPE::LANE_MODEL_PT_SIMPLE) // only for pt_simple
					{
						cv::Mat full_binary_mask = get_lane_full_binary_mask(binary_mask); // 1080,1920, 1
						//cv::imwrite("full_binary_mask.jpg",full_binary_mask);
						cv::Mat image_with_surface = OpencvUtil::merge_mask(origin_image, full_binary_mask, surface_color); // 1080,1920 + 1080,1920
						NumpyUtil::cv_add_weighted(image_with_surface, out, config.safe_area_alpha);
					}
				}
				

				// (1) draw detection boxs
				if (config.b_draw_boxs){
					cv::Mat image = out;
					DisplayUtil::draw_detection_boxs(image, detection_boxs, box_invasion_results, 5, out);
				}


				// (2.1) left/right lane
				if (config.b_draw_left_right_lane)
				{
					if (id_left>=0 && id_right>=0){
						cvpoints_t clipped_left_lane_cvpoints = clip_lane_by_nearest_box_point(
							v_src_lane_points[id_left], near_x, near_y
						);
						cvpoints_t clipped_right_lane_cvpoints = clip_lane_by_nearest_box_point(
							v_src_lane_points[id_right], near_x, near_y
						);
						DisplayUtil::draw_lane(out, clipped_left_lane_cvpoints, blue2);
						DisplayUtil::draw_lane(out, clipped_right_lane_cvpoints, red2);
					}
				}
				// (2.2) other lane
				if (config.b_draw_other_lane){
					for(int lane_id=0; lane_id<v_src_lane_points.size();lane_id++){
						if (lane_id != id_left && lane_id != id_right){
							int color_index = lane_id % color_count;
							cv::Scalar color = color_map[color_index];

							DisplayUtil::draw_lane(out, v_src_lane_points[lane_id], blue);
						}
					}
				}
				
				// (3.1) draw left/right fitted lane
				if (config.b_draw_left_right_fitted_lane){
					if (id_left>=0 && id_right>=0){
						cvpoints_t clipped_left_fitted_lane_cvpoints = clip_lane_by_nearest_box_point(
							v_fitted_lane_points[id_left], near_x, near_y
						);
						cvpoints_t clipped_right_fitted_lane_cvpoints = clip_lane_by_nearest_box_point(
							v_fitted_lane_points[id_right], near_x, near_y
						);
						DisplayUtil::draw_lane_line(out, clipped_left_fitted_lane_cvpoints, fitted_color);
						DisplayUtil::draw_lane_line(out, clipped_right_fitted_lane_cvpoints, fitted_color);
					}
					// if (id_left>=0){
					// 	// DisplayUtil::draw_lane(out, v_fitted_lane_points[id_left], fitted_color, 3);
					// 	DisplayUtil::draw_lane_line(out, v_fitted_lane_points[id_left], fitted_color);
					// }
					// if (id_right>=0){
					// 	// DisplayUtil::draw_lane(out, v_fitted_lane_points[id_right], fitted_color, 3);
					// 	DisplayUtil::draw_lane_line(out, v_fitted_lane_points[id_right], fitted_color);
					// }
				}
				// (3.2) draw other fitted lane
				if (config.b_draw_other_fitted_lane){
					for(int lane_id=0; lane_id<v_fitted_lane_points.size();lane_id++){
						if (lane_id != id_left && lane_id != id_right){
							DisplayUtil::draw_lane(out, v_fitted_lane_points[lane_id], fitted_color, 3);
						}
					}
				}

				// (4) draw expand left and right lanes
				if (config.b_draw_expand_left_right_lane && id_left>=0 && id_right>=0){
					// DisplayUtil::draw_lane(out, left_expand_lane_cvpoints, blue, 3);
					// DisplayUtil::draw_lane(out, right_expand_lane_cvpoints, red, 3);
					cvpoints_t clipped_left_expand_lane_cvpoints = clip_lane_by_nearest_box_point(
						left_expand_lane_cvpoints, near_x, near_y
					);
					cvpoints_t clipped_right_expand_lane_cvpoints = clip_lane_by_nearest_box_point(
						right_expand_lane_cvpoints, near_x, near_y
					);
					DisplayUtil::draw_lane_line(out, clipped_left_expand_lane_cvpoints, blue);
					DisplayUtil::draw_lane_line(out, clipped_right_expand_lane_cvpoints, red);
				}	
				// (5) draw lane image points
				if (config.b_draw_lane_keypoint){
					for(int lane_id=0; lane_id<v_lane_keypoint.size();lane_id++){
						DisplayUtil::draw_circle_point_with_text(out, v_lane_keypoint[lane_id].top,    yellow, lane_id, 20);
						DisplayUtil::draw_circle_point_with_text(out, v_lane_keypoint[lane_id].bottom, yellow, lane_id, 20);
					}
				}
				
				// (6) draw left-right lane safe area  and corners
				if (config.b_draw_safe_area && id_left>=0 && id_right>=0){
					cvpoints_t clipped_left_lane_cvpoints = clip_lane_by_nearest_box_point(
						left_expand_lane_cvpoints, near_x, near_y
					);
					cvpoints_t clipped_right_lane_cvpoints = clip_lane_by_nearest_box_point(
						right_expand_lane_cvpoints, near_x, near_y
					);
					
					int y_upper = near_y; 
					int y_lower = origin_image.rows; // 1080

					cv::Mat safe_area = draw_lane_safe_area(
						config,
						out, 
						nearest_point,
						clipped_left_lane_cvpoints, 
						clipped_right_lane_cvpoints, 
						y_upper, 
						y_lower,
						lane_safe_area_corner
					);

					cv::Mat safe_area_1 = draw_lane_safe_area(
						config,
						out, 
						nearest_point,
						clipped_left_lane_cvpoints, 
						clipped_right_lane_cvpoints
					);

					NumpyUtil::cv_add_weighted(safe_area, out, 0.8);
					NumpyUtil::cv_add_weighted(safe_area_1, out, config.safe_area_alpha);

				}
				// (7) draw train cvpoints
				if (config.b_draw_train_cvpoints){
				 	for(int i=0; i<trains_cvpoints.size();i++){
				 		DisplayUtil::draw_lane(out, trains_cvpoints[i], fitted_color, 2);
				 	}
				}
				// (8) draw lane stats on image 
				if (config.b_draw_stats){
					int lane_count = v_src_lane_points.size();
					std::stringstream ss;
					ss<<"lane="<<lane_count<<", id_left="<<id_left<<", id_right="<<id_right;
					std::string display_text = ss.str();
					//std::cout<< display_text << std::endl;

					cv::Point2i origin(540,20);
					cv::putText(out, display_text, origin, 
						cv::FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2  // yellow 
					);
				}
			}
			
#pragma endregion


#pragma region do lane invasion detect

			bool lanesegutil::lane_invasion_detect(
				int lane_model_type, // caffe, pt_simple, pt_complex
				CAMERA_TYPE camera_type, // long, short
				const cv::Mat& origin_image, 
				const cv::Mat& binary_mask, 
				const channel_mat_t& instance_mask,
				const detection_boxs_t& detection_boxs,
				const std::vector<cvpoints_t>& trains_cvpoints,
				const cvpoints_t& lidar_cvpoints,
				// const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, // lidar pointscloud
				const std::vector<cv::Point3f> cloud, // lidar pointscloud
				const LaneInvasionConfig& config_,
				cv::Mat& image_with_color_mask,
				int& lane_count,
				int& id_left, 
				int& id_right,
				box_invasion_results_t& box_invasion_results,
				std::vector<int>& lidar_invasion_status,
				lane_safe_area_corner_t& lane_safe_area_corner,
				bool& is_open_long_camera,
				std::vector<lidar_invasion_cvbox>& cv_obstacle_box // lidar invasion object cv box
			)
			{
				//std::cout<<"lanesegutil::lane_invasion_detect for lane_model_type = "<<lane_model_type << std::endl;

				// # binary_mask = (256, 1024) v=[0,1];  instance_mask = (8, 256, 1024) float
				// # binary_mask = (128, 480) v=[0,1];  instance_mask = (2, 256, 1024) v=[0,1];

#ifdef DEBUG_TIME
				static int64_t pre_cost = 0;
				static int64_t forward_cost = 0;
				static int64_t post_cost = 0;
				static int64_t total_cost = 0;

				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::ptime pt2;
				int64_t cost;
#endif // DEBUG_TIME

				LaneInvasionConfig config = config_;
				//if (config.use_tr34){
				//	config.tr33 = get_tr33_from_tr34(config.tr34);
				//}

				// (0) init invasion results with default values
				int box_count = detection_boxs.size();
				box_invasion_results.clear();
				box_invasion_results.resize(box_count);

				
				/*
				if (false){ // test case
					//cvpoint_t point(995,955);
					cvpoint_t point(800,568);
					cvpoints_t cvpoints{point};
					dpoints_t dpoints = cvpoints_to_dpoints(cvpoints);
					dpoints_t tmp_coord_trans = image_points_to_distance_points(config, dpoints);
					cvpoints_t cvpoints2 = distance_points_to_image_points(config, tmp_coord_trans);
					
					print_points("cvpoints",cvpoints);
					//print_points("tmp_coord_trans",tmp_coord_trans);
					print_points("cvpoints2",cvpoints2);
					for(int i=0;i<cvpoints.size();i++){
						if(cvpoints[i] == cvpoints2[i]){
							std::cout<<" equal \n";
						} else {
							std::cout<< cvpoints[i]<<" != "<<cvpoints2[i]<<std::endl;
						}
					}

					return true;
				}
				

				if (true){
					cvpoint_t point(995,1080);
					dpoint_t point_trans = image_point_to_dist_point(config, point);
					cvpoint_t point2 = dist_point_to_image_point(config, point_trans);
					std::cout<<"point = "<<point<<std::endl;
					for(auto& dim: point_trans){
						std::cout<<dim <<",";
					}
					std::cout<<std::endl;
					std::cout<<"point2 = "<<point2<<std::endl;
					return true;
				}

				// point = [995, 955]
				// 0.308023,0.0766408,12.3097,
				// point2 = [995, 955]
				*/
				static int m_counter = 0;
				
				// (1) get clustered lane points (256,1024)
				std::vector<dpoints_t> v_src_lane_points;
				x_get_clustered_lane_points(
					lane_model_type,
					config,
					binary_mask, 
					instance_mask,
					v_src_lane_points
				);

				std::cout << "v_src_lane_points size:" << v_src_lane_points.size() << std::endl;
				// std::cout << "v_src_lane_points[0] size:" << v_src_lane_points[0].size() << std::endl;
				// std::cout << "v_src_lane_points[1] size:" << v_src_lane_points[1].size() << std::endl;

				// (2) transform lane points from binary image to origin image 
				// (256,1024)===>(1080,1920) 
				x_transform_binary_lanes_to_origin_lanes(
					lane_model_type, 
					v_src_lane_points
				);

				std::cout << "v_src_lane_points size:" << v_src_lane_points.size() << std::endl;
				// std::cout << "v_src_lane_points[0] size:" << v_src_lane_points[0].size() << std::endl;
				// std::cout << "v_src_lane_points[1] size:" << v_src_lane_points[1].size() << std::endl;
				// std::cout << "cvpoints -> distance_points end" << std::endl;

#ifdef DEBUG_INFO_FWC
				for(int lane_id=0; lane_id< v_src_lane_points.size(); lane_id++){
					const dpoints_t& one_lane_points = v_src_lane_points[lane_id]; // (1080,1920)
					std::cout << "./"+std::to_string(lane_id)+"_source.csv" << std::endl;
					ofstream file("./"+std::to_string(lane_id)+"_source.csv");
					if (file){
						file << "x" << "," << "y" << "\n";
						for(int idx=0; idx<one_lane_points.size(); idx++){
							// tmp(Point(lane_trans[0][idx], lane_trans[2][idx])) = 255;
							file << one_lane_points[idx][0] << "," << one_lane_points[idx][1] << "\n";
						}
					}
					file.close();
				}

				// #pragma region fwc
				// cv::Mat result(1080, 1920, CV_8UC1, Scalar::all(0));
				// cv::Mat_<uchar> tmp = result;
				for(int lane_id=0; lane_id< v_src_lane_points.size(); lane_id++){
					const dpoints_t& one_lane_points = v_src_lane_points[lane_id]; // (1080,1920)
					// for lane point: image coord 1080,1920  ===> distance coord  x=[-5,5] y=[15,60]
					dpoints_t lane_trans = image_points_to_distance_points(camera_type, config, one_lane_points); // 3*n
					// X= lane_trans[0,:], Y= lane_trans[2,:]
					std::cout << "./"+std::to_string(lane_id)+".csv" << std::endl;
					ofstream file("./"+std::to_string(lane_id)+".csv");
					if (file){
						file << "x" << "," << "y" << "\n";
						for(int idx=0; idx<lane_trans[0].size(); idx++){
							// tmp(Point(lane_trans[0][idx], lane_trans[2][idx])) = 255;
							file << lane_trans[0][idx] << "," << lane_trans[2][idx] << "\n";
						}
					}
					file.close();
				}
#endif

				// // cv::imwrite("./result.jpg", result);
				// #pragma endregion

				// (3) polyfit lanes
				// std::vector<dpoints_t> v_auto_range_lane_points;
				// std::vector<dpoints_t> v_user_range_lane_points;

				// lane_polyfit(
				// 	config, 
				// 	m_counter, 
				// 	origin_image.size(), 
				// 	v_src_lane_points,
				// 	v_auto_range_lane_points,
				// 	v_user_range_lane_points
				// );

				// (3) polyfit lanes
				std::vector<dpoints_t> v_auto_range_dist_lane_points;
				std::vector<dpoints_t> v_user_range_dist_lane_points;
				
				// cvpoints -> distance_points
				std::vector<dpoints_t> v_src_dist_lane_points;
				// std::cout << "v_src_lane_points size:" << v_src_lane_points.size() << std::endl;
				for(int lane_id=0; lane_id < v_src_lane_points.size(); lane_id++){
					const dpoints_t& one_lane_points = v_src_lane_points[lane_id]; 
					dpoints_t lane_trans = image_points_to_distance_points(camera_type, config, one_lane_points); // 3*n
					// std::cout << "lane_trans size:" << lane_trans.size() << " " << lane_trans[0].size() << std::endl;
					// X= lane_trans[0,:], Y= lane_trans[2,:]
					dpoints_t tmp_ps;
					for(int idx=0; idx<lane_trans[0].size(); idx++){
						dpoint_t tmp_p;
						tmp_p.push_back(lane_trans[0][idx]);
						tmp_p.push_back(lane_trans[2][idx]);
						tmp_ps.push_back(tmp_p);
					}
					v_src_dist_lane_points.push_back(tmp_ps);
				}

				std::cout << "v_src_dist_lane_points size:" << v_src_dist_lane_points.size() << std::endl;
				// std::cout << "v_src_dist_lane_points[0] size:" << v_src_dist_lane_points[0].size() << std::endl;

				std::vector<cv::Mat> v_left_right_polyfit_matk; // add left right lane polyfit matk
				// in distance_points polyfit
				lane_polyfit(
					camera_type,
					config, 
					m_counter, 
					origin_image.size(), 
					v_src_dist_lane_points,
					v_auto_range_dist_lane_points,
					v_user_range_dist_lane_points,
					v_left_right_polyfit_matk
				);

				// std::cout << "polyfit end" << std::endl;
				std::cout << "v_auto_range_dist_lane_points size:" << v_auto_range_dist_lane_points.size() << std::endl;
				// std::cout << "point num:" << v_auto_range_dist_lane_points[0].size() << std::endl;


				// v_auto_range_dist_lane_points is distance_points, -> cvpoints
				std::vector<dpoints_t> v_auto_range_lane_points;
				std::vector<dpoints_t> v_user_range_lane_points;

				std::vector<dpoints_t> v_auto_range_dist_lane_points_new;
				for(int lane_id=0; lane_id < v_auto_range_dist_lane_points.size(); lane_id++){
					dpoints_t& one_lane_points = v_auto_range_dist_lane_points[lane_id];
					
					dpoint_t tmp_x;
					dpoint_t tmp_z;
					dpoint_t tmp_y;
					for(int idx=0; idx<one_lane_points.size(); idx++){
						tmp_x.push_back(one_lane_points[idx][0]);
						tmp_z.push_back(0.0);
						tmp_y.push_back(one_lane_points[idx][1]);
					}
					dpoints_t tmp_ps{tmp_x, tmp_z, tmp_y};
					v_auto_range_dist_lane_points_new.push_back(tmp_ps);
				}

				std::vector<cvpoints_t> v_left_right_expand_cvpoints;
				for(int lane_id=0; lane_id < v_auto_range_dist_lane_points_new.size(); lane_id++){
					const dpoints_t& one_lane_points = v_auto_range_dist_lane_points_new[lane_id]; 
					cvpoints_t lane_trans = distance_points_to_image_points(camera_type, config, one_lane_points); // 3*n
					dpoints_t lane_trans_d = cvpoints_to_dpoints(lane_trans);
					v_left_right_expand_cvpoints.push_back(lane_trans);
					v_auto_range_lane_points.push_back(lane_trans_d);
				}


				// for(int lane_id=0; lane_id < v_auto_range_dist_lane_points_new.size(); lane_id++){
				// 	const dpoints_t& one_lane_points = v_auto_range_dist_lane_points_new[lane_id]; 
				// 	cvpoints_t lane_trans = distance_points_to_image_points(camera_type, config, one_lane_points); // 3*n
				// 	dpoints_t lane_trans_d = cvpoints_to_dpoints(lane_trans);
				// 	v_auto_range_lane_points.push_back(lane_trans_d);
				// }

#ifdef DEBUG_INFO_FWC
				for(int lane_id=0; lane_id< v_auto_range_lane_points.size(); lane_id++){
					const dpoints_t& one_lane_points = v_auto_range_lane_points[lane_id]; // (1080,1920)
					std::cout << "./"+std::to_string(lane_id)+"_new.csv" << std::endl;
					ofstream file("./"+std::to_string(lane_id)+"_new.csv");
					if (file){
						file << "x" << "," << "y" << "\n";
						for(int idx=0; idx<one_lane_points.size(); idx++){
							// tmp(Point(lane_trans[0][idx], lane_trans[2][idx])) = 255;
							file << one_lane_points[idx][0] << "," << one_lane_points[idx][1] << "\n";
						}
					}
					file.close();
				}
#endif



				// (4) get left right main lane (auto range)
				std::vector<dpoints_t> v_merged_lane_points;
				std::vector<LaneKeypoint> v_lane_keypoint;
				lane_count = 0;
				id_left = -1;
				id_right = -1;
				double x_left = 0;
				double x_right = 0; 
				dpoints_t coord_left, coord_right; // full range 

				get_left_right_lane(
					camera_type,
					config, 
					v_auto_range_lane_points, 
					v_user_range_lane_points,
					v_merged_lane_points,
					v_lane_keypoint,
					lane_count,
					id_left, 
					id_right, 
					x_left, 
					x_right, 
					coord_left, 
					coord_right
				);

				//std::cout << "get_left_right_lane" << std::endl;

				// (5) expand left and right lane X by 0.7825
				// X= lane_trans[0,:], Y= lane_trans[2,:]
				dpoints_t coord_expand_left = coord_left;
				dpoints_t coord_expand_right = coord_right;

				// if (coord_expand_left.size()>0){
				// 	for(auto& x: coord_expand_left[0]){
				// 		x -= config.railway_delta_width;
				// 	}
				// }
				
				// if (coord_expand_right.size()>0){
				// 	for(auto& x: coord_expand_right[0]){
				// 		x += config.railway_delta_width;
				// 	}
				// }

				//std::cout << "expand left and right lane" << std::endl;


				// (6) get left and right expand image points in origin image
				// cvpoints_t left_expand_lane_cvpoints  = distance_points_to_image_points(
				// 	camera_type,
				// 	config, 
				// 	coord_expand_left
				// );
				// cvpoints_t right_expand_lane_cvpoints = distance_points_to_image_points(
				// 	camera_type,
				// 	config, 
				// 	coord_expand_right
				// );
				cvpoints_t left_expand_lane_cvpoints, right_expand_lane_cvpoints;
				if (v_left_right_expand_cvpoints.size() == 2){
					left_expand_lane_cvpoints = v_left_right_expand_cvpoints[0];
					right_expand_lane_cvpoints = v_left_right_expand_cvpoints[1];
				}

#ifdef DEBUG_INFO_FWC
				std::vector<cvpoints_t> v_expand_lane_cvpoints;
				v_expand_lane_cvpoints.push_back(left_expand_lane_cvpoints);
				v_expand_lane_cvpoints.push_back(right_expand_lane_cvpoints);
				for(int lane_id=0; lane_id< v_expand_lane_cvpoints.size(); lane_id++){
					const cvpoints_t& one_lane_points = v_expand_lane_cvpoints[lane_id]; // (1080,1920)
					std::cout << "./"+std::to_string(lane_id)+"_expand.csv" << std::endl;
					ofstream file("./"+std::to_string(lane_id)+"_expand.csv");
					if (file){
						file << "x" << "," << "y" << "\n";
						for(int idx=0; idx<one_lane_points.size(); idx++){
							// tmp(Point(lane_trans[0][idx], lane_trans[2][idx])) = 255;
							file << one_lane_points[idx].x << "," << one_lane_points[idx].y << "\n";
						}
					}
					file.close();
				}
#endif

				if (box_count<1){
					printf("[WARNING-1] no detecton boxs. \n");
				}
				if (id_left<0){
					printf("[WARNING-2a] can not find left lane \n");
				} else if (id_right<0){
					printf("[WARNING-2b] can not find right lane \n");
				} 

				cvpoint_t nearest_point; // nearest invasion box point

				// fwc test
				TABLE_TYPE table_type = TABLE_LONG_A;
				if (camera_type == CAMERA_SHORT){
					table_type = TABLE_SHORT_A;
				}

				std::vector<LidarBox> obstacle_box; // lidar invasion object 3d box
				// std::vector<lidar_invasion_cvbox> cv_obstacle_box; // lidar invasion object cv box
				
				int status_type; // out
				dpoints_t curved_point_list; // out
				std::vector<double> curved_r_list; // out

				
				if (id_left>=0 && id_right>=0){
					// (7) box image points invasion
					box_invasion_results = box_image_points_invasion_detect(
						camera_type,
						config, 
						coord_expand_left,  // for other common box
						coord_expand_right, 
						left_expand_lane_cvpoints, // for train class box
						right_expand_lane_cvpoints,
						detection_boxs,
						trains_cvpoints
					);

					//  (8) lidar point invasion detect
					InvasionData  invasion;
					for (int idx=config.polyfit_order; idx>=0; idx--){
						invasion.coeff_left.push_back(v_left_right_polyfit_matk[0].at<double>(idx, 0));
						invasion.coeff_right.push_back(v_left_right_polyfit_matk[1].at<double>(idx, 0));
					}

					// CAMERA_TYPE::CAMERA_LONG  1 
					if (table_type != 1){
						if (config.use_lidar_pointcloud_smallobj_invasion_detect){
							lidar_pointcloud_smallobj_invasion_detect(
								cloud, invasion, left_expand_lane_cvpoints, right_expand_lane_cvpoints,
								obstacle_box, cv_obstacle_box
							);

							//std::cout << "obstacle_box size:" << obstacle_box.size() << std::endl;
							// << "cv_obstacle_box size:" << cv_obstacle_box.size() << std::endl;
							lidar_invasion_status = lidar_image_points_invasion_detect(
								camera_type,
								config, 
								coord_expand_left, 
								coord_expand_right, 
								lidar_cvpoints
							);
						}

						if (config.use_lane_status){
							//  (10) get lane status
							status_type = get_lane_status(invasion, v_src_dist_lane_points, curved_point_list, curved_r_list);
							// std::cout << "status_type:" << status_type << std::endl;
						}
					} 

					// (9) get nearest invasion box point
					nearest_point = get_nearest_invasion_box_point(
						detection_boxs, 
						box_invasion_results
					);
				}
				


				// (10) draw image with lane mask, safe area
				x_draw_lane_with_mask(
					lane_model_type,
					config,
					origin_image, 
					binary_mask,
					instance_mask,
					v_src_lane_points, 
					v_merged_lane_points,
					v_lane_keypoint,
					id_left, 
					id_right, 
					nearest_point,
					left_expand_lane_cvpoints, 
					right_expand_lane_cvpoints, 
					detection_boxs,
					box_invasion_results,
					trains_cvpoints,
					image_with_color_mask,
					lane_safe_area_corner
				);
				// CAMERA_TYPE::CAMERA_LONG  1 
				if (table_type != 1){
					// bool is_show_lidarbox = true;
					if (config.use_lidar_pointcloud_smallobj_invasion_detect){
						for (size_t i = 0; i < cv_obstacle_box.size(); i++)
						{
							const lidar_invasion_cvbox cv_box = cv_obstacle_box[i];
							int xmin = cv_box.xmin;
							int xmax = cv_box.xmax;
							int ymin = cv_box.ymin;
							int ymax = cv_box.ymax;
							cv::Rect box(xmin,ymin, xmax-xmin,ymax-ymin);
							std::stringstream ss;
							ss << fixed << setprecision(2) << cv_box.dist;
							std::string str = "Dis: " + ss.str() + "m";
							cv::putText(image_with_color_mask, str, cv::Point(xmin+(xmax-xmin)/2, ymin-10), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(0, 128, 255), 1);
							cv::rectangle(image_with_color_mask, box.tl(), box.br(), cv::Scalar(0,128,255), 5, 8, 0);
						}
						cv::putText(image_with_color_mask, std::to_string(cv_obstacle_box.size()), cv::Point(100, 100), cv::FONT_HERSHEY_TRIPLEX, 2.0, cv::Scalar(0, 128, 255), 5);
					}
					// bool is_show_lane_curved_points = true;
					if (config.use_lane_status){
						// status_type
						// status0: empty
						// status1: Turn Left
						// status2: Turn Right
						// status3: Curved point
						// status4: Straight Turn close
						// status5: Straight Turn long
						// status6: Straight
						switch (status_type)
						{
						case 0:
							cv::putText(image_with_color_mask, "empty", cv::Point(100, 200), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
							is_open_long_camera = false;
							break;
						case 1:
							cv::putText(image_with_color_mask, "Turn Left", cv::Point(100, 200), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
							is_open_long_camera = false;
							break;
						case 2:
							cv::putText(image_with_color_mask, "Turn Right", cv::Point(100, 200), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
							is_open_long_camera = false;
							break;
						case 3:{
							// dpoints to cvpoints
							dpoint_t tmp_x;
							dpoint_t tmp_z;
							dpoint_t tmp_y;
							for(int idx=0; idx<curved_point_list.size(); idx++){
								tmp_x.push_back(curved_point_list[idx][1]);
								tmp_z.push_back(0.0);
								tmp_y.push_back(curved_point_list[idx][0]);
							}
							dpoints_t tmp_ps{tmp_x, tmp_z, tmp_y};
							cvpoints_t curved_cvpoints = distance_points_to_image_points(camera_type, config, tmp_ps); // 3*n
							for (int idx=0; idx<curved_cvpoints.size(); idx++)
								cv::circle(image_with_color_mask, curved_cvpoints[idx], 8, cv::Scalar(255, 0, 255), -1);
							
							std::string str_flag;
							std::stringstream ss1, ss2;
							ss1 << fixed << setprecision(2) << curved_point_list[0][0];
							ss2 << fixed << setprecision(2) << curved_point_list[1][0];
							str_flag = "Dis:"+ss1.str()+" "+ss2.str();
							cv::putText(image_with_color_mask, str_flag, cv::Point(100, 200), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
							std::stringstream ss3, ss4;
							ss3 << fixed << setprecision(2) << curved_r_list[0];
							ss4 << fixed << setprecision(2) << curved_r_list[1];
							// std::cout << curved_r_list[0] << " " << curved_r_list[1] << std::endl;
							str_flag = "CurvedR:"+ss3.str()+" "+ss4.str();
							cv::putText(image_with_color_mask, str_flag, cv::Point(100, 250), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
							
							if (curved_point_list[0][0] <=30 || curved_point_list[1][0] <= 30){
								is_open_long_camera = false;
							}
							else{
								is_open_long_camera = true;
							}

							break;
						}
						case 4:
							cv::putText(image_with_color_mask, "Straight Turn close", cv::Point(100, 200), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
							is_open_long_camera = false;
							break;
						case 5:
							cv::putText(image_with_color_mask, "Straight Turn long", cv::Point(100, 200), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
							is_open_long_camera = true;
							break;
						case 6:
							cv::putText(image_with_color_mask, "Straight", cv::Point(100, 200), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
							is_open_long_camera = true;
							break;
						default:
							is_open_long_camera = false;
							break;
						}// end switch	
						if (is_open_long_camera){
							cv::putText(image_with_color_mask, "open_long_camera:True", cv::Point(100, 150), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
						}
						else{
							cv::putText(image_with_color_mask, "open_long_camera:False", cv::Point(100, 150), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255, 0, 0));
						}
					}
				}// end if table type
				m_counter++;

				return true;
			}
#pragma endregion

		}
	}
}// end namespace