/*
* Software License Agreement
*
*  WATRIX.AI - www.watrix.ai
*  Copyright (c) 2016-2018, Watrix Technology, Inc.
*
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the copyright holder(s) nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: zunlin.ke@watrix.ai (Zunlin Ke)
*
*/
#pragma once
#pragma warning( disable: 4819 ) 
#pragma warning( disable: 4244 ) 
#pragma warning( disable: 4267 ) 
#pragma warning( disable: 4251 )
#include "algorithm_shared_export.h" 

// std
#include <string>
#include <vector>

// opencv
#include <opencv2/core.hpp> // Mat
#include <opencv2/features2d.hpp> // KeyPoint

namespace watrix {
	namespace algorithm {

		// opencv def
		typedef cv::Point2i cvpoint_t; // point in image 
		typedef std::vector<cvpoint_t> cvpoints_t;

		typedef std::vector<cv::Rect> boxs_t;
		typedef std::vector<cv::KeyPoint> keypoints_t;

		typedef std::vector<cv::Point> contour_t;
		typedef std::vector<contour_t> contours_t;

		// for caffe blob mat
		typedef std::vector<cv::Mat> channel_mat_t; //[1,256,256],[3,256,256],[4,256,256]
		typedef std::vector<channel_mat_t> blob_channel_mat_t; // [5,1,256,256],[5,3,256,256],[5,4,256,256]

		typedef std::pair<cv::Mat, cv::Mat> mat_pair_t; // 2-pair
				
		// define 
		#define BETWEEN(x,a,b) ( std::min(a,b) <= (x) ) && ( (x) <= std::max(a,b) )
		#define MINV(v) (*std::min_element(v.begin(),v.end()))
		#define MAXV(v) (*std::max_element(v.begin(),v.end()))

		#define CV_BGR(b,g,r) CV_RGB(r,g,b)

		#define COLOR_YELLOW CV_RGB(255, 255, 0)
		#define COLOR_RED CV_RGB(255, 0, 0)
		#define COLOR_GREEN CV_RGB(0, 255, 0)
		#define COLOR_BLUE CV_RGB(0, 0, 255)


		// caffe net 
		struct SHARED_EXPORT caffe_net_file_t {
			std::string deploy_file;
			std::string model_file;
		};

		// for object detection box result
		// 1 image ===> n output
		// b image ===> b
		struct SHARED_EXPORT detection_box_t {
			int xmin, ymin, xmax, ymax;
			float confidence;
			int class_index;
			std::string class_name;
			// dist from ditance table[cy][cx]
			bool valid_dist;
			float dist_x;
			float dist_y;
		};
		typedef std::vector<detection_box_t> detection_boxs_t; // for 1 image

		struct SHARED_EXPORT component_stat_t {
			int x,y,w,h, area;
		};


		struct SHARED_EXPORT table_param_t {
			std::string long_a;
			std::string long_b; 
			std::string short_a;
			std::string short_b;
		};

		struct SHARED_EXPORT lidar_camera_param_t {
			std::string camera_long;
			std::string camera_short; 
			std::string lidar_short;
		};

		enum TABLE_TYPE {
			TABLE_LONG_A = 1,
			TABLE_LONG_B = 2,
			TABLE_SHORT_A = 3,
			TABLE_SHORT_B = 4
		};

		enum CAMERA_TYPE {
			CAMERA_LONG = 11,
			CAMERA_SHORT = 22
		};

		// for carv2
		struct SHARED_EXPORT LahuParam {
			std::string model_path;
			int pixel_threshold_for_count = 185; // >= 185, count++
			int lahu_count_threshold = 5; // count>=5, mark as handian
			int pixel_threshold_for_box = 128; // for binary to get box
			float score_threshold = 0.65555; 
		};

		struct SHARED_EXPORT MosunParam {
			std::string model_path;
			float real_width = 90.8; // mm 
			float left_right_area_max_ratio = 10; //  if area <= 10
			float default_pixel_meter_factor = 0.0780069;
		};

		struct SHARED_EXPORT MosunResult {
			bool success = false;
			cv::Mat rotated;
			
			// area 
			int left_area;
			int right_area; 
			int middle_area;
			float left_right_area_ratio;
			bool left_right_symmetrical; // if left/right area ratio <= 2, then true; otherwise false

			cvpoint_t top_point;
			cvpoint_t bottom_point;
			cvpoint_t left_point;
			cvpoint_t right_point;
			int mosun_in_pixel = 0; // in pixel
			float mosun_in_meter = 0.0; // in mm
		};

		// for autotrain
		// for lane cluster 
		typedef std::vector<double> dpoint_t; // point (feature point)
		typedef dpoint_t ftpoint_t; // feature point
		typedef std::vector<dpoint_t> dpoints_t; // multiple point in grid

		enum CLUSTER_TYPE {
			USER_MEANSHIFT = 1,
			MLPACK_MEANSHIFT = 2,
			MLPACK_DBSCAN = 3
		};

		struct mean_shift_result_t {
			std::vector<dpoint_t> original_points;
			std::vector<dpoint_t> cluster_centers;
			std::vector<int> cluster_ids;
		};

		enum INVASION_STATUS {
			UNKNOW = -1,
			NO_INVASION = 0, 
			YES_INVASION = 1
		};

		// for lane box invasion detection
		struct SHARED_EXPORT box_invasion_result_t {
			int invasion_status = UNKNOW; // -1 UNKNOW, 0 NOT Invasion, 1 Yes Invasion
			float invasion_distance = 0; // meters
		};
		typedef std::vector<box_invasion_result_t> box_invasion_results_t; // for 1 image

		struct SHARED_EXPORT YoloNetConfig {
			int net_count;
			std::string proto_filepath;
			std::string weight_filepath;
			std::string label_filepath;
			cv::Size input_size;
			bool resize_keep_flag; // true, use ResizeKP; false use cv::resize
			std::vector<float> bgr_means;
			float normalize_value; // 1/255.0
			float confidence_threshold; // 0.4 for filter out box
		};

		struct SHARED_EXPORT LaneKeypoint{
			cvpoint_t top;
			cvpoint_t bottom;
		};

		struct SHARED_EXPORT LaneInvasionConfig {
			std::string output_dir = "";// set output dir for debug temp results
			bool b_save_temp_images = true; // save temp image results
			
			bool b_draw_lane_surface = true; // draw lane surface
			bool b_draw_boxs = false; // draw detection boxs

			bool b_draw_left_right_lane = true; // draw left right lane
			bool b_draw_other_lane = true; // draw other lane

			bool b_draw_left_right_fitted_lane = true; // draw left/right fitted lane
			bool b_draw_other_fitted_lane = false; // draw other fitted lane

			bool b_draw_expand_left_right_lane = true; // draw expand left right lane
			bool b_draw_lane_keypoint = false; // draw left/right lane top/bottom keypoints

			bool b_draw_safe_area = true; // draw safe area
			bool b_draw_safe_area_corner = true; // draw 4 corner

			bool b_draw_train_cvpoints = true; // draw train-class cvpoints
			bool b_draw_stats = true; // draw stats 

			unsigned int safe_area_y_step = 1; // y step for drawing safe area  >=1
			double safe_area_alpha = 0.5; // overlay aplpa

			bool use_tr34 = true; // true, use tr34; false, use tr33
			dpoints_t tr33; // for caffe version
			
			// projection matrix
			dpoints_t tr34_long_b;
			dpoints_t tr34_short_b; 

			//dpoints_t tr33_long_b;
			//dpoints_t tr33_short_b;

			double z_height;
			
			// cluster grid related params
			unsigned int grid_size = 8; // default 8
			unsigned int min_grid_count_in_cluster = 10; // if grid_count <=10 then filter out this cluster

			// cluster algorithm params
			unsigned int cluster_type = CLUSTER_TYPE::USER_MEANSHIFT; // (1 USER_MEANSHIFT,2 MLPACK_MEANSHIFT, 3 MLPACK_DBSCAN)
			double user_meanshift_kernel_bandwidth = 0.52; 
			double user_meanshift_cluster_epsilon = 1.5;
			
			double mlpack_meanshift_radius = 1.5;
			unsigned int mlpack_meanshift_max_iterations  = 1000; // max iterations
			double mlpack_meanshift_bandwidth = 0.52 ;

			double mlpack_dbscan_cluster_epsilon = 0.7; // not same
			unsigned int mlpack_dbscan_min_pts  = 3; // cluster at least >=3 pts

			// filter out lane noise params
			bool filter_out_lane_noise = false; // filter out lane noise
			int min_area_threshold = 300; // min area for filter lane noise
			int min_lane_pts = 10; // at least >=10 points for one lane

			// polyfit lane 
			int polyfit_order = 2; // by default 4;  value range = 1,2,...9
			bool reverse_xy = true; // whethe reverse xy
			//bool use_user_range = false; // false, auto set range depends on points
			int x_range_min = 0;
			int x_range_max = 1920;
			int y_range_min = 568;
			int y_range_max = 1080;
			bool fit_lane_bottom = true; // fit lane bottom for distortion image

			// standard limit params (m)
			double railway_standard_width = 1.435;
			double railway_half_width = 0.7175;
			double railway_limit_width = 1.500;
			double railway_delta_width = 0.7825; 

			// case1 params
			double case1_x_threshold = 1.500; // default 
			double case1_y_threshold = 15.0; // long: 33-34m;  short: 9-10m;
		};

		struct SHARED_EXPORT lane_safe_area_corner_t {
			bool valid;
			cvpoint_t left_upper;
			cvpoint_t right_upper;
			cvpoint_t left_lower;
			cvpoint_t right_lower;
		};

		//==================================================================
		// pytorch net 
		//==================================================================
		struct SHARED_EXPORT PtSimpleLaneSegNetParams {
			std::string model_path;
			int surface_id = 0;
			int left_id = 1;
			int right_id = 2;
		};

		struct SHARED_EXPORT PtComplexLaneSegNetParams {
			std::string feature_model_path;
			std::string binary_model_path;
		};

		enum LANE_MODEL_TYPE {
			LANE_MODEL_CAFFE = 1, // caffe 8-dim feature map
			LANE_MODEL_PT_SIMPLE = 2, // pt simple (zhaoshengchu)
			LANE_MODEL_PT_COMPLEX = 3 // pt complex (tongrengling)
		};

	}
} // end namespace