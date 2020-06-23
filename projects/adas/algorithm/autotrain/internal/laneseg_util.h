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
#include "projects/adas/algorithm/algorithm_shared_export.h" // SHARED_EXPORT
#include "projects/adas/algorithm/algorithm_type.h" //caffe_net_file_t,  Mat, Keypoint, box_t, detection_boxs_t


namespace watrix {
	namespace algorithm {
		namespace internal {

			class lanesegutil
			{
			public:
				static cv::Size ORIGIN_SIZE;
				static cv::Size UPPER_SIZE;
				static cv::Size CLIP_SIZE;

				static cv::Size CAFFE_INPUT_SIZE;
				static cv::Size PT_SIMPLE_INPUT_SIZE;

				static const int PT_SIMPLE_LANE_COUNT = 2; // only left + right lane for simple lane

	#pragma region get_lane_full_binary_mask

				static cv::Mat get_lane_full_binary_mask(
					const cv::Mat& binary_mask
				);

	#pragma endregion


	#pragma region point cvpoint/dpoints
				
				static dpoints_t cvpoints_to_dpoints(const cvpoints_t& cvpoints);
				static cvpoints_t dpoints_to_cvpoints(const dpoints_t& points);
				static void print_points(const std::string& name, const dpoints_t& points);
				static void print_points(const std::string& name, const cvpoints_t& cvpoints);

	#pragma endregion


	#pragma region transform points
				
				static dpoints_t get_tr33_from_tr34(const dpoints_t& tr34);

	#pragma region transform v0
				static dpoints_t image_points_to_distance_points_v0(
					const dpoints_t& points, 
					const dpoints_t& tr33,
					double z_height
				);


				static cvpoints_t distance_points_to_image_points_v0(
					const dpoints_t& coord_trans_, 
					const dpoints_t& tr33,
					double z_height
				);

	#pragma endregion

	#pragma region transform v1
				static dpoints_t nouse_image_points_to_distance_points_v1(
					const dpoints_t& points, 
					const dpoints_t& tr33,
					double z_height
				);

				static dpoints_t image_points_to_distance_points_v1(
					TABLE_TYPE table_type, // long_a, short_a
					const dpoints_t& points
				);


				static cvpoints_t distance_points_to_image_points_v1(
					const dpoints_t& coord_trans_, 
					const dpoints_t& tr34,
					double z_height
				);

				#pragma endregion

	#pragma region transform vector
				static dpoints_t image_points_to_distance_points(
					CAMERA_TYPE camera_type, // long_a, short_a
					const LaneInvasionConfig& config,
					const dpoints_t& points
				);

				static cvpoints_t distance_points_to_image_points(
					CAMERA_TYPE camera_type, // long_b, short_b
					const LaneInvasionConfig& config,
					const dpoints_t& coord_trans
				);

	#pragma endregion

	#pragma region transform sigle
				static dpoint_t image_point_to_dist_point(
					CAMERA_TYPE camera_type, // long_a, short_a
					const LaneInvasionConfig& config,
					cvpoint_t point
				);

				static cvpoint_t dist_point_to_image_point(
					CAMERA_TYPE camera_type, // long_b, short_b
					const LaneInvasionConfig& config,
					dpoint_t point_trans
				);

				static void bound_cvpoint(cvpoint_t& point, cv::Size size, int delta = 0);
	#pragma endregion


	#pragma endregion			


	#pragma region find x points by y
				
				static std::vector<double> get_lane_x_coords(const dpoints_t& lane_coord_trans, double y);
				static std::vector<int> get_lane_x_image_points(const cvpoints_t& lane_image_points, double y);

	#pragma endregion 


	#pragma region get clustered lane points

				static void x_get_clustered_lane_points(
					int lane_model_type, // caffe, pt_simple, pt_complex
					const LaneInvasionConfig& config,
					const cv::Mat& binary_mask,  // [256,1024] v=[0,1]
					const channel_mat_t& instance_mask, // [8, 256,1024] 8-dim float feature map
					std::vector<dpoints_t>& v_src_lane_points
				);
	#pragma endregion


	#pragma region binary lanes to origin lanes			

				static void __transform_binary_lanes_to_origin_lanes(
					std::vector<dpoints_t>& v_lane_points,
					cv::Size input_size // (256,1024) / (128,480)
				);

				static void x_transform_binary_lanes_to_origin_lanes(
					int lane_model_type, // caffe, pt_simple, pt_complex
					std::vector<dpoints_t>& v_lane_points
				);
				
	#pragma endregion


	#pragma region polyfit
				static void lane_polyfit(
					CAMERA_TYPE camera_type,
					const LaneInvasionConfig& config,
					int image_index,
					cv::Size origin_size,
					const std::vector<dpoints_t>& v_lane_points,
					std::vector<dpoints_t>& v_auto_range_lane_points,
					std::vector<dpoints_t>& v_user_range_lane_points,
					std::vector<cv::Mat>& v_left_right_polyfit_matk
				);

	#pragma endregion


	#pragma region polyline intersect

				static bool polyline_intersect(
					const LaneInvasionConfig& config,
					const cvpoints_t& lane_cvpoints,
					const cvpoints_t& train_points
				);
	#pragma endregion


	#pragma region get left/right lane 

				static void get_left_right_lane(
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
				);

	#pragma endregion


	#pragma region detection boxs invasion detect


	#pragma region case1 box
				static bool is_case1_box(
					const detection_box_t& box,
					double case1_x_threshold, 
					double case1_y_threshold
				);

				static box_invasion_result_t do_case1_box_invasion_detect(const detection_box_t& box);
	#pragma endregion

	/* 
	#pragma region train box
				static bool is_train_box(
					const detection_box_t& box
				);

				static box_invasion_result_t do_train_box_invasion_detect(
					CAMERA_TYPE camera_type, // long, short
					const LaneInvasionConfig& config,
					const cvpoints_t& left_expand_lane_cvpoints,
					const cvpoints_t& right_expand_lane_cvpoints,
					const cvpoints_t& train_points
				);
	#pragma endregion
	*/

	#pragma region other/common box invasion detect
				static box_invasion_result_t __do_box_invasion_detect(
					const LaneInvasionConfig& config,
					const dpoints_t& coord_expand_left,
					const dpoints_t& coord_expand_right,
					double x1, double y1, 
					double x2, double y2
				);

				static box_invasion_result_t do_box_invasion_detect(
					CAMERA_TYPE camera_type, // long, short
					const LaneInvasionConfig& config,
					const dpoints_t& coord_expand_left,
					const dpoints_t& coord_expand_right,
					const cvpoints_t& left_expand_lane_cvpoints,
					const cvpoints_t& right_expand_lane_cvpoints,
					const std::vector<dpoints_t>& v_src_dist_lane_points,
					const detection_box_t& detection_box
				);
	#pragma endregion

				static box_invasion_results_t box_image_points_invasion_detect(
					CAMERA_TYPE camera_type, // long, short
					const LaneInvasionConfig& config,
					const dpoints_t& coord_expand_left,
					const dpoints_t& coord_expand_right,
					const cvpoints_t& left_expand_lane_cvpoints,
					const cvpoints_t& right_expand_lane_cvpoints,
					const std::vector<dpoints_t>& v_src_dist_lane_points,
					const detection_boxs_t& detection_boxs,
					const std::vector<cvpoints_t>& trains_cvpoints
				);
	#pragma endregion


	#pragma region lidar invasion detect		

				static INVASION_STATUS do_point_invasion_detect(
					const LaneInvasionConfig& config,
					const dpoints_t& coord_expand_left,
					const dpoints_t& coord_expand_right,
					double x, double y 
				);

				static std::vector<int> lidar_image_points_invasion_detect(
					CAMERA_TYPE camera_type, // long, short
					const LaneInvasionConfig& config,
					const dpoints_t& coord_expand_left,
					const dpoints_t& coord_expand_right,
					const cvpoints_t& cvpoints
				);

	#pragma endregion


	#pragma region lidar pointcloud small objects invasion detect
				static void lidar_pointcloud_smallobj_invasion_detect(
					// pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
					const std::vector<cv::Point3f>& cloud, 
					InvasionData  invasion,
					float distance_limit,
					cvpoints_t& input_l,
					cvpoints_t& input_r,
					std::vector<LidarBox>& obstacle_box,
					std::vector<lidar_invasion_cvbox>& cv_obstacle_box
				);
	#pragma endregion


	#pragma region get lane status
				static int get_lane_status(
					InvasionData  invasion,
					std::vector<dpoints_t>& v_src_dist_lane_points,
					dpoints_t& curved_point_list, // out
					std::vector<double>& curved_r_list // out
				);
	#pragma endregion


	#pragma region draw lane with mask
				
				static cvpoint_t get_nearest_invasion_box_point(
					const detection_boxs_t& detection_boxs,
					const std::vector<lidar_invasion_cvbox>& cv_obstacle_box, // lidar invasion object cv box
					const box_invasion_results_t& box_invasion_results
				);

	#pragma region clip lane by nearest point
				static cvpoints_t clip_lane_by_nearest_box_point(
					const dpoints_t& lane,
					int near_x, int near_y
				);

				static cvpoints_t clip_lane_by_nearest_box_point(
					const cvpoints_t& lane,
					int near_x, int near_y
				);

	#pragma endregion

	#pragma region draw safe area

				static cv::Mat draw_lane_safe_area(
					const LaneInvasionConfig& config,
					const cv::Mat& image_,
					const cvpoint_t& nearest_point,
					const cvpoints_t& left_lane_cvpoints,
					const cvpoints_t& right_lane_cvpoints,
					int y_upper, int y_lower,
					lane_safe_area_corner_t& lane_safe_area_corner
				);

				static cv::Mat draw_lane_safe_area(
					const LaneInvasionConfig& config,
					const cv::Mat& image_,
					const cvpoint_t& nearest_point,
					const cvpoints_t& left_lane_cvpoints,
					const cvpoints_t& right_lane_cvpoints
				);

	#pragma endregion

	#pragma region draw final lane 
				
				static void x_draw_lane_with_mask(
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
				);

	#pragma endregion

	#pragma endregion


	#pragma region do lane invasion detect

				static bool lane_invasion_detect(
					int lane_model_type, // caffe, pt_simple, pt_complex
					CAMERA_TYPE camera_type, // long, short
					const cv::Mat& origin_image, 
					const cv::Mat& binary_mask, 
					const channel_mat_t& instance_mask,
					const detection_boxs_t& detection_boxs,
					const std::vector<cvpoints_t>& trains_cvpoints,
					const cvpoints_t& lidar_cvpoints,
					// const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, // lidar pointscloud
					const std::vector<cv::Point3f>& cloud, // lidar pointscloud
					const LaneInvasionConfig& config,
					cv::Mat& image_with_color_mask,
					int& lane_count,
					int& id_left, 
					int& id_right,
					box_invasion_results_t& box_invasion_results,
					std::vector<int>& lidar_invasion_status,
					lane_safe_area_corner_t& lane_safe_area_corner,
					bool& is_open_long_camera,
					std::vector<lidar_invasion_cvbox>& cv_obstacle_box, // lidar invasion object cv box
					cvpoint_t& touch_point, // 侵界障碍物和轨道的接触点
					cvpoints_t& left_fitted_lane_cvpoints, // 拟合后的左轨道图像坐标点列
					cvpoints_t& right_fitted_lane_cvpoints // 拟合后的右轨道图像坐标点列
				);
	#pragma endregion


			};

		}
	}
}// end namespace