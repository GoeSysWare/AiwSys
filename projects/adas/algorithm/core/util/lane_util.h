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

#include <vector>

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT LaneUtil
		{
		public:
			// for laneseg
			static cv::Mat connected_component_binary_mask(
				const cv::Mat& binary_mask,
				int min_area_threshold 
			);


			// for mosun detect
			static void get_largest_connected_component(
				const cv::Mat& binary_mask, // 0-255
				cv::Mat& out,
				component_stat_t& largest_stat
			);


			static dpoints_t filter_out_noise(
				const cv::Mat& binary_mask, 
				const dpoints_t& one_lane_points,
				int min_area_threshold 
			);



			static ftpoint_t get_feature(const channel_mat_t& instance_mask, int y, int x);

			static void get_grid_features_and_points(
				const cv::Mat& binary_mask, 
				const channel_mat_t& instance_mask,
				const int grid_size, // default = 8
				std::vector<ftpoint_t>& grid_features,
				std::vector<dpoints_t>& grid_points
			);

			static void cvHilditchThin1(cv::Mat& src, cv::Mat& dst); 


			static void get_clustered_lane_points_from_features(
				const LaneInvasionConfig& config,
				const cv::Mat& binary_mask, 
				const channel_mat_t& instance_mask,
				std::vector<dpoints_t>& v_src_lane_points
			);

			static void get_clustered_lane_points_from_left_right(
				const LaneInvasionConfig& config,
				const cv::Mat& binary_mask,   // [128,480] v=[0,1]  surface 
				const channel_mat_t& instance_mask, // [2, 128,480]  v=[0,1] left/right lane points
				std::vector<dpoints_t>& v_src_lane_points
			);

			static cv::Mat get_lane_binary_image(
				const cv::Size& size, 
				const dpoints_t& one_lane_points
			);

			static dpoints_t filter_out_lane_nosie(
				const cv::Mat& binary_mask, 
				int min_area_threshold,
				const dpoints_t& one_lane_points,
				unsigned int lane_id
			);


			static cv::Mat get_largest_lane_mask(
				const channel_mat_t& instance_mask
			);

			static cvpoints_t get_lane_cvpoints(
				const cv::Mat& lane_binary
			);

			static int get_average_x(
				const cv::Mat& lane_binary
			);



		};

	}
}// end namespace