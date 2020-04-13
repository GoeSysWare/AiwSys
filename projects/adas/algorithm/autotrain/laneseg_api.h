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
#include "projects/adas/algorithm/algorithm_shared_export.h" 
#include "projects/adas/algorithm/algorithm_type.h" 

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT LaneSegApi
		{
			public:
				static int lane_model_type; 

				static void set_model_type(int lane_model_type);

				static void init(
					const caffe_net_file_t& detect_net_params,
					int feature_dim,
					int net_count
				);

				static void init(
					const PtSimpleLaneSegNetParams& params,
					int net_count
				);

				static void free();

				static void set_bgr_mean(
					const std::vector<float>& bgr_mean
				);

				static bool lane_seg(
					int net_id,
					const std::vector<cv::Mat>& v_image,
					int min_area_threshold, 
					std::vector<cv::Mat>& v_binary_mask, 
					std::vector<channel_mat_t>& v_instance_mask
				);

				static bool lane_seg_sequence(
					int net_id,
					const std::vector<cv::Mat>& v_image_front_result,
					const std::vector<cv::Mat>& v_image_cur,
					int min_area_threshold, 
					std::vector<cv::Mat>& v_binary_mask, 
					std::vector<channel_mat_t>& v_instance_mask
				);

				static bool lane_seg(
					int net_id,
					const cv::Mat& image,
					int min_area_threshold, 
					cv::Mat& binary_mask, 
					channel_mat_t& instance_mask
				);

				static cv::Mat get_lane_full_binary_mask(const cv::Mat& binary_mask);

				static bool lane_invasion_detect(
					CAMERA_TYPE camera_type, // long, short
					const cv::Mat& origin_image, 
					const cv::Mat& binary_mask, 
					const channel_mat_t& instance_mask,
					const detection_boxs_t& detection_boxs,
					//const std::vector<cvpoints_t>& trains_cvpoints,
					const cvpoints_t& lidar_cvpoints,
					// const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, // lidar pointscloud
					const std::vector<cv::Point3f> cloud, // lidar pointscloud
					const LaneInvasionConfig& config,
					cv::Mat& image_with_color_mask,
					int& lane_count,
					int& id_left, 
					int& id_right,
					box_invasion_results_t& box_invasion_results,
					std::vector<int>& lidar_invasion_status,
					lane_safe_area_corner_t& lane_safe_area_corner,
					bool& is_open_long_camera,
					std::vector<lidar_invasion_cvbox>& cv_obstacle_box // lidar invasion object cv box
				);
		};

	}
}// end namespace