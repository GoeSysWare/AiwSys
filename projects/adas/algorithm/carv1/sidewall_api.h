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
#include "algorithm_shared_export.h" 
#include "algorithm_type.h" //caffe_net_file_t,  Mat, Keypoint, box_t, refinedet_boxs_t

#include "sidewall_type.h" // sidewall_param_t

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT SidewallApi
		{
			public:
				static void init(
					const caffe_net_file_t& caffe_net_params
				);
				static void free();

				// set net image size
				static void set_image_size(int height, int width);

				static void init_sidewall_param(
					const SidewallType::sidewall_param_t& param
				);

				static void detect_and_compute(
					const bool enable_gpu,
					const cv::Mat& image,
					keypoints_t& keypoints, 
					cv::Mat& descriptor
				);

				static bool sidewall_match(
					const bool enable_gpu,
					const cv::Mat& image_object,
					const keypoints_t& keypoint_object,
					const cv::Mat& descriptor_object,
					const std::vector<cv::Mat>& images_history,
					const std::vector<keypoints_t>& keypoints_history,
					const std::vector<cv::Mat>& descriptors_history,
					bool& match_success,
					cv::Mat& best_roi,
					uint32_t& image_index,
					uint32_t& y_offset
				);

				static bool sidewall_detect(
					const int& net_id,
					const std::vector<cv::Mat>& v_image,
					const std::vector<cv::Mat>& v_roi,
					const cv::Size& blur_size,
					const bool fix_distortion_flag,
					const float box_min_binary_threshold,
					const int box_min_width,
					const int box_min_height,
					std::vector<bool>& v_has_anomaly,
					std::vector<boxs_t>& v_anomaly_boxs
				);

				static bool sidewall_detect(
					const int& net_id,
					const cv::Mat& image,
					const cv::Mat& roi,
					const cv::Size& blur_size,
					const bool fix_distortion_flag,
					const float box_min_binary_threshold,
					const int box_min_width,
					const int box_min_height,
					bool& has_anomaly,
					boxs_t& anomaly_boxs
				);


				static void sidewall_match_and_detect(
					const bool enable_gpu_match,
					const cv::Mat& image_object,
					const keypoints_t& keypoint_object,
					const cv::Mat& descriptor_object,
					const std::vector<cv::Mat>& images_history,
					const std::vector<keypoints_t>& keypoints_history,
					const std::vector<cv::Mat>& descriptors_history,
					const int& net_id,
					const cv::Size& blur_size,
					const bool fix_distortion_flag,
					const float box_min_binary_threshold,
					const int box_min_width,
					const int box_min_height,
					bool& match_success,
					cv::Mat& best_roi,
					uint32_t& image_index,
					uint32_t& y_offset,
					bool& has_anomaly,
					boxs_t& boxs
				);
		};

	}
}// end namespace