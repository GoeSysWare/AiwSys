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
*  ANY WAY OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: zunlin.ke@watrix.ai (Zunlin Ke)
*
*/
#pragma once
#include "algorithm_type.h" //caffe_net_file_t,  Mat, Keypoint, box_t, refinedet_boxs_t

#include "../sidewall_type.h" // sidewall_param_t

namespace watrix {
	namespace algorithm {
		namespace internal{

			class  SidewallUtil {

			public:
				static std::stringstream ss;

			private:
				static SidewallType::sidewall_param_t sidewall_param;

				static std::vector<cv::Point2f> keypoints_to_points2f(keypoints_t keypoints);

				static int detect_and_compute(
					const cv::Mat& image,
					keypoints_t& keypoints,
					cv::Mat& descriptor
				);

				static void concat_images(
					const std::vector<cv::Mat>& image_history,
					const std::vector<keypoints_t> &keypoints_history,
					const std::vector<cv::Mat> & descriptors_history,
					cv::Mat& img_scene,
					keypoints_t& keypoint_scene,
					cv::Mat& descriptor_scene
				);

				static void match_and_filter(
					const cv::Mat& descriptor_object, 
					const cv::Mat& descriptor_scene,
					const keypoints_t& keypoint_object, 
					const keypoints_t& keypoint_scene, 
					const float nn_match_ratio,
					keypoints_t& matched1, 
					keypoints_t& matched2
				);

				static bool find_homography(
					const keypoints_t& matched1, 
					const keypoints_t& matched2,
					cv::Mat& homography, 
					cv::Mat& inlier_mask
				);

				static void get_average_move(
					const keypoints_t& matched1, 
					const keypoints_t& matched2,
					const cv::Mat& inlier_mask,
					int& count, 
					float& avg_x_move, 
					float& avg_y_move
				);

				static void select_best_roi(
					const cv::Mat& image_scene, 
					const cv::Mat& image_object, 
					uint32_t y,
					uint32_t& best_y, 
					cv::Mat& best_roi
				);

				static void get_image_index_and_offset(
					const std::vector<cv::Mat>& images_history, 
					uint32_t best_y,
					uint32_t& image_index, 
					uint32_t& y_offset
				);

				static void match_and_filter(
					const bool enable_gpu,
					const cv::Mat& descriptor_object,
					const cv::Mat& descriptor_scene,
					const keypoints_t& keypoint_object,
					const keypoints_t& keypoint_scene,
					const float nn_match_ratio,
					keypoints_t& matched1,
					keypoints_t& matched2
				);

			public:
				static void init_sidewall_param(
					const SidewallType::sidewall_param_t& param
				);

				static int detect_and_compute(
					const bool enable_gpu,
					const cv::Mat& image,
					keypoints_t& keypoints,
					cv::Mat& descriptor
				);

				static bool sidewall_match(
					const bool enable_gpu,
					const cv::Mat& image_object,
					const keypoints_t keypoint_object,
					const cv::Mat descriptor_object,
					const std::vector<cv::Mat>& images_history,
					const std::vector<keypoints_t>& keypoints_history,
					const std::vector<cv::Mat>& descriptors_history,
					cv::Mat& best_roi,
					uint32_t& image_index,
					uint32_t& y_offset
				);

				// use history roi to fix current image distortion
				static void sidewall_fix(
					const cv::Mat& roi,
					const cv::Mat& image,
					cv::Mat& result_image
				);

				static void sidewall_fix(
					const std::vector<cv::Mat>& v_roi,
					const std::vector<cv::Mat>& v_image,
					std::vector<cv::Mat>& v_result_image
				);
			};

		}
	}
}// end namespace