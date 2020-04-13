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
#include "algorithm_type.h" //caffe_net_file_t,  Mat, Keypoint, box_t, refinedet_boxs_t
#include "algorithm/core/caffe/internal/caffe_def.h"

namespace watrix {
	namespace algorithm {
		namespace internal{

			class SidewallApiImpl
			{
			public:
				static const int m_max_batch_size = 4; // >=4 out of memory 

				static int m_counter;
	
				static void init(
					const caffe_net_file_t& caffe_net_params
				);
				static void free();

				// set sidewall net image size
				static void set_image_size(int height, int width);
				
				/*
				for XxxNet: use v_input1,v_input2,v_output
				for XxxApi: use v_image,v_roi,v_diff,v_anomaly_boxs
				*/
				static bool sidewall_detect(
					const int net_id,
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

			protected:

				static bool sidewall_detect(
					shared_caffe_net_t sidewall_net,
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

			};


		}
	}
}// end namespace