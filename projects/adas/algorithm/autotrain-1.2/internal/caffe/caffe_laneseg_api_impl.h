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
#include "projects/adas/algorithm/algorithm_type.h" 

#include "projects/adas/algorithm/core/caffe/internal/caffe_def.h"


namespace watrix {
	namespace algorithm {
		namespace internal{

			class  CaffeLaneSegApiImpl
			{
			protected:
				static int m_counter;
				static std::vector<float> m_bgr_mean;

			public:
				static void init(
					const caffe_net_file_t& detect_net_params,
					int feature_dim,
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
					std::vector<cv::Mat>& v_binary_mask, // 256,1024
					std::vector<channel_mat_t>& v_instance_mask // 8,256,1024
				);
			};


		}
	}
}// end namespace