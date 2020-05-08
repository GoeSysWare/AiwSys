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

#include "projects/adas/algorithm/core/caffe/internal/caffe_net_v2.h"

namespace watrix {
	namespace algorithm {
		namespace internal {

			class YoloNet : public CaffeNetV2
			{
			public:
				static YoloNetConfig config_;
				static std::vector<shared_caffe_net_t> v_net_;

				/*
				num_inputs()=1  data (1,3,416,416)
				num_outputs()=1 detection_out (1,1,N,7)
				*/
				static const int input_channel_ = 3;
				static int input_height_;
				static int input_width_; // default 416

				// 1,1,N,7  (N boxs of 7 elements)
				static const int output_channel_ = 1;
				static int output_height_; //
				static const int output_width_ = 7;

			public:
				static void Init(const YoloNetConfig& config);
				static void Free();

				static bool Detect(
					int net_id,
					const cv::Mat& image,
					detection_boxs_t& output
				);
				
			private:
				static void SetClassLabels(const std::string& filepath);

				static void SetMean(const std::vector<float>& bgr_means);

				static void WrapInputLayer(
					shared_caffe_net_t net, 
					std::vector<cv::Mat>* input_channels
				);

				static void ResizeKP(const cv::Mat& img, cv::Mat& dst, const cv::Size& input_size); 
				static void ResizeKPV2(const cv::Mat& img, cv::Mat& dst, const cv::Size& input_size);

				static void Preprocess(
					shared_caffe_net_t net,
					const cv::Mat& img,
					std::vector<cv::Mat>* input_channels,
					double normalize_value
				);

				static void Postprocess(
					shared_caffe_net_t net,
					cv::Size origin_size,
					float confidence_threshold,
					detection_boxs_t& output
				);


				static int m_counter;
				static int class_count_; // 80 or 20
				static std::vector<std::string> class_labels_; // class labels: person,car,...
				static cv::Size input_size_; // 416,416
				static cv::Mat mean_; // bgr or 
			};

		}
	}
} // end of namespace
