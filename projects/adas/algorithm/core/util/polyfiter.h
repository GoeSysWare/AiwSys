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
#include "projects/adas/algorithm/algorithm_type.h" //caffe_net_file_t,  Mat, Keypoint, box_t, refinedet_boxs_t


namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT Polyfiter {
			public:
				Polyfiter(
					const std::vector<cvpoint_t>& src_points_, 
					int n = 4,
					bool reverse_xy_ = true, /* reverse xy depends on data points*/
					int x_range_min = 0,
					int x_range_max = 1980,
					int y_range_min = 568, // 512,568;   640,440
					int y_range_max = 1080
				);

				Polyfiter(
					const std::vector<dpoint_t>& src_d_points_, 
					int n = 4,
					bool reverse_xy_ = true, /* reverse xy depends on data points*/
					int x_range_min = 0,
					int x_range_max = 1980,
					int y_range_min = 568, // 512,568;   640,440
					int y_range_max = 1080
				);

				std::vector<cvpoint_t> fit(bool use_auto_range = true);
				std::vector<dpoint_t> fit_dpoint(CAMERA_TYPE camera_type, dpoints_t d_src_points, bool use_auto_range = true);
				cv::Mat draw_image(const cv::Size& image_size);
				cv::Mat cal_mat_add_points(dpoints_t d_src_points);
				void set_mat_k(cv::Mat mat_t_new);
				
			protected:
				void _reverse_points_xy();
				cv::Mat _cal_mat();
				double _predict(double x);

			private:
				std::vector<cvpoint_t> src_points;
				std::vector<cvpoint_t> fitted_points;
				bool is_valid_input;
				int n;
				bool reverse_xy;

				int auto_x_range_min;
				int auto_x_range_max;
				int auto_y_range_min;
				int auto_y_range_max;

				int user_x_range_min;
				int user_x_range_max;
				int user_y_range_min;
				int user_y_range_max;

				cv::Mat mat_k; // coeff
		};


	}
}// end namespace