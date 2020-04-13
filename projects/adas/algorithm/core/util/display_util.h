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

		class SHARED_EXPORT DisplayUtil
		{
		public:

#pragma region draw box contour

			// draw boxs
			static void draw_box(
				const cv::Mat& image,
				const cv::Rect& box,
				const unsigned int thickness,
				cv::Mat& image_with_boxs
			);

			static void draw_boxs(
				const cv::Mat& image,
				const std::vector<cv::Rect>& boxs,
				const unsigned int thickness,
				cv::Mat& image_with_boxs
			);

			static void draw_contour(
				const cv::Mat& image,
				const std::vector<cv::Point>& contour,
				const unsigned int thickness,
				cv::Mat& image_with_contours
			);

			static void draw_contours(
				const cv::Mat& image,
				const std::vector<std::vector<cv::Point> >& contours,
				const unsigned int thickness,
				cv::Mat& image_with_contours
			);
#pragma endregion

			static void draw_detection_boxs(
				const cv::Mat& image,
				const detection_boxs_t& boxs,
				const box_invasion_results_t& box_invasion_results,
				const unsigned int thickness,
				cv::Mat& image_with_boxs
			);

			static void draw_mosun_result(
				const MosunResult& mosun_result,
				cv::Mat& image_with_result
			);


#pragma region draw point/circles
				
			static void draw_circle_point(
				cv::Mat& image, int x, int y, cv::Scalar color, int radius= 1
			);

			static void draw_circle_point(
				cv::Mat& image, cv::Point2i pt, cv::Scalar color, int radius= 1
			);

			static void draw_circle_point_with_text(
				cv::Mat& image, cv::Point2i pt, cv::Scalar color, const std::string& text, int radius= 1
			);

			static void draw_circle_point_with_text(
				cv::Mat& image, cv::Point2i pt, cv::Scalar color, int value, int radius= 1
			);

			static void draw_lane(
				cv::Mat& out, const dpoints_t& one_lane_dpoints, cv::Scalar color, int radius=1
			);

			static void draw_lane(
				cv::Mat& out, const cvpoints_t& one_lane_cvpoints, cv::Scalar color, int radius=1
			);
#pragma endregion

			static void draw_lane_line(
				cv::Mat& out, const dpoints_t& one_lane_dpoints, cv::Scalar color
			);

			static void draw_lane_line(
				cv::Mat& out, const cvpoints_t& one_lane_cvpoints, cv::Scalar color
			);
		};

	}
}// end namespace