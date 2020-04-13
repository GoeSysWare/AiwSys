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

		class SHARED_EXPORT OpencvUtil
		{
		public:


			static void cv_resize(const cv::Mat& input, cv::Mat& output, const cv::Size& size);


			// CV_8UC1
			static cv::Mat get_channel_mat(
				const cv::Mat& color_mat,
				const unsigned int channel
			);

			// CV_32FC1 
			static cv::Mat get_float_channel_mat(
				const cv::Mat& color_mat,
				const unsigned int channel
			);

			// hwc,bgr,0-255 ===> chw,bgr, -mean
			static void bgr_subtract_mean(
				const cv::Mat& bgr,
				const std::vector<float>& bgr_mean,
				float scale,
				channel_mat_t& channel_mat
			);

			// 8UC3 + 8UC1
			static cv::Mat merge_mask(
				const cv::Mat& image,
				const cv::Mat& binary_mask,
				int b, int g, int r
			);

			// 8UC3 + 8UC1
			static cv::Mat merge_mask(
				const cv::Mat& image,
				const cv::Mat& binary_mask,
				cv::Scalar bgr
			);

			// 8UC3
			static void print_mat(const cv::Mat& image);

			// 8UC3
			static cv::Mat generate_mat(
				int height, int width, int b, int g, int r
			);
			
			static void mat_add_value(
				cv::Mat& mat,
				float value
			);

			static void mat_subtract_value(
				cv::Mat& mat,
				float value
			);

			static cv::Mat concat_mat(
				const cv::Mat& first_image,
				const cv::Mat& second_image
			);

			static void split_mat_horizon(
				const cv::Mat& image,
				cv::Mat& upper,
				cv::Mat& lower
			);

			static cv::Mat clip_mat(
				const cv::Mat& image,
				const float start_ratio,
				const float end_ratio
			);

			static cv::Mat rotate_mat(const cv::Mat& image);
			static cv::Mat rotate_mat2(const cv::Mat& image);

			static size_t get_value_count(const cv::Mat& image);

			static cv::Mat get_binary_image(
				const cv::Mat& image, 
				const float min_binary_threshold
			);

			static void dilate_mat(
				cv::Mat& diff,
				const int dilate_size
			);

			static void erode_mat(
				cv::Mat& diff,
				const int erode_size
			);

			static cv::Rect bounding_box(const boxs_t& boxs);

			static cv::Rect boundary(
				const cv::Rect& rect_in,
				const cv::Size& max_size
			);

			static cv::Rect expand_box(
				const cv::Rect& box,
				const int expand_width,
				const int expand_height,
				const cv::Size& max_size
			);

			static bool contour_compare(
				const std::vector<cv::Point>& contour1,
				const std::vector<cv::Point>& contour2
			);

			static bool rect_compare(
				const cv::Rect& left_rect,
				const cv::Rect& right_rect
			);

			static cv::Rect diff_box_to_origin_box(
				const cv::Rect& box_in_diff,
				const cv::Size& diff_size,
				const cv::Size& origin_size,
				const int origin_x_offset
			);

			static void diff_boxs_to_origin_boxs(
				const std::vector<cv::Rect>& v_box_in_diff,
				const cv::Size& diff_size,
				const cv::Size& origin_size,
				const int origin_x_offset,
				std::vector<cv::Rect>& v_box_in_origin
			);

			static bool get_contours(
				const cv::Mat& diff,
				const float box_min_binary_threshold,
				contours_t& contours
			);

			static bool get_contours(
				const cv::Mat& diff,
				const float box_min_binary_threshold,
				const float contour_min_area,
				contours_t& contours
			);

			static bool get_boxs(
				const cv::Mat& diff,
				const float box_min_binary_threshold,
				boxs_t& boxs
			);

			static bool get_boxs_and(
				const cv::Mat& diff,
				const float box_min_binary_threshold,
				const int box_min_height,
				const int box_min_width,
				boxs_t& boxs
			);

			static bool get_boxs_or(
				const cv::Mat& diff,
				const float box_min_binary_threshold,
				const int box_min_height,
				const int box_min_width,
				boxs_t& boxs
			);

			static bool get_boxs_and(
				const cv::Mat& diff,
				const cv::Size& origin_size,
				const int origin_x_offset,
				const float box_min_binary_threshold,
				const int box_min_height,
				const int box_min_width,
				boxs_t& boxs
			);

			static bool get_boxs_or(
				const cv::Mat& diff,
				const cv::Size& origin_size,
				const int origin_x_offset,
				const float box_min_binary_threshold,
				const int box_min_height,
				const int box_min_width,
				boxs_t& boxs
			);

			static bool get_boxs_and(
				const boxs_t& boxs,
				const int box_min_height,
				const int box_min_width,
				boxs_t& results
			);

			static bool get_boxs_or(
				const boxs_t& boxs,
				const int box_min_height,
				const int box_min_width,
				boxs_t& results
			);

			static bool get_contours_and_boxs(
				const cv::Mat& diff,
				const float box_min_binary_threshold,
				contours_t& contours,
				boxs_t& boxs
			);



			static bool filter_railway_boxs(
				const cv::Mat& origin,
				const boxs_t& boxs,
				const float box_pixel_threshold,
				boxs_t& result_boxs
			);

			static bool filter_railway_boxs2(
				const cv::Mat& origin,
				const boxs_t& boxs,
				const int box_expand_width,
				const int box_expand_height,
				const float box_stdev_threshold,
				boxs_t& result_boxs
			);

			static float get_average_pixel(const cv::Mat& image);
			static float get_average_pixel(const cv::Mat& image,const cv::Rect& box);
			static float get_image_pixel_stdev(const cv::Mat& image);

			static bool filter_topwire_boxs(
				const cv::Mat& origin_topwire_roi,
				const boxs_t& boxs_in_diff,
				const cv::Mat& diff,
				const float box_min_binary_threshold,
				boxs_t& result_boxs
			);

			static void get_average_white_black_piexl(
				const cv::Mat& binary_diff, 
				const cv::Mat& gray,
				float& avg_white_pixel,
				float& avg_black_pixel
			);


			static bool get_railway_box(
				const cv::Mat& diff,
				const cv::Size& origin_size,
				const int origin_x_offset,
				const float railway_box_ratio,
				cv::Rect& box
			);

			static bool get_topwire_box(
				const cv::Mat& diff,
				const cv::Size& origin_size,
				const int origin_x_offset,
				const float topwire_box_ratio,
				cv::Rect& box
			);


			static bool get_horizontal_gap_box(
				const cv::Mat& diff,
				const float gap_ratio,
				cv::Rect& gap_box_for_fill_white,
				cv::Rect& gap_box_in_diff
			);

			static void fill_mat(
				cv::Mat& mat,
				const cv::Rect& fill_rect,
				const int delta_height,
				const int fill_value
			);

			static bool get_col_white_pixel_count(
				const cv::Mat& diff,
				std::vector<int>& v_col_white_pixel_count
			);

			static bool get_row_white_pixel_count(
				const cv::Mat& diff,
				std::vector<int>& v_row_white_pixel_count
			);

			static cv::Mat get_horizontal_project_mat(
				const cv::Mat& origin_mat
			);

			static cv::Mat get_vertical_project_mat(
				const cv::Mat& origin_mat
			);

			static bool get_horizontal_distance_roi(
				const cv::Mat& origin_mat,
				cv::Mat& roi
			);

			static bool get_vertical_project_distance(
				const cv::Mat& origin_mat,
				int& left_col,
				int& right_col
			);

		};

	}
}// end namespace