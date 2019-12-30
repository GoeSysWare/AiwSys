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
#include "algorithm_shared_export.h" // SHARED_EXPORT
#include "algorithm_type.h" //caffe_net_file_t,  Mat, Keypoint, box_t, refinedet_boxs_t

// std
#include <vector>

// opencv
#include <opencv2/core.hpp> // Mat Rect
#include <opencv2/features2d.hpp> // KeyPoint

namespace watrix {
	namespace algorithm {

		class NumpyUtil
		{
		public:
			// math
			static float get_rad_angle(int x1, int y1, int x2, int y2);
			static float rad_to_degree(float rad);
			static float degree_to_rad(float degree);
			static float get_rad_angle(const cv::Point2i& left, const cv::Point2i& right);
			static float get_degree_angle(const cv::Point2i& left, const cv::Point2i& right);

			static void max_min_of_vector(const std::vector<int>& v, int& max_value, int& min_value);
			static int get_index(const std::vector<int>& v, int value);
			static int max_index(const std::vector<int>& v);
			static int max_index(const std::vector<float>& v);

			static std::vector<int> add(const std::vector<int>& a, const std::vector<int>& b);
			static std::vector<int> sub(const std::vector<int>& a, const std::vector<int>& b);
			static std::vector<int> multipy_by(const std::vector<int>& v, int value);
			static std::vector<int> devide_by(const std::vector<int>& v, int value);

			static int dot(const std::vector<int>& a,const std::vector<int>& b);
			static float dot(const std::vector<float>& a,const std::vector<float>& b);
			static float module(const std::vector<float>& v);
			static float cosine(const std::vector<float>& v1,const std::vector<float>& v2);

			// cv 
			// (1) basic 
			static void cv_resize(const cv::Mat& input, cv::Mat& output, const cv::Size& size);
			static void cv_cvtcolor_to_gray(const cv::Mat& image, cv::Mat& gray);
			static void cv_cvtcolor_to_bgr(const cv::Mat& image, cv::Mat& bgr);
			static void cv_add_weighted(const cv::Mat& overlay, cv::Mat& image, double alpha);
			static void cv_threshold(const cv::Mat& gray, cv::Mat& mask);
			static void cv_houghlinesp(const cv::Mat& mask, std::vector<cv::Vec4i>& lines);
			static void cv_canny(const cv::Mat& image, cv::Mat& edges, int min_val, int max_val);
			static void cv_findcounters(const cv::Mat& mask, contours_t& contours);
			static void cv_dilate_mat(cv::Mat& diff,int dilate_size);
			static void cv_erode_mat(cv::Mat& diff,int erode_size);

			// (2) extension
			static cv::Mat cv_connected_component_mask(const cv::Mat& binary_mask,int min_area_threshold);
			static cv::Mat cv_rotate_image(const cv::Mat& image, float degree);
			static cv::Mat cv_rotate_image_keep_size(const cv::Mat& image, float degree);
			static cv::Mat cv_vconcat(const cv::Mat& first,const cv::Mat& second);

			static void cv_split_mat_horizon(const cv::Mat& image,cv::Mat& upper,cv::Mat& lower);
			static cv::Mat cv_clip_mat(const cv::Mat& image,float start_ratio, float end_ratio);
			static cv::Rect cv_max_bounding_box(const std::vector<cv::Rect>& boxs);
			static cv::Rect cv_boundary(const cv::Rect& rect_in,const cv::Size& max_size);
			static size_t cv_get_value_count(const cv::Mat& image);

			// np
			static void np_where_g(const cv::Mat& image, int value, std::vector<int>& v_y,std::vector<int>& v_x);
			static void np_where_eq(const cv::Mat& image, int value, std::vector<int>& v_y,std::vector<int>& v_x);
			static cv::Mat np_get_roi(const cv::Mat& image, int y1, int y2, int x1, int x2);

			// out = image.argmax(axis=0) [0,1] 256,256
			static cv::Mat np_argmax_axis_channel2(const channel_mat_t& channel_mat);
			// out = image.argmax(axis=0) [0,1,2] 256,256
			static cv::Mat np_argmax_axis_channel3(const channel_mat_t& channel_mat);
			static cv::Mat np_argmax(const channel_mat_t& channel_mat);
			static cv::Mat np_binary_mask_as255(const cv::Mat& binary_mask);

			// return vindex
			static std::vector<int> np_unique_return_index(const std::vector<int>& v);

			static std::vector<int> np_argwhere_eq(const std::vector<int>& v, int value);
			static std::vector<int> np_argwhere_eq(const std::vector<double>& v, double value);

			static std::vector<dpoint_t> np_insert(const std::vector<dpoint_t>& points, float value);
			static std::vector<dpoint_t> np_delete(const std::vector<dpoint_t>& points, int column);
			static void np_insert(std::vector<dpoint_t>& points, float value);
			static std::vector<dpoint_t> np_transpose(const std::vector<dpoint_t>& points);
			static std::vector<dpoint_t> np_inverse(const std::vector<dpoint_t>& points);
			static std::vector<dpoint_t> np_matmul(const std::vector<dpoint_t>& a,const std::vector<dpoint_t>& b);
		
			// algorithms
			// nms
			static bool sort_score_pair_descend(const std::pair<float, int>& pair1,const std::pair<float, int>& pair2);
			static float jaccard_overlap(const cv::Rect& bbox1,const cv::Rect& bbox2);

			static int nms_fast(
				const std::vector<cv::Rect>& boxs,
				const float overlapThresh, // 0.45
				//const float min_area, // 400
				std::vector<cv::Rect>& new_boxs
			);
		};

	}
}// end namespace