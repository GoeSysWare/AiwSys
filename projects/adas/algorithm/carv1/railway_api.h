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

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT RailwayApi
		{
			public:
				static void init(
					const caffe_net_file_t& crop_net_params,
					const caffe_net_file_t& detect_net_params
				);
				static void free();

				/*
				�������ܣ����ֹ�ͼ���쳣�͹��

				������
				net_id���ֹ�����id����Զ��߳�ʹ��
				v_image: �ֹ�ͼ��
				box_min_binary_threshold:ͼ���ֵ������ֵ
				box_min_width:���ο����С���
				box_min_height:���ο����С�߶�
				filter_box_flag:�Ƿ���box�����ع���
				filter_box_piexl_threshold:box���ع��˵���ֵ
				gap_ratio:���ĳ��Ⱥ�ͼ���ܳ��ȵı�����ֵ
				v_crop_success:�ü�ֹ��Ƿ�ɹ�
				v_has_gap: �Ƿ��й��
				v_gap_boxs:�����������
				v_has_anomaly: �Ƿ����쳣
				v_anomaly_boxs: �쳣��������ο�

				����ֵ����
				*/
				static void railway_detect(
					const int& net_id,
					const std::vector<cv::Mat>& v_image,
					const cv::Size& blur_size,
					const int dilate_size,
					const float box_min_binary_threshold,
					const int box_min_width,
					const int box_min_height,
					const bool filter_box_by_avg_pixel, // filter box 1 by avg pixel 
					const float filter_box_piexl_threshold, // [0,1]
					const bool filter_box_by_stdev_pixel, // filter box 2 by stdev
					const int box_expand_width,
					const int box_expand_height,
					const float filter_box_stdev_threshold,// [0,
					const float gap_ratio,
					std::vector<bool>& v_crop_success,
					std::vector<bool>& v_has_gap,
					std::vector<cv::Rect>& v_gap_boxs,
					std::vector<bool>& v_has_anomaly,
					std::vector<boxs_t>& v_anomaly_boxs
				);
		};

	}
}// end namespace