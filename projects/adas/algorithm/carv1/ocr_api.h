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

#ifdef USE_DLIB 

#include "algorithm_shared_export.h" 
#include "algorithm_type.h" //caffe_net_file_t,  Mat, Keypoint, box_t, refinedet_boxs_t

#include "ocr_type.h"

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT OcrApi
		{
		public:
			static void init(
				const caffe_net_file_t& detect_net_params
			);
			static void free();

			/*
			�������ܣ���ʼ��ocr�Ĳ���

			������
			param��ocr�Ĳ���

			����ֵ����
			*/
			static void init_ocr_param(const OcrType::ocr_param_t& param);

			/*
			��������: ocr��⣬batch�汾

			������
			net_id��ocr����id����Զ��߳�ʹ��
			v_pair_image: ocrͼ��
			v_up:ͼ�����Ƿ�������
			v_has_roi: �Ƿ��⵽��roi
			v_box: roi��Ӧ���������
			v_roi: roi��Ӧ��ͼ������

			����ֵ����
			*/
			static void detect(
				const int& net_id,
				const std::vector<OcrType::ocr_mat_pair_t>& v_pair_image,
				const std::vector<bool>& v_up,
				std::vector<bool>& v_has_roi,
				std::vector<cv::Rect>& v_box,
				std::vector<cv::Mat>& v_roi
			);

			/*
			�������ܣ�ocr��⣬����ͼ��汾

			������
			net_id: ocr����id����Զ��߳�ʹ��
			pair_image: ocrͼ��
			up:ͼ�����Ƿ�������
			has_roi: �Ƿ��⵽��roi
			box: roi��Ӧ���������
			roi: roi��Ӧ��ͼ������

			����ֵ����
			*/
			static void detect(
				const int& net_id,
				const OcrType::ocr_mat_pair_t& pair_image,
				const bool& up,
				bool& has_roi,
				cv::Rect& box,
				cv::Mat& roi
			);

			/*
			�������ܣ���ȡ����roi�����ͼ������

			������
			roi: roi��Ӧ��ͼ������
			feature: roi�����ͼ������

			����ֵ��true��ʾ�ɹ�,false��ʾʧ��
			*/
			static bool get_feature(
				const cv::Mat& roi,
				OcrType::feature_t& feature
			);

			/*
			�������ܣ�ocrʶ��batch�汾

			������
			net_id:ocr����id����Զ��߳�ʹ��
			v_pair_image: ocrͼ��
			v_up:ͼ�����Ƿ�������
			v_features:��ƥ��������б�
			min_similarity:����ƥ�����С���ƶ���ֵ
			v_has_roi: �Ƿ��⵽��roi
			v_box: roi��Ӧ���������
			v_roi: roi��Ӧ��ͼ������
			v_success:�Ƿ�ʶ��ɹ�
			v_result:ʶ��Ľ��

			����ֵ����
			*/
			static void recognise(
				const int& net_id,
				const std::vector<OcrType::ocr_mat_pair_t>& v_pair_image,
				const std::vector<bool>& v_up,
				const OcrType::features_t& v_features,
				const float min_similarity,
				std::vector<bool>& v_has_roi,
				std::vector<cv::Rect>& v_box,
				std::vector<cv::Mat>& v_roi,
				std::vector<bool>& v_success,
				std::vector<float>& v_similarity,
				std::vector<std::string>& v_result
			);

			/*
			�������ܣ�ocrʶ�𣬵���ͼ��汾

			������
			net_id:ocr����id����Զ��߳�ʹ��
			pair_image: ocrͼ��
			up:ͼ�����Ƿ�������
			v_features:��ƥ��������б�
			min_similarity:����ƥ�����С���ƶ���ֵ
			has_roi: �Ƿ��⵽��roi
			box: roi��Ӧ���������
			roi: roi��Ӧ��ͼ������
			success:�Ƿ�ʶ��ɹ�
			result:ʶ��Ľ��

			����ֵ����
			*/
			static void recognise(
				const int& net_id,
				const OcrType::ocr_mat_pair_t& pair_image,
				const bool& up,
				const OcrType::features_t& v_features,
				const float min_similarity,
				bool& has_roi,
				cv::Rect& box,
				cv::Mat& roi,
				bool& success,
				float& similarity,
				std::string& result
			);

			/*
			�������ܣ�ocr roiʶ��

			������
			roi: ����roiͼ��
			v_features:��ƥ��������б�
			min_similarity:����ƥ�����С���ƶ���ֵ
			success:�Ƿ�ʶ��ɹ�
			result:ʶ��Ľ��

			����ֵ����
			*/
			static void roi_recognise(
				const cv::Mat& roi,
				const OcrType::features_t& v_features,
				const float min_similarity,
				bool& success,
				float& similarity,
				std::string& result
			);

		};

	}
}// end namespace

#endif