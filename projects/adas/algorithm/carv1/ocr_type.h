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

		class SHARED_EXPORT OcrType 
		{
		public:
			typedef std::vector<float> feature_t;

			typedef mat_pair_t ocr_mat_pair_t;

			struct id_feature_t
			{
				std::string m_id;
				feature_t feature;
			};

			typedef std::vector<OcrType::id_feature_t> features_t;

			struct ocr_param_t {
				// �ü�ԭͼ[0,4,0.9]�������
				float clip_start_ratio = 0.4f;
				float clip_end_ratio = 0.9f;

				int roi_image_width = 200;
				int roi_image_height = 200;

				float box_binary_threshold = 0.5f;
				float contour_min_area = 20.f;
				int box_min_width = 10;
				int box_min_height = 10;

				// roi�Ĵ�С��Χ
				int roi_min_width = 80;
				int roi_max_width = 180;
				int roi_min_height = 100;
				int roi_max_height = 240;
				int roi_width_delta = 4; // ��һ���Ż�roi�����width
				int roi_height_delta = 25; // ��һ���Ż�roi�����height

				// roi height/width ������Χ
				float height_width_min_ratio = 0.9f;
				float height_width_max_ratio = 2.5f;

				//����õ�roi�ٽ���ƽ����ֵ���ˣ�ͨ����������Ƚ�����
				float roi_avg_pixel_min_threshold = 100.f; 
			};
		};

		SHARED_EXPORT std::ostream& operator<<(std::ostream& cout, const OcrType::ocr_param_t& ocr_param);
	}
}
