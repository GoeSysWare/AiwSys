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

#include "lockcatch_type.h"

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT LockcatchApi
		{
			public:
				static void init(
					const caffe_net_file_t& crop_net_params,
					const caffe_net_file_t& refine_net_params
				);
				static void free();

				/*
				�������ܣ��������ͼ���쳣

				������
				net_id����������id����Զ��߳�ʹ��
				v_lockcatch: ����ͼ��
				lockcatch_threshold:�ж������쳣����ֵ
				v_has_lockcatch: ����ͼ���Ƿ������������
				v_status: ��⵽���۵�״̬(��������,����ȱʧ��ȱʧ)
				v_roi_boxs: ���۵�������ο�

				����ֵ��true��ʾ�ɹ���false��ʾʧ��
				*/
				static bool lockcatch_detect(
					const int& net_id,
					const std::vector<LockcatchType::lockcatch_mat_pair_t>& v_lockcatch,
					const LockcatchType::lockcatch_threshold_t& net1_lockcatch_threshold,
					const LockcatchType::lockcatch_threshold_t& net2_lockcatch_threshold,
					const cv::Size& blur_size,
					std::vector<bool>& v_has_lockcatch,
					std::vector<LockcatchType::lockcatch_status_t>& v_status,
					std::vector<boxs_t>& v_roi_boxs
				);

				static bool lockcatch_detect_v0(
					const int& net_id,
					const std::vector<LockcatchType::lockcatch_mat_pair_t>& v_lockcatch,
					const LockcatchType::lockcatch_threshold_t& lockcatch_threshold,
					const cv::Size& blur_size,
					std::vector<bool>& v_has_lockcatch,
					std::vector<LockcatchType::lockcatch_status_t>& v_status,
					std::vector<boxs_t>& v_roi_boxs
				);

				/*
				�������ܣ���ȡ����״̬��string�ַ���

				������
				lockcatch_status������״̬

				����ֵ������״̬��Ӧ��string�ַ���
				*/
				static std::string get_lockcatch_status_string(
					const LockcatchType::lockcatch_status_t& lockcatch_status
				);

				/*
				�������ܣ��ж�����״̬�Ƿ����쳣

				������
				lockcatch_status������״̬

				����ֵ��true��ʾ���쳣��false��ʾû���쳣
				*/
				static bool has_anomaly(
					const LockcatchType::lockcatch_status_t& lockcatch_status
				);
		};

	}
}// end namespace