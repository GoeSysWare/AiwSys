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

		class SHARED_EXPORT LockcatchType
		{
		public:
			// ���۰������+�Ҳ�2��roi��һ��roi����(��ñ����˿)
			enum LOCKCATCH_ROI_COMPONENT {
				COM_LUOMAO, // ��ñ
				COM_TIESI // ��˿
			};

			enum COMPONENT_STATUS {
				STATUS_NORMAL = 0, //����״̬
				STATUS_PARTLOSE = 1, // ����ȱʧ
				STATUS_MISSING = 2, // ������
			};

			struct SHARED_EXPORT lockcatch_status_t {
				// struct has methods,so we must use SHARED_EXPORT

				bool refine_flag;
				// �����+�Ҳ�ROI�����ڵ�ʱ��refine_flag=true��ʾ��Ҫ��ϸ����������refine_flag=false
				COMPONENT_STATUS left_luomao_status;
				COMPONENT_STATUS left_tiesi_status;
				COMPONENT_STATUS right_luomao_status;
				COMPONENT_STATUS right_tiesi_status;

				bool equals(const LockcatchType::lockcatch_status_t& other);
			};

			struct lockcatch_threshold_t {
				int luomao_exist_min_area;
				int	luomao_missing_min_area;
				int	tiesi_exist_min_area;
				int	tiesi_missing_min_area;
			};

			typedef mat_pair_t lockcatch_mat_pair_t; // 2-pair
		};

	}
}// end namespace