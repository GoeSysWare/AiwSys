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
#include "projects/adas/algorithm/algorithm_shared_export.h" 

// std
#include <string>

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT CaffeApi {

		public:
			/*
			�������ܣ�����caffe��ģʽ

			������
			gpu_mode��trueʹ��GPUģʽ��falseʹ��CPUģʽ
			device_id: GPU��ID�ţ���0��ʼ
			seed: caffe���������������

			����ֵ����
			*/
			static void set_mode(bool gpu_mode, int device_id, unsigned int seed);

			/*
			�������ܣ���ʼ��caffe��log�����ø�������log��·��

			������
			log_name��log������
			log_fatal_prefix: fatal����log��·��ǰ׺
			log_error_prefix: error����log��·��ǰ׺
			log_warning_prefix: warning����log��·��ǰ׺
			log_info_prefix: info����log��·��ǰ׺
			
			����ֵ����
			*/
			static void init_glog(
				const std::string& log_name,
				const std::string& log_fatal_prefix,
				const std::string& log_error_prefix,
				const std::string& log_warning_prefix,
				const std::string& log_info_prefix
			);


			/*
			�������ܣ��ر�caffe��log

			��������

			����ֵ����
			*/
			static void shutdown_glog();
		};

	}
}// end namespace
