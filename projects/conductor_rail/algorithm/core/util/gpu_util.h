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

#ifdef ENABLE_OPENCV_CUDA

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT GpuUtil
		{
		public:
			static int test_gpu(bool enable_gpu);
			static int gpu_orb_demo();
			static int cpu_orb_demo();

			static int gpu_detect_and_compute(
				const cv::Mat& image,
				keypoints_t& keypoints,
				cv::Mat& descriptor
			);

			static int gpu_match_and_filter(
				const cv::Mat& descriptor_object, 
				const cv::Mat& descriptor_scene,
				const keypoints_t& keypoint_object, 
				const keypoints_t& keypoint_scene,
				const float nn_match_ratio,
				keypoints_t& matched1, 
				keypoints_t& matched2
			);

			static int gpu_orb_and_match(
				const cv::Mat& image_object,
				const cv::Mat& image_scene,
				const float nn_match_ratio,
				keypoints_t& matched1,
				keypoints_t& matched2
			);

		};

	}
}// end namespace

#endif // ENABLE_OPENCV_CUDA