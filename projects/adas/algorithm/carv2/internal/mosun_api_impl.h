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
#include "algorithm_type.h" //caffe_net_file_t,  Mat, Keypoint, box_t, refinedet_boxs_t
#include "algorithm/core/caffe/internal/caffe_def.h"


//==================================================
// torch
#include <torch/script.h> // One-stop header.
#include <torch/torch.h> // One-stop header.
//#include <ATen/Tensor.h>
//==================================================


namespace watrix {
	namespace algorithm {
		namespace internal{

			typedef torch::jit::script::Module pt_module_t;

			class  MosunApiImpl
			{
			protected:
				static MosunParam params;
				static std::vector<pt_module_t> v_net;

				static int m_counter;

				static const int ORIGIN_HEIGHT = 964;
				static const int ORIGIN_WIDTH = 1288;
				static const int INPUT_HEIGHT = 384;
				static const int INPUT_WIDTH = 512;  

			public:
				static void init(
					const MosunParam& params,
					int net_count
				);
				static void free();

				
				static MosunResult detect(
					int net_id,
					const cv::Mat& image
				);

			protected:

				static bool seg(
					int net_id,
					const cv::Mat& image,
					cv::Mat& binary_mask, 
					cv::Mat& binary3_mask 
				);
			};


		}
	}
}// end namespace