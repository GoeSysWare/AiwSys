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
#include "algorithm/core/caffe/internal/caffe_net.h"

namespace watrix {
	namespace algorithm {
		namespace internal {

			class LahuNet : public CaffeNet
			{
			public:

				/*
				input_height=964
				input_width=1288

				# data ---> prob 
				# input N,1,256,256 (chw)
				# output N,2,1,1
				*/
				static const int input_channel = 1;
				static const int input_height = 300; 
				static const int input_width = 300;

				// feature  2,1,1
				static const int output_channel = 2;
				static const int output_height = 1;
				static const int output_width = 1;

			public:
				static void init(
					const caffe_net_file_t& caffe_net_file,
					int net_count
				);
				static void free();

				static std::vector<shared_caffe_net_t> v_net;
				
				static void get_inputs_outputs(
					CaffeNet::caffe_net_n_inputs_t& n_inputs,
					CaffeNet::caffe_net_n_outputs_t& n_outputs
				); 

			};

		}
	}
} // end of namespace
