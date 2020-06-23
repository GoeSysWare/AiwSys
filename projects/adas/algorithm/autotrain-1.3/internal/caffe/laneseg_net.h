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
#include "projects/adas/algorithm/algorithm_type.h" 
#include "projects/adas/algorithm/core/caffe/internal/caffe_def.h"
#include "projects/adas/algorithm/core/caffe/internal/caffe_net.h"
namespace watrix {
	namespace algorithm {
		namespace internal {

			class LaneSegNet : public CaffeNet
			{
			public:
				/*
				num_inputs()=1
				num_outputs()=2
				input_blob shape_string:1 3 256 1024 (786432)
				output_blob shape_string:1 8 256 1024 (2097152)
				output_blob shape_string:1 2 256 1024 (524288)
				*/
				static const int input_channel = 3;
				static const int input_height = 256;
				static const int input_width = 1024;

				// mask 
				static int output_channel; // 8-->16
				static const int output_height = 256;
				static const int output_width = 1024;

				// binary
				static const int output_channel2 = 2;
				static const int output_height2 = 256;
				static const int output_width2 = 1024;

			public:
				static void init(
					const caffe_net_file_t& caffe_net_file,
					int feature_dim,
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
