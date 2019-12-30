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

#include "caffe_def.h" // caffe_net_t, caffe_blob_t, 

namespace watrix {
	namespace algorithm {
		namespace internal {

			class  CaffeNet {
				// ================================================
				// data model
				// ================================================
			public:
				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1
				input_blob shape_string:5 1 128 256 (163840)
				blob shape_string:5 1 128 256 (163840)
				*/

				/*
				num_inputs()=2   // data_1,data_2
				num_outputs()=2 // score_1, score_2
				input_blob1, shape_string : 5 1 512 128 (65536)
				input_blob2, shape_string : 5 1 512 128 (65536)
				output_blob1, shape_string : 5 3 512 128 (196608)
				output_blob2, shape_string : 5 1 512 128 (65536)
				*/

				struct caffe_net_input_t {
					int input_channel;
					int input_height;
					int input_width;
					blob_channel_mat_t blob_channel_mat;
				};

				struct caffe_net_output_t {
					int output_channel;
					int output_height;
					int output_width;
					blob_channel_mat_t blob_channel_mat;
				};

				typedef std::vector<caffe_net_input_t> caffe_net_n_inputs_t;
				typedef std::vector<caffe_net_output_t> caffe_net_n_outputs_t;

			public:

				static void init(
					const caffe_net_file_t& caffe_net_file,
					const int net_count,
					std::vector<shared_caffe_net_t>& v_net
				);

				static void free(std::vector<shared_caffe_net_t>& v_net);

				static void get_inputs_outputs(
					CaffeNet::caffe_net_n_inputs_t& n_inputs,
					CaffeNet::caffe_net_n_outputs_t& n_outputs
				); // sub-class to implement

				static bool forward(
					shared_caffe_net_t m_net,
					caffe_net_n_inputs_t& n_inputs,
					caffe_net_n_outputs_t& n_outputs,
					bool get_float_output = false
				);

			protected:
				static void set_input(
					blob_channel_mat_t& blob_channel_mat,
					caffe_blob_t& blob
				);

				
				static void get_output_uint8(
					caffe_blob_t& blob,
					blob_channel_mat_t& blob_channel_mat
				);

				static void get_output_float(
					caffe_blob_t& blob,
					blob_channel_mat_t& blob_channel_mat
				);

				static void get_output(
					caffe_blob_t& blob,
					blob_channel_mat_t& blob_channel_mat,
					bool keep_float = false
				);


			};

		}
	}
}// end namespace
