#include "topwire_net.h"

#include <boost/date_time/posix_time/posix_time.hpp>  

namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region init and free for TopwireCropNet
			std::vector<shared_caffe_net_t> TopwireCropNet::v_net;

			void TopwireCropNet::init(
				const caffe_net_file_t& caffe_net_file
			)
			{
				CaffeNet::init(caffe_net_file, net_count, v_net);
			}

			void TopwireCropNet::free()
			{
				CaffeNet::free(v_net);
			}

			void TopwireCropNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1
				input_blob shape_string:5 1 128 256 (163840)
				output blob shape_string:5 1 128 256 (163840)
				*/
				// 1-inputs
				CaffeNet::caffe_net_input_t crop_caffe_net_input;
				crop_caffe_net_input.input_channel = input_channel;
				crop_caffe_net_input.input_height = input_height;
				crop_caffe_net_input.input_width = input_width;
				//crop_caffe_net_input.blob_channel_mat;

				// 1-outputs
				CaffeNet::caffe_net_output_t crop_caffe_net_output;
				crop_caffe_net_output.output_channel = output_channel;
				crop_caffe_net_output.output_height = output_height;
				crop_caffe_net_output.output_width = output_width;
				//crop_caffe_net_output.blob_channel_mat;

				n_inputs.push_back(crop_caffe_net_input); // 1-inputs
				n_outputs.push_back(crop_caffe_net_output); // 1-outpus
			}

#pragma endregion 

#pragma region init and free for TopwireDetectNet
			std::vector<shared_caffe_net_t> TopwireDetectNet::v_net;

			void TopwireDetectNet::init(
				const caffe_net_file_t& caffe_net_file
			)
			{
				CaffeNet::init(caffe_net_file, net_count, v_net);
			}

			void TopwireDetectNet::free()
			{
				CaffeNet::free(v_net);
			}	

			void TopwireDetectNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				/*
				num_inputs()=2   // data_1,data_2
				num_outputs()=2 // score_1, score_2
				input_blob1, shape_string : 5 1 512 128 (65536)
				input_blob2, shape_string : 5 1 512 128 (65536)
				output_blob1, shape_string : 5 3 512 128 (196608)
				output_blob2, shape_string : 5 1 512 128 (65536)
				*/
				//==============================================================================
				// 2-inputs
				CaffeNet::caffe_net_input_t detect_caffe_net_input;
				detect_caffe_net_input.input_channel = input_channel;
				detect_caffe_net_input.input_height = input_height;
				detect_caffe_net_input.input_width = input_width;
				//detect_caffe_net_input.blob_channel_mat;

				CaffeNet::caffe_net_input_t detect_caffe_net_input2;
				detect_caffe_net_input2.input_channel = input_channel2;
				detect_caffe_net_input2.input_height = input_height2;
				detect_caffe_net_input2.input_width = input_width2;
				//detect_caffe_net_input2.blob_channel_mat;

				// 2-outputs
				CaffeNet::caffe_net_output_t detect_caffe_net_output;
				detect_caffe_net_output.output_channel = output_channel;
				detect_caffe_net_output.output_height = output_height;
				detect_caffe_net_output.output_width = output_width;
				//detect_caffe_net_output.blob_channel_mat;

				CaffeNet::caffe_net_output_t detect_caffe_net_output2;
				detect_caffe_net_output2.output_channel = output_channel2;
				detect_caffe_net_output2.output_height = output_height2;
				detect_caffe_net_output2.output_width = output_width2;
				//detect_caffe_net_output2.blob_channel_mat;

				n_inputs.push_back(detect_caffe_net_input);
				n_inputs.push_back(detect_caffe_net_input2); // 2-inputs

				n_outputs.push_back(detect_caffe_net_output);
				n_outputs.push_back(detect_caffe_net_output2); // 2-outputs
			}
#pragma endregion 

		}
	}
}


