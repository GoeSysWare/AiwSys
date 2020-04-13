#include "sidewall_net.h"

namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region init and free for SidewallNet
			std::vector<shared_caffe_net_t> SidewallNet::v_net;

			int SidewallNet::input_height = 512;
			int SidewallNet::input_width = 1024;
			int SidewallNet::output_height = 512;
			int SidewallNet::output_width = 1024;

			void SidewallNet::init(
				const caffe_net_file_t& caffe_net_file
			)
			{
				CaffeNet::init(caffe_net_file, net_count, v_net);
			}

			void SidewallNet::free()
			{
				CaffeNet::free(v_net);
			}

			void SidewallNet::set_image_size(int height, int width)
			{
				SidewallNet::input_height = height;
				SidewallNet::input_width = width;
				SidewallNet::output_height = height;
				SidewallNet::output_width = width;
			}

			void SidewallNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				/*
				num_inputs()=1
				num_outputs()=1
				input_blob shape_string:3 2 1024 2048 (12582912)
				output_blob shape_string:3 1 1024 2048 (6291456)
				*/
				// 1-inputs
				CaffeNet::caffe_net_input_t sidewall_caffe_net_input;
				sidewall_caffe_net_input.input_channel = SidewallNet::input_channel;
				sidewall_caffe_net_input.input_height = SidewallNet::input_height;
				sidewall_caffe_net_input.input_width = SidewallNet::input_width;
				//sidewall_caffe_net_input.blob_channel_mat;

				// 1-outputs
				CaffeNet::caffe_net_output_t sidewall_caffe_net_output;
				sidewall_caffe_net_output.output_channel = SidewallNet::output_channel;
				sidewall_caffe_net_output.output_height = SidewallNet::output_height;
				sidewall_caffe_net_output.output_width = SidewallNet::output_width;
				//sidewall_caffe_net_output.blob_channel_mat;

				n_inputs.push_back(sidewall_caffe_net_input); // 1-inputs
				n_outputs.push_back(sidewall_caffe_net_output); // 1-outpus
			}

			
#pragma endregion 

		}
	}
}


