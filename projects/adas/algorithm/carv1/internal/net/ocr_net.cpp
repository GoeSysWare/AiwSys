#include "ocr_net.h"

namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region init and free for OcrNet
			std::vector<shared_caffe_net_t> OcrNet::v_net;

			void OcrNet::init(
				const caffe_net_file_t& caffe_net_file
			)
			{
				CaffeNet::init(caffe_net_file, net_count, v_net);
			}

			void OcrNet::free()
			{
				CaffeNet::free(v_net);
			}

			void OcrNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				/*
				num_inputs()=1
				num_outputs()=1
				input_blob shape_string:2 1 512 512 (524288)
				output_blob shape_string:2 1 512 512 (524288)
				*/
				// 1-inputs
				CaffeNet::caffe_net_input_t sidewall_caffe_net_input;
				sidewall_caffe_net_input.input_channel = OcrNet::input_channel;
				sidewall_caffe_net_input.input_height = OcrNet::input_height;
				sidewall_caffe_net_input.input_width = OcrNet::input_width;
				//sidewall_caffe_net_input.blob_channel_mat;

				// 1-outputs
				CaffeNet::caffe_net_output_t sidewall_caffe_net_output;
				sidewall_caffe_net_output.output_channel = OcrNet::output_channel;
				sidewall_caffe_net_output.output_height = OcrNet::output_height;
				sidewall_caffe_net_output.output_width = OcrNet::output_width;
				//sidewall_caffe_net_output.blob_channel_mat;

				n_inputs.push_back(sidewall_caffe_net_input); // 1-inputs
				n_outputs.push_back(sidewall_caffe_net_output); // 1-outpus
			}
#pragma endregion 

		}
	}
}


