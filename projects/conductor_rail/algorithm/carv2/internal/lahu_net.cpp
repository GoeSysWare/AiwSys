#include "lahu_net.h"

// caffe
#include <caffe/caffe.hpp>
using namespace caffe;

namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region init and free for LahuNet
			std::vector<shared_caffe_net_t> LahuNet::v_net;

			void LahuNet::init(
				const caffe_net_file_t& caffe_net_file,
				int net_count
			)
			{
				CaffeNet::init(caffe_net_file, net_count, v_net);
			}

			void LahuNet::free()
			{
				CaffeNet::free(v_net);
			}

			
			void LahuNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				// 1-inputs
				CaffeNet::caffe_net_input_t caffe_net_input;
				caffe_net_input.input_channel = input_channel;
				caffe_net_input.input_height = input_height;
				caffe_net_input.input_width = input_width;
				//caffe_net_input.blob_channel_mat;

				// 1-outputs
				CaffeNet::caffe_net_output_t caffe_net_output;
				caffe_net_output.output_channel = output_channel;
				caffe_net_output.output_height = output_height;
				caffe_net_output.output_width = output_width;
				//caffe_net_output.blob_channel_mat;
				
				n_inputs.push_back(caffe_net_input); // 1-inputs
				n_outputs.push_back(caffe_net_output); // 1-outpus
			}
#pragma endregion 

		}
	}
}


