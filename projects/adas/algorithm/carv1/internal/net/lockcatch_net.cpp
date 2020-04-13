#include "lockcatch_net.h"

namespace watrix {
	namespace algorithm {
		namespace internal {


#pragma region init and free for CropNet
			std::vector<shared_caffe_net_t> LockcatchCropNet::v_net;

			void LockcatchCropNet::init(
				const caffe_net_file_t& caffe_net_file
			)
			{
				CaffeNet::init(caffe_net_file, net_count, v_net);
			}

			void LockcatchCropNet::free()
			{
				CaffeNet::free(v_net);
			}

			void LockcatchCropNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1
				input_blob 1 1 256 256 (65536)
				output_blob 1 3 256 256 (196608)
				*/
				// 1-inputs
				CaffeNet::caffe_net_input_t crop_caffe_net_input;
				crop_caffe_net_input.input_channel = LockcatchCropNet::input_channel;
				crop_caffe_net_input.input_height = LockcatchCropNet::input_height;
				crop_caffe_net_input.input_width = LockcatchCropNet::input_width;
				//crop_caffe_net_input.blob_channel_mat;

				// 1-outputs
				CaffeNet::caffe_net_output_t crop_caffe_net_output;
				crop_caffe_net_output.output_channel = LockcatchCropNet::output_channel;
				crop_caffe_net_output.output_height = LockcatchCropNet::output_height;
				crop_caffe_net_output.output_width = LockcatchCropNet::output_width;
				//crop_caffe_net_output.blob_channel_mat;

				n_inputs.push_back(crop_caffe_net_input); // 1-inputs
				n_outputs.push_back(crop_caffe_net_output); // 1-outpus
			}
#pragma endregion 


#pragma region init and free for DetectNet
			std::vector<shared_caffe_net_t> LockcatchRefineNet::v_net;

			void LockcatchRefineNet::init(
				const caffe_net_file_t& caffe_net_file
			)
			{
				CaffeNet::init(caffe_net_file, net_count, v_net);
			}

			void LockcatchRefineNet::free()
			{
				CaffeNet::free(v_net);
			}

			void LockcatchRefineNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1

				input_blob  2 1 256 256 (131072)
				output_blob 2 4 256 256 (524288)
				*/
				// 1-inputs
				CaffeNet::caffe_net_input_t refine_caffe_net_input;
				refine_caffe_net_input.input_channel = LockcatchRefineNet::input_channel;
				refine_caffe_net_input.input_height = LockcatchRefineNet::input_height;
				refine_caffe_net_input.input_width = LockcatchRefineNet::input_width;
				//refine_caffe_net_input.blob_channel_mat;

				// 1-outputs
				CaffeNet::caffe_net_output_t refine_caffe_net_output;
				refine_caffe_net_output.output_channel = LockcatchRefineNet::output_channel;
				refine_caffe_net_output.output_height = LockcatchRefineNet::output_height;
				refine_caffe_net_output.output_width = LockcatchRefineNet::output_width;
				//refine_caffe_net_output.blob_channel_mat;

				n_inputs.push_back(refine_caffe_net_input); // 1-inputs
				n_outputs.push_back(refine_caffe_net_output); // 1-outpus
			}
#pragma endregion 


		}
	}
}


