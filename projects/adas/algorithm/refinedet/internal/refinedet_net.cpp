#include "refinedet_net.h"

// caffe
#include <caffe/caffe.hpp>
using namespace caffe;

namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region init and free for RefineDetNet
			std::vector<shared_caffe_net_t> RefineDetNet::v_net;

			int RefineDetNet::input_height = 1024;
			int RefineDetNet::input_width = 1024;

			void RefineDetNet::init(
				const caffe_net_file_t& caffe_net_file
			)
			{
				CaffeNet::init(caffe_net_file, net_count, v_net);
			}

			void RefineDetNet::free()
			{
				CaffeNet::free(v_net);
			}

			void RefineDetNet::set_image_size(int height, int width)
			{
				RefineDetNet::input_height = height;
				RefineDetNet::input_width = width;
			}

			void RefineDetNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				// 1-inputs
				CaffeNet::caffe_net_input_t caffe_net_input;
				caffe_net_input.input_channel = RefineDetNet::input_channel;
				caffe_net_input.input_height = RefineDetNet::input_height;
				caffe_net_input.input_width = RefineDetNet::input_width;
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

			/*
			bool RefineDetNet::forward(
				shared_caffe_net_t m_net,
				const std::vector<cv::Mat>& v_image,
				std::vector<refinedet_outputs_t>& v_output
			)
			{
				int batch_size = v_image.size();

				std::cout << "num_inputs()=" << m_net->num_inputs() << std::endl;
				std::cout << "num_outputs()=" << m_net->num_outputs() << std::endl;


				blob_channel_mat_t blob_channel_mat;

				//transformer = Transformer({ 'data': net.blobs['data'].data.shape })
				//transformer.set_transpose('data', (2, 0, 1))
				//transformer.set_mean('data', np.array([104, 117, 123])) 

				caffe_blob_t* input_blob = m_net->input_blobs()[0];
				input_blob->Reshape(
					batch_size,
					3,
					512,
					512
				);
				std::cout << "input_blob shape_string:" << input_blob->shape_string() << std::endl;
				// forward dimension change to all layers
				m_net->Reshape();

				// for data preprocess
				shared_ptr<caffe::DataTransformer<float>> data_transformer;
				caffe::TransformationParameter params;
				params.set_mean_value(0, 104);
				params.set_mean_value(1, 117);
				params.set_mean_value(2, 123); // bgr 
				// params.set_mean_file("/path/to/image_mean.binaryproto");
				// instantiate a DataTransformer using trans_para for image preprocess
				data_transformer.reset(new caffe::DataTransformer<float>(params, caffe::TEST));
				
				// maybe you need to resize image before this step
				data_transformer->Transform(v_image, m_net->input_blobs()[0]);

				// (3) forward through net to get n output blobs
				m_net->Forward();

				caffe_blob_t* output_blob = m_net->output_blobs()[0];
				std::cout << "output_blob shape_string:" << output_blob->shape_string() << std::endl;

				return true;
			}
			*/

		}
	}
}


