#include "caffe_net.h"

// glog
#include <glog/logging.h>

using namespace caffe;
using namespace std;

//#define RELEASE_DEBUG

namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region init and free nets

			void CaffeNet::init(
				const caffe_net_file_t& caffe_net_file,
				const int net_count,
				std::vector<shared_caffe_net_t>& v_net
			)
			{
				enum caffe::Phase phase = caffe::Phase::TEST;

				v_net.resize(net_count);
				for (int i = 0; i < v_net.size(); i++)
				{
					v_net[i] = shared_caffe_net_t(
						new caffe_net_t(caffe_net_file.deploy_file, phase)
					);
					v_net[i]->CopyTrainedLayersFrom(caffe_net_file.model_file);
				};
			}

			void CaffeNet::free(std::vector<shared_caffe_net_t>& v_net)
			{
				//LOG(INFO)<<"[API]  CaffeNet::free \n";
				for (int i = 0; i < v_net.size(); i++)
				{
					v_net[i] = nullptr;
				}
			}
#pragma endregion

#pragma region forward

			void CaffeNet::get_inputs_outputs(
				CaffeNet::caffe_net_n_inputs_t& n_inputs,
				CaffeNet::caffe_net_n_outputs_t& n_outputs
			)
			{
				// sub-class to implement
			}

			bool CaffeNet::forward(
				shared_caffe_net_t m_net,
				caffe_net_n_inputs_t& n_inputs,
				caffe_net_n_outputs_t& n_outputs,
				bool get_float_output
			)
			{
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

				/*
				num_inputs()=1
				num_outputs()=2
				input_blob shape_string:1 1 1024 512 (524288)
				output_blob shape_string:1 1 1024 512 (524288)
				output_blob shape_string:1 1 1024 512 (524288)
				*/

#ifdef DEBUG_TIME
				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::ptime pt2;
				int64_t cost;
#endif // DEBUG_TIME

#ifdef RELEASE_DEBUG
				//LOG(INFO)<<"[API] =============================================\n";
				//LOG(INFO)<<"[API] CaffeNet::forward 11111111111111111111111111111111111111111111\n";
				//LOG(INFO)<<"[API] =============================================\n";
#endif // RELEASE_DEBUG

				CHECK_EQ(m_net->num_inputs(), n_inputs.size()) <<"m_net->num_inputs() must ="<< n_inputs.size(); 
				CHECK_EQ(m_net->num_outputs(), n_outputs.size()) << "m_net->num_outputs() must =" << n_outputs.size();

				int input_n = n_inputs.size();
				int output_n = n_outputs.size();

#ifdef RELEASE_DEBUG
				LOG(INFO)<<"num_inputs()=" << m_net->num_inputs() << std::endl;
				LOG(INFO)<<"num_outputs()=" << m_net->num_outputs() << std::endl;
#endif // RELEASE_DEBUG

				// (1) reshape for n input blob
				for (int i = 0; i < input_n; i++)
				{
					caffe_blob_t* input_blob = m_net->input_blobs()[i];
					input_blob->Reshape(
						n_inputs[i].blob_channel_mat.size(),
						n_inputs[i].input_channel,
						n_inputs[i].input_height,
						n_inputs[i].input_width
					);
					//LOG(INFO)<<"[API] n_inputs[i].input_channel="<< n_inputs[i].input_channel << std::endl;
					//LOG(INFO)<<"[API] n_inputs[i].input_height="<< n_inputs[i].input_height << std::endl;
					//LOG(INFO)<<"[API] n_inputs[i].input_width="<< n_inputs[i].input_width << std::endl;


					//std::cout<<"input_blob shape_string:" << input_blob->shape_string() << std::endl;
#ifdef RELEASE_DEBUG
					LOG(INFO)<<"input_blob shape_string:" << input_blob->shape_string() << std::endl;
					//LOG(INFO) << "[API] CaffeNet::forward 2222222222 before Reshape \n";
#endif // 
				}

				// forward dimension change to all layers
				m_net->Reshape(); 

				// (2) set input data for n blob
				for (int i = 0; i < input_n; i++)
				{
					caffe_blob_t* input_blob = m_net->input_blobs()[i];

#ifdef RELEASE_DEBUG
					//LOG(INFO) << "[API] CaffeNet::forward 444444444444444444 before set_input \n";
#endif // shared_DEBUG

					set_input(n_inputs[i].blob_channel_mat, *input_blob);
				
#ifdef RELEASE_DEBUG
					//LOG(INFO) << "[API] CaffeNet::forward 555555555555555555 after set_input \n";
#endif // shared_DEBUG
				}

#ifdef RELEASE_DEBUG
				//LOG(INFO) << "[API] CaffeNet::forward 6666666666 before Forward \n";
#endif // shared_DEBUG

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				//LOG(INFO) << "[API-CAFFE] [1] set_input: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


				// (3) forward through net to get n output blobs
				m_net->Forward();

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				//LOG(INFO) << "[API-CAFFE] [2] forward : cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


#ifdef RELEASE_DEBUG
				//LOG(INFO) << "[API] CaffeNet::forward 7777777777 after Forward \n";
#endif // shared_DEBUG
				
				// (4) get output data for n blob
				for (int i = 0; i < output_n; i++)
				{
					caffe_blob_t* output_blob = m_net->output_blobs()[i];

					//LOG(INFO)<<"[API] n_outputs[i].output_channel=" << n_outputs[i].output_channel << std::endl;
					//LOG(INFO)<<"[API] n_outputs[i].output_height=" << n_outputs[i].output_height << std::endl;
					//LOG(INFO)<<"[API] n_outputs[i].output_width=" << n_outputs[i].output_width << std::endl;

					//CHECK_EQ( (output_blob->channels()), n_outputs[i].output_channel)<< "outpub_blob->channels() must =" << n_outputs[i].output_channel;

					//std::cout<<"output_blob shape_string:" << output_blob->shape_string() << std::endl;
#ifdef RELEASE_DEBUG
					LOG(INFO)<<"output_blob shape_string:" << output_blob->shape_string() << std::endl;
					//LOG(INFO) << "[API] CaffeNet::forward 88888888888888 before get_output \n";
#endif // shared_DEBUG

					get_output(*output_blob, n_outputs[i].blob_channel_mat, get_float_output);

#ifdef RELEASE_DEBUG
					//LOG(INFO) << "[API] CaffeNet::forward 9999999999999999 after get_output \n";
#endif // shared_DEBUG
					
#ifdef DEBUG_TIME
					pt2 = boost::posix_time::microsec_clock::local_time();

					cost = (pt2 - pt1).total_milliseconds();
					//LOG(INFO) << "[API-CAFFE] [3] get_output: cost=" << cost*1.0 << std::endl;

					pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				}

#ifdef RELEASE_DEBUG
				//LOG(INFO) << "[API] =============================================\n";
				//LOG(INFO) << "[API] CaffeNet::forward finished.\n";
				//LOG(INFO) << "[API] =============================================\n";
#endif // shared_DEBUG

				return true;
			}

#pragma endregion


#pragma region set input

			/*
			void CaffeNet::set_double_input(
				const std::vector<channel_mat_t>& v_input1,
				const std::vector<channel_mat_t>& v_input2,
				caffe_blob_t& blob
			)
			{
				int num = blob.num();
				int channels = blob.channels();
				//(channels == 2);
				int height = blob.height();
				int width = blob.width();

				float* dst_data = blob.mutable_cpu_data();
				for (int n = 0; n < num; ++n)
				{
					const cv::Mat& first_image = v_input1[n][0];
					const cv::Mat& second_image = v_input2[n][0];

					for (int c = 0; c < channels; ++c)
					{
						if (c == 0)
						{
							for (int h = 0; h < height; ++h)
							{
								for (int w = 0; w < width; ++w)
								{
									*(dst_data++) = static_cast<float>(first_image.at<uchar>(h, w));
								}
							}
						}
						else if (c == 1) {
							for (int h = 0; h < height; ++h)
							{
								for (int w = 0; w < width; ++w)
								{
									*(dst_data++) = static_cast<float>(second_image.at<uchar>(h, w));
								}
							}
						}

					}
				}

			}
			*/

			void CaffeNet::set_input(
				blob_channel_mat_t& blob_channel_mat,
				caffe_blob_t& blob
			)
			{
				//LOG(INFO)<<"[API] =================================\n";
				//LOG(INFO)<<"[API] set_input\n";
				//LOG(INFO)<<"[API] shape_string:"<< blob.shape_string()<<std::endl;
				//LOG(INFO)<<"[API] =================================\n";

				int batch_size = blob.num();
				int channels = blob.channels();
				int height = blob.height();
				int width = blob.width();
				int count = blob.count(); // n*c*w*w

				float* blob_data = blob.mutable_cpu_data();

				for (int b = 0; b < batch_size; b++) {
					channel_mat_t& channel_mat = blob_channel_mat[b];
					
					CHECK_EQ((channel_mat.size()), channels) << "channel_mat.size() must =" << channels;
					for (int c = 0; c < channels; ++c)
					{
						//cv::Mat float_mat;
						//channel_mat[c].convertTo(float_mat, CV_32FC1);

						cv::Mat& mat = channel_mat[c];
						mat.convertTo(mat, CV_32FC1); // faster

						float* ptr;
						for (int h = 0; h < height; ++h)
						{
							ptr = mat.ptr<float>(h); // row ptr
							for (int w = 0; w < width; ++w)
							{
								// *(blob_data++)= float_mat.at<float>(h, w); // at cost time,so remove 
								*blob_data++ = *ptr++;
							}
						}
					}
				}
			}

#pragma endregion

		//! Note: data_ptr指向已经处理好（去均值的，符合网络输入图像的长宽和Batch Size）的数据
		void caffe_forward(boost::shared_ptr< Net<float> > & net, float *data_ptr)
		{
			Blob<float>* input_blobs = net->input_blobs()[0];
			switch (Caffe::mode())
			{
			case Caffe::CPU:
				memcpy(input_blobs->mutable_cpu_data(), data_ptr,
					sizeof(float) * input_blobs->count());
				break;
			case Caffe::GPU:
				cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,
					sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
				break;
			default:
				LOG(FATAL) << "Unknown Caffe mode.";
			} 
			net->ForwardPrefilled();
		}



#pragma region get output

			void CaffeNet::get_output_uint8(
				caffe_blob_t& blob,
				blob_channel_mat_t& blob_channel_mat
			)
			{
				//LOG(INFO)<<"[API] =================================\n";
				//LOG(INFO)<<"[API] get_output\n";
				//LOG(INFO)<<"[API] =================================\n";

				//input_layer  shape_string:5 3 128 256 (32768)
				//output_layer shape_string:5 3 128 256 (32768)
				float* blob_data = blob.mutable_cpu_data();

				int batch_size = blob.num();
				int channels = blob.channels();
				int output_mat_type = CV_8UC1;
				int height = blob.height();
				int width = blob.width();

				/*
				std::cout<<"[API] =================================\n";
				std::cout<<"[API] get_output\n";
				std::cout<<"[API] =================================\n";
				std::cout<<"height = "<< height << std::endl;
				std::cout<<"width = "<< width << std::endl;
				*/

				//float* ptr = blob_data;
				for (int b = 0; b < batch_size; b++)
				{
					channel_mat_t channel_mat;
					for (int c = 0; c < channels; ++c)
					{
						cv::Mat mat(height, width, output_mat_type);

						for (int h = 0; h < mat.rows; h++)
						{
							uchar *p = mat.ptr<uchar>(h);
							for (int w = 0; w < mat.cols; w++)
							{
								//float f_value = blob_data[blob.offset(b, c, h, w)]; // offset cost time, so remove
								float f_value = *blob_data++;
								*p++ = static_cast<uchar>(f_value * 255); // [0,1] ===>[0,255]
							}
						}
						channel_mat.push_back(mat);
					}

					blob_channel_mat.push_back(channel_mat);
				}
			}

			void CaffeNet::get_output_float(
				caffe_blob_t& blob,
				blob_channel_mat_t& blob_channel_mat
			)
			{
				//input_layer  shape_string:5 3 128 256 (32768)
				//output_layer shape_string:5 3 128 256 (32768)
				float* blob_data = blob.mutable_cpu_data();

				int batch_size = blob.num();
				int channels = blob.channels();
				int output_mat_type = CV_32FC1;
				int height = blob.height();
				int width = blob.width();

				/*
				std::cout<<"[API] =================================\n";
				std::cout<<"[API] get_output float\n";
				std::cout<<"[API] =================================\n";
				std::cout<<"height = "<< height << std::endl;
				std::cout<<"width = "<< width << std::endl;
				*/
				

				//float* ptr = blob_data;
				for (int b = 0; b < batch_size; b++)
				{
					channel_mat_t channel_mat;
					for (int c = 0; c < channels; ++c)
					{
						cv::Mat mat(height, width, output_mat_type);

						for (int h = 0; h < mat.rows; h++)
						{
							float *p = mat.ptr<float>(h);
							for (int w = 0; w < mat.cols; w++)
							{
								//float f_value = blob_data[blob.offset(b, c, h, w)]; // offset cost time, so remove
								float f_value = *blob_data++;
								*p++ = f_value; // [0,1]  keep float value
							}
						}
						channel_mat.push_back(mat);
					}

					blob_channel_mat.push_back(channel_mat);
				}
			}


			void CaffeNet::get_output(
				caffe_blob_t& blob,
				blob_channel_mat_t& blob_channel_mat,
				bool keep_float
			)
			{
				if (keep_float){
					get_output_float(blob, blob_channel_mat);
				} else {
					get_output_uint8(blob, blob_channel_mat);
				}
			}


#pragma endregion


			/*
			void CaffeNet::__get_output_1_channel(
			caffe_blob_t& blob,
			std::vector<cv::Mat>& v_output
			)
			{
			//input_layer  shape_string:5 1 128 256 (32768)
			//output_layer shape_string:5 1 128 256 (32768)

			float* output_data = blob.mutable_cpu_data();

			int num = blob.num();
			int channels = blob.channels();
			//(channels == 1);
			int output_mat_type = CV_8UC1;
			int height = blob.height();
			int width = blob.width();

			for (int n = 0; n < num; n++)
			{
			cv::Mat result_mat(height, width, output_mat_type);

			for (int c = 0; c < channels; ++c)
			{
			for (int h = 0; h < result_mat.rows; h++)
			{
			uchar *p = result_mat.ptr<uchar>(h);
			for (int w = 0; w < result_mat.cols; w++)
			{
			float f_value = output_data[blob.offset(n, c, h, w)]; // [0,1]
			p[w] = static_cast<uchar>(f_value * 255);
			}
			}
			}

			v_output.push_back(result_mat);
			}

			}

			void CaffeNet::__get_output_3_channel(
			caffe_blob_t& blob,
			std::vector<cv::Mat>& v_output
			)
			{
			//input_layer  shape_string:5 3 128 256 (32768)
			//output_layer shape_string:5 3 128 256 (32768)

			float* output_data = blob.mutable_cpu_data();

			int num = blob.num();
			int channels = blob.channels();
			//(channels == 3); // 3 or 4
			int output_mat_type = CV_8UC(channels);
			int height = blob.height();
			int width = blob.width();

			for (int n = 0; n < num; n++)
			{
			cv::Mat result_mat(height, width, output_mat_type);

			for (int c = 0; c < channels; ++c)
			{
			for (int h = 0; h < result_mat.rows; h++)
			{
			cv::Vec3b *p = result_mat.ptr<cv::Vec3b>(h);
			for (int w = 0; w < result_mat.cols; w++)
			{
			float f_value = output_data[blob.offset(n, c, h, w)]; // [0,1]
			p[w][c] = static_cast<uchar>(f_value * 255);
			}
			}
			}

			v_output.push_back(result_mat);
			}
			}

			void CaffeNet::__get_output_4_channel(
			caffe_blob_t& blob,
			std::vector<cv::Mat>& v_output
			)
			{
			// not work as expected ???
			//input_layer  shape_string:5 3 128 256 (32768)
			//output_layer shape_string:5 3 128 256 (32768)
			LOG(INFO)<<"[API] __get_output_4_channel\n" << std::endl;
			float* output_data = blob.mutable_cpu_data();

			int num = blob.num();
			int channels = blob.channels();
			//(channels == 4); // 3 or 4
			int output_mat_type = CV_8UC(channels);
			int height = blob.height();
			int width = blob.width();

			for (int n = 0; n < num; n++)
			{
			cv::Mat result_mat(height, width, output_mat_type);

			for (int c = 0; c < channels; ++c)
			{
			for (int h = 0; h < result_mat.rows; h++)
			{
			cv::Vec4b *p = result_mat.ptr<cv::Vec4b>(h);
			for (int w = 0; w < result_mat.cols; w++)
			{
			float f_value = output_data[blob.offset(n, c, h, w)]; // [0,1]
			p[w][c] = static_cast<uchar>(f_value * 255);
			}
			}
			}

			v_output.push_back(result_mat);
			}
			}
			*/


		}
	}
}// end namespace


 /*
 (1)获取net的blob name
 //==============================================================
 const vector<string>& blob_names = m_net->blob_names();
 for (unsigned int i = 0; i != blob_names.size(); ++i)
 {
 std::cout << i << ", blob name=" << blob_names[i] << std::endl;
 }
 //==============================================================

 (2)根据name获取对应的blob
 boost::shared_ptr<Blob<float> > input_blob = m_net->blob_by_name("data");

 (3)获取input,output blob

 LOG(INFO)<<"[API] num_inputs()=" << m_net->num_inputs() << std::endl; // data_1,data_2
 LOG(INFO)<<"[API] num_outputs()=" << m_net->num_outputs() << std::endl; // score_1, score_2

 Blob<float>* input_blob = m_net->input_blobs()[0];
 Blob<float>* blob = m_net->output_blobs()[0];

 (4)输出blob shape_string
 LOG(INFO)<<"[API] shape_string:" << input_blob->shape_string()<< std::endl;

 LOG(INFO)<<"[API] num:" << input_blob->num()
 << " channels:" << input_blob->channels()
 << " width:" << input_blob->width()
 << " height:" << input_blob->height()
 << std::endl;


 */