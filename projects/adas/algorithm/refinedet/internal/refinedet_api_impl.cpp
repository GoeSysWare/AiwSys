#include "refinedet_api_impl.h"
#include "refinedet_net.h"

#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

namespace watrix {
	namespace algorithm {
		namespace internal {

			int RefineDetApiImpl::m_counter = 0; // init
			std::vector<float> RefineDetApiImpl::m_bgr_mean; // init (keystep, otherwise linker error)

			int RefineDetApiImpl::class_count_ = 2; // by default 2 (background,person)
			std::vector<std::string> RefineDetApiImpl::class_labels_;

			void RefineDetApiImpl::init_class_labels()
			{
				class_labels_.push_back("background");
				class_labels_.push_back("person");

				class_count_ = (int)class_labels_.size();
			}

			void RefineDetApiImpl::init(
				const caffe_net_file_t& detect_net_params
			)
			{
				RefineDetNet::init(detect_net_params);
				init_class_labels();
			}

			void RefineDetApiImpl::free()
			{
				RefineDetNet::free();
			}

			void RefineDetApiImpl::set_image_size(int height, int width)
			{
				RefineDetNet::set_image_size(height,width);
				LOG(INFO) << "[API-RefineDet] set_image_size to (height,width)= (" << height <<","<< width << ")" << std::endl;
			}

			void RefineDetApiImpl::set_bgr_mean(const std::vector<float>& bgr_mean)
			{
				for (size_t i = 0; i < bgr_mean.size(); i++)
				{
					RefineDetApiImpl::m_bgr_mean.push_back(bgr_mean[i]);
				}
			}

			bool RefineDetApiImpl::detect(
				const int& net_id,
				const std::vector<cv::Mat>& v_image,
				float threshold,
				std::vector<detection_boxs_t>& v_output
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, 0) << "net_id invalid";
				shared_caffe_net_t refinedet_net = RefineDetNet::v_net[net_id];
				return RefineDetApiImpl::detect(
					refinedet_net,
					v_image,
					threshold,
					v_output
				);
			}

			bool RefineDetApiImpl::detect(
				shared_caffe_net_t refinedet_net,
				const std::vector<cv::Mat>& v_image, // bgr image
				float threshold,
				std::vector<detection_boxs_t>& v_output
			)
			{

#ifdef DEBUG_TIME
				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::ptime pt2;
				int64_t cost;
#endif // DEBUG_TIME

				int batch_size = v_image.size();
				int image_width = v_image[0].cols;
				int image_height = v_image[0].rows;
				for (size_t i = 0; i < v_image.size(); i++)
				{
					CHECK(!v_image[i].empty()) << "invalid mat";
					CHECK(v_image[i].channels()==3) << "mat channels must ==3";
				}

				// ============================================================
				int input_width = RefineDetNet::input_width;
				int input_height = RefineDetNet::input_height;
				cv::Size input_size(input_width, input_height);
				// ============================================================

				std::vector<cv::Mat> v_resized_input; // resized image
				
				for (size_t i = 0; i < v_image.size(); i++)
				{
					cv::Mat float_image;
					v_image[i].convertTo(float_image, CV_32FC3);

					cv::Mat resized_image;
					cv::resize(float_image, resized_image, input_size); // 512,512,3

					//std::cout<<"resized_image.channels()= "<< resized_image.channels() << std::endl; // 3
					//std::cout<<"resized_image.size()=" << resized_image.size() << std::endl; // [512 x 512]
					//std::cout<<"resized_image.type()=" << resized_image.type() << std::endl; // CV_32FC3=21

					/*
					std::cout << "================resize===================\n";
					for (int row = 0; row <= 5; row++) {
						for (int col = 0; col <= 5; col++) {
							std::cout << resized_image.at<cv::Vec3f>(row, col)[0] << " ";
						}
						std::cout << std::endl; // 80.0322 73.0322 70.0322 75.4189 68.4189 65.4189
					}
					std::cout << "=================resize==================\n";
					*/

					v_resized_input.push_back(resized_image);
				}

				CaffeNet::caffe_net_n_inputs_t n_inputs;
				CaffeNet::caffe_net_n_outputs_t n_outputs;
				RefineDetNet::get_inputs_outputs(n_inputs, n_outputs); // 1-inputs,1-outputs

				for (size_t i = 0; i < batch_size; i++)
				{
					channel_mat_t channel_mat;
					OpencvUtil::bgr_subtract_mean(
						v_resized_input[i], 
						m_bgr_mean, 
						1.0, // no scale
						channel_mat
					);

					/*
					for(int c=0; c<3; c++){
						// bgr  512,512,3 float
						cv::Mat c_mat = OpencvUtil::get_float_channel_mat(v_resized_input[i], c);
						OpencvUtil::mat_subtract_value(c_mat, bgr_mean[c]); // image -= mean
						channel_mat.push_back(c_mat); // b, g, r
					}
					*/
				
					n_inputs[0].blob_channel_mat.push_back(channel_mat);
				}

				internal::RefineDetNet::forward(
					refinedet_net,
					n_inputs,
					n_outputs,
					true // get float output
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-SIDEWALL] [3] forward net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				//========================================================================
				// very special output here, NOT [N,1,500,7] but [1,1,500*N,7] 
				//========================================================================
				cv::Mat float_box_mat = n_outputs[0].blob_channel_mat[0][0]; // 1,1,500*N,7 

				//std::cout<<"box_mat.channels()= "<< float_box_mat.channels() << std::endl; // 1
				//std::cout<<"box_mat.size()=" << float_box_mat.size() << std::endl; // [7 x 500] w h
				//std::cout<<"box_mat.type()=" << float_box_mat.type() << std::endl; // CV_32FC1=5

				LOG(INFO)<<"box_mat.size()=" << float_box_mat.size() << std::endl; // [7 x 1000] w h

				get_detection_boxs(
					image_height,
					image_width,
					float_box_mat,
					threshold,
					v_output
				);

				//std::cout << "detection_boxs.size()= " << detection_boxs.size() << std::endl;
				LOG(INFO) << "v_output.size()= " << v_output.size() << std::endl;

				m_counter++;

				return true;
			}


			void RefineDetApiImpl::get_detection_boxs(
					int height, int width,
					const cv::Mat& float_box_mat,
					float threshold,
					std::vector<detection_boxs_t>& v_output
			)
			{
				/*
				box_mat: 500*N,7  CV_32FC1
				*/
				int box_count_per_image = internal::RefineDetNet::output_height; // 500 
				int batch_size = float_box_mat.rows / box_count_per_image; // 2
				/*
				std::cout<<" box_count_per_image = "<<box_count_per_image<<std::endl;
				std::cout<<" batch_size = "<<batch_size<<std::endl;
				*/

				for(int b=0; b< batch_size; b++) {
					int start = b * box_count_per_image;
					int end = (b+1) * box_count_per_image;

					/*
					std::cout<<" start = "<<start<<std::endl;
					std::cout<<" end = "<<end<<std::endl;
					*/

					detection_boxs_t detection_boxs;
					for (int row = start; row < end; row++)
					{
						const float *p = float_box_mat.ptr<float>(row);
						// p [0,1,2,3,4,5,6,7]  (_, class_index, confidence, xmin,ymin,xmax,ymax) 
						
						int class_index = int(p[1]); // 1 person 0 background
						float confidence = p[2]; // 0-1
						float xmin = p[3]; // 0-1
						float ymin = p[4]; // 0-1
						float xmax = p[5]; // 0-1
						float ymax = p[6]; // 0-1
						
						//std::cout<<" class_index, confidence = "<< class_index <<","<< confidence << std::endl;

						if (class_index> 0 && confidence >= threshold) {
							detection_box_t detection_box{ 
								int(xmin*width),
								int(ymin*height),
								int(xmax*width),
								int(ymax*height),
								confidence,
								class_index,
								class_labels_[class_index]
							};

							detection_boxs.push_back(detection_box);
						}
					}
					v_output.push_back(detection_boxs);
				}
			}

		}
	}
}// end namespace

