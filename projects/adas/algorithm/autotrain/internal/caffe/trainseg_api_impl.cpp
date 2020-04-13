#include "trainseg_api_impl.h"
#include "trainseg_net.h"

#include "projects/adas/algorithm/core/util/display_util.h"
#include "projects/adas/algorithm/core/util/opencv_util.h"
#include "projects/adas/algorithm/core/util/numpy_util.h"

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

namespace watrix {
	namespace algorithm {
		namespace internal {

			int TrainSegApiImpl::m_counter = 0; // init
			std::vector<float> TrainSegApiImpl::m_bgr_mean; // init (keystep, otherwise linker error)

			void TrainSegApiImpl::init(
				const caffe_net_file_t& detect_net_params,
				int net_count
			)
			{
				TrainSegNet::init(detect_net_params,net_count);
			}

			void TrainSegApiImpl::free()
			{
				TrainSegNet::free();
			}

			void TrainSegApiImpl::set_bgr_mean(const std::vector<float>& bgr_mean)
			{
				for (size_t i = 0; i < bgr_mean.size(); i++)
				{
					TrainSegApiImpl::m_bgr_mean.push_back(bgr_mean[i]);
				}
			}

			bool TrainSegApiImpl::train_seg(
				int net_id,
				const std::vector<cv::Mat>& v_image,
				std::vector<cv::Mat>& v_output
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, TrainSegNet::v_net.size()) << "net_id invalid";
				shared_caffe_net_t trainseg_net = TrainSegNet::v_net[net_id];

				return TrainSegApiImpl::train_seg(
					trainseg_net,
					v_image,
					v_output
				);
			}

			bool TrainSegApiImpl::train_seg(
				shared_caffe_net_t net,
				const std::vector<cv::Mat>& v_image,
				std::vector<cv::Mat>& v_output
			)
			{
				m_counter++;
#ifdef DEBUG_TIME
				static int64_t pre_cost = 0;
				static int64_t forward_cost = 0;
				static int64_t post_cost = 0;
				static int64_t total_cost = 0;

				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::ptime pt2;
				int64_t cost;
#endif // DEBUG_TIME

				int batch_size = v_image.size();
				for (size_t i = 0; i < v_image.size(); i++)
				{
					CHECK(!v_image[i].empty()) << "invalid mat";
					CHECK(v_image[i].channels()==3) << "mat channels must ==3";
				}

				// ============================================================
				int input_width = TrainSegNet::input_width;
				int input_height = TrainSegNet::input_height;
				cv::Size input_size(input_width, input_height);

				int image_width = v_image[0].cols;
				int image_height = v_image[0].rows;
				cv::Size origin_size(image_width, image_height);
				// ============================================================

				std::vector<cv::Mat> v_resized_input; // resized image
				
				for (size_t i = 0; i < v_image.size(); i++)
				{
					cv::Mat float_image;
					v_image[i].convertTo(float_image, CV_32FC3);

					cv::Mat resized_image;
					cv::resize(float_image, resized_image, input_size); // 512,512,3
					// 80.0322 73.0322 70.0322 75.4189 68.4189 65.4189

					v_resized_input.push_back(resized_image);
				}

				CaffeNet::caffe_net_n_inputs_t n_inputs;
				CaffeNet::caffe_net_n_outputs_t n_outputs;
				TrainSegNet::get_inputs_outputs(n_inputs, n_outputs); // 1-inputs,1-outputs

				for (size_t i = 0; i < batch_size; i++)
				{
					channel_mat_t channel_mat;
					// hwc ===> chw 
					OpencvUtil::bgr_subtract_mean(
						v_resized_input[i], 
						m_bgr_mean, 
						1.0, // 1 for trainseg, 1/255.0 for laneseg
						channel_mat
					);
					n_inputs[0].blob_channel_mat.push_back(channel_mat);
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-TRAINSEG] [1] pre-process data: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				pre_cost += cost;
				LOG(INFO) << "[API-TRAINSEG] #counter ="<<m_counter<<" pre_cost=" << pre_cost/(m_counter*1.0) << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				internal::TrainSegNet::forward(
					net,
					n_inputs,
					n_outputs,
					true // get float output  value range=0,1
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-TRAINSEG] [3] forward net1: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				forward_cost += cost;
				LOG(INFO) << "[API-TRAINSEG] #counter ="<<m_counter<<" forward_cost=" << forward_cost/(m_counter*1.0) << std::endl;

				
				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				//========================================================================
				// get output  N,3,256,256
				//========================================================================
				
				blob_channel_mat_t& blob_output = n_outputs[0].blob_channel_mat; // 1-outputs

				for (size_t i = 0; i < batch_size; i++)
				{
					// dl = 2,256,256; gz = 3,256,256 ===> binary_mask 256,256, [0,255] ===> resize(1080,1920)
					channel_mat_t channel_mat = blob_output[i]; 
					//std::cout<< channel_mat.size() << std::endl;

					cv::Mat binary_mask_1 = NumpyUtil::np_argmax(channel_mat); // 256,256 value=0,1
					cv::Mat binary_mask_255 = NumpyUtil::np_binary_mask_as255(binary_mask_1);

					cv::Mat origin_binary_mask = get_trainseg_binary_mask(binary_mask_255, origin_size);
					v_output.push_back(origin_binary_mask);
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-TRAINSEG] [3] post process: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();

				total_cost += cost;
				post_cost += cost;
				LOG(INFO) << "[API-TRAINSEG] #counter ="<<m_counter<<" post_cost=" << post_cost/(m_counter*1.0) << std::endl;

				LOG(INFO) << "[API-TRAINSEG] #counter ="<<m_counter<<" total_cost=" << total_cost/(m_counter*1.0) << std::endl;
#endif // DEBUG_TIME
				
				return true;
			}

			cv::Mat TrainSegApiImpl::get_trainseg_binary_mask(
				const cv::Mat& binary_mask, 
				const cv::Size& origin_size
			)
			{
				/*
				binary_mask:  256,256   CV_8FC1   bgr   0,255
				*/
				cv::Mat origin_binary_mask;
				cv::resize(binary_mask, origin_binary_mask, origin_size);
				return origin_binary_mask;
			}

		}
	}
}// end namespace

