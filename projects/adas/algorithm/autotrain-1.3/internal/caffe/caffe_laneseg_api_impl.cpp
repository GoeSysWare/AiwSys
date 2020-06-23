#include "caffe_laneseg_api_impl.h"
#include "laneseg_net.h"

#include "projects/adas/algorithm/core/util/filesystem_util.h"
#include "projects/adas/algorithm/core/util/display_util.h"
#include "projects/adas/algorithm/core/util/opencv_util.h"
#include "projects/adas/algorithm/core/util/numpy_util.h"
#include "projects/adas/algorithm/core/util/polyfiter.h"
#include "projects/adas/algorithm/core/util/lane_util.h"

// user defined mean shift 
//#include "algorithm/third/NumCpp.hpp"
// user defined mean shift 
//#include "algorithm/third/NumCpp.hpp"
#include "projects/adas/algorithm/third/cluster_util.h"


#include "projects/adas/algorithm/autotrain/monocular_distance_api.h"


// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;

namespace watrix {
	namespace algorithm {
		namespace internal {

			int CaffeLaneSegApiImpl::m_counter = 0; // init
			std::vector<float> CaffeLaneSegApiImpl::m_bgr_mean; // init (keystep, otherwise linker error)

			void CaffeLaneSegApiImpl::init(
				const caffe_net_file_t& detect_net_params,
				int feature_dim,
				int net_count
			)
			{
				LaneSegNet::init(detect_net_params, feature_dim, net_count);
			}

			void CaffeLaneSegApiImpl::free()
			{
				LaneSegNet::free();
			}

			void CaffeLaneSegApiImpl::set_bgr_mean(const std::vector<float>& bgr_mean)
			{
				for (size_t i = 0; i < bgr_mean.size(); i++)
				{
					CaffeLaneSegApiImpl::m_bgr_mean.push_back(bgr_mean[i]);
				}
			}

			bool CaffeLaneSegApiImpl::lane_seg(
				int net_id,
				const std::vector<cv::Mat>& v_image,
				int min_area_threshold, 
				std::vector<cv::Mat>& v_binary_mask, // 256,1024
				std::vector<channel_mat_t>& v_instance_mask // 8,256,1024
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, LaneSegNet::v_net.size()) << "net_id invalid";

				shared_caffe_net_t net = LaneSegNet::v_net[net_id];
				

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

				/*
				 input image size: 1920,1080 ===> cropped 1920,512 ===> resize 1024, 256 
				 ===> forward net ===> binary mask
				*/
				cv::Size origin_size(1920,1080);
				cv::Size clip_size(1920,512); // 
				cv::Size upper_size(1920,568); // 
				// cv::Size input_size(1024,256);
				// ============================================================
				int input_width = LaneSegNet::input_width;
				int input_height = LaneSegNet::input_height;
				cv::Size input_size(input_width, input_height);
				// ============================================================

				std::vector<cv::Mat> v_resized_input; // resized image
				
				for (size_t i = 0; i < v_image.size(); i++)
				{
					cv::Rect rect(0, 1080-512, 1920, 512);
					cv::Mat clip_image = v_image[i](rect); // clip image 

					cv::Mat float_clip_image;
					clip_image.convertTo(float_clip_image, CV_32FC3);

					cv::Mat resized_image;
					cv::resize(float_clip_image, resized_image, input_size); // 256,1024,3
					// 80.0322 73.0322 70.0322 75.4189 68.4189 65.4189

					v_resized_input.push_back(resized_image);
				}

				CaffeNet::caffe_net_n_inputs_t n_inputs;
				CaffeNet::caffe_net_n_outputs_t n_outputs;
				LaneSegNet::get_inputs_outputs(n_inputs, n_outputs); // 1-inputs,2-outputs

				for (size_t i = 0; i < batch_size; i++)
				{
					channel_mat_t channel_mat;
					OpencvUtil::bgr_subtract_mean(
						v_resized_input[i], 
						m_bgr_mean, 
						1/255.0, // 1 for trainseg, 1/255.0 for laneseg
						channel_mat
					);
					n_inputs[0].blob_channel_mat.push_back(channel_mat);
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LANESEG] [1] pre-process data: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				pre_cost += cost;
				LOG(INFO) << "[API-LANESEG] #counter ="<<m_counter<<" pre_cost=" << pre_cost/(m_counter*1.0) << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				internal::LaneSegNet::forward(
					net,
					n_inputs,
					n_outputs,
					true // get float output value range=0,1
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LANESEG] [2] forward net1: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				forward_cost += cost;
				LOG(INFO) << "[API-LANESEG] #counter ="<<m_counter<<" forward_cost=" << forward_cost/(m_counter*1.0) << std::endl;

				
				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				//========================================================================
				// get output1  N,2,256,1024
				//     output2  N,8,256,1024
				//========================================================================
				
				blob_channel_mat_t& blob_output = n_outputs[0].blob_channel_mat; // 2-outputs
				blob_channel_mat_t& blob_output2 = n_outputs[1].blob_channel_mat; // 2-outputs

				for (size_t i = 0; i < batch_size; i++)
				{
					// output[0].argmax(axis=0)  # (2,256, 1024) ===> argmax (256, 1024) ===> resize (512,1920)
					channel_mat_t output_binary_seg = blob_output2[i]; // 2,256,1024
					channel_mat_t output_instance_seg = blob_output[i]; // 8,256,1024

					//std::cout<<" instance_seg.size() = "<<instance_seg.size()<<std::endl; 
					//std::cout<<" binary_seg.size() = "<<binary_seg.size()<<std::endl; 
					cv::Mat binary_mask_01 = NumpyUtil::np_argmax(output_binary_seg); //  256,1024 value=0,1
					// post process binary mask to filter out noises
					cv::Mat filtered_binary_mask_01 = LaneUtil::connected_component_binary_mask(
						binary_mask_01, 
						min_area_threshold
					); // 256,1024  (remove noise)
					v_binary_mask.push_back(filtered_binary_mask_01); // (256, 1024) v=[0,1]
					v_instance_mask.push_back(output_instance_seg);// (8, 256, 1024) float value

					// binary_mask + instance_mask
					// # binary_seg = (256, 1024) v=[0,1];   instance_seg = (8, 256, 1024)
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LANESEG] [3] post process: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();

				total_cost+=cost;
				post_cost += cost;
				LOG(INFO) << "[API-LANESEG] #counter ="<<m_counter<<" post_cost=" << post_cost/(m_counter*1.0) << std::endl;

				LOG(INFO) << "[API-LANESEG] #counter ="<<m_counter<<" total_cost=" << total_cost/(m_counter*1.0) << std::endl;
#endif // DEBUG_TIME
				
				return true;
			}



		}	
	}
}// end namespace
