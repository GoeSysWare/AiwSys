#include "sidewall_api_impl.h"
#include "../net/sidewall_net.h"
#include "../sidewall_util.h"
#include "../ocr_util.h" // OcrUtil::nms_fast

#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"

// std
#include <iostream>
using namespace std;

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;


namespace watrix {
	namespace algorithm {
		namespace internal{

			int SidewallApiImpl::m_counter = 0;
			
			void SidewallApiImpl::init(
				const caffe_net_file_t& caffe_net_params
			)
			{
				SidewallNet::init(caffe_net_params);
			}

			void SidewallApiImpl::free()
			{
				SidewallNet::free();
			}

			void SidewallApiImpl::set_image_size(int height, int width)
			{
				SidewallNet::set_image_size(height,width);
				LOG(INFO) << "[API-SIDEWALL] set_image_size to (height,width)= (" << height <<","<< width << ")" << std::endl;
			}

			bool SidewallApiImpl::sidewall_detect(
				shared_caffe_net_t sidewall_net,
				const std::vector<cv::Mat>& v_image,
				const std::vector<cv::Mat>& v_roi,
				const cv::Size& blur_size,
				const bool fix_distortion_flag,
				const float box_min_binary_threshold,
				const int box_min_width,
				const int box_min_height,
				std::vector<bool>& v_has_anomaly,
				std::vector<boxs_t>& v_anomaly_boxs
			)
			{
#ifdef DEBUG_TIME
				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::ptime pt2;
				int64_t cost;
#endif // DEBUG_TIME

				int batch_size = v_image.size();
				CHECK_LE(batch_size, m_max_batch_size) << "sidewall batch_size must <" << m_max_batch_size;
				CHECK_EQ(v_image.size(), v_roi.size()) << "v_image.size() must = v_roi.size() ";
				for (size_t i = 0; i < v_image.size(); i++)
				{
					CHECK(!v_image[i].empty()) << "invalid mat";
					CHECK(!v_roi[i].empty()) << "invalid mat";
				}

				std::vector<cv::Mat> v_input_image,v_input_roi; // resized image
				if (fix_distortion_flag) {
					SidewallUtil::sidewall_fix(v_roi, v_image, v_input_image);

#ifdef shared_DEBUG
					LOG(INFO)<<"[API-SIDEWALL] v_input_image.size() = " << v_image.size() << std::endl;
#endif // shared_DEBUG

				}
				else {
					for (size_t i = 0; i < v_image.size(); i++)
					{
						cv::Mat image = v_image[i].clone();
						v_input_image.push_back(image);
					}
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-SIDEWALL] [2-1] fix distortion: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				for (size_t i = 0; i < v_roi.size(); i++)
				{
					cv::Mat roi = v_roi[i].clone();
					v_input_roi.push_back(roi);
				}

				// ============================================================
				int input_width = SidewallNet::input_width;
				int input_height = SidewallNet::input_height;
				cv::Size input_size(input_width, input_height);

				// resize sidewall image to standard net input (width,height) to reduct time cost
				for (size_t i = 0; i < v_input_image.size(); i++)
				{
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] resize sidewall image/roi to (" << input_width<<","<< input_height<<")" << std::endl;
#endif // shared_DEBUG

					cv::resize(v_input_image[i], v_input_image[i], input_size);
					cv::resize(v_input_roi[i], v_input_roi[i], input_size);
				}
				// ============================================================

				/*
				num_inputs()=1
				num_outputs()=1
				input_blob shape_string:3 2 1024 2048 (12582912)
				output_blob shape_string:3 1 1024 2048 (6291456)
				*/
				CaffeNet::caffe_net_n_inputs_t sidewall_n_inputs;
				CaffeNet::caffe_net_n_outputs_t sidewall_n_outputs;
				SidewallNet::get_inputs_outputs(sidewall_n_inputs, sidewall_n_outputs); // 1-inputs,1-outputs

				for (size_t i = 0; i < batch_size; i++)
				{
					cv::Mat blur1, blur2; // blur image to remove tiny anomaly
					cv::blur(v_input_image[i], blur1, blur_size);
					cv::blur(v_input_roi[i], blur2, blur_size);

					channel_mat_t channel_mat_blur;
					channel_mat_blur.push_back(blur1);
					channel_mat_blur.push_back(blur2);

#ifdef shared_DEBUG
					//cv::imwrite("sidewall/results/pair/" + to_string(m_counter) + "_input1.jpg", blur1);
					//cv::imwrite("sidewall/results/pair/" + to_string(m_counter) + "_input2.jpg", blur2);
#endif // shared_DEBUG

					sidewall_n_inputs[0].blob_channel_mat.push_back(channel_mat_blur);
				}
#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-SIDEWALL] [2-2] before net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


				SidewallNet::forward(
					sidewall_net,
					sidewall_n_inputs,
					sidewall_n_outputs
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-SIDEWALL] [3] forward net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				blob_channel_mat_t& v_diff = sidewall_n_outputs[0].blob_channel_mat; // 1-outputs

				for (size_t i = 0; i < v_diff.size(); i++)
				{
					cv::Mat& diff = v_diff[i][0];
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] diff.size()=" << diff.size() << std::endl; // [2048 x 1024]
					LOG(INFO)<<"[API] diff.type()=" << diff.type() << std::endl; // CV_8UC1=0
					cv::imwrite("sidewall/results/pair/" + to_string(m_counter) + "_diff.jpg", diff);
#endif // shared_DEBUG

					boxs_t boxs;
					bool has_anomaly = OpencvUtil::get_boxs_or(
						diff, // [1024,1024]
						v_image[i].size(), // origin image size [2048,1024]
						0,
						box_min_binary_threshold,
						box_min_width,
						box_min_height,
						boxs
					);

#ifdef shared_DEBUG
					LOG(INFO) << "[API] #" + to_string(m_counter) + " boxs.size()=" << boxs.size() << std::endl;
#endif // shared_DEBUG

					// 调试
					int box_count = 20;
					if (boxs.size()>box_count)
					{
						cv::Mat blur1, blur2; // blur image to remove tiny anomaly
						cv::blur(v_input_image[i], blur1, blur_size);
						cv::blur(v_input_roi[i], blur2, blur_size);

						LOG(ERROR) << "[API-SIDEWALL-ERROR] error begin=====================================";
						LOG(ERROR) << "[API-SIDEWALL-ERROR] box_count>20 #counter=" + to_string(m_counter) + " boxs.size()=" << boxs.size() << std::endl;

						cv::imwrite("./error/" + to_string(m_counter) + "_error_diff.jpg", diff);
						cv::imwrite("./error/" + to_string(m_counter) + "_input1.jpg", blur1);
						cv::imwrite("./error/" + to_string(m_counter) + "_input2.jpg", blur2);

						LOG(ERROR) << SidewallUtil::ss.str();
						
						LOG(ERROR) << "[API-SIDEWALL-ERROR] error end=====================================";
					}
					SidewallUtil::ss = stringstream("");//

					bool enable_nms = false;
					if (enable_nms)
					{
						boxs_t nms_boxs;
						internal::OcrUtil::nms_fast(boxs, 0.1, nms_boxs);
						boxs = nms_boxs;
						has_anomaly = boxs.size() > 0;

#ifdef shared_DEBUG
						LOG(INFO) << "[API] #" + to_string(m_counter) + " nms_boxs.size()=" << boxs.size() << std::endl;
#endif // shared_DEBUG
					}
					
					v_has_anomaly.push_back(has_anomaly);
					v_anomaly_boxs.push_back(boxs);
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-SIDEWALL] [4] after net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				m_counter++;

				return true;
			}


			bool SidewallApiImpl::sidewall_detect(
				const int net_id,
				const std::vector<cv::Mat>& v_image,
				const std::vector<cv::Mat>& v_roi,
				const cv::Size& blur_size,
				const bool fix_distortion_flag,
				const float box_min_binary_threshold,
				const int box_min_width,
				const int box_min_height,
				std::vector<bool>& v_has_anomaly,
				std::vector<boxs_t>& v_anomaly_boxs
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, 3) << "net_id invalid";
				shared_caffe_net_t sidewall_net = SidewallNet::v_net[net_id];
				return SidewallApiImpl::sidewall_detect(
					sidewall_net,
					v_image,
					v_roi,
					blur_size,
					fix_distortion_flag,
					box_min_binary_threshold,
					box_min_width,
					box_min_height,
					v_has_anomaly,
					v_anomaly_boxs
				);
			}

		}
	}
}// end namespace

