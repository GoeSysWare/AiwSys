#include "railway_api_impl.h"
#include "../net/railway_net.h"

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

			int RailwayApiImpl::m_counter = 0;
			
#pragma region RailwayApiImpl

			void RailwayApiImpl::init(
				const caffe_net_file_t& crop_net_params,
				const caffe_net_file_t& detect_net_params
			)
			{
				RailwayCropNet::init(crop_net_params);
				RailwayDetectNet::init(detect_net_params);
			}

			void RailwayApiImpl::free()
			{
				RailwayCropNet::free();
				RailwayDetectNet::free();
			}

			bool RailwayApiImpl::crop_and_detect(
				const int net_id,
				const std::vector<cv::Mat>& v_image,
				const cv::Size& blur_size,
				const int dilate_size,
				const float box_min_binary_threshold,
				const int box_min_width,
				const int box_min_height,
				const bool filter_box_by_avg_pixel, // filter box 1 by avg pixel 
				const float filter_box_piexl_threshold, // [0,1]
				const bool filter_box_by_stdev_pixel, // filter box 2 by stdev pixel
				const int box_expand_width,
				const int box_expand_height,
				const float filter_box_stdev_threshold, // [0,
				const float gap_ratio,
				std::vector<bool>& v_crop_success,
				std::vector<bool>& v_has_gap,
				std::vector<cv::Rect>& v_gap_boxs,
				std::vector<bool>& v_has_anomaly,
				std::vector<boxs_t>& v_anomaly_boxs
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, 1) << "net_id invalid";
				shared_caffe_net_t crop_net = RailwayCropNet::v_net[net_id];
				shared_caffe_net_t detect_net = RailwayDetectNet::v_net[net_id];

				return RailwayApiImpl::crop_and_detect(
					crop_net,
					detect_net,
					v_image,
					blur_size,
					dilate_size,
					box_min_binary_threshold,
					box_min_width,
					box_min_height,
					filter_box_by_avg_pixel,
					filter_box_piexl_threshold,
					filter_box_by_stdev_pixel,
					box_expand_width,
					box_expand_height,
					filter_box_stdev_threshold,
					gap_ratio,
					v_crop_success,
					v_has_gap,
					v_gap_boxs,
					v_has_anomaly,
					v_anomaly_boxs
				);
			}

			bool RailwayApiImpl::post_process_for_gap(
				cv::Mat& merge_diff_output,
				const float gap_ratio,
				const cv::Rect& origin_rect,
				cv::Rect& gap_box_in_origin
			)
			{
				cv::Rect gap_box_for_fill_white;//用于存储分割出来的可能的唯一的轨缝区域框,用于空白填充
				cv::Rect gap_box_in_diff;//用于存储分割出来的可能的满足投影阈值的最终唯一轨缝区域框
				bool has_gap = OpencvUtil::get_horizontal_gap_box(
					merge_diff_output, gap_ratio,
					gap_box_for_fill_white, gap_box_in_diff
				);

				if (has_gap)
				{
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] gap_box_in_diff=" << gap_box_in_diff << std::endl;
					LOG(INFO)<<"[API] gap_box_for_fill_white=" << gap_box_for_fill_white << std::endl;
					LOG(INFO)<<"[API] has_gap=" << has_gap << std::endl;
#endif // shared_DEBUG

					// get gap box in origin
					gap_box_in_origin = OpencvUtil::diff_box_to_origin_box(
						gap_box_in_diff,
						merge_diff_output.size(),
						origin_rect.size(),
						origin_rect.x
					);

					// fill white for merge_diff_output
					OpencvUtil::fill_mat(
						merge_diff_output,
						gap_box_for_fill_white,
						m_gap_fill_white_delta_height,
						0  // 0 for black
					);
				}
				return has_gap;
			}

			bool RailwayApiImpl::crop_and_detect(
				shared_caffe_net_t crop_net,
				shared_caffe_net_t detect_net,
				const std::vector<cv::Mat>& v_image,
				const cv::Size& blur_size,
				const int dilate_size, // in pixels for diff to dilate
				const float box_min_binary_threshold,
				const int box_min_width,
				const int box_min_height,
				const bool filter_box_by_avg_pixel, // filter box 1 by avg pixel 
				const float filter_box_piexl_threshold, // [0,1]
				const bool filter_box_by_stdev_pixel, // filter box 2 by stdev
				const int box_expand_width,
				const int box_expand_height,
				const float filter_box_stdev_threshold, // [0,
				const float gap_ratio,
				std::vector<bool>& v_crop_success,
				std::vector<bool>& v_has_gap,
				std::vector<cv::Rect>& v_gap_boxs,
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
				CHECK_LE(batch_size, m_max_batch_size) << "railway batch_size must <" << m_max_batch_size;
				for (size_t i = 0; i < v_image.size(); i++)
				{
					CHECK(!v_image[i].empty()) << "invalid mat";
				}

				// origin 标准输入大小 for crop
				const int origin_input_height1 = RailwayCropNet::input_height;
				const int origin_input_width1 = RailwayCropNet::input_width;

				// railway_roi 标准输入大小 for detect
				const int railway_roi_input_height2 = RailwayDetectNet::input_height;
				const int railway_roi_input_width2 = RailwayDetectNet::input_width;

				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1
				input_blob shape_string:5 1 128 256 (163840)
				output blob shape_string:5 1 128 256 (163840)
				*/
				CaffeNet::caffe_net_n_inputs_t crop_n_inputs;
				CaffeNet::caffe_net_n_outputs_t crop_n_outputs;
				RailwayCropNet::get_inputs_outputs(crop_n_inputs, crop_n_outputs); // 1-inputs, 1-outpus

				for (int i = 0; i < batch_size; i++)
				{
					channel_mat_t channel_mat;
					cv::Mat origin_input;
					cv::resize(v_image[i], origin_input, cv::Size(origin_input_width1, origin_input_height1));
					channel_mat.push_back(origin_input); //  1 128 256

					crop_n_inputs[0].blob_channel_mat.push_back(channel_mat); // 5 1 128 256
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-RAILWAY] [1] before net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				//根据net1获取256,128的railway_diff_output,获取原图boxs,获取原图钢轨rect,从原图中截取获得钢轨origin_railway_roi
				// 1*256*128 ===>1*256*128 钢轨分割图railway_diff
				RailwayCropNet::forward(
					crop_net,
					crop_n_inputs,
					crop_n_outputs
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-RAILWAY] [2] forward net1: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				blob_channel_mat_t& v_output_mat = crop_n_outputs[0].blob_channel_mat; // 1-outputs

				CaffeNet::caffe_net_n_inputs_t detect_n_inputs;
				CaffeNet::caffe_net_n_outputs_t detect_n_outputs;
				RailwayDetectNet::get_inputs_outputs(detect_n_inputs, detect_n_outputs); // 2-inputs,2-outpus

				int flip_code = 0; // 上下翻转
				
				std::vector<cv::Rect> v_origin_railway_rect; // for later use
				for (size_t i = 0; i < v_output_mat.size(); i++)
				{
					const cv::Mat& origin_mat = v_image[i];
					cv::Mat& railway_diff_output = v_output_mat[i][0]; //CV_8UC1 (256*128) same as net-input

					// 256*128 ===>2048*1024 获取原图中boxs
					cv::Rect origin_railway_box;
					const float railway_box_ratio = 0.85f;
					bool has_railway_box = OpencvUtil::get_railway_box(
						railway_diff_output, 
						origin_mat.size(), 
						0, 
						railway_box_ratio,
						origin_railway_box
					);

#ifdef shared_DEBUG
					{
						cv::imwrite("railway/" + to_string(m_counter) + "_railway_diff_output.jpg", railway_diff_output);

						cv::Mat origin_railway_boxs_mat;
						DisplayUtil::draw_box(origin_mat, origin_railway_box, 5, origin_railway_boxs_mat);
						cv::imwrite("railway/" + to_string(m_counter) + "_origin_railway_boxs_mat.jpg", origin_railway_boxs_mat);
					}
#endif

					v_crop_success.push_back(has_railway_box);
					if (has_railway_box == false) { // 如果没有railway box设置默认值，最后skip网络结果
						origin_railway_box = cv::Rect(1400, 0, 224, 1024);// 
					}
					v_origin_railway_rect.push_back(origin_railway_box); // @@@@@@@@@@@@@@@@@@@@@

					//根据box裁剪出原图2048*1024中的钢轨
					cv::Mat origin_railway_roi = origin_mat(origin_railway_box);
					// 对裁剪的钢轨区域blur
					if (blur_size != cv::Size(1, 1)) {
						cv::blur(origin_railway_roi, origin_railway_roi, blur_size);
					}

					//LOG(INFO)<<"[API] origin_railway_roi.size()=" << origin_railway_roi.size() << std::endl; // 224*1024

					/*
					// Method 1:
					// origin_railway_roi(224*1024)进行resize到标准大小128*512
					// 上下flip得到2张图像，送入net2,网络出来4张图,1,2,3,4

					// 1对应railway_roi_input1的响应图，2对应railway_roi_input2的响应图
					// 3是12的混合响应图(只用于训练，暂时不用结果),4是为轨缝设计的响应图(暂时不用结果)
					
					cv::Mat railway_roi_input1;
					cv::resize(origin_railway_roi, railway_roi_input1, cv::Size(railway_roi_input_width2, railway_roi_input_height2));
					
					cv::Mat railway_roi_input2;
					cv::flip(railway_roi_input1, railway_roi_input2, flip_code); // >0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
					*/

					/*
					// Method 2:
					origin_railway_roi(224*1024)进行resize到标准大小128*512 railway_roi_input
					从中间裁剪为上下2张128*256 作为一对送入net
					*/
					cv::Mat railway_roi_input;
					int double_height = 2;
					cv::resize(
						origin_railway_roi, 
						railway_roi_input,
						cv::Size(railway_roi_input_width2, railway_roi_input_height2*(double_height)) // 128,512
					);
					
					cv::Mat railway_roi_input1, railway_roi_input2;
					OpencvUtil::split_mat_horizon(
						railway_roi_input,  // 128,512
						railway_roi_input1, // 128*256
						railway_roi_input2  // 128*256
					);


					// input1,input2
					channel_mat_t channel_mat_railway_roi_input1;
					channel_mat_railway_roi_input1.push_back(railway_roi_input1);

					channel_mat_t channel_mat_railway_roi_input2;
					channel_mat_railway_roi_input2.push_back(railway_roi_input2);

#ifdef shared_DEBUG
					{
						cv::imwrite("railway/diff/" + to_string(m_counter) + "_origin_railway_roi.jpg", origin_railway_roi);
						cv::imwrite("railway/diff/" + to_string(m_counter) + "_railway_roi_input.jpg", railway_roi_input);
						cv::imwrite("railway/diff/" + to_string(m_counter) + "_railway_roi_input1.jpg", railway_roi_input1);
						cv::imwrite("railway/diff/" + to_string(m_counter) + "_railway_roi_input2.jpg", railway_roi_input2);
					}
#endif

					detect_n_inputs[0].blob_channel_mat.push_back(channel_mat_railway_roi_input1);
					detect_n_inputs[1].blob_channel_mat.push_back(channel_mat_railway_roi_input2);
				}
#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-RAILWAY] [3] before net2: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				//2张128*512的mat===>4张输出128*512 mat
				RailwayDetectNet::forward(
					detect_net,
					detect_n_inputs,
					detect_n_outputs
				); // 2-outputs

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-RAILWAY] [4] forward net2: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				blob_channel_mat_t& v_output1 = detect_n_outputs[0].blob_channel_mat;
				blob_channel_mat_t& v_output2 = detect_n_outputs[1].blob_channel_mat;

				//LOG(INFO)<<"[API] ******************************************************" << std::endl;
				//LOG(INFO)<<"[API] v_output1.size()=" << v_output1.size() << std::endl;
				//LOG(INFO)<<"[API] v_output2.size()=" << v_output2.size() << std::endl;
				//LOG(INFO)<<"[API] ******************************************************" << std::endl;

				for (size_t i = 0; i < v_output1.size(); i++)
				{
					if (!v_crop_success[i]) //crop failed,then skip this image
					{
						//====================================================================================
						cv::Rect gap_box_in_origin;
						bool has_gap = false;
						v_has_gap.push_back(has_gap);
						v_gap_boxs.push_back(gap_box_in_origin);
						//=============================================================================
						boxs_t boxs;
						bool has_anomaly = false;
						v_has_anomaly.push_back(has_anomaly);
						v_anomaly_boxs.push_back(boxs);
					}
					else {

						const cv::Mat& origin_mat = v_image[i];
						channel_mat_t& channel_mat3 = v_output1[i]; // 3-channel  3*512*128 ===>3*256*128
						channel_mat_t& channel_mat1 = v_output2[i]; // 1-channel  1*512*128 ===>1*256*128

						cv::Mat result_mat_0 = channel_mat3[0];  // 256*128
						cv::Mat result_mat_1 = channel_mat3[1];  // 256*128
						cv::Mat result_mat_2 = channel_mat3[2];  // 256*128
						cv::Mat result_mat_3 = channel_mat1[0];

						/*
						// Method 1:
						// 只使用4张图像的前2张(512*128,512*128),上下翻转求bit_or得到 merge_diff_output  512*128
						cv::Mat merge_diff_output;
						cv::Mat flip_result1;
						cv::flip(result_mat_1, flip_result1, flip_code); // >0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
						bitwise_or(result_mat_0, flip_result1, merge_diff_output);
						*/
						
						// Method 2:
						//只使用4张图像的前2张(256*128,256*128),上下concat得到 merge_diff_output  512*128
						cv::Mat merge_diff_output = OpencvUtil::concat_mat(result_mat_0, result_mat_1);

#ifdef shared_DEBUG
						LOG(INFO) << "[API-RAILWAY] result_mat_0.size()=" << result_mat_0.size() << std::endl;
						LOG(INFO) << "[API-RAILWAY] result_mat_1.size()=" << result_mat_1.size() << std::endl;
						LOG(INFO) << "[API-TOPWIRE] merge_diff_output.size()=" << merge_diff_output.size() << std::endl;

						{
							cv::imwrite("railway/diff/" + to_string(m_counter) + "_result_0.jpg", result_mat_0);
							cv::imwrite("railway/diff/" + to_string(m_counter) + "_result_1.jpg", result_mat_1);
							//cv::imwrite("railway/diff/" + to_string(m_counter) + "_result_2.jpg", result_mat_2);
							//cv::imwrite("railway/diff/" + to_string(m_counter) + "_result_3.jpg", result_mat_3);

							//cv::imwrite("railway/diff/" + to_string(m_counter) + "_result_1_flip.jpg", flip_result1);
							cv::imwrite("railway/diff/" + to_string(m_counter) + "_z_merge_diff_output.jpg", merge_diff_output);
						}
#endif

						// dilate diff 对diff图进行膨胀操作，连接细小的相应区域成为较大的区域
						//OpencvUtil::dilate_mat(merge_diff_output, dilate_size); // dilate_size

						cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_size, dilate_size));
						cv::morphologyEx(merge_diff_output, merge_diff_output, cv::MORPH_CLOSE, element);

#ifdef shared_DEBUG
						{
							cv::imwrite("railway/diff/" + to_string(m_counter) + "_z_merge_diff_output_2dilate.jpg", merge_diff_output);
							//LOG(INFO)<<"[API] merge_diff_output.size()=" << merge_diff_output.size() << std::endl;//128*512
						}
#endif

						//====================================================================================
						cv::Rect gap_box_in_origin;
						bool has_gap = post_process_for_gap(
							merge_diff_output,
							gap_ratio,
							v_origin_railway_rect[i],
							gap_box_in_origin
						);
						v_has_gap.push_back(has_gap);
						v_gap_boxs.push_back(gap_box_in_origin);
						//=============================================================================

						//根据merge_diff_output获取原图中的异常boxs
						boxs_t boxs_in_origin;
						OpencvUtil::get_boxs_or(
							merge_diff_output,
							v_origin_railway_rect[i].size(),
							v_origin_railway_rect[i].x,
							box_min_binary_threshold,
							box_min_height,
							box_min_width,
							boxs_in_origin
						);

						// (1) filter railway box by avg pixel
						boxs_t filtered_boxs_in_origin;
						if (filter_box_by_avg_pixel)
						{
							OpencvUtil::filter_railway_boxs(
								origin_mat,
								boxs_in_origin,
								filter_box_piexl_threshold, // 0.75f;
								filtered_boxs_in_origin
							);
						}
						else {
							filtered_boxs_in_origin = boxs_in_origin;
						}

						// (2) filter railway box by stdev
						boxs_t filtered_boxs_in_origin2;
						if (filter_box_by_stdev_pixel)
						{
							OpencvUtil::filter_railway_boxs2(
								origin_mat,
								filtered_boxs_in_origin,
								box_expand_width,
								box_expand_height,
								filter_box_stdev_threshold,
								filtered_boxs_in_origin2
							);
						}
						else {
							filtered_boxs_in_origin2 = filtered_boxs_in_origin;
						}


						bool has_anomaly = filtered_boxs_in_origin2.size() > 0;

						v_has_anomaly.push_back(has_anomaly);
						v_anomaly_boxs.push_back(filtered_boxs_in_origin2);
					}
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-RAILWAY] [5] after net2: cost= " << cost*1.0 << std::endl;
#endif // DEBUG_TIME

				m_counter++;
				return true;
			}
#pragma endregion

		}
	}
}// end namespace

