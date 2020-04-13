#include "lockcatch_api_impl.h"
#include "../net/lockcatch_net.h"

#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"

// std
#include <iostream>
#include <map>
using namespace std;

// glog
#include <glog/logging.h>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

namespace watrix {
	namespace algorithm {
		namespace internal {

			int LockcatchApiImpl::m_counter = 0;
#pragma region init and free 
			
			void LockcatchApiImpl::init(
				const caffe_net_file_t& crop_net_params,
				const caffe_net_file_t& refine_net_params
			)
			{
				LockcatchCropNet::init(crop_net_params);
				LockcatchRefineNet::init(refine_net_params);
			}

			void LockcatchApiImpl::free()
			{
				LockcatchCropNet::free();
				LockcatchRefineNet::free();
			}

#pragma endregion 

			bool LockcatchApiImpl::lockcatch_detect_v0(
				shared_caffe_net_t crop_net,
				shared_caffe_net_t refine_net,
				const std::vector<LockcatchType::lockcatch_mat_pair_t>& v_lockcatch,
				const LockcatchType::lockcatch_threshold_t& lockcatch_threshold,
				const cv::Size& blur_size,
				std::vector<bool>& v_has_lockcatch,
				std::vector<LockcatchType::lockcatch_status_t>& v_status,
				std::vector<boxs_t>& v_roi_boxs
			)
			{

#ifdef DEBUG_TIME
				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				int batch_size = v_lockcatch.size();
				CHECK_LE(batch_size, m_max_batch_size) << "lockcatch batch_size must <" << m_max_batch_size;
				for (size_t i = 0; i < v_lockcatch.size(); i++)
				{
					CHECK(!v_lockcatch[i].first.empty()) << "invalid mat";
					CHECK(!v_lockcatch[i].second.empty()) << "invalid mat";
				}

				//lockcatch_threshold = { 5000, 200, 10000, 200 };
				const int input_height = LockcatchCropNet::input_height;
				const int input_width = LockcatchCropNet::input_width;
				cv::Size size1(input_width, input_height); // 256,256
				
				const int input_height2 = LockcatchRefineNet::input_height;
				const int input_width2 = LockcatchRefineNet::input_width;
				cv::Size size2(input_width2, input_height2);

				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1
				input_blob 1 1 256 256 (65536)
				output_blob 1 3 256 256 (196608)
				*/
				CaffeNet::caffe_net_n_inputs_t crop_n_inputs;
				CaffeNet::caffe_net_n_outputs_t crop_n_outputs;
				LockcatchCropNet::get_inputs_outputs(crop_n_inputs, crop_n_outputs);// 1-inputs,1-outputs

																							  // (1) 输入网络进行粗分割，获取锁扣，铁丝，锁扣+铁丝 3种diff图
																							  //std::vector<channel_mat_t> v_input1;
				std::vector<cv::Mat> v_origin_image;
				for (int n = 0; n < batch_size; n++)
				{
					const LockcatchType::lockcatch_mat_pair_t& lockcatch_pair = v_lockcatch[n];

					const cv::Mat& first_image = lockcatch_pair.first;
					const cv::Mat& second_image = lockcatch_pair.second;

					cv::Mat full_image = OpencvUtil::concat_mat(first_image, second_image);
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] full_image.type()=" << full_image.type() << std::endl; // CV_8UC1
					LOG(INFO)<<"[API] full_image.size()=" << full_image.size() << std::endl; //full_image.size()=[2048 x 2048]
#endif
					v_origin_image.push_back(full_image); // @@@@@@@@@@@@@@@@@@ for later use

					// 对锁扣全图blur
					if (blur_size != cv::Size(1, 1)) {
						cv::blur(full_image, full_image, blur_size);
					}

					cv::Mat resized_full_image_input;
					cv::resize(full_image, resized_full_image_input, size1); // 256*256

#ifdef shared_DEBUG
					std::cout << "resized_full_image_input.size()=" << resized_full_image_input.size() << std::endl; 
					{
						imwrite("lockcatch/" + to_string(m_counter) + "_image_1.jpg", first_image);
						imwrite("lockcatch/" + to_string(m_counter) + "_image_2.jpg", second_image);
						imwrite("lockcatch/" + to_string(m_counter) + "_image_3_full.jpg", full_image);
						imwrite("lockcatch/" + to_string(m_counter) + "_image_4_resized_input.jpg", resized_full_image_input);
					}
#endif

					channel_mat_t channel_mat;
					channel_mat.push_back(resized_full_image_input);

					crop_n_inputs[0].blob_channel_mat.push_back(channel_mat);
				}

#ifdef DEBUG_TIME
				boost::posix_time::ptime pt2 = boost::posix_time::microsec_clock::local_time();

				int64_t cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LOCKCATCH] [1] before net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				LockcatchCropNet::forward(
					crop_net,
					crop_n_inputs,
					crop_n_outputs
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LOCKCATCH] [2] forward net1: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


				blob_channel_mat_t& v_output1 = crop_n_outputs[0].blob_channel_mat; // 1-outpus

#ifdef shared_DEBUG
				LOG(INFO)<<"[API-LOCKCATCH] v_output1.size()=" << v_output1.size() << std::endl;
#endif
				// (2) 获取锁扣+铁丝 左右侧ROI区域，输入网络进行精细分割
				// 获取left:  锁扣图1，锁扣缺失图2，铁丝图3，铁丝缺失图4
				// 获取right: 锁扣图1，锁扣缺失图2，铁丝图3，铁丝缺失图4

				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1

				input_blob  2 1 256 256 (131072)
				output_blob 2 4 256 256 (524288)
				*/
				CaffeNet::caffe_net_n_inputs_t refine_n_inputs;
				CaffeNet::caffe_net_n_outputs_t refine_n_outputs;
				LockcatchRefineNet::get_inputs_outputs(refine_n_inputs, refine_n_outputs);// 1-inputs,1-outputs

				for (int n = 0; n < batch_size; n++)
				{
					const channel_mat_t& channel_mat = v_output1[n]; // 3-channel, 8UC1 // 256*256

					const Mat& diff_0 = channel_mat[0]; // luomao
					const Mat& diff_1 = channel_mat[1]; // tiesi
					const Mat& diff_2 = channel_mat[2]; // luomao+tiesi roi

#ifdef shared_DEBUG
					std::cout << "diff_2.size()= " << diff_2.size() << std::endl;

					{
						imwrite("lockcatch/" + to_string(m_counter) + "_diff_0.jpg", diff_0);
						imwrite("lockcatch/" + to_string(m_counter) + "_diff_1.jpg", diff_1);
						imwrite("lockcatch/" + to_string(m_counter) + "_diff_2.jpg", diff_2);
					}
#endif

					LockcatchType::lockcatch_status_t lockcatch_status;
					cv::Mat left_roi, right_roi;
					boxs_t roi_boxs;
					bool has_lockcatch;
					LockcatchApiImpl::get_lockcatch_status_and_roi(
						diff_2,  // use luomao+tiesi roi
						v_origin_image[n],
						has_lockcatch,
						lockcatch_status,
						left_roi,
						right_roi,
						roi_boxs
					);
					v_has_lockcatch.push_back(has_lockcatch);
					v_status.push_back(lockcatch_status);
					v_roi_boxs.push_back(roi_boxs);
					
					// left and right roi may be invalid for result.
					cv::resize(left_roi, left_roi, size2);
					cv::resize(right_roi, right_roi, size2);

#ifdef shared_DEBUG
					{
						cv::imwrite("lockcatch/roi/" + to_string(m_counter) + "_resize_left_roi.jpg", left_roi);
						cv::imwrite("lockcatch/roi/" + to_string(m_counter) + "_resize_right_roi.jpg", right_roi);
					}
#endif

					channel_mat_t channel_mat_left_roi;
					channel_mat_left_roi.push_back(left_roi);

					channel_mat_t channel_mat_right_roi;
					channel_mat_right_roi.push_back(right_roi);

					refine_n_inputs[0].blob_channel_mat.push_back(channel_mat_left_roi);
					refine_n_inputs[0].blob_channel_mat.push_back(channel_mat_right_roi);
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LOCKCATCH] [3] before net2: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


				LockcatchRefineNet::forward(
					refine_net,
					refine_n_inputs,
					refine_n_outputs
				); // 1-outpus

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LOCKCATCH] [4] forward net2: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


				blob_channel_mat_t&  v_output2 = refine_n_outputs[0].blob_channel_mat;

#ifdef shared_DEBUG
				LOG(INFO)<<"[API-LOCKCATCH] batch_size=" << batch_size << ", v_output2.size()=" << v_output2.size() << std::endl;
#endif

				// batch_size=4, v_output2.size()=8, [0,1],[2,3],[4,5],[6,7]
				// batch_size=4, v_output2.size()=6, [x,x],[0,1],[2,3],[4,5]
				int k = 0;
				for (int n = 0; n < batch_size; n++) {
					if (v_status[n].refine_flag) //
					{
						// k = 0,  [0,1]
						// k = 2,  [2,3] 
						// k = 4,  [4,5]
						const channel_mat_t& channel_mat_left = v_output2[k];    //  4-channel
						const channel_mat_t& channel_mat_right = v_output2[k + 1]; //  4-channel

#ifdef shared_DEBUG
						for (int c = 0; c < channel_mat_left.size(); c++) // 0,1,2,3
						{
							const cv::Mat& left_roi_diff = channel_mat_left[c];
							const cv::Mat& right_roi_diff = channel_mat_right[c];
							imwrite("lockcatch/roi/" + to_string(m_counter) + "_" + to_string(n) + "_left_roi_diff_c" + to_string(c) + ".jpg", left_roi_diff);
							imwrite("lockcatch/roi/" + to_string(m_counter) + "_" + to_string(n) + "_right_roi_diff_c" + to_string(c) + ".jpg", right_roi_diff);
						}
#endif // shared_DEBUG

						LockcatchApiImpl::get_lockcatch_status(
							channel_mat_left,
							channel_mat_right,
							lockcatch_threshold,
							v_status[n] // refine status
						);

						k += 2; // step 2
					}
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LOCKCATCH] [5] after net2: cost= " << cost*1.0 << std::endl;
#endif // DEBUG_TIME

				m_counter++;
				return true;
			}


			bool LockcatchApiImpl::lockcatch_detect_v0(
				const int net_id,
				const std::vector<LockcatchType::lockcatch_mat_pair_t>& v_lockcatch,
				const LockcatchType::lockcatch_threshold_t& lockcatch_threshold,
				const cv::Size& blur_size,
				std::vector<bool>& v_has_lockcatch,
				std::vector<LockcatchType::lockcatch_status_t>& v_status,
				std::vector<boxs_t>& v_roi_boxs
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, 1) << "net_id invalid";
				shared_caffe_net_t crop_net = LockcatchCropNet::v_net[net_id];
				shared_caffe_net_t refine_net = LockcatchRefineNet::v_net[net_id];

				return LockcatchApiImpl::lockcatch_detect_v0(
					crop_net,
					refine_net,
					v_lockcatch,
					lockcatch_threshold,
					blur_size,
					v_has_lockcatch,
					v_status,
					v_roi_boxs
					);
			}

			
			/*
			net1进行检测，
			optional 使用net2进行优化(missing,*,*,*)===>(normal,*,*,*)
			*/
			bool LockcatchApiImpl::lockcatch_detect(
				shared_caffe_net_t crop_net,
				shared_caffe_net_t refine_net,
				const std::vector<LockcatchType::lockcatch_mat_pair_t>& v_lockcatch,
				const LockcatchType::lockcatch_threshold_t& net1_lockcatch_threshold,
				const LockcatchType::lockcatch_threshold_t& net2_lockcatch_threshold,
				const cv::Size& blur_size,
				std::vector<bool>& v_has_lockcatch,
				std::vector<LockcatchType::lockcatch_status_t>& v_status,
				std::vector<boxs_t>& v_roi_boxs
			)
			{

#ifdef DEBUG_TIME
				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::ptime pt2;
#endif // DEBUG_TIME

				int batch_size = v_lockcatch.size();
				CHECK_LE(batch_size, m_max_batch_size) << "lockcatch batch_size must <" << m_max_batch_size;
				for (size_t i = 0; i < v_lockcatch.size(); i++)
				{
					CHECK(!v_lockcatch[i].first.empty()) << "invalid mat";
					CHECK(!v_lockcatch[i].second.empty()) << "invalid mat";
				}

				//lockcatch_threshold = { 5000, 200, 10000, 200 };
				const int input_height = LockcatchCropNet::input_height;
				const int input_width = LockcatchCropNet::input_width;
				cv::Size size1(input_width, input_height); // 256,256

				const int input_height2 = LockcatchRefineNet::input_height;
				const int input_width2 = LockcatchRefineNet::input_width;
				cv::Size size2(input_width2, input_height2);

				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1
				input_blob 1 1 256 256 (65536)
				output_blob 1 3 256 256 (196608)
				*/
				CaffeNet::caffe_net_n_inputs_t crop_n_inputs;
				CaffeNet::caffe_net_n_outputs_t crop_n_outputs;
				LockcatchCropNet::get_inputs_outputs(crop_n_inputs, crop_n_outputs);// 1-inputs,1-outputs

				// (1) 输入网络进行粗分割，获取锁扣，铁丝，锁扣+铁丝 3种diff图
				std::vector<cv::Mat> v_origin_image;
				for (int n = 0; n < batch_size; n++)
				{
					const LockcatchType::lockcatch_mat_pair_t& lockcatch_pair = v_lockcatch[n];

					const cv::Mat& first_image = lockcatch_pair.first;
					const cv::Mat& second_image = lockcatch_pair.second;

					cv::Mat full_image = OpencvUtil::concat_mat(first_image, second_image);
					v_origin_image.push_back(full_image); //for later use [2048 x 2048]

					// 对锁扣全图blur
					if (blur_size != cv::Size(1, 1)) {
						cv::blur(full_image, full_image, blur_size);
					}

					cv::Mat resized_full_image_input;
					cv::resize(full_image, resized_full_image_input, size1); // 256*256

#ifdef shared_DEBUG
					//std::cout << "resized_full_image_input.size()=" << resized_full_image_input.size() << std::endl;
					{
						imwrite("lockcatch/" + to_string(m_counter) + "_image_1.jpg", first_image);
						imwrite("lockcatch/" + to_string(m_counter) + "_image_2.jpg", second_image);
						imwrite("lockcatch/" + to_string(m_counter) + "_image_3_full.jpg", full_image);
						imwrite("lockcatch/" + to_string(m_counter) + "_image_4_resized_input.jpg", resized_full_image_input);
					}
#endif

					channel_mat_t channel_mat;
					channel_mat.push_back(resized_full_image_input);

					crop_n_inputs[0].blob_channel_mat.push_back(channel_mat);
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				int64_t cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LOCKCATCH] [1] before net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				LockcatchCropNet::forward(
					crop_net,
					crop_n_inputs,
					crop_n_outputs
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LOCKCATCH] [2] forward net1: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				blob_channel_mat_t& v_output1 = crop_n_outputs[0].blob_channel_mat; // 1-outpus

#ifdef shared_DEBUG
				LOG(INFO) << "[API-LOCKCATCH] v_output1.size()=" << v_output1.size() << std::endl;
#endif

				// (2) 获取锁扣+铁丝 左右侧ROI区域，输入网络进行精细分割
				// 获取left:  锁扣图1，锁扣缺失图2，铁丝图3，铁丝缺失图4
				// 获取right: 锁扣图1，锁扣缺失图2，铁丝图3，铁丝缺失图4

				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1

				input_blob  2 1 256 256 (131072)
				output_blob 2 4 256 256 (524288)
				*/
				CaffeNet::caffe_net_n_inputs_t refine_n_inputs;
				CaffeNet::caffe_net_n_outputs_t refine_n_outputs;
				LockcatchRefineNet::get_inputs_outputs(refine_n_inputs, refine_n_outputs);// 1-inputs,1-outputs

				bool refine_by_net2 = false; 

				// 对diff图进行后处理获取status
				for (int n = 0; n < batch_size; n++)
				{
					const channel_mat_t& channel_mat = v_output1[n]; // 3-channel, 8UC1 // 256*256

					const Mat& diff_0 = channel_mat[0]; // luomao
					const Mat& diff_1 = channel_mat[1]; // tiesi
					const Mat& diff_2 = channel_mat[2]; // luomao+tiesi roi

#ifdef shared_DEBUG
					//std::cout << "diff_2.size()= " << diff_2.size() << std::endl;

					{
						imwrite("lockcatch/" + to_string(m_counter) + "_diff_0.jpg", diff_0);
						imwrite("lockcatch/" + to_string(m_counter) + "_diff_1.jpg", diff_1);
						imwrite("lockcatch/" + to_string(m_counter) + "_diff_2.jpg", diff_2);
					}
#endif

					LockcatchType::lockcatch_status_t lockcatch_status;
					cv::Mat left_roi, right_roi;
					boxs_t roi_boxs;
					bool has_lockcatch;
					LockcatchApiImpl::get_lockcatch_status_and_roi(
						diff_2,  // use luomao+tiesi roi
						v_origin_image[n],
						has_lockcatch,
						lockcatch_status,
						left_roi,
						right_roi,
						roi_boxs
					);
					v_has_lockcatch.push_back(has_lockcatch);
					v_status.push_back(lockcatch_status);
					v_roi_boxs.push_back(roi_boxs);

#ifdef shared_DEBUG
					{
						cv::imwrite("lockcatch/" + to_string(m_counter) + "_roi_left.jpg", left_roi);
						cv::imwrite("lockcatch/" + to_string(m_counter) + "_roi_right.jpg", right_roi);
					}
#endif

					if (v_status[n].refine_flag) 
					{ // refine status, 2 roi boxs
						float box_min_binary_threshold = 0.5f;
						Mat binary_diff_0 = OpencvUtil::get_binary_image(diff_0, box_min_binary_threshold);
						Mat binary_diff_1 = OpencvUtil::get_binary_image(diff_1, box_min_binary_threshold);
						Mat binary_diff_2 = OpencvUtil::get_binary_image(diff_2, box_min_binary_threshold);

						boxs_t roi_boxs_in_diff;
						OpencvUtil::diff_boxs_to_origin_boxs(roi_boxs, 
							v_origin_image[n].size(),
							diff_2.size(),
							0, 
							roi_boxs_in_diff
						);
						
#ifdef shared_DEBUG
						{
							//cv::imwrite("lockcatch/" + to_string(m_counter) + "_binary_diff_0.jpg", binary_diff_0);
							//cv::imwrite("lockcatch/" + to_string(m_counter) + "_binary_diff_1.jpg", binary_diff_1);
							//cv::imwrite("lockcatch/" + to_string(m_counter) + "_binary_diff_2.jpg", binary_diff_2);

							Mat diff_0_with_boxs;
							DisplayUtil::draw_boxs(binary_diff_0, roi_boxs_in_diff, 2, diff_0_with_boxs);
							cv::imwrite("lockcatch/" + to_string(m_counter) + "_diff_0_with_boxs.jpg", diff_0_with_boxs);

							Mat diff_1_with_boxs;
							DisplayUtil::draw_boxs(binary_diff_1, roi_boxs_in_diff, 2, diff_1_with_boxs);
							cv::imwrite("lockcatch/" + to_string(m_counter) + "_diff_1_with_boxs.jpg", diff_1_with_boxs);

							Mat diff_2_with_boxs;
							DisplayUtil::draw_boxs(binary_diff_2, roi_boxs_in_diff, 2, diff_2_with_boxs);
							cv::imwrite("lockcatch/" + to_string(m_counter) + "_diff_2_with_boxs.jpg", diff_2_with_boxs);
						}
#endif

						// refine luomao
						get_component_status(
							binary_diff_0, 
							roi_boxs_in_diff,
							net1_lockcatch_threshold.luomao_exist_min_area,
							net1_lockcatch_threshold.luomao_missing_min_area,
							v_status[n].left_luomao_status,
							v_status[n].right_luomao_status
						);
						

						// refine tiesi
						get_component_status(
							binary_diff_1,
							roi_boxs_in_diff,
							net1_lockcatch_threshold.tiesi_exist_min_area,
							net1_lockcatch_threshold.tiesi_missing_min_area,
							v_status[n].left_tiesi_status,
							v_status[n].right_tiesi_status
						);
						
						//==========================================================================
						// 如果net1的结果有异常，则需要net2进一步refine，有可能消除异常。
						// if missing,normal,normal,normal, then refine by net2, ===> normal
						if (has_anomaly(v_status[n])) {
							refine_by_net2 = true; // refine by net2

							v_status[n].refine_flag = true;
							string str_status = get_lockcatch_status_string(v_status[n]);
							
							LOG(INFO) << "[API-LOCKCATCH] refine "<< str_status<<" by net2 \n";

							// left and right roi may be invalid for result.
							cv::resize(left_roi, left_roi, size2);
							cv::resize(right_roi, right_roi, size2);

#ifdef shared_DEBUG
							{
								cv::imwrite("lockcatch/roi/" + to_string(m_counter) + "_resize_left_roi.jpg", left_roi);
								cv::imwrite("lockcatch/roi/" + to_string(m_counter) + "_resize_right_roi.jpg", right_roi);
							}
#endif

							channel_mat_t channel_mat_left_roi;
							channel_mat_left_roi.push_back(left_roi);

							channel_mat_t channel_mat_right_roi;
							channel_mat_right_roi.push_back(right_roi);

							refine_n_inputs[0].blob_channel_mat.push_back(channel_mat_left_roi);
							refine_n_inputs[0].blob_channel_mat.push_back(channel_mat_right_roi);

						}
						else {
							v_status[n].refine_flag = false;
						}
						//==========================================================================

					}
				}

				LOG(INFO) << "[API-LOCKCATCH] net2 input size = " << refine_n_inputs[0].blob_channel_mat.size() << std::endl;

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LOCKCATCH] [3] before net2: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				// optional refine by net2
				if (refine_by_net2)
				{
					LockcatchRefineNet::forward(
						refine_net,
						refine_n_inputs,
						refine_n_outputs
					); // 1-outpus

#ifdef DEBUG_TIME
					pt2 = boost::posix_time::microsec_clock::local_time();
					cost = (pt2 - pt1).total_milliseconds();
					LOG(INFO) << "[API-LOCKCATCH] [4] forward net2: cost= " << cost*1.0 << std::endl;

					pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

					blob_channel_mat_t&  v_output2 = refine_n_outputs[0].blob_channel_mat;

					LOG(INFO) << "[API-LOCKCATCH] net2 output size = " << refine_n_outputs[0].blob_channel_mat.size() << std::endl;

					// batch_size=4, v_output2.size()=8, [0,1],[2,3],[4,5],[6,7]
					// batch_size=4, v_output2.size()=6, [x,x],[0,1],[2,3],[4,5]
					int k = 0;
					for (int n = 0; n < batch_size; n++) {
						if (v_status[n].refine_flag) //
						{
							// k = 0,  [0,1]
							// k = 2,  [2,3] 
							// k = 4,  [4,5]
							const channel_mat_t& channel_mat_left = v_output2[k];    //  4-channel
							const channel_mat_t& channel_mat_right = v_output2[k + 1]; //  4-channel

#ifdef shared_DEBUG
							for (int c = 0; c < channel_mat_left.size(); c++) // 0,1,2,3
							{
								const cv::Mat& left_roi_diff = channel_mat_left[c];
								const cv::Mat& right_roi_diff = channel_mat_right[c];
								imwrite("lockcatch/roi/" + to_string(m_counter) + "_" + to_string(n) + "_left_roi_diff_c" + to_string(c) + ".jpg", left_roi_diff);
								imwrite("lockcatch/roi/" + to_string(m_counter) + "_" + to_string(n) + "_right_roi_diff_c" + to_string(c) + ".jpg", right_roi_diff);
							}
#endif // shared_DEBUG

							
							LockcatchApiImpl::get_lockcatch_status(
								channel_mat_left,
								channel_mat_right,
								net2_lockcatch_threshold,
								v_status[n] // refine status
							);

							k += 2; // step 2
						}
					}

#ifdef DEBUG_TIME
					pt2 = boost::posix_time::microsec_clock::local_time();
					cost = (pt2 - pt1).total_milliseconds();
					LOG(INFO) << "[API-LOCKCATCH] [5] after net2: cost= " << cost*1.0 << std::endl;
#endif // DEBUG_TIME
				}
				else {
					LOG(INFO) << "[API-LOCKCATCH] [5] NOT refine by net2 \n";
				}

				m_counter++;

				return true;
			}

			bool LockcatchApiImpl::lockcatch_detect(
				const int net_id,
				const std::vector<LockcatchType::lockcatch_mat_pair_t>& v_lockcatch,
				const LockcatchType::lockcatch_threshold_t& net1_lockcatch_threshold,
				const LockcatchType::lockcatch_threshold_t& net2_lockcatch_threshold,
				const cv::Size& blur_size,
				std::vector<bool>& v_has_lockcatch,
				std::vector<LockcatchType::lockcatch_status_t>& v_status,
				std::vector<boxs_t>& v_roi_boxs
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, 1) << "net_id invalid";
				shared_caffe_net_t crop_net = LockcatchCropNet::v_net[net_id];
				shared_caffe_net_t refine_net = LockcatchRefineNet::v_net[net_id];

				return LockcatchApiImpl::lockcatch_detect(
					crop_net,
					refine_net,
					v_lockcatch,
					net1_lockcatch_threshold,
					net2_lockcatch_threshold,
					blur_size,
					v_has_lockcatch,
					v_status,
					v_roi_boxs
				);
			}


#pragma region get lockcatch roi

			void LockcatchApiImpl::get_lockcatch_status_and_roi(
				const cv::Mat& diff,
				const cv::Mat& origin_image,
				bool& has_lockcatch,
				LockcatchType::lockcatch_status_t& lockcatch_status,
				cv::Mat& left_roi,
				cv::Mat& right_roi,
				boxs_t& roi_boxs
			)
			{
				float box_min_binary_threshold = 0.5f;
				int lockcatch_box_width = 150;
				int lockcatch_box_height = 500;
				boxs_t origin_boxs;
				OpencvUtil::get_boxs_and(
					diff, 
					origin_image.size(), 
					0, 
					box_min_binary_threshold,
					lockcatch_box_height, 
					lockcatch_box_width, 
					origin_boxs
				);

#ifdef shared_DEBUG
				LOG(INFO)<<"[API] [lockcatch] origin_boxs.size()=" << origin_boxs.size() << std::endl;
#endif // shared_DEBUG

				has_lockcatch = false;
				/*
				full lockcatch size:
				size = [224 x 560]
				size = [192 x 592]

				size = [224 x 576]
				size = [192 x 584]
				*/

				// 过滤掉一些临近最上面，最下面的box 上下30个像素。
				int delta_height = 30;
				int min_height = delta_height; 
				int max_height = origin_image.size().height - delta_height;
				std::vector<bool> v_filtered_out;
				boxs_t keep_origin_boxs;
				for (size_t i = 0; i < origin_boxs.size(); i++)
				{
					cv::Rect& box = origin_boxs[i];
					if (box.y>min_height && box.y+box.height <max_height)
					{
						keep_origin_boxs.push_back(box);
					}
					else {
#ifdef shared_DEBUG
						LOG(INFO)<<"[API] [lockcatch] box is filtered out \n";
#endif // shared_DEBUG
					}
				}
				int box_size = (int)keep_origin_boxs.size();

#ifdef shared_DEBUG
				LOG(INFO)<<"[API] [lockcatch] keep_origin_boxs.size()=" << box_size << std::endl;
#endif // shared_DEBUG

				cv::Rect left_rect, right_rect;

				// keep_origin_boxs可能会有4个box，roi_boxs只需要面积最大的2个。
				if ( 0 == origin_boxs.size())
				{
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] [bad image 1] left AND right roi is missing.\n";
#endif // shared_DEBUG

					// 左侧+右侧 ROI(螺帽，铁丝)都丢失
					lockcatch_status.refine_flag = false;
					lockcatch_status.left_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
					lockcatch_status.left_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
					lockcatch_status.right_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
					lockcatch_status.right_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
				}
				else if ( 1 == origin_boxs.size()) {
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] [bad image 2] left OR right roi is missing.\n";
#endif // shared_DEBUG

					Rect& rect = origin_boxs[0];

					// rect的中心位于左边，或者右边 
					if (rect.x + rect.width / 2 <= origin_image.cols / 2) {
						// 左侧ROI(螺帽，铁丝)存在；右侧ROI(螺帽，铁丝)都丢失。
						lockcatch_status.refine_flag = false;
						lockcatch_status.left_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.left_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.right_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
						lockcatch_status.right_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
					}
					else {
						//左侧ROI(螺帽，铁丝)都丢失；右侧ROI(螺帽，铁丝)存在。
						lockcatch_status.refine_flag = false;
						lockcatch_status.left_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
						lockcatch_status.left_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
						lockcatch_status.right_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.right_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
					}

					roi_boxs.push_back(rect);// save only 1 roi box
				}
				else{ // origin_boxs.size() >=2
					has_lockcatch = true;
					if (keep_origin_boxs.size() >=2)
					{
						//左侧+右侧ROI(螺帽，铁丝)都存在；需要获取左侧+右侧ROI区域精细化处理。 left+right
						lockcatch_status.refine_flag = true;
						lockcatch_status.left_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.left_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.right_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.right_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;

						// 对N个boxs的面积进行sort，只使用面积最大的前2个
						std::sort(keep_origin_boxs.begin(), keep_origin_boxs.end(), OpencvUtil::rect_compare);

#ifdef shared_DEBUG
						for (size_t i = 0; i < keep_origin_boxs.size(); i++)
						{
							//LOG(INFO)<<"[API] # " << i << " area = " << keep_origin_boxs[i].area() << std::endl;
						}
#endif // shared_DEBUG

						left_rect = keep_origin_boxs[0];
						right_rect = keep_origin_boxs[1];
						// （左右判断需要优化）
						// 确保left_rect是左边的,right_rect是右边的。
						if (left_rect.x + left_rect.width / 2 > right_rect.x + right_rect.width / 2) {
							//left_rect在右边，则swap
							//LOG(INFO)<<"[API] swap left_rect and right_rect" << std::endl;
							std::swap(left_rect, right_rect);
						}
					}
					else {
						// 左侧 or 右侧 ROI(螺帽，铁丝)都filtered out
						lockcatch_status.refine_flag = false;
						lockcatch_status.left_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.left_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.right_luomao_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
						lockcatch_status.right_tiesi_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
					}
				}

				// get left and right roi
				Rect skip_rect(0, 0, 50, 50); // skip result
				if (!lockcatch_status.refine_flag)
				{
					// left and right roi result are skipped
					left_rect = skip_rect;
					right_rect = skip_rect; 
				}
				left_roi = origin_image(left_rect);
				right_roi = origin_image(right_rect);
				roi_boxs.push_back(left_rect);
				roi_boxs.push_back(right_rect);

#ifdef shared_DEBUG
				{
					Mat origin_all_boxs_mat;
					DisplayUtil::draw_boxs(origin_image, origin_boxs, 3, origin_all_boxs_mat);

					Mat origin_keep_boxs_mat;
					DisplayUtil::draw_boxs(origin_image, keep_origin_boxs, 3, origin_keep_boxs_mat);

					Mat origin_roi_boxs_mat;
					DisplayUtil::draw_boxs(origin_image, roi_boxs, 3, origin_roi_boxs_mat);

					imwrite("lockcatch/origin_1_all_boxs_mat.jpg", origin_all_boxs_mat);
					imwrite("lockcatch/origin_2_keep_boxs_mat.jpg", origin_keep_boxs_mat);
					imwrite("lockcatch/origin_3_roi_boxs_mat.jpg", origin_roi_boxs_mat);
				}
#endif

			}

#pragma endregion 

#pragma region get lockcatch status

			LockcatchType::COMPONENT_STATUS LockcatchApiImpl::get_component_status(
				const cv::Mat& exist_component,
				const int exist_area,
				const cv::Mat& missing_component,
				const int missing_area
			)
			{
				//==========================================
				// exist + missing ===>然后进行处理
				cv::Mat exist_missing;
				bitwise_or(exist_component, missing_component, exist_missing);
				//==========================================
				
				LockcatchType::COMPONENT_STATUS status;
				contours_t exist_contours;
				OpencvUtil::get_contours(exist_missing,0.5f, exist_contours);

#ifdef shared_DEBUG
				LOG(INFO)<<"[API]  exist_contours.size()= " << exist_contours.size() << std::endl;
#endif
				if (exist_contours.size() > 0) {
					std::sort(exist_contours.begin(), exist_contours.end(), OpencvUtil::contour_compare);

					double area = abs(contourArea(exist_contours[0], false));
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] area = " << area << std::endl;
#endif

					if (area > exist_area) {
						status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL; // 正常
#ifdef shared_DEBUG
						LOG(INFO)<<"[API] [STATUS_NORMAL] area = " << area << " >" << exist_area << std::endl;
#endif
					}
					else if(area< missing_area) {
						status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;   // 认为不存在
#ifdef shared_DEBUG
						LOG(INFO)<<"[API] [STATUS_MISSING] area = " << area << " <" << missing_area << std::endl;
#endif
					}
					else {
						status = LockcatchType::COMPONENT_STATUS::STATUS_PARTLOSE; // 部分缺失

#ifdef shared_DEBUG
						LOG(INFO)<<"[API] [STATUS_PARTLOSE] area = "<<area << std::endl;
#endif
					}
				}
				else {
					status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING; // 不存在
				}
				return status;
			}

			void LockcatchApiImpl::get_lockcatch_status(
				const bool left,
				const channel_mat_t& channel_mat,
				const LockcatchType::lockcatch_threshold_t& lockcatch_threshold,
				LockcatchType::lockcatch_status_t& lockcatch_status
			)
			{
				std::string luomao_filename;
				std::string tiesi_filename;
				if (left) {
					//LOG(INFO)<<"[API] get left status\n";
					luomao_filename = "lockcatch/roi/aaa_left_luomao.jpg";
					tiesi_filename = "lockcatch/roi/aaa_left_tiesi.jpg";
				}
				else {
					//LOG(INFO)<<"[API] get right status\n";
					luomao_filename = "lockcatch/roi/aaa_right_luomao.jpg";
					tiesi_filename = "lockcatch/roi/aaa_right_tiesi.jpg";
				}

#ifdef shared_DEBUG
				{
					cv::Mat luomao_exist_missing;
					bitwise_or(channel_mat[0], channel_mat[1], luomao_exist_missing);

					cv::Mat tiesi_exist_missing;
					bitwise_or(channel_mat[2], channel_mat[3], tiesi_exist_missing);

					cv::imwrite(luomao_filename, luomao_exist_missing);
					cv::imwrite(tiesi_filename, tiesi_exist_missing);
				}
#endif

				CHECK_EQ(channel_mat.size(), 4) << "channel_mat.size() must = 4 ";
				LockcatchType::COMPONENT_STATUS x_luomao_status = get_component_status(
					channel_mat[0], lockcatch_threshold.luomao_exist_min_area,
					channel_mat[1], lockcatch_threshold.luomao_missing_min_area
				);
				LockcatchType::COMPONENT_STATUS x_tiesi_status = get_component_status(
					channel_mat[2], lockcatch_threshold.tiesi_exist_min_area,
					channel_mat[3], lockcatch_threshold.tiesi_missing_min_area
				);

				if (left) {
					lockcatch_status.left_luomao_status = x_luomao_status;
					lockcatch_status.left_tiesi_status = x_tiesi_status;
				}
				else {
					lockcatch_status.right_luomao_status = x_luomao_status;
					lockcatch_status.right_tiesi_status = x_tiesi_status;
				}
			}

			void LockcatchApiImpl::get_lockcatch_status(
				const channel_mat_t& channel_mat_left,
				const channel_mat_t& channel_mat_right,
				const LockcatchType::lockcatch_threshold_t& lockcatch_threshold,
				LockcatchType::lockcatch_status_t& lockcatch_status
			)
			{
				const int count = 2;
				bool flag_array[] = {
					true,
					false
				};
				channel_mat_t channel_mat_array[] = {
					channel_mat_left,
					channel_mat_right
				};
				for (int i = 0; i < count; i++)
				{
					get_lockcatch_status(
						flag_array[i],
						channel_mat_array[i],
						lockcatch_threshold,
						lockcatch_status
					);
				}
			}

#pragma endregion

#pragma region print lockcatch status
			std::string LockcatchApiImpl::get_component_status_string(LockcatchType::COMPONENT_STATUS status)
			{
				std::map<LockcatchType::COMPONENT_STATUS, std::string> map_status_string;
				map_status_string[LockcatchType::STATUS_NORMAL] = "normal";
				map_status_string[LockcatchType::STATUS_PARTLOSE] = "partlose";
				map_status_string[LockcatchType::STATUS_MISSING] = "missing";
				return map_status_string[status];
			}

			std::string LockcatchApiImpl::get_lockcatch_status_string(
				const LockcatchType::lockcatch_status_t& lockcatch_status
			)
			{
				stringstream ss;
				ss << "(";
				ss << get_component_status_string(lockcatch_status.left_luomao_status);
				ss << ", " << get_component_status_string(lockcatch_status.right_luomao_status);
				ss << ", " << get_component_status_string(lockcatch_status.left_tiesi_status);
				ss << ", " << get_component_status_string(lockcatch_status.right_tiesi_status);
				ss << ")";
				return  ss.str();
			}

			/*
			bool LockcatchApiImpl::has_anomaly_v1(const LockcatchType::lockcatch_status_t& lockcatch_status)
			{
				bool normal = (lockcatch_status.left_luomao_status == LockcatchType::STATUS_NORMAL)
					&& (lockcatch_status.left_tiesi_status == LockcatchType::STATUS_NORMAL)
					&& (lockcatch_status.right_luomao_status == LockcatchType::STATUS_NORMAL)
					&& (lockcatch_status.right_tiesi_status == LockcatchType::STATUS_NORMAL);
				return !normal;
			}
			*/


			bool LockcatchApiImpl::has_anomaly(const LockcatchType::lockcatch_status_t& lockcatch_status)
			{
				bool missing = (lockcatch_status.left_luomao_status == LockcatchType::STATUS_MISSING)
					|| (lockcatch_status.left_tiesi_status == LockcatchType::STATUS_MISSING)
					|| (lockcatch_status.right_luomao_status == LockcatchType::STATUS_MISSING)
					|| (lockcatch_status.right_tiesi_status == LockcatchType::STATUS_MISSING);
				return missing;
			}
#pragma endregion


#pragma region v2-status 

			LockcatchType::COMPONENT_STATUS LockcatchApiImpl::get_component_status(
				const float area,
				const float exist_area,
				const float missing_area
			)
			{
				//std::cout << "exist_area=" << exist_area << std::endl;
				//std::cout << "missing_area=" << missing_area << std::endl;

				LockcatchType::COMPONENT_STATUS component_status;
				if (area>= exist_area)
				{
					component_status = LockcatchType::COMPONENT_STATUS::STATUS_NORMAL;
				}
				else if (area<= missing_area) {
					component_status = LockcatchType::COMPONENT_STATUS::STATUS_MISSING;
				}
				else {
					component_status = LockcatchType::COMPONENT_STATUS::STATUS_PARTLOSE;
				}
				return component_status;
			}

			void LockcatchApiImpl::get_component_status(
				const cv::Mat& diff,
				const boxs_t& boxs_in_diff,
				const float exist_avg_pixel,
				const float missing_avg_pixel,
				LockcatchType::COMPONENT_STATUS& left_com_status,
				LockcatchType::COMPONENT_STATUS& right_com_status
			)
			{
				/*
				[API] left_box_avg_pixel=0
				[API] right_box_avg_pixel=21.7949
				[API] left_box_avg_pixel=56.6818
				[API] right_box_avg_pixel=57.0753


				[API] left_box_avg_pixel=0
				[API] right_box_avg_pixel=22.6054
				[API] left_box_avg_pixel=56.4137
				[API] right_box_avg_pixel=59.5459		


				[API] left_box_avg_pixel=0
				[API] right_box_avg_pixel=22.8961
				[API] left_box_avg_pixel=54.9569
				[API] right_box_avg_pixel=61.6549


				[API] left_box_avg_pixel=21.5691
				[API] right_box_avg_pixel=20.9877
				[API] left_box_avg_pixel=0.510511
				[API] right_box_avg_pixel=55.8796


				[API] left_box_avg_pixel=22.1
				[API] right_box_avg_pixel=4.76
				[API] left_box_avg_pixel=55.3714
				[API] right_box_avg_pixel=49.64

				[API] left_box_avg_pixel=22.9396
				[API] right_box_avg_pixel=1.68734
				[API] left_box_avg_pixel=56.7696
				[API] right_box_avg_pixel=43.871
				*/

				// boxs_in_diff.size() == 2
				CHECK_EQ(boxs_in_diff.size(), 2) << "not 2 lockcatch roi boxs";

				Rect left_box = boxs_in_diff[0];
				Rect right_box = boxs_in_diff[1];

				float scale = 1.0f;
				float left_box_avg_pixel =  OpencvUtil::get_average_pixel(diff, left_box)  / scale; // 0-255
				float right_box_avg_pixel = OpencvUtil::get_average_pixel(diff, right_box) / scale; // 0-255

#ifdef shared_DEBUG
				std::cout << "[API] left_box_avg_pixel=" << left_box_avg_pixel << std::endl;
				std::cout << "[API] right_box_avg_pixel=" << right_box_avg_pixel << std::endl;
#endif // shared_DEBUG

				left_com_status = get_component_status(left_box_avg_pixel, exist_avg_pixel, missing_avg_pixel);
				right_com_status = get_component_status(right_box_avg_pixel, exist_avg_pixel, missing_avg_pixel);
			}

#pragma endregion


		}
	}
}// end namespace

