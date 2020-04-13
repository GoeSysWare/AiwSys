#include "topwire_api_impl.h"
#include "../net/topwire_net.h"

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

			int TopwireApiImpl::m_counter = 0;
			
#pragma region TopwireApiImpl
			void TopwireApiImpl::init(
				const caffe_net_file_t& crop_net_params,
				const caffe_net_file_t& detect_net_params
			)
			{
				TopwireCropNet::init(crop_net_params);
				TopwireDetectNet::init(detect_net_params);
			}

			void TopwireApiImpl::free()
			{
				TopwireCropNet::free();
				TopwireDetectNet::free();
			}

			bool TopwireApiImpl::crop_and_detect(
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
				std::vector<bool>& v_crop_success,
				std::vector<bool>& v_has_hole,
				std::vector<boxs_t>& v_hole_boxs,
				std::vector<bool>& v_has_anomaly,
				std::vector<boxs_t>& v_anomaly_boxs
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, 0) << "net_id invalid";
				shared_caffe_net_t crop_net = TopwireCropNet::v_net[net_id];
				shared_caffe_net_t detect_net = TopwireDetectNet::v_net[net_id];

				return TopwireApiImpl::crop_and_detect(
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
					v_crop_success,
					v_has_hole,
					v_hole_boxs,
					v_has_anomaly,
					v_anomaly_boxs
				);
			}

			bool TopwireApiImpl::crop_and_detect(
				shared_caffe_net_t crop_net,
				shared_caffe_net_t detect_net,
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
				std::vector<bool>& v_crop_success,
				std::vector<bool>& v_has_hole,
				std::vector<boxs_t>& v_hole_boxs,
				std::vector<bool>& v_has_anomaly,
				std::vector<boxs_t>& v_anomaly_boxs
			)
			{

#ifdef DEBUG_TIME
				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::ptime pt2;
#endif // DEBUG_TIME

				int batch_size = v_image.size();
				CHECK_LE(batch_size, m_max_batch_size) << "topwire batch_size must <" << m_max_batch_size;
				for (size_t i = 0; i < v_image.size(); i++)
				{
					CHECK(!v_image[i].empty()) << "invalid mat";
				}

				// origin 标准输入大小 for crop
				const int origin_input_height1 = TopwireCropNet::input_height;
				const int origin_input_width1 = TopwireCropNet::input_width;

				// topwire_roi 标准输入大小 for detect
				const int topwire_roi_input_height2 = TopwireDetectNet::input_height;
				const int topwire_roi_input_width2 = TopwireDetectNet::input_width;

				/*
				num_inputs()=1   // data
				num_outputs()=1  // seg-score1
				input_blob shape_string:5 1 128 256 (163840)
				output blob shape_string:5 1 128 256 (163840)
				*/
				CaffeNet::caffe_net_n_inputs_t crop_n_inputs;
				CaffeNet::caffe_net_n_outputs_t crop_n_outputs;
				TopwireCropNet::get_inputs_outputs(crop_n_inputs, crop_n_outputs); // 1-inputs, 1-outpus

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

				int64_t cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO)<<"[API-TOPWIRE] [1] before net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				//根据net1获取256,128的topwire_diff_output,获取原图boxs,获取原图接触网rect,从原图中截取获得接触网origin_topwire_roi
				// 1*256*128 ===>1*256*128 接触网分割图topwire_diff
				TopwireCropNet::forward(
					crop_net,
					crop_n_inputs,
					crop_n_outputs
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO)<<"[API-TOPWIRE] [2] forward net1: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


				blob_channel_mat_t& v_output_mat = crop_n_outputs[0].blob_channel_mat; // 1-outputs

				CaffeNet::caffe_net_n_inputs_t detect_n_inputs;
				CaffeNet::caffe_net_n_outputs_t detect_n_outputs;
				TopwireDetectNet::get_inputs_outputs(detect_n_inputs, detect_n_outputs); // 2-inputs,2-outpus

				int flip_code = 0; // 上下翻转
				
				std::vector<cv::Rect> v_origin_topwire_rect; // for later use
				std::vector<cv::Mat> v_origin_topwire_roi; // for later use

				for (size_t i = 0; i < v_output_mat.size(); i++)
				{
					const cv::Mat& origin_mat = v_image[i];
					cv::Mat& topwire_diff_output = v_output_mat[i][0];

#ifdef shared_DEBUG
					LOG(INFO)<<"[API] topwire_diff_output.type()=" << topwire_diff_output.type() << std::endl; // CV_8UC1
					LOG(INFO)<<"[API] topwire_diff_output.size()=" << topwire_diff_output.size() << std::endl; // (256*128) same as net-input
#endif

					// 256*128 ===>2048*1024 获取原图中boxs
					cv::Rect origin_topwire_box;
					const float topwire_box_ratio = 0.85f;
					bool has_topwire_box = OpencvUtil::get_topwire_box(
						topwire_diff_output,
						origin_mat.size(),
						0,
						topwire_box_ratio,
						origin_topwire_box
					);

#ifdef shared_DEBUG
					{
						cv::imwrite("topwire/diff/" + to_string(m_counter) + "_topwire_diff_output.jpg", topwire_diff_output);

						cv::Mat origin_topwire_boxs_mat;
						DisplayUtil::draw_box(origin_mat, origin_topwire_box, 5, origin_topwire_boxs_mat);
						cv::imwrite("topwire/diff/" + to_string(m_counter) + "_origin_topwire_boxs_mat.jpg", origin_topwire_boxs_mat);
					}
#endif

					v_crop_success.push_back(has_topwire_box);
					if (has_topwire_box == false) { // 如果没有topwire box设置默认值，最后skip网络结果
						origin_topwire_box = cv::Rect(1400, 0, 224, 1024);// 
					}
					v_origin_topwire_rect.push_back(origin_topwire_box); // @@@@@@@@@@@@@@@@@@@@@

					//根据box裁剪出原图2048*1024中的接触网
					cv::Mat origin_topwire_roi = origin_mat(origin_topwire_box);
					// 对裁剪的接触网区域blur
					if (blur_size != cv::Size(1, 1)) {
						cv::blur(origin_topwire_roi, origin_topwire_roi, blur_size);
					}

					v_origin_topwire_roi.push_back(origin_topwire_roi);
					//LOG(INFO)<<"[API] origin_topwire_roi.size()=" << origin_topwire_roi.size() << std::endl; // 224*1024

					// origin_topwire_roi(224*1024)进行resize到标准大小128*512
					// 上下flip得到2张图像，送入net2,网络出来4张图,1,2,3,4
					// 1对应topwire_roi_input1的响应图，2对应topwire_roi_input2的响应图
					// 3是12的混合响应图(只用于训练，暂时不用结果),4是为轨缝设计的响应图(暂时不用结果)
					channel_mat_t channel_mat_topwire_roi_input1;
					cv::Mat topwire_roi_input1;
					cv::resize(origin_topwire_roi, topwire_roi_input1, cv::Size(topwire_roi_input_width2, topwire_roi_input_height2));
					channel_mat_topwire_roi_input1.push_back(topwire_roi_input1);

					channel_mat_t channel_mat_topwire_roi_input2;
					cv::Mat topwire_roi_input2;
					cv::flip(topwire_roi_input1, topwire_roi_input2, flip_code); // >0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
					//topwire_roi_input2 = topwire_roi_input1; // hack 

					channel_mat_topwire_roi_input2.push_back(topwire_roi_input2);

#ifdef shared_DEBUG
					{
						cv::imwrite("topwire/diff/" + to_string(m_counter) + "_origin_topwire_roi.jpg", origin_topwire_roi);
						cv::imwrite("topwire/diff/" + to_string(m_counter) + "_topwire_roi_input1.jpg", topwire_roi_input1);
						cv::imwrite("topwire/diff/" + to_string(m_counter) + "_topwire_roi_input2.jpg", topwire_roi_input2);
					}
					LOG(INFO)<<"[API] topwire_roi_input1.channels()=" << topwire_roi_input1.channels() << std::endl;
					LOG(INFO)<<"[API] topwire_roi_input2.channels()=" << topwire_roi_input2.channels() << std::endl;
#endif
					
					detect_n_inputs[0].blob_channel_mat.push_back(channel_mat_topwire_roi_input1);
					detect_n_inputs[1].blob_channel_mat.push_back(channel_mat_topwire_roi_input2);
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO)<<"[API-TOPWIRE] [3] before net2: cost= " << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


				//2张128*512的mat===>4张输出128*512 mat
				TopwireDetectNet::forward(
					detect_net,
					detect_n_inputs,
					detect_n_outputs
				); // 2-outputs


#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO)<<"[API-TOPWIRE] [4] forward net2: cost= " << cost*1.0 << std::endl;

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
						/*cv::Rect gap_box_in_origin;
						bool has_gap = false;
						v_has_gap.push_back(has_gap);
						v_gap_boxs.push_back(gap_box_in_origin);*/
						//=============================================================================
						boxs_t boxs;
						bool has_anomaly = false;
						v_has_anomaly.push_back(has_anomaly);
						v_anomaly_boxs.push_back(boxs);
					}
					else {
						const cv::Mat& origin_mat = v_image[i];
						channel_mat_t& channel_mat3 = v_output1[i]; // 3-channel  3*512*128
						channel_mat_t& channel_mat1 = v_output2[i]; // 1-channel  1*512*128

						cv::Mat result_mat_0 = channel_mat3[0];
						cv::Mat result_mat_1 = channel_mat3[1];
						cv::Mat result_mat_2 = channel_mat3[2];
						cv::Mat result_mat_3 = channel_mat1[0];

						//只使用4张图像的前2张
						cv::Mat merge_diff_output;
						cv::Mat flip_result1;
						cv::flip(result_mat_1, flip_result1, flip_code); // >0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
						//flip_result1 = result_mat_1; // hack

						bitwise_or(result_mat_0, flip_result1, merge_diff_output);

#ifdef shared_DEBUG
						{
							cv::imwrite("topwire/diff/" + to_string(m_counter) + "_result_0.jpg", result_mat_0);
							cv::imwrite("topwire/diff/" + to_string(m_counter) + "_result_1.jpg", result_mat_1);
							cv::imwrite("topwire/diff/" + to_string(m_counter) + "_result_2.jpg", result_mat_2);
							cv::imwrite("topwire/diff/" + to_string(m_counter) + "_result_3.jpg", result_mat_3);

							cv::imwrite("topwire/diff/" + to_string(m_counter) + "_result_1_flip.jpg", flip_result1);
							cv::imwrite("topwire/diff/" + to_string(m_counter) + "_merge_diff_output.jpg", merge_diff_output);
							LOG(INFO)<<"[API] merge_diff_output.size()=" << merge_diff_output.size() << std::endl;//128*512
						}
#endif

						// dilate diff 对diff图进行膨胀操作，连接细小的相应区域成为较大的区域
						//OpencvUtil::dilate_mat(merge_diff_output, dilate_size); // dilate_size
						
						//=========================================================
						//根据merge_diff_output获取diff中的异常boxs
						boxs_t boxs_in_diff;
						bool has_anomaly_with_holes = OpencvUtil::get_boxs_and(
							merge_diff_output,
							box_min_binary_threshold,
							5,
							5,
							boxs_in_diff
						);

						// (1) filter railway box by avg pixel
						boxs_t filtered_boxs_in_diff;
						if (filter_box_by_avg_pixel)
						{
							OpencvUtil::filter_topwire_boxs(
								v_origin_topwire_roi[i],
								boxs_in_diff,
								merge_diff_output,
								box_min_binary_threshold,
								filtered_boxs_in_diff
							);
						}
						else {
							filtered_boxs_in_diff = boxs_in_diff;
						}

						bool has_hole;
						boxs_t hole_boxs_in_diff;
						boxs_t normal_boxs_in_diff;
						remove_hole_boxs(
							filtered_boxs_in_diff,
							has_hole,
							hole_boxs_in_diff,
							normal_boxs_in_diff
						);

						// (2) filter railway box by stdev
						boxs_t normal_boxs_in_diff2;
						if (filter_box_by_stdev_pixel)
						{
							OpencvUtil::filter_railway_boxs2(
								v_origin_topwire_roi[i],
								normal_boxs_in_diff,
								box_expand_width,
								box_expand_height,
								filter_box_stdev_threshold,
								normal_boxs_in_diff2
							);
						}
						else {
							normal_boxs_in_diff2 = normal_boxs_in_diff;
						}

						//=========================================================
						// hole boxs
						boxs_t hole_boxs_in_origin;
						OpencvUtil::diff_boxs_to_origin_boxs(
							hole_boxs_in_diff,
							merge_diff_output.size(),
							v_origin_topwire_rect[i].size(),
							v_origin_topwire_rect[i].x,
							hole_boxs_in_origin
						);
						//=========================================================

						// (2) get normal boxs in origin
						boxs_t normal_boxs_in_origin;
						OpencvUtil::diff_boxs_to_origin_boxs(
							normal_boxs_in_diff2,
							merge_diff_output.size(),
							v_origin_topwire_rect[i].size(),
							v_origin_topwire_rect[i].x,
							normal_boxs_in_origin
						);

						// (3) filter origin boxs by height and width
						boxs_t filtered_normal_boxs_in_origin;
						OpencvUtil::get_boxs_or(
							normal_boxs_in_origin,
							box_min_height,
							box_min_width,
							filtered_normal_boxs_in_origin
						);

						bool has_anomaly = filtered_normal_boxs_in_origin.size() > 0;

						// hole
						v_has_hole.push_back(has_hole);
						v_hole_boxs.push_back(hole_boxs_in_origin);
						// anomaly
						v_has_anomaly.push_back(has_anomaly);
						v_anomaly_boxs.push_back(filtered_normal_boxs_in_origin);
					}
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO)<<"[API-TOPWIRE] [5] after net2: cost= " << cost*1.0 << std::endl;
#endif // DEBUG_TIME

				m_counter++;

				return true;
			}

			bool TopwireApiImpl::remove_hole_boxs(
				const boxs_t& boxs_with_holes,
				bool& has_hole,
				boxs_t& hole_boxs,
				boxs_t& normal_boxs
			)
			{
#ifdef shared_DEBUG
				// diff:  128;  hole pos:  
				LOG(INFO)<<"[API] boxs_with_holes.size()=" << boxs_with_holes.size() << std::endl;
#endif 
				has_hole = false;
				if (boxs_with_holes.size()== m_hole_count)
				{

#ifdef shared_DEBUG
					for (size_t i = 0; i < boxs_with_holes.size(); i++)
					{
						LOG(INFO)<<"[API] hole size = " << boxs_with_holes[i].size() << std::endl;
					}
#endif 
					/*
					diff box size:
					hole size = [11 x 15]
					hole size = [11 x 14]

					hole size = [15 x 14]
					hole size = [15 x 14]

					hole size = [14 x 18]
					hole size = [12 x 19]

					hole size = [21 x 20]
					hole size = [15 x 16]
					*/

					// remove 2 holes
					cv::Rect left_box = boxs_with_holes[0];
					cv::Rect right_box = boxs_with_holes[1];

					// left on the right side,then swap
					if (left_box.x+ left_box.width/2 > right_box.x + right_box.width / 2)
					{
						std::swap(left_box, right_box);
					}

					// (1) filter with hole size 
					bool is_hole_box_flag = false;
					if ( is_hole_box(left_box) && is_hole_box(right_box) )
					{
						is_hole_box_flag = true;
					}
					else {

#ifdef shared_DEBUG
						LOG(INFO)<<"[API]  (1) is_hole_box_flag = false \n";
#endif
					}

					// (2) make sure 2 holes distance >= 10
					
					bool hole_distance_ok = false;
					int hole_distance_threshold = 10;
					int hole_distance = abs(right_box.x - left_box.x - left_box.width);
					if (hole_distance >= hole_distance_threshold)
					{
						hole_distance_ok = true;
					}
					else {

#ifdef shared_DEBUG
						LOG(INFO)<<"[API]  (2) hole_distance_ok = false \n";
#endif
					}


					// (3) make sure 2 holes are on the same row
					bool is_on_same_row_flag = false;
					int row_delta_up = abs(left_box.y - right_box.y);
					int row_delta_down = abs(left_box.y+ left_box.height - right_box.y - right_box.height);
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] row_delta_up = " << row_delta_up << std::endl;
					LOG(INFO)<<"[API] row_delta_down = " << row_delta_down << std::endl;
#endif
					// or 
					if (row_delta_up <= row_delta_threshold || row_delta_down <= row_delta_threshold)
					{
						is_on_same_row_flag = true;
					}
					else 
					{
#ifdef shared_DEBUG
						LOG(INFO)<<"[API]  (3) is_on_same_row_flag = false \n";
#endif
					}

					// (4) make sure 2 holes are symmetric
					bool is_sysmmetric_flag = false;

#ifdef shared_DEBUG
					std::cout <<  "x1 = "<<left_box.x << std::endl;
					std::cout <<  "x2 = " << left_box.x + left_box.width << std::endl;
					std::cout <<  "x3 = " << right_box.x << std::endl;
					std::cout <<  "x4 = " << right_box.x + right_box.width << std::endl;
#endif
					/*
					x1 = 32
					x2 = 44
					x3 = 74
					x4 = 88

					x1 = 30
					x2 = 45
					x3 = 74
					x4 = 87
					*/
					if (left_box.x>=25 && left_box.x+ left_box.width<=65 &&
						right_box.x>=65 && right_box.x + right_box.width<=105
						)
					{
						is_sysmmetric_flag = true;
					}
					else
					{
#ifdef shared_DEBUG
						LOG(INFO)<<"[API]  (4) is_sysmmetric_flag = false \n";
#endif
					}

					if (is_hole_box_flag && hole_distance_ok && is_on_same_row_flag && is_sysmmetric_flag)
					{
						has_hole = true;
						hole_boxs.push_back(left_box);
						hole_boxs.push_back(right_box);
#ifdef shared_DEBUG
						// remove 2 holes at last, so normal_boxs are null.
						LOG(INFO)<<"[API] ======================================================\n";
						LOG(INFO)<<"[API] remove 2 holes at last \n";
						LOG(INFO)<<"[API] ======================================================\n";
#endif
					}
					else {
						// 2 boxs are not holes, kept with results
						normal_boxs = boxs_with_holes;
					}
				}
				else {
					normal_boxs = boxs_with_holes;
				}

				return normal_boxs.size() > 0;
			}

			bool TopwireApiImpl::is_hole_box(
				const cv::Rect& hole_box
			)
			{
				return (hole_box.width >= hole_width - hole_width_delta) &&
					(hole_box.width <= hole_width + hole_width_delta) &&
					(hole_box.height >= hole_height - hole_height_delta) &&
					(hole_box.height <= hole_height + hole_height_delta);
			}
#pragma endregion


		}
	}
}// end namespace

