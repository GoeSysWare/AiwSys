﻿#include "lahu_api_impl.h"
#include "lahu_net.h"

#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"
#include "algorithm/core/util/numpy_util.h"
#include "algorithm/core/profiler.h"

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

namespace watrix {
	namespace algorithm {
		namespace internal {

			int LahuApiImpl::m_counter = 0; // init
			std::vector<float> LahuApiImpl::m_bgr_mean = {0,0,0}; // init (keystep, otherwise linker error)
			LahuParam LahuApiImpl::m_param; // init

			void LahuApiImpl::init(
				const caffe_net_file_t& detect_net_params,
				int net_count,
				const LahuParam& lahu_param
			)
			{
				LahuNet::init(detect_net_params,net_count);
				LahuApiImpl::m_param = lahu_param;
			}

			void LahuApiImpl::free()
			{
				LahuNet::free();
			}

			void LahuApiImpl::set_bgr_mean(const std::vector<float>& bgr_mean)
			{
				for (size_t i = 0; i < bgr_mean.size(); i++)
				{
					LahuApiImpl::m_bgr_mean.push_back(bgr_mean[i]);
				}
			}

			bool LahuApiImpl::detect(
				int net_id,
				const std::vector<cv::Mat>& v_image,
				std::vector<bool>& v_has_lahu,
				std::vector<float>& v_score1,
				std::vector<float>& v_score2,
				std::vector<cv::Rect>& v_boxes
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, LahuNet::v_net.size()) << "net_id invalid";
				shared_caffe_net_t lahu_net = LahuNet::v_net[net_id];

				return LahuApiImpl::detect(
					lahu_net,
					v_image,
					v_has_lahu,
					v_score1,
					v_score2,
					v_boxes
				);
			}

			bool LahuApiImpl::detect(
				shared_caffe_net_t net,
				const std::vector<cv::Mat>& v_image,
				std::vector<bool>& v_has_lahu,
				std::vector<float>& v_score1,
				std::vector<float>& v_score2,
				std::vector<cv::Rect>& v_boxes
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
				CHECK(batch_size>=1) << "invalid batch_size";

				for (size_t i = 0; i < v_image.size(); i++)
				{
					CHECK(!v_image[i].empty()) << "invalid mat";
					CHECK(v_image[i].channels()==3) << "mat channels must ==3";
				}

				// ============================================================
				int input_width = LahuNet::input_width;
				int input_height = LahuNet::input_height;
				cv::Size input_size(input_width, input_height);

				int output_width = LahuNet::output_width;
				int output_height = LahuNet::output_height;
				cv::Size output_size(output_width, output_height);

				int image_width = v_image[0].cols;
				int image_height = v_image[0].rows;
				cv::Size origin_size(image_width, image_height);
				// ============================================================

				std::vector<cv::Mat> v_resized_input; // resized image
				
				/*
				1288,964,3 ===>get gray roi ===>resize/scale to 256,256,1 ===> 
				n,1,256,256 ===> forward ===> n,2,1,1 
				 */
				std::vector<bool> v_roi_success;
				v_roi_success.resize(batch_size, false);

				double normalize_value = 1;
				for (size_t i = 0; i < v_image.size(); i++)
				{
					cv::Rect roi_box;
					cv::Mat roi_image; // gray image
					bool success = get_lahu_roi_box(v_image[i], roi_box, roi_image);
					v_roi_success[i] = success; // roi may failed

					v_boxes.push_back(roi_box);

#ifdef DEBUG_INFO 
					std::cout<<" success = "<< success << std::endl;
					std::cout<<" roi_image.empty() = "<< roi_image.empty() << std::endl;
					std::cout<<" roi_box = "<< roi_box << std::endl;
#endif 

					cv::Mat float_image; 
					roi_image.convertTo(float_image, CV_32FC1, normalize_value); 
				
					cv::Mat resized_image;
					cv::resize(float_image, resized_image, input_size); // 256,256,1

					v_resized_input.push_back(resized_image);
				}

				CaffeNet::caffe_net_n_inputs_t n_inputs;
				CaffeNet::caffe_net_n_outputs_t n_outputs;
				LahuNet::get_inputs_outputs(n_inputs, n_outputs); // 1-inputs,1-outputs

				// hwc ===> chw  (c==1)
				for (size_t i = 0; i < batch_size; i++)
				{
					channel_mat_t channel_mat{v_resized_input[i]};
					n_inputs[0].blob_channel_mat.push_back(channel_mat);
				}			

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LAHU] [1] pre-process data: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				pre_cost += cost;
				LOG(INFO) << "[API-LAHU] #counter ="<<m_counter<<" pre_cost=" << pre_cost/(m_counter*1.0) << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				internal::LahuNet::forward(
					net,
					n_inputs,
					n_outputs,
					true // get float output  (prob score)
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LAHU] [3] forward net1: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				forward_cost += cost;
				LOG(INFO) << "[API-LAHU] #counter ="<<m_counter<<" forward_cost=" << forward_cost/(m_counter*1.0) << std::endl;

				
				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME


				//========================================================================
				// get output  N,2,1,1 (float prob)
				//========================================================================

				blob_channel_mat_t& blob_output = n_outputs[0].blob_channel_mat; // 1-outputs

				for (size_t i = 0; i < batch_size; i++)
				{
					// 2,1,1 
					channel_mat_t channel_mat = blob_output[i]; 

#ifdef DEBUG_INFO
					std::cout<<"channel_mat.size() = "<< channel_mat.size() << std::endl;
#endif

					cv::Mat& feature_0 = channel_mat[0];
					cv::Mat& feature_1 = channel_mat[1];  
					float score_0 = feature_0.at<float>(0,0);
					float score_1 = feature_1.at<float>(0,0);

#ifdef DEBUG_INFO
					printf("roi_success = %d, score_0 = %f, score_1 = %f \n",
						int(v_roi_success[i]), score_0, score_1
					);
#endif

					cv::Mat max_prob = NumpyUtil::np_argmax_axis_channel2(channel_mat); // 1,1  [0,1]
					int max_prob_index = max_prob.at<uchar>(0,0); // 0 or 1 for c=2
					
#ifdef DEBUG_INFO
					std::cout<<" max_prob_index = "<< max_prob_index << std::endl;
#endif 	
					bool has_lahu = v_roi_success[i] && (score_0 >=m_param.score_threshold);
					v_score1.push_back(score_0);
					v_score2.push_back(score_1);
					v_has_lahu.push_back(has_lahu);
				}

				/*
				feature_0 = [0.99976975]
 				feature_1 = [0.00023029452]
				 */

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-LAHU] [3] post process: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();

				total_cost += cost;
				post_cost += cost;
				LOG(INFO) << "[API-LAHU] #counter ="<<m_counter<<" post_cost=" << post_cost/(m_counter*1.0) << std::endl;

				LOG(INFO) << "[API-LAHU] #counter ="<<m_counter<<" total_cost=" << total_cost/(m_counter*1.0) << std::endl;
#endif // DEBUG_TIME
				
				return true;
			}


			/* 
			bool LahuApiImpl::postprocess_binary_mask(
				const cv::Mat& feature_map,
				const cv::Size& origin_size,
				const cv::Size& input_size,
				const cv::Size& output_size,
				cv::Rect& box 
			)
			{
				//binary_mask:  24,34   CV_8FC1   0-255
				//origin_size: wh 1080,1920
				//input_size: wh 964,1288
				//output_size:wh 24,34
				//return: has_lahu, box
				
				bool has_lahu = false;
				int count = 0;
				int height = feature_map.rows;
				int width = feature_map.cols;

				for (int h = 0; h < height; h++)
				{
					const uchar *p = feature_map.ptr<uchar>(h);
					for (int w = 0; w < width; w++)
					{
						if (*p >= LahuApiImpl::m_param.pixel_threshold_for_count) {  
							count ++;
						}
						p++;
					}
				}

#ifdef DEBUG_INFO
				printf("count = %d \n", count);
#endif

				if (count >= LahuApiImpl::m_param.lahu_count_threshold){
					has_lahu = true;

					float width_scale = origin_size.width / (output_size.width*1.0);
					float height_scale = origin_size.height / (output_size.height*1.0);

					cv::Mat binary; // 0, 255
					cv::threshold(feature_map, binary, LahuApiImpl::m_param.pixel_threshold_for_box, 255, CV_THRESH_BINARY); 

					// 进行连通域分析
					cv::Mat labels; // w*h  label = 0,1,2,3,...N-1 (0- background)       CV_32S = 4
					cv::Mat stats; // N*5  表示每个连通区域的外接矩形和面积 [x,y,w,h, area]   CV_32S = 4
					cv::Mat centroids; // N*2  (x,y)                                     CV_32S = 4
					int num_components = connectedComponentsWithStats(binary, labels, stats, centroids, 4, CV_32S);
					

#ifdef DEBUG_INFO
					//cv::imwrite("binary.jpg", binary);
					std::cout<<" num_components =" << num_components << std::endl; // 8
					std::cout<<" stats hw =" << stats.cols <<","<<stats.rows << std::endl; // 8*5
					std::cout<<" stats =" << stats << std::endl;
#endif

					int min_x1 = 0;
					int min_y1 = 0;
					int max_x1 = 0;
					int max_y1 = 0;
					if (num_components>=2){ // at least 2 
						int row=1;
						int s0 = stats.at<int>(row,0);
						int s1 = stats.at<int>(row,1);
						int s2 = stats.at<int>(row,2);
						int s3 = stats.at<int>(row,3);

						min_x1 = s0;
						min_y1 = s1;
						max_x1 = s0+s2;
						max_y1 = s1+s3;

						for(row=2;row<num_components;row++){
							int s0 = stats.at<int>(row,0);
							int s1 = stats.at<int>(row,1);
							int s2 = stats.at<int>(row,2);
							int s3 = stats.at<int>(row,3);

							if (s0<min_x1){
								min_x1 = s0;
							}
								
							if (s1<min_y1){ 
								min_y1 = s1;
							}

							if (s0+s2>max_x1){
								max_x1 = s0+s2;
							}

							if (s1+s3>max_y1){
								max_y1 = s1+s3;
							}
						}
					}

#ifdef DEBUG_INFO
					printf("%d,%d,%d,%d,\n", min_x1, min_y1, max_x1, max_y1);
					// (20, 7, 23, 23)
#endif
					//x1 = min_x1*964/24-150
					//y1 = min_y1*1288/34
					//x2 = max_x1*964/24+50
					//y2 = max_y1*1288/34+50
					
					int x1 = int(min_x1*width_scale-150); // -150
					int y1 = int(min_y1*height_scale);
					int x2 = int(max_x1*width_scale+50);
					int y2 = int(max_y1*height_scale+50);

					if (x1<=0){
						x1 = 1; 
					}
					if (y1<=0){
						y1 = 1;
					}
					if (x2>=origin_size.width){
						x2 = origin_size.width-1;
					}
					if (y2>=origin_size.height){
						x2 = origin_size.height-1;
					}

					box.x = x1;
					box.y = y1;
					box.width = x2 - x1;
					box.height = y2 - y1;

#ifdef DEBUG_INFO
					printf("box(x1,y1,x2,y2) %d,%d,%d,%d,\n", x1, y1, x2, y2);
#endif				
				}

				return has_lahu;
			}
			*/


			bool LahuApiImpl::sort_rect_by_width(
				const cv::Rect& left_rect,
				const cv::Rect& right_rect
			)
			{
				return left_rect.width > right_rect.width;
			}


			bool LahuApiImpl::get_lahu_roi_box(
				const cv::Mat& image, cv::Rect& box, cv::Mat& roi
			)
			{
				// roi (256,256,1)
				box = cv::Rect(0,0,0,0);
				// may be failed, so we must init roi with default
				roi = cv::Mat(256, 256, CV_8UC1, cv::Scalar(0)); 

				cv::Mat gray;
				NumpyUtil::cv_cvtcolor_to_gray(image, gray);

				float factor = 4.;
				int minVal=10;
				int maxVal=160;
				
				int h = image.rows;
				int w = image.cols; 
				int resized_w = int(w/factor);
				int resized_h = int(h/factor);

				cv::Mat resized;
				NumpyUtil::cv_resize(gray, resized, cv::Size(resized_w,resized_h));

				cv::Mat edges;
				NumpyUtil::cv_canny(resized, edges, minVal, maxVal);

				NumpyUtil::cv_dilate_mat(edges, 3);

				contours_t contours;
				NumpyUtil::cv_findcounters(edges, contours);

				// get boxs
				boxs_t boxs;
				int roi_box_width_threshold =  20;
				for(auto& contour: contours){
					cv::Rect box = cv::boundingRect(contour); // x,y,w,h
					if (box.width<45 && (box.x<2 || box.x+box.width > w-2)){
						continue;
					} else if (box.width > roi_box_width_threshold){
						cv::Rect origin_box;
						origin_box.x = int(box.x * factor);
						origin_box.y = int(box.y * factor);
						origin_box.width = int(box.width* factor);
						origin_box.height = int(box.height* factor);

						NumpyUtil::cv_boundary(origin_box, image.size());

						boxs.push_back(origin_box);
					}
				}

#ifdef DEBUG_INFO
				std::cout<<"boxs = "<<boxs.size()<<std::endl;
#endif 

				boxs_t new_boxs;
				NumpyUtil::nms_fast(boxs, 0.45, new_boxs);

				if (new_boxs.size()<1){
					return false;
				}

#ifdef DEBUG_INFO
				std::cout<<"after NMS, new_boxs = "<<new_boxs.size()<<std::endl;
#endif
				// Sort the score pair according to the scores in descending order
				std::stable_sort(
					new_boxs.begin(),
					new_boxs.end(),
					sort_rect_by_width
				);

#ifdef DEBUG_IMAGE 
				{
					cv::Mat image_with_boxs;
					DisplayUtil::draw_boxs(image, new_boxs, 2, image_with_boxs);
					cv::imwrite("3_image_with_boxs1_all.png",image_with_boxs);
				}
#endif
			

				cv::Rect first_box = new_boxs[0];
				int _x1 = first_box.x;
				//int _y1 = first_box.y;
				int _x2 = first_box.x + first_box.width;
				//int _y2 = first_box.y + first_box.height;

#ifdef DEBUG_IMAGE 
				{
					cv::Mat image_with_boxs2;
					DisplayUtil::draw_box(image, first_box, 2, image_with_boxs2);
					cv::imwrite("3_image_with_boxs2_first.png",image_with_boxs2);
				}
#endif

				// left expand min_x and right expand max_x
				int min_x = _x1;
				int max_x = _x2;
				int min_y, max_y; 

				for(int i=1; i<new_boxs.size(); ++i){
					cv::Rect new_box = new_boxs[i];
					int x1 = new_box.x;
					//int y1 = new_box.y;
					int x2 = new_box.x + new_box.width;
					//int y2 = new_box.y + new_box.height;
					if (first_box.width/new_box.width<=2){
						min_x = min(min_x, x1);
						max_x = max(max_x, x2);
					}
				}
				

#ifdef DEBUG_IMAGE 
				{
					//printf("[bbb] x1 = %d, x2 = %d, y1 = %d, y2 = %d  \n", min_x, max_x, min_y, max_y);
					cv::Rect tmp_box_bbb{min_x,min_y, max_x-min_x, max_y-min_y};
					cv::Mat image_with_boxs_bbb;
					DisplayUtil::draw_box(image, tmp_box_bbb, 2, image_with_boxs_bbb);
					cv::imwrite("3_image_with_boxs3_bbb.png",image_with_boxs_bbb);
				}
#endif

				min_x = min_x + (max_x - min_x)/2.0;
				if ((max_x - min_x) < 250 ){
					max_x += 50;
				} else {
					max_x += 20;
				}

				if (min_x<0){
					min_x = 0;
				}
				if (max_x>=w){
					max_x = w-1;
				}

				min_y = 241;
				max_y = 241+max_x-min_x;

				//printf("[ccc] x1 = %d, x2 = %d, y1 = %d, y2 = %d  \n", min_x, max_x, min_y, max_y);
				if (min_y> max_y){
					return false;
				}

#ifdef DEBUG_IMAGE 
				{
					cv::Rect tmp_box_ccc{min_x,min_y, max_x-min_x, max_y-min_y};
					cv::Mat image_with_boxs_ccc;
					DisplayUtil::draw_box(image, tmp_box_ccc, 2, image_with_boxs_ccc);
					cv::imwrite("3_image_with_boxs3_ccc.png",image_with_boxs_ccc);
				}
#endif

				box.x = min_x;
				box.y = min_y;
				box.width = max_x - min_x;
				box.height = max_y - min_y;

				roi = gray(box); // pass out roi image (gray)

#ifdef DEBUG_IMAGE
				{
					cv::Mat image_with_boxs3;
					DisplayUtil::draw_box(image, box, 2, image_with_boxs3);
					cv::imwrite("3_image_with_boxs4_roi.png",image_with_boxs3);
				}
#endif

				return true;
			}


		}
	}
}// end namespace


/*
[[  0   0  34  24 794]
 [ 22   7   1   1   1]
 [ 20  12   3  11  21]]
(20, 7, 23, 23)
('box', 653, 265, 973, 921)

*/