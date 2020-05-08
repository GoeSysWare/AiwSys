#include "pt_simple_laneseg_api_impl.h"
#include "projects/adas/algorithm/autotrain/internal/laneseg_util.h"

#include "projects/adas/algorithm/core/util/filesystem_util.h"
#include "projects/adas/algorithm/core/util/display_util.h"
#include "projects/adas/algorithm/core/util/opencv_util.h"
#include "projects/adas/algorithm/core/util/numpy_util.h"
#include "projects/adas/algorithm/core/util/polyfiter.h"

// third
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

			PtSimpleLaneSegNetParams PtSimpleLaneSegApiImpl::params;
			std::vector<pt_module_t> PtSimpleLaneSegApiImpl::v_net;

			int PtSimpleLaneSegApiImpl::m_counter = 0; // init
			std::vector<float> PtSimpleLaneSegApiImpl::m_bgr_mean; // init (keystep, otherwise linker error)

			void PtSimpleLaneSegApiImpl::init(
				const PtSimpleLaneSegNetParams& params,
				int net_count
			)
			{
				std::cout<<"PtSimpleLaneSegApiImpl::init \n";
				PtSimpleLaneSegApiImpl::params = params;
				v_net.resize(net_count);
				for (int i = 0; i < v_net.size(); i++)
				{
					v_net[i] = torch::jit::load(params.model_path);
					//assert(v_net[i] != nullptr);
    				v_net[i].to(at::kCUDA);
				}
			}

			void PtSimpleLaneSegApiImpl::free()
			{
				for (int i = 0; i < v_net.size(); i++)
				{
					//v_net[i] = nullptr;
				}
			}

			void PtSimpleLaneSegApiImpl::set_bgr_mean(const std::vector<float>& bgr_mean)
			{
				for (size_t i = 0; i < bgr_mean.size(); i++)
				{
					PtSimpleLaneSegApiImpl::m_bgr_mean.push_back(bgr_mean[i]);
				}
			}

#pragma region lane seg
			bool PtSimpleLaneSegApiImpl::lane_seg(
				int net_id,
				const std::vector<cv::Mat>& v_image,
				int min_area_threshold, 
				std::vector<cv::Mat>& v_binary_mask, // 256,1024
				std::vector<channel_mat_t>& v_instance_mask // 8,256,1024
			)
			{
				//std::cout<<"===================================111\n";
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, v_net.size()) << "net_id invalid";
				pt_module_t& net = v_net[net_id];
				
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
				int ORIGIN_WIDTH = lanesegutil::ORIGIN_SIZE.width; // (1920,1080)  1920
				int ORIGIN_HEIGHT = lanesegutil::ORIGIN_SIZE.height; // (1920,1080)  1080
				//pt_simple v1
				// int CLIP_HEIGHT = lanesegutil::CLIP_SIZE.height; // 512 --->640

				// // pt_simple v2 v3 v4 resize 160 480
				// int HEIGHT = lanesegutil::PT_SIMPLE_INPUT_SIZE.height; // 160
				// int WIDTH = lanesegutil::PT_SIMPLE_INPUT_SIZE.width; // 480

				// pt_simple v5 240 480
				int HEIGHT = lanesegutil::PT_SIMPLE_INPUT_SIZE.height; // 240
				int WIDTH = lanesegutil::PT_SIMPLE_INPUT_SIZE.width; // 480
				
				int batch_count = v_image.size();
				int image_size = WIDTH*HEIGHT*3;
				//std::cout<<"batch_count :  "<<batch_count<<std::endl;
				int batch_h = batch_count * HEIGHT;
				int batch_w = WIDTH;
				cv::Mat batch_image(batch_h, batch_w, CV_8UC3);

				for (size_t i = 0; i < batch_count; i++)
				{
					cv::Mat image = v_image[i];

					// clip 568*1920 
					cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // hwc
					// 568*1920 ===>128*480
					cv::cvtColor(image,image, cv::COLOR_BGR2RGB); // bgr--->rgb
					//image = image(cv::Range(1080-CLIP_HEIGHT, 1080),cv::Range(0,1920));

					//pt_simple v1
					// image = image(cv::Range(ORIGIN_HEIGHT-CLIP_HEIGHT, ORIGIN_HEIGHT),cv::Range(0,ORIGIN_WIDTH));

					//pt_simple v2 v3 v4
					// image = image(cv::Range(440, 1080),cv::Range(0,ORIGIN_WIDTH));
					// //resize 160 480
					// cv::resize(image, image, cv::Size(WIDTH, HEIGHT));

					//pt_simple v5
					//resize 240 480
					cv::resize(image, image, cv::Size(WIDTH, HEIGHT));

					memcpy(batch_image.data+i*image_size, (char*)image.data, image_size);
				}

				/*
				//get instance mask = 8,384,512   8-dim features, float 
				int dims = 8;
				for(int c=0; c< dims; c++)
				{
					cv::Mat instance_feature_map(HEIGHT, WIDTH, CV_32FC1, cv::Scalar(0.)); // float 
					memcpy(instance_feature_map.data, (char*)(data_ins + channel_step*c) , 4*channel_step);
					instance_mask.push_back(instance_feature_map);
				}
				 */

				//cv::imwrite("batch_image.jpg",batch_image);

				at::TensorOptions options(at::ScalarType::Byte);
				torch::Tensor batch_tensor = torch::from_blob(batch_image.data, {batch_count, HEIGHT, WIDTH, 3}, options);
				//must put here
				batch_tensor = batch_tensor.to(torch::kCUDA);
				batch_tensor = batch_tensor.permute({0, 3, 1, 2}); // nhwc ---> nchw
				batch_tensor = batch_tensor.toType(torch::kFloat32);// uin8--->float
				//normalize [0-255]--->[0-1]---> substract bgr 
				batch_tensor = batch_tensor.div(255.0);
				
				for (size_t i = 0; i < batch_count; i++)
				{
					batch_tensor[i][0] = batch_tensor[i][0].sub(0.5).div(0.5);
					batch_tensor[i][1] = batch_tensor[i][1].sub(0.5).div(0.5);
					batch_tensor[i][2] = batch_tensor[i][2].sub(0.5).div(0.5);
				}
				//std::vector<torch::jit::IValue> inputs; // batch inputs 
				// inputs.push_back(torch::ones({2, 3, 128, 480}).to(torch::kCUDA));
				//torch::Tensor out_tensor = net.forward({batch_tensor}).toTensor(); // nchw
#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] [1] pre-process data: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				pre_cost += cost;
				//std::cout<<"[API-PTSIMPLE-LANESEG] #counter =="<<pre_cost<<std::endl;
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] #counter ="<<m_counter<<" pre_cost=" << pre_cost/(m_counter*1.0) << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				torch::Tensor out_tensor = net.forward({batch_tensor}).toTensor(); // nchw
				// out_tensor = out_tensor.squeeze();
				std::cout<<"out_tensor.sizes() = "<<out_tensor.sizes() << std::endl;
				// std::cout<<"out_tensor[0].sizes() = "<<out_tensor[0].sizes() << std::endl;
			
			
#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] [2] forward net1: cost=" << cost*1.0 << std::endl;
				total_cost+=cost;
				forward_cost += cost;
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] #counter ="<<m_counter<<" forward_cost=" << forward_cost/(m_counter*1.0) << std::endl;			
#endif // DEBUG_TIME
				
				//pt_simple v1
				// max for dim 1  [0,1,2,3]
				// std::tuple<torch::Tensor,torch::Tensor> result = out_tensor.max(1, true);

#ifdef DEBUG_INFO
			if (false){
				std::cout<<"out_tensor.sizes() = "<< out_tensor.sizes()<<std::endl;  // [2, 4, 128, 480]
				std::cout<<"0 ---"<< std::get<0>(result).sizes()<<std::endl;  // [2, 1, 128, 480]
				std::cout<<"1 ---"<< std::get<1>(result).sizes()<<std::endl;  // [2, 1, 128, 480]
			}
#endif

				for (size_t i = 0; i < batch_count; i++)
				{
					//pt_simple v1
					// torch::Tensor top_scores = std::get<0>(result)[i]; // [1, 128, 480]
					// torch::Tensor top_idxs = std::get<1>(result)[i].toType(torch::kInt32).cpu(); // [1, 128, 480]

					//pt_simple v2
					torch::Tensor top_idxs = out_tensor[i].toType(torch::kInt32).cpu(); // [1, 160, 480]

					//std::cout<<"top_scores ---"<< top_scores.sizes()<<std::endl; 
					//std::cout<<"top_idxs ---"<< top_idxs.sizes()<<std::endl; 

					cv::Mat filtered_binary_mask_01(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
					
					channel_mat_t output_instance_seg;
					cv::Mat left_lane_instance (HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
					cv::Mat right_lane_instance(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));

					int* data = (int*)top_idxs.data_ptr(); // 1*128*420

					// char tmp[WIDTH * HEIGHT];
					// memset(tmp, 0, WIDTH * HEIGHT);

					// 0 background, 1 left, 2 right
					for (int h = 0; h < HEIGHT; h++) {
						for (int w = 0; w < WIDTH; w++) {
							int val = 0;
							val = *(data+h*WIDTH+w);
							// std::cout << "val:" << val << std::endl;
							if (val == params.surface_id){ // surface  (no use)
								// tmp[h * WIDTH + w] = 0; // image[y,x] = 255
								filtered_binary_mask_01.at<uchar>(h,w) = 1; // mark as black
							}
							else if (val == params.left_id){ // left lane
								// tmp[h * WIDTH + w] = 128;
								left_lane_instance.at<uchar>(h,w) = 1; // mark as black
							}
							else if (val == params.right_id){ // right lane
								// tmp[h * WIDTH + w] = 255;
								right_lane_instance.at<uchar>(h,w) = 1; // mark as black
							}
						}
					}

					// 进行连通域分析
					cv::Mat labels; // w*h  label = 0,1,2,3,...N-1 (0- background)       CV_32S = 4
					cv::Mat stats; // N*5  表示每个连通区域的外接矩形和面积 [x,y,w,h, area]   CV_32S = 4
					cv::Mat centroids; // N*2  (x,y)                                     CV_32S = 4
					int num_components; // 连通区域number
					
					num_components = connectedComponentsWithStats(left_lane_instance, labels, stats, centroids, 4, CV_32S);
					cv::Mat left_lane_instance_new(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
					for (int h = 0; h < HEIGHT; h++) {
						int min_label = num_components;
						for (int w = 0; w < WIDTH; w++){
							int label = labels.at<int>(h,w);
							if (label != 0 && label <= min_label) min_label = label;
						}

						if (min_label != num_components){
							for (int w = 0; w < WIDTH; w++) {
								int label = labels.at<int>(h,w);
								if (label == min_label) left_lane_instance_new.at<uchar>(h,w) = 1;
							}
						}
					}

					num_components = connectedComponentsWithStats(right_lane_instance, labels, stats, centroids, 4, CV_32S);
					cv::Mat right_lane_instance_new(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
					for (int h = 0; h < HEIGHT; h++) {
						int min_label = num_components;
						for (int w = 0; w < WIDTH; w++){
							int label = labels.at<int>(h,w);
							if (label != 0 && label <= min_label) min_label = label;
						}

						if (min_label != num_components){
							for (int w = 0; w < WIDTH; w++) {
								int label = labels.at<int>(h,w);
								if (label == min_label) right_lane_instance_new.at<uchar>(h,w) = 1;
							}
						}
					}


					output_instance_seg.push_back(left_lane_instance_new);
					output_instance_seg.push_back(right_lane_instance_new);

#ifdef DEBUG_INFO_FWC
					char tmp[WIDTH * HEIGHT];
					memset(tmp, 0, WIDTH * HEIGHT);
					// 0 background, 1 left, 2 right
					for (int h = 0; h < HEIGHT; h++) {
						for (int w = 0; w < WIDTH; w++) {
							int val_left = left_lane_instance_new.at<uchar>(h,w);
							int val_right = right_lane_instance_new.at<uchar>(h,w);
							// std::cout << "val:" << val << std::endl;
							if (val_left == 1){ // left lane
								tmp[h * WIDTH + w] = 128;
							}
							if (val_right == 1){ // right lane
								tmp[h * WIDTH + w] = 255;
							}
						}
					}
					cv::Mat segment_image(HEIGHT, WIDTH, CV_8UC1, tmp);
					cv::imwrite("./segmentation"+std::to_string(i)+"_.png", segment_image);
#endif
					v_binary_mask.push_back(filtered_binary_mask_01); // (128, 420) v=[0,1]
					v_instance_mask.push_back(output_instance_seg);// (2, 128, 420) int value v=[0,1]
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] [3] post process: cost=" << cost*1.0 << std::endl;
				pt1 = boost::posix_time::microsec_clock::local_time();

				total_cost+=cost;
				post_cost += cost;
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] #counter ="<<m_counter<<" post_cost=" << post_cost/(m_counter*1.0) << std::endl;

				LOG(INFO) << "[API-PTSIMPLE-LANESEG] #counter ="<<m_counter<<" total_cost=" << total_cost/(m_counter*1.0) << std::endl;
#endif // DEBUG_TIME
				
				return true;
			}

#pragma region lane seg sequence
			bool PtSimpleLaneSegApiImpl::lane_seg_sequence(
				int net_id,
				const std::vector<cv::Mat>& v_image_front_result, 
				const std::vector<cv::Mat>& v_image_cur,
				int min_area_threshold, 
				std::vector<cv::Mat>& v_binary_mask, // 256,1024
				std::vector<channel_mat_t>& v_instance_mask // 8,256,1024
			)
			{
				//std::cout<<"===================================111\n";
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, v_net.size()) << "net_id invalid";
				pt_module_t& net = v_net[net_id];
				
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
				int ORIGIN_WIDTH = lanesegutil::ORIGIN_SIZE.width; // (1920,1080)  1920
				int ORIGIN_HEIGHT = lanesegutil::ORIGIN_SIZE.height; // (1920,1080)  1080
				//pt_simple v1
				// int CLIP_HEIGHT = lanesegutil::CLIP_SIZE.height; // 512 --->640
				int HEIGHT = lanesegutil::PT_SIMPLE_INPUT_SIZE.height; // 272
				int WIDTH = lanesegutil::PT_SIMPLE_INPUT_SIZE.width; // 480
				
				int batch_count = v_image_cur.size();
				int image_size = WIDTH*HEIGHT*3;
				//std::cout<<"batch_count :  "<<batch_count<<std::endl;
				int batch_h = batch_count * HEIGHT;
				int batch_w = WIDTH;
				cv::Mat batch_image(batch_h, batch_w, CV_8UC3);

				for (size_t i = 0; i < batch_count; i++)
				{
					cv::Mat image = v_image_cur[i];

					// clip 568*1920 
					cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // hwc
					// 568*1920 ===>128*480
					cv::cvtColor(image,image, cv::COLOR_BGR2RGB); // bgr--->rgb
					//image = image(cv::Range(1080-CLIP_HEIGHT, 1080),cv::Range(0,1920));

					//pt_simple v1
					// image = image(cv::Range(ORIGIN_HEIGHT-CLIP_HEIGHT, ORIGIN_HEIGHT),cv::Range(0,ORIGIN_WIDTH));

					//pt_simple v2
					// image = image(cv::Range(440, 1080),cv::Range(0,ORIGIN_WIDTH));
					cv::resize(image, image, cv::Size(WIDTH, HEIGHT));

					memcpy(batch_image.data+i*image_size, (char*)image.data, image_size);
				}


				at::TensorOptions options(at::ScalarType::Byte);
				torch::Tensor batch_tensor = torch::from_blob(batch_image.data, {batch_count, HEIGHT, WIDTH, 3}, options);
				//must put here
				batch_tensor = batch_tensor.to(torch::kCUDA);
				batch_tensor = batch_tensor.permute({0, 3, 1, 2}); // nhwc ---> nchw
				batch_tensor = batch_tensor.toType(torch::kFloat32);// uin8--->float
				//normalize [0-255]--->[0-1]---> substract bgr 
				batch_tensor = batch_tensor.div(255.0);
				
				for (size_t i = 0; i < batch_count; i++)
				{
					batch_tensor[i][0] = batch_tensor[i][0].sub(0.5).div(0.5);
					batch_tensor[i][1] = batch_tensor[i][1].sub(0.5).div(0.5);
					batch_tensor[i][2] = batch_tensor[i][2].sub(0.5).div(0.5);
				}
				//std::vector<torch::jit::IValue> inputs; // batch inputs 
				// inputs.push_back(torch::ones({2, 3, 128, 480}).to(torch::kCUDA));
				//torch::Tensor out_tensor = net.forward({batch_tensor}).toTensor(); // nchw


				// batch_tensor_front_result
				batch_count = v_image_front_result.size();
				image_size = WIDTH*HEIGHT*5;
				batch_h = batch_count * HEIGHT;
				batch_w = WIDTH;
				cv::Mat batch_image_front_result(batch_h, batch_w, CV_32FC(5));

				for (size_t i = 0; i < batch_count; i++)
				{
					cv::Mat image = v_image_front_result[i];
					memcpy(batch_image_front_result.data+i*image_size*sizeof(float), image.data, image_size*sizeof(float));
				}
				
				at::TensorOptions options_1(at::ScalarType::Float);
				// std::cout << batch_image_front_result.rows << " " << batch_image_front_result.cols << " " << batch_image_front_result.channels() << std::endl;
				torch::Tensor batch_tensor_front_result = torch::from_blob(batch_image_front_result.data, {batch_count, HEIGHT, WIDTH, 5}, options_1);

				// std::cout << format(v_image_front_result[0], Formatter::FMT_NUMPY) << std::endl;

				//must put here
				batch_tensor_front_result = batch_tensor_front_result.to(torch::kCUDA);
				batch_tensor_front_result = batch_tensor_front_result.permute({0, 3, 1, 2}); // nhwc ---> nchw
				batch_tensor_front_result = batch_tensor_front_result.toType(torch::kFloat32);// uin8--->float


				// std::cout << batch_tensor << std::endl;
				// std::cout << format(batch_image_front_result, Formatter::FMT_NUMPY) << std::endl;
				// std::cout << batch_tensor_front_result << std::endl;




#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] [1] pre-process data: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				pre_cost += cost;
				//std::cout<<"[API-PTSIMPLE-LANESEG] #counter =="<<pre_cost<<std::endl;
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] #counter ="<<m_counter<<" pre_cost=" << pre_cost/(m_counter*1.0) << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME
				
				std::cout << "batch_tensor:" << batch_tensor.sizes() << std::endl;
				std::cout << "batch_tensor_front_result:" << batch_tensor_front_result.sizes() << std::endl;
				torch::Tensor out_tensor = net.forward({batch_tensor, batch_tensor_front_result}).toTensor(); // nchw

				// int total_1 = out_tensor.eq(1).sum().item<int>();
				// int total_2 = out_tensor.eq(2).sum().item<int>();
				// if (total_1 <= 50 || total_2 <= 50){
				// 	out_tensor.zero_();
				// 	// if (!batch_tensor_front_result.is_nonzero()){
				// 	// 	std::cout << "zero" << std::endl;
				// 	// }
				// }
				


				// out_tensor = out_tensor.squeeze();
				std::cout<<"out_tensor.sizes() = "<<out_tensor.sizes() << std::endl;
				// std::cout<<"out_tensor[0].sizes() = "<<out_tensor[0].sizes() << std::endl;
			
			
#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] [2] forward net1: cost=" << cost*1.0 << std::endl;
				total_cost+=cost;
				forward_cost += cost;
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] #counter ="<<m_counter<<" forward_cost=" << forward_cost/(m_counter*1.0) << std::endl;			
#endif // DEBUG_TIME
				
				//pt_simple v1
				// max for dim 1  [0,1,2,3]
				// std::tuple<torch::Tensor,torch::Tensor> result = out_tensor.max(1, true);

#ifdef DEBUG_INFO
			if (false){
				std::cout<<"out_tensor.sizes() = "<< out_tensor.sizes()<<std::endl;  // [2, 4, 128, 480]
				std::cout<<"0 ---"<< std::get<0>(result).sizes()<<std::endl;  // [2, 1, 128, 480]
				std::cout<<"1 ---"<< std::get<1>(result).sizes()<<std::endl;  // [2, 1, 128, 480]
			}
#endif

				for (size_t i = 0; i < batch_count; i++)
				{
					//pt_simple v1
					// torch::Tensor top_scores = std::get<0>(result)[i]; // [1, 128, 480]
					// torch::Tensor top_idxs = std::get<1>(result)[i].toType(torch::kInt32).cpu(); // [1, 128, 480]

					//pt_simple v2
					// torch::Tensor top_idxs = out_tensor[i].toType(torch::kInt32).cpu(); // [1, 160, 480]

					//pt_simple v7_sequence
					torch::Tensor tensor_soft =  torch::softmax(out_tensor[i], 0);
					torch::Tensor tensor_index = torch::argmax(tensor_soft, 0);

					torch::Tensor tensor_soft_cpu = tensor_soft.toType(torch::kFloat32).cpu();
					torch::Tensor top_idxs = tensor_index.toType(torch::kInt32).cpu();

					float* data_soft = (float*)tensor_soft_cpu.data_ptr();
					int* data = (int*)top_idxs.data_ptr(); // 5*272*480
					
					// softmax argmax
					cv::Mat result_img_softmax = cv::Mat::zeros(HEIGHT, WIDTH, CV_32FC(5)); //装softmax后的结果 1*5*272*480
					cv::Mat result_img(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0)); //原始softmax+argmax后的结果图 1*272*480
					cv::Mat result_img_bp(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0)); //将result_img中 h <= 170 的地方置为0. >170中1,2,3标签的地方置为1
#ifdef DEBUG_INFO_FWC
					cv::Mat result_left_right_img(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0));
#endif
					for (int h = 0; h < HEIGHT; h++) {
						for (int w = 0; w < WIDTH; w++) {
							int val = *(data+h*WIDTH+w);
							result_img.at<uchar>(h,w) = val;

							int idx = h*WIDTH + w;
							for (int t_idx=0; t_idx<5; t_idx++){
								result_img_softmax.at<cv::Vec<float,5>>(h,w)[t_idx] = *(data_soft + idx + t_idx * WIDTH * HEIGHT);
							}
#ifdef DEBUG_INFO_FWC
							if (val == 1){
								result_left_right_img.at<uchar>(h,w) = 128;
							}
							else if (val == 2){
								result_left_right_img.at<uchar>(h,w) = 255;
							}
							else if (val == 3){
								result_left_right_img.at<uchar>(h,w) = 50;
							}
#endif
							// 只有长焦做截断操作,i = 1时
							if (val == 1 || val == 2 || val == 3){
								result_img_bp.at<uchar>(h,w) = 1;
							}
							
							if (i == 1 && h < 171){
								result_img_bp.at<uchar>(h,w) = 0;
							}
							
					
						}
					}

#ifdef DEBUG_INFO_FWC
					// std::cout << format(result_img_softmax, Formatter::FMT_NUMPY) << std::endl;
					const std::string outputFileName = "./result_left_right_img"+std::to_string(i)+".png";
					cv::imwrite(outputFileName, result_left_right_img);
#endif



#pragma region my softmax argmax
// 					torch::Tensor top_idxs = out_tensor[i].toType(torch::kFloat32).cpu(); // [1, 160, 480]

// 					float* data = (float*)top_idxs.data_ptr(); // 5*272*480
					
// 					// softmax argmax
// 					cv::Mat result_img_softmax = cv::Mat::zeros(HEIGHT, WIDTH, CV_32FC(5)); //装softmax后的结果 1*5*272*480
// 					cv::Mat result_img(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0)); //原始softmax+argmax后的结果图 1*272*480
// 					cv::Mat result_img_bp(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0)); //将result_img中 h <= 180 的地方置为0. >180中1,2,3标签的地方置为1
// #ifdef DEBUG_INFO_FWC
// 					cv::Mat result_left_right_img(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0));
// #endif
// 					for (int h=0; h < HEIGHT; h++){
// 						for (int w=0; w < WIDTH; w++){
// 							int idx = h*WIDTH + w;
// 							float result_0 = *(data + idx);
// 							float result_1 = *(data + idx + WIDTH * HEIGHT);
// 							float result_2 = *(data + idx + 2 * WIDTH * HEIGHT);
// 							float result_3 = *(data + idx + 3 * WIDTH * HEIGHT);
// 							float result_4 = *(data + idx + 4 * WIDTH * HEIGHT);

// 							std::vector<float> v_result_0 = {result_0, result_1, result_2, result_3, result_4};
// 							float max_val = *max_element(v_result_0.begin(), v_result_0.end());

// 							// std::cout << result_0 << " " << result_1 << " " << result_2 << " " << result_3 << " " << result_4 << std::endl;
// 							std::vector<float> v_result = {exp(result_0-max_val), exp(result_1-max_val), exp(result_2-max_val), exp(result_3-max_val), exp(result_4-max_val)};
// 							float sum = v_result[0] + v_result[1] + v_result[2] + v_result[3] + v_result[4];
// 							float max_value = 0.0;
// 							int argmax_idx = 0;
// 							for (int t_idx=0; t_idx<5; t_idx++){
// 								result_img_softmax.at<cv::Vec<float,5>>(h,w)[t_idx] = v_result[t_idx] / sum;
// 								if (v_result[t_idx] / sum >= max_value){
// 									max_value = v_result[t_idx];
// 									argmax_idx = t_idx;
// 								}
// 							}

// 							result_img.at<uchar>(h,w) = argmax_idx;
// #ifdef DEBUG_INFO_FWC
// 							if (argmax_idx == 1){
// 								result_left_right_img.at<uchar>(h,w) = 128;
// 							}
// 							else if (argmax_idx == 2){
// 								result_left_right_img.at<uchar>(h,w) = 255;
// 							}
// 							else if (argmax_idx == 3){
// 								result_left_right_img.at<uchar>(h,w) = 50;
// 							}
// #endif
// 							if (h >= 181){
// 								if (argmax_idx == 1 || argmax_idx == 2 || argmax_idx == 3){
// 									result_img_bp.at<uchar>(h,w) = 1;
// 								}
// 							}
// 						}
// 					}
// 					// end softmax argmax
// #ifdef DEBUG_INFO_FWC
//					// std::cout << format(result_img_softmax, Formatter::FMT_NUMPY) << std::endl;
// 					const std::string outputFileName = "./result_left_right_img.png";
// 					cv::imwrite(outputFileName, result_left_right_img);
// #endif
#pragma endregion

					//连通区后处理
					cv::Mat labels; // w*h  label = 0,1,2,3,...N-1 (0- background)       CV_32S = 4
					cv::Mat stats; // N*5  表示每个连通区域的外接矩形和面积 [x,y,w,h, area]   CV_32S = 4
					cv::Mat centroids; // N*2  (x,y)                                     CV_32S = 4
					int num_components; // 连通区域number

					
					num_components = connectedComponentsWithStats(result_img_bp, labels, stats, centroids, 4, CV_32S);

#ifdef DEBUG_INFO_FWC
					std::cout << format(stats, Formatter::FMT_NUMPY) << std::endl;
#endif

					cv::Mat result_img_mask(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0));

					if (i == 0){
						// 对于短焦，选择大于1000的连通区域
						std::vector<int> v_unsave_idx;
						for (int idx_label=0; idx_label<num_components; idx_label++){
							if ((stats.at<int>(idx_label, 4) < 1000) || (stats.at<int>(idx_label, 4) > 8000)){
								v_unsave_idx.push_back(idx_label);
							}
						}
						
						if (v_unsave_idx.size()){
							for (int h = 0; h < HEIGHT; h++) {
								for (int w = 0; w < WIDTH; w++){
									int label = labels.at<int>(h,w);
									for(int t_idx=0; t_idx<v_unsave_idx.size(); t_idx++){
										if (label == v_unsave_idx[t_idx]){
											result_img.at<uchar>(h,w) = 0;
											break;
										}
									}
								}
							}
						}

					}
					else if (i == 1){
						// 对于长焦，选择最大的连通区域，如果最大连同区area小于2000则置为0
						int final_label=0;
						int max_area=0;
						for (int idx_label=0; idx_label<num_components; idx_label++){
							if ((stats.at<int>(idx_label, 4) > max_area) && (stats.at<int>(idx_label, 4) <= 8400)){
								max_area = stats.at<int>(idx_label, 4);
								final_label = idx_label;
							}
						}
						// 如果最大连通区area大于2000，则修改result_img,将最大连同区的标签保留，其余置为0；
						// 反之，将result_img全置为0；
	#ifdef DEBUG_INFO_FWC
						std::cout << "max area:" << max_area << std::endl;
	#endif
						if (max_area >= 2000){
							for (int h = 0; h < HEIGHT; h++) {
								for (int w = 0; w < WIDTH; w++){
									int label = labels.at<int>(h,w);
									if (label != final_label){
										result_img.at<uchar>(h,w) = 0;
									}
								}
							}
						}
						else{
							cv::Mat result_img_tmp(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0));
							result_img = result_img_tmp.clone();
						}
					}

#ifdef DEBUG_INFO_FWC
						cv::Mat result_left_right_img_new(HEIGHT, WIDTH, CV_8UC1, Scalar::all(0));
						for (int h=0; h < HEIGHT; h++){
							for (int w=0; w < WIDTH; w++){
								int val = result_img.at<uchar>(h,w);

								if (val == 1){
									result_left_right_img_new.at<uchar>(h,w) = 128;
								}
								else if (val == 2){
									result_left_right_img_new.at<uchar>(h,w) = 255;
								}
								else if (val == 3){
									result_left_right_img_new.at<uchar>(h,w) = 50;
								}

							}
						}
						const std::string outputFileName1 = "./result_left_right_img_new"+std::to_string(i)+".png";
						cv::imwrite(outputFileName1, result_left_right_img_new);
#endif



					channel_mat_t output_instance_seg;
					cv::Mat left_lane_instance (HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
					cv::Mat right_lane_instance(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));

					// 0 background, 1 left, 2 right
					for (int h = 0; h < HEIGHT; h++) {
						for (int w = 0; w < WIDTH; w++) {
							int val = result_img.at<uchar>(h,w);
							if (val == params.left_id){ // left lane
								left_lane_instance.at<uchar>(h,w) = 1; // mark as black
							}
							else if (val == params.right_id){ // right lane
								right_lane_instance.at<uchar>(h,w) = 1; // mark as black
							}
						}
					}

					// 进行连通域分析
					// cv::Mat labels; // w*h  label = 0,1,2,3,...N-1 (0- background)       CV_32S = 4
					// cv::Mat stats; // N*5  表示每个连通区域的外接矩形和面积 [x,y,w,h, area]   CV_32S = 4
					// cv::Mat centroids; // N*2  (x,y)                                     CV_32S = 4
					// int num_components; // 连通区域number
					
					num_components = connectedComponentsWithStats(left_lane_instance, labels, stats, centroids, 4, CV_32S);
					cv::Mat left_lane_instance_new(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
					for (int h = 0; h < HEIGHT; h++) {
						int min_label = num_components;
						for (int w = 0; w < WIDTH; w++){
							int label = labels.at<int>(h,w);
							if (label != 0 && label <= min_label) min_label = label;
						}

						if (min_label != num_components){
							for (int w = 0; w < WIDTH; w++) {
								int label = labels.at<int>(h,w);
								if (label == min_label) left_lane_instance_new.at<uchar>(h,w) = 1;
							}
						}
					}

					num_components = connectedComponentsWithStats(right_lane_instance, labels, stats, centroids, 4, CV_32S);
					cv::Mat right_lane_instance_new(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
					for (int h = 0; h < HEIGHT; h++) {
						int min_label = num_components;
						for (int w = 0; w < WIDTH; w++){
							int label = labels.at<int>(h,w);
							if (label != 0 && label <= min_label) min_label = label;
						}

						if (min_label != num_components){
							for (int w = 0; w < WIDTH; w++) {
								int label = labels.at<int>(h,w);
								if (label == min_label) right_lane_instance_new.at<uchar>(h,w) = 1;
							}
						}
					}


					output_instance_seg.push_back(left_lane_instance_new);
					output_instance_seg.push_back(right_lane_instance_new);

#ifdef DEBUG_INFO_FWC
					char tmp[WIDTH * HEIGHT];
					memset(tmp, 0, WIDTH * HEIGHT);
					// 0 background, 1 left, 2 right
					for (int h = 0; h < HEIGHT; h++) {
						for (int w = 0; w < WIDTH; w++) {
							int val_left = left_lane_instance_new.at<uchar>(h,w);
							int val_right = right_lane_instance_new.at<uchar>(h,w);
							// std::cout << "val:" << val << std::endl;
							if (val_left == 1){ // left lane
								tmp[h * WIDTH + w] = 128;
							}
							if (val_right == 1){ // right lane
								tmp[h * WIDTH + w] = 255;
							}
						}
					}
					cv::Mat segment_image(HEIGHT, WIDTH, CV_8UC1, tmp);
					cv::imwrite("./segmentation"+std::to_string(i)+"_.png", segment_image);
#endif
					v_binary_mask.push_back(result_img_softmax); // (128, 420) v=[0,1]
					v_instance_mask.push_back(output_instance_seg);// (2, 128, 420) int value v=[0,1]
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] [3] post process: cost=" << cost*1.0 << std::endl;
				pt1 = boost::posix_time::microsec_clock::local_time();

				total_cost+=cost;
				post_cost += cost;
				LOG(INFO) << "[API-PTSIMPLE-LANESEG] #counter ="<<m_counter<<" post_cost=" << post_cost/(m_counter*1.0) << std::endl;

				LOG(INFO) << "[API-PTSIMPLE-LANESEG] #counter ="<<m_counter<<" total_cost=" << total_cost/(m_counter*1.0) << std::endl;
#endif // DEBUG_TIME
				
				return true;
			}
#pragma endregion lane seg sequence

			

		}	
	}
}// end namespace
