#include "mosun_api_impl.h"

#include "algorithm/core/util/filesystem_util.h"
#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"
#include "algorithm/core/util/numpy_util.h"
#include "algorithm/core/util/lane_util.h"
#include "algorithm/core/profiler.h"


// cuda 
#include <cuda.h>
#include <cuda_runtime.h>
#include<algorithm>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

namespace watrix {
	namespace algorithm {
		namespace internal {

			MosunParam MosunApiImpl::params;
			std::vector<pt_module_t> MosunApiImpl::v_net;

			int MosunApiImpl::m_counter = 0; // init
			
			void MosunApiImpl::init(
				const MosunParam& params,
				int net_count
			)
			{
				MosunApiImpl::params = params;
				v_net.resize(net_count);

				torch::DeviceType device_type = torch::kCUDA;  //torch::kCUDA  and torch::kCPU
				torch::Device device(device_type, 0);

				for (int i = 0; i < v_net.size(); i++)
				{
					v_net[i] = torch::jit::load(params.model_path);
					//assert(v_net[i] != nullptr);
    				v_net[i].to(device);
					std::cout<<" pytorch load module OK \n";
				}
				std::cout<<" v_net.size()  = "<< v_net.size() << std::endl;
			}

			void MosunApiImpl::free()
			{
				for (int i = 0; i < v_net.size(); i++)
				{
					//v_net[i] = nullptr;
				}
			}

#pragma region cluster 

		torch::Tensor cluster(
			torch::Tensor binary_seg_pred, 
			torch::Tensor pix_embedding, 
			int feature_dim, 
			float delta_v, 
			float delta_d
		){
			torch::Tensor b_seg = binary_seg_pred.squeeze(0);
			pix_embedding = pix_embedding.permute({1, 2, 0});
			//std::cout << pix_embedding<< std::endl;
								
			int count = 0;
			while (1)
			{
				torch::Tensor remaining = b_seg.eq(1).nonzero();
				if (remaining.numel() == 0)
				{
					break;
				}
				torch::Tensor dist;
				float dist_;
				torch::Tensor center = remaining[0];
				torch::Tensor center_emb; 
				torch::Tensor seg_mean;

				center_emb = pix_embedding[center[0]][center[1]];

				//exit(0);
				float eps = 0.001;
				float var = 1.0;
				while (var > eps)
				{
					dist = pix_embedding - center_emb;
					dist = torch::norm(dist, 2, -1, false);
					
					torch::Tensor mask = (dist <= delta_d) * b_seg.eq(1);
					mask = mask.unsqueeze(-1).repeat({1, 1, feature_dim});     
					seg_mean = pix_embedding.masked_select(mask).view({-1, feature_dim}).mean(0);

					var = *((seg_mean.cpu() - center_emb.cpu()).norm(2)).data<float>();
					center_emb = seg_mean;
				}

				dist = pix_embedding - seg_mean;
				dist = torch::norm(dist, 2, -1, false);
				count -= 1;
				torch::Tensor mask_ = (((dist <= delta_d)*(b_seg.eq(1))).toType(torch::kLong))*(count-1); 

				//exit(0);
				b_seg = b_seg + mask_;
				//b_seg[(dist <= delta_d)*b_seg.eq(1)] = count;
				//}
			}
      std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> result = torch::_unique2(b_seg,1,1,1);
      torch::Tensor Tensor_id = std::get<0>(result);
      torch::Tensor Tensor_num = std::get<2>(result);
      std::tuple<torch::Tensor,torch::Tensor> num_sort = Tensor_num.topk(1-count, -1);
      torch::Tensor top_scores = std::get<0>(num_sort).view(-1);//{1,10} ��� {10}
      torch::Tensor top_idxs = std::get<1>(num_sort).view(-1);      
      //std::cout<<"Tensor_id = "<< Tensor_id <<std::endl;          
      //std::cout<<"Tensor_num = "<< Tensor_num <<std::endl;
      //std::cout<<"top_scores = "<< top_scores <<std::endl;          
      //std::cout<<"top_idxs = "<< top_idxs <<std::endl; 
      int c = std::min(-count,3);
      for (int i=1; i<=c; i++)
      {
        torch::Tensor mask_a = ((b_seg.eq(Tensor_id[top_idxs[i]])).toType(torch::kLong))*(i-Tensor_id[top_idxs[i]]); 
        b_seg += mask_a;
      }
			return b_seg;
		}

#pragma endregion

#pragma region lane seg
			bool MosunApiImpl::seg(
				int net_id,
				const cv::Mat& image_,
				cv::Mat& binary_mask, 
				cv::Mat& binary3_mask
			)
			{
				//std::cout<<"===================================111\n";
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, v_net.size()) << "net_id invalid";
				CHECK_EQ(image_.channels(), 3) << "image must have 3 channels";

				pt_module_t& module = v_net[net_id];

				torch::DeviceType device_type = torch::kCUDA;  //torch::kCUDA  and torch::kCPU
				torch::Device device(device_type, 0);

				
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
				
        cv::Mat roiImage = image_(cv::Rect(20, 100, 710, 100));
        roiImage.setTo(cv::Scalar(0,0,0));
				cv::Mat image;
				cv::resize(image_, image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

				at::TensorOptions options(at::ScalarType::Byte);
				torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, options);
				tensor_image = tensor_image.to(device);
				tensor_image = tensor_image.permute({0, 3, 1, 2});
				tensor_image = tensor_image.toType(torch::kFloat);
				tensor_image = tensor_image.sub(116.779).div(128);
    

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();
				cost = (pt2 - pt1).total_milliseconds();
				std::cout << "[API-MOSUN] [1] pre-process data: cost1=" << cost*1.0 << std::endl;

				total_cost+=cost;
				pre_cost += cost;
				//std::cout<<"[API-MOSUN] #counter =="<<pre_cost<<std::endl;
				LOG(INFO) << "[API-MOSUN] #counter ="<<m_counter<<" pre_cost=" << pre_cost/(m_counter*1.0) << std::endl;
				
				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				//cudaDeviceSynchronize();
				auto out_tensor = module.forward({tensor_image}).toTuple()->elements();
				//cudaDeviceSynchronize();

				torch::Tensor out_tensor_bin = out_tensor[0].toTensor(); // [1, 2, 384, 512]
    			torch::Tensor out_tensor_ins = out_tensor[1].toTensor()[0];

				//std::cout<< "out_tensor_bin ="<<out_tensor_bin.sizes() << std::endl;
#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				std::cout << "[API-MOSUN] [2] forward net1: cost2=" << cost*1.0 << std::endl;
				total_cost+=cost;
				forward_cost += cost;
				LOG(INFO) << "[API-MOSUN] #counter ="<<m_counter<<" forward_cost=" << forward_cost/(m_counter*1.0) << std::endl;			
#endif // DEBUG_TIME

				torch::Tensor binary_tensor = torch::argmax(torch::softmax(out_tensor_bin,1),1,true)[0]; // [1, 384, 512]
				
				//BEGIN_PROFILE(cluster)
				torch::Tensor instance_cluster = cluster(binary_tensor, out_tensor_ins, 8, 0.5, 1.5).cpu(); // [384, 512]
				//END_PROFILE(cluster)

				int64_t* data_cls = (int64_t *)instance_cluster.data_ptr();
				int64_t* data_bin = (int64_t *)binary_tensor.data_ptr();    

				uchar binary_mask_data[INPUT_WIDTH * INPUT_HEIGHT]; // 0, 255
				memset(binary_mask_data, 0, INPUT_WIDTH * INPUT_HEIGHT);

				uchar binary3_mask_data[INPUT_WIDTH * INPUT_HEIGHT]; //0, 64,128,255
				memset(binary3_mask_data, 0, INPUT_WIDTH * INPUT_HEIGHT);

				int channel_step = INPUT_HEIGHT*INPUT_WIDTH;
				for (int h = 0; h < INPUT_HEIGHT; h++) 
				{
					int widthstep = h * INPUT_WIDTH;
					for (int w = 0; w < INPUT_WIDTH; w++) 
					{
						int64_t val0 = *(data_cls + widthstep + w);
            //int64_t val0_bin = *(data_bin + widthstep + w); 
						// -1, -2, -3
            //if (val0_bin>0){
						//	binary_mask_data[widthstep + w] = 255;            
            //}
						if (val0 == 1){ // left
							binary_mask_data[widthstep + w] = 255;
							binary3_mask_data[widthstep + w] = 255;

							//color_mask.at<cv::Vec3b>(h,w) = cv::Vec3b(255,0,0);
							//lane1.at<uchar>(h,w) = 255;
						}    
						else if (val0 == 2){   // right 
							binary_mask_data[widthstep + w] = 255;
							binary3_mask_data[widthstep + w] = 128;

							//color_mask.at<cv::Vec3b>(h,w) = cv::Vec3b(0,255,0);
							//lane2.at<uchar>(h,w) = 255;
						}   
						else if (val0 == 3){   // middle
							binary_mask_data[widthstep + w] = 255;
							binary3_mask_data[widthstep + w] = 64;

							//color_mask.at<cv::Vec3b>(h,w) = cv::Vec3b(0,0,255);
							//lane3.at<uchar>(h,w) = 255;
						}           
            //else{std::cout<<"val0 = "<< val0 <<std::endl;}  
					}
				}

				cv::Mat binary_mask_(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1, binary_mask_data);
				cv::Mat binary3_mask_(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1, binary3_mask_data);

				binary_mask = binary_mask_.clone(); 
				binary3_mask = binary3_mask_.clone(); 
#ifdef DEBUG_IMAGE
				{
					cv::imwrite("./3_1_binary_mask.jpg", binary_mask);
					cv::imwrite("./3_2_binary3_mask.jpg", binary3_mask);
				}
#endif


#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				std::cout << "[API-MOSUN] [3] post process: cost3=" << cost*1.0 << std::endl;
				pt1 = boost::posix_time::microsec_clock::local_time();

				total_cost+=cost;
				post_cost += cost;
				LOG(INFO) << "[API-MOSUN] #counter ="<<m_counter<<" post_cost=" << post_cost/(m_counter*1.0) << std::endl;
				LOG(INFO) << "[API-MOSUN] #counter ="<<m_counter<<" total_cost=" << total_cost/(m_counter*1.0) << std::endl;
#endif // DEBUG_TIME
				
				return true;
			}

			void transform_binary_points_to_origin_points(
				cvpoints_t& lane_cvpoints, 
				float h_rad, float w_rad
			)
			{
				// v1: autotrain (256,1024) ===(512,1920)===>(1080,1920) 
				// v2: mosun (384,512) ===(964,1288) 
				for(auto& cvpoint: lane_cvpoints)
				{
					cvpoint.x = round(cvpoint.x * w_rad);
					cvpoint.y = round(cvpoint.y * h_rad);
				}
			}

			bool sort_cvpoint_by_x(const cvpoint_t& a, const cvpoint_t& b)
			{
				return a.x < b.x; 
			}


#pragma region get_top_bottom_point
			bool get_top_bottom_point(
				const cv::Mat& rotated_mask,
				cvpoint_t& top_point,
				cvpoint_t& bottom_point,                    
				cvpoint_t& left_point,
				cvpoint_t& right_point       
			)
			{
				int image_height = rotated_mask.rows;
        int center_x = (left_point.x+right_point.x)/2;
        int center_y = (left_point.y+right_point.y)/2;
				int top = image_height;
        int bottom1 = 0; 
        int bottom2 = 0; 
        int bottom3 = 0;  
        int bottom = 0;                
				cvpoint_t place_top;
				cvpoint_t place_bottom;
        int center_sys_x1 =left_point.x;
        int center_sys_x2 =0;
        int center_sys_x3 =0;
        int center_sys_x4 =right_point.x;        
				std::vector<int> vy,vx;
				NumpyUtil::np_where_g(rotated_mask,0,vy,vx);	 // dst >0 
				if (vy.size()<10){
					return false;
				}
        cvpoint_t place_bottom1;  
        cvpoint_t place_bottom2; 
        cvpoint_t place_bottom3;         
				for(int i=0;i<vy.size();i++){
					int y = vy[i];
					int x = vx[i];
					if (y< top ){
						top = y;  
						place_top.x = x;
						place_top.y = y;
					}
				}
        if (place_top.x<center_x)
        {
            center_sys_x2 = place_top.x;
            center_sys_x3 = center_x+(center_x-place_top.x);
        }
        else
        {
            center_sys_x2 = center_x-(place_top.x-center_x);
            center_sys_x3 = place_top.x;  
        }
				for(int i=0;i<vy.size();i++){
					int y = vy[i];
					int x = vx[i];        
					if (y>bottom1 && x>center_sys_x1 && x<center_sys_x2){
						bottom1 = y; 
						place_bottom1.x = x;
						place_bottom1.y = y;
					}  
					if (y>bottom2 && x>center_sys_x2 && x<center_sys_x3){
						bottom2 = y; 
						place_bottom2.x = x;
						place_bottom2.y = y;
					}  
					if (y>bottom3 && x>center_sys_x3 && x<center_sys_x4){
						bottom3 = y; 
						place_bottom3.x = x;
						place_bottom3.y = y;
					}                   
        }   
        double w12 = 1.0*(place_bottom1.x-left_point.x)/(center_sys_x2-left_point.x);
        double w34 = 1.0*(right_point.x-place_bottom3.x)/(right_point.x-center_sys_x3);
        double y_aver_left = left_point.y - w12*(left_point.y-place_top.y);
        double y_aver_right = right_point.y - w34*(right_point.y-place_top.y);
        if (place_bottom1.x-left_point.x>10 && place_bottom1.y-y_aver_left>5)
        {
            bottom = bottom1;
            place_bottom = place_bottom1;
        } 
        else if (right_point.x-place_bottom3.x>10 && place_bottom3.y-y_aver_right>5)
        {
            bottom = bottom3;
            place_bottom = place_bottom3;        
        }  
        else
        {
            bottom = bottom2;
            place_bottom = place_bottom2;        
        }  
#ifdef DEBUG_INFO 
				{
					printf(" top = %d, bottom = %d \n",top, bottom);
					std::cout<<"place_top = "<< place_top <<std::endl;
					std::cout<<"place_bottom = "<< place_bottom <<std::endl;
				}
#endif
				

				int max_top = 0;
				int top_boundary = std::min(place_top.y+60, image_height);
				for(int y=place_top.y; y< top_boundary; y++){
					if (rotated_mask.at<uchar>(y, place_top.x) > 0){
						max_top = y ;
					}
				}

				int min_bottom = 0;
				int bottom_boundary = std::max(place_bottom.y-60, 0);
				for(int y=place_bottom.y; y> bottom_boundary; y--){
					if (rotated_mask.at<uchar>(y, place_bottom.x) > 0){
						min_bottom = y ;
					}
				}

#ifdef DEBUG_INFO 
				{
					printf(" max_top = %d, min_bottom = %d \n",max_top, min_bottom);
				}
#endif

				if (min_bottom == 0){
					min_bottom = place_bottom.y; 
				}
				if (max_top == 0){ // ??? image_height
					max_top = place_top.y;
				}

				int top_result = (max_top + place_top.y) / 2 ;
				int bottom_result = (min_bottom + place_bottom.y) / 2;
				int result = bottom_result - top_result;

#ifdef DEBUG_INFO 
				{
					printf(" top_result = %d, bottom_result = %d, result = %d \n",
						top_result, bottom_result, result
					);
				}
#endif

				cvpoint_t top_point_(place_top.x, top_result);
				cvpoint_t bottom_point_(place_bottom.x, bottom_result);

				top_point = top_point_;
				bottom_point = bottom_point_;
				return true;
			}
#pragma endregion


			void get_largest_middle_area(
				const cv::Mat& image_largest_area,
				const cv::Mat& binary3_mask,
				int left_right_area_max_ratio,
				cv::Mat& out_largest_middle,
				MosunResult& mosun_result
			)
			{
				int height = image_largest_area.rows;
				int width = image_largest_area.cols;
				cv::Mat image_largest_middle(height,width, CV_8UC1, cv::Scalar(0)); 

				//channel_mat_t channel_mat;
				cv::Mat lane1(height, width, CV_8UC1, cv::Scalar(0));
				cv::Mat lane2(height, width, CV_8UC1, cv::Scalar(0));
				cv::Mat lane3(height, width, CV_8UC1, cv::Scalar(0));
        int num_p_lane[3] = {0,0,0};
				for (int h = 0; h < height; h++) 
				{
					for (int w = 0; w < width; w++) 
					{
						int val = image_largest_area.at<uchar>(h,w); // 0-255
						if (val > 0){  
							int val2 = binary3_mask.at<uchar>(h,w); //0- left-255, right-128, middle-64
							// NO ORDER, so we need to find the largest middle area
							if (val2 == 255){
								lane1.at<uchar>(h,w) = 255;
                num_p_lane[0] = num_p_lane[0]+1;
							} else if (val2 == 128){
								lane2.at<uchar>(h,w) = 255;
                num_p_lane[1] = num_p_lane[1]+1;
							}
							else if (val2 == 64){
								lane3.at<uchar>(h,w) = 255;
                num_p_lane[2] = num_p_lane[2]+1;
							}
						}
					}
				}
#ifdef DEBUG_IMAGE
				{
					cv::imwrite("./3_5_lane1.jpg", lane1);
					cv::imwrite("./3_5_lane2.jpg", lane2);
					cv::imwrite("./3_5_lane3.jpg", lane3);
				}
#endif
				//channel_mat.push_back(lane1);
				//channel_mat.push_back(lane2);
				//channel_mat.push_back(lane3);
        int p_max = *max_element(num_p_lane,num_p_lane+3);
        int p_min = *min_element(num_p_lane,num_p_lane+3);
#ifdef DEBUG_INFO 
  			{
        std::cout<<"num_p_lane =  "<< num_p_lane[0] << " " << num_p_lane[1] << " " << num_p_lane[2] << std::endl;
        std::cout<<"num_p_lane =  "<< p_max << " " << p_min << std::endl;
 				}
#endif
        channel_mat_t left_right;
        if (p_min<50){
           if (p_max==num_p_lane[0]){
               image_largest_middle = lane1;
			         left_right.push_back(lane2);
  					   left_right.push_back(lane3);  
                            
           }
           if (p_max==num_p_lane[1]){
               image_largest_middle = lane2;
  					   left_right.push_back(lane1);
  					   left_right.push_back(lane3);           
           }
           if (p_max==num_p_lane[2]){
               image_largest_middle = lane3;
			         left_right.push_back(lane1);
  					   left_right.push_back(lane2);            
           }
        }
        else{
  				int x1 = LaneUtil::get_average_x(lane1);
  				int x2 = LaneUtil::get_average_x(lane2);
  				int x3 = LaneUtil::get_average_x(lane3);
  
#ifdef DEBUG_INFO 
  				{
  					std::cout<<"x1 = "<< x1 << std::endl;
  					std::cout<<"x2 = "<< x2 << std::endl;
  					std::cout<<"x3 = "<< x3 << std::endl;
  				}
#endif 
  				// 275, 90, 489
  				if (BETWEEN(x1,x2,x3)){
  					image_largest_middle = lane1;
  					std::cout<<"HIT x1 = "<< x1 << std::endl;
  
  					left_right.push_back(lane2);
  					left_right.push_back(lane3);
  				} else if(BETWEEN(x2,x1,x3))
  				{
  					image_largest_middle = lane2;
  					std::cout<<"HIT x2 = "<< x2 << std::endl;
  
  					left_right.push_back(lane1);
  					left_right.push_back(lane3);
  				} else {
  					image_largest_middle = lane3;
  					std::cout<<"HIT x3 = "<< x3 << std::endl;
  
  					left_right.push_back(lane1);
  					left_right.push_back(lane2);
  				}
        }
				// (4.1) filter out nosie for middle area
				component_stat_t middle_stat;
				LaneUtil::get_largest_connected_component(
					image_largest_middle,
					out_largest_middle,
					middle_stat
				);

				// (4.2) left or right area 	
				cv::Mat out1,out2;
				component_stat_t stat1, stat2;
				LaneUtil::get_largest_connected_component(left_right[0], out1, stat1);
				LaneUtil::get_largest_connected_component(left_right[1], out2, stat2);

#ifdef DEBUG_IMAGE 
				{
					cv::imwrite("3_6_largest_middle.png", image_largest_middle);
					cv::imwrite("3_7_filtered_largest_middle.png", out_largest_middle);

					cv::imwrite("3_8_lane1.png", out1);
					cv::imwrite("3_8_lane2.png", out2);
				}
#endif 

				float area_ratio = 1; 
				if (stat1.area<stat2.area)
				{
					area_ratio = stat2.area / (stat1.area*1.0);
				} else {
					area_ratio = stat1.area / (stat2.area*1.0);
				}

#ifdef DEBUG_INFO 
				{
					std::cout<<"\n ================left or right missing==================== "<< std::endl;
					std::cout<<"\n stat1.area = " << stat1.area << std::endl;
					std::cout<<" stat2.area = " << stat2.area << std::endl;
          std::cout<<" middle_stat.area = " << middle_stat.area << std::endl;             
					std::cout<<" area_ratio = " << area_ratio << std::endl;
					//stat1.area = 395
 					//stat2.area = 2
					std::cout<<"\n ================left or right missing==================== "<< std::endl;
				}
#endif 
				mosun_result.left_area = stat1.area;
				mosun_result.right_area = stat2.area;
				mosun_result.middle_area = middle_stat.area;
				mosun_result.left_right_area_ratio = area_ratio;
				mosun_result.left_right_symmetrical = (area_ratio <= left_right_area_max_ratio);
			}

      
			MosunResult MosunApiImpl::detect(
				int net_id,
				const cv::Mat& image
			)
			{
				MosunResult mosun_result;
				
				// (1) seg for image
				cv::Mat binary_mask;  // 0-255 
				cv::Mat binary3_mask;  // left-255, right-128, middle-64
				seg(net_id, image, binary_mask, binary3_mask);

				// (2) get middle lane 
				cv::Mat image_largest_area;
				component_stat_t largest_stat;
				LaneUtil::get_largest_connected_component(binary_mask, image_largest_area, largest_stat);
				
#ifdef DEBUG_IMAGE
				{
					cv::imwrite("3_4_largest_area.png", image_largest_area);
				}
#endif

				// (3) get left/right point in origin image
				// (384,512) ===(964,1288) 
				float h_rad = ORIGIN_HEIGHT/(INPUT_HEIGHT*1.f);
				float w_rad = ORIGIN_WIDTH /(INPUT_WIDTH*1.f);

				//cvpoint_t left_point(largest_stat.x, ORIGIN_HEIGHT/2);
				//cvpoint_t right_point(largest_stat.x+largest_stat.w, ORIGIN_HEIGHT/2);
				// (384,512) ===(964,1288)  binary point ===> origin point
				//left_point.x = left_point.x * w_rad;
				//right_point.x = right_point.x * w_rad;
        
				// (4) get largest middle area 
				cv::Mat image_largest_middle;
				get_largest_middle_area(
					image_largest_area, 
					binary3_mask,
					params.left_right_area_max_ratio,
					image_largest_middle,
					mosun_result
				);
				//return mosun_result;

				// (5) get left/right point(rotate angle) for middle area
				// (5.1) get lane points from largest lane 
        if (mosun_result.left_area + mosun_result.right_area + mosun_result.middle_area<200){
					return mosun_result;
				}
				cvpoints_t middle_cvpoints = LaneUtil::get_lane_cvpoints(image_largest_middle);
				int n = middle_cvpoints.size();
				if (n < 10){
					return mosun_result;
				}
				// (5.3) sort lane points by x-axis and get middle left/right point
				std::stable_sort(
					middle_cvpoints.begin(),
					middle_cvpoints.end(), 
					sort_cvpoint_by_x
				);
				cvpoint_t middle_left = middle_cvpoints[0];
				cvpoint_t middle_right = middle_cvpoints[n-1];

				// (5.4) get rotate angle
				float degree_resize = NumpyUtil::get_degree_angle(middle_left,middle_right);

        middle_left.x = round(middle_left.x * w_rad);
        middle_left.y = round(middle_left.y * h_rad);
        middle_right.x = round(middle_right.x * w_rad);
        middle_right.y = round(middle_right.y * h_rad);
        float degree = NumpyUtil::get_degree_angle(middle_left,middle_right);
				float rad = NumpyUtil::get_rad_angle(middle_left,middle_right);       
        
        // (5.4) rotate the ending points
        float center_x = ORIGIN_WIDTH/2;
        float center_y = ORIGIN_HEIGHT/2;
        cvpoint_t middle_left_org;
        cvpoint_t middle_right_org;
        middle_left_org.x = (middle_left.x - center_x)*cos(-rad) - (middle_left.y - center_y)*sin(-rad) + center_x ;
        middle_left_org.y = (middle_left.x - center_x)*sin(-rad) + (middle_left.y - center_y)*cos(-rad) + center_y ;    
        middle_right_org.x = (middle_right.x - center_x)*cos(-rad) - (middle_right.y - center_y)*sin(-rad) + center_x ;
        middle_right_org.y = (middle_right.x - center_x)*sin(-rad) + (middle_right.y - center_y)*cos(-rad) + center_y ;                  
				// rotate middle mask
         
				// (6) get rotated middle mask              
				cv::Mat rotated_middle_mask_resize = NumpyUtil::cv_rotate_image_keep_size(image_largest_middle, degree_resize);                              
				cvpoints_t middle_cvpoints_orgsize = LaneUtil::get_lane_cvpoints(rotated_middle_mask_resize);    
				// (7) transform lane points from binary image to origin image 
				transform_binary_points_to_origin_points(middle_cvpoints_orgsize, h_rad, w_rad);
        
        // (8) get rotated binary mask and the true distance of tramline
        cv::Mat rotated_binary_resize = NumpyUtil::cv_rotate_image_keep_size(image_largest_area, degree_resize);
        cvpoints_t binary_cvpoints_resize = LaneUtil::get_lane_cvpoints(rotated_binary_resize);
		
        int n_binary = binary_cvpoints_resize.size();

		if (n_binary == 0) { return mosun_result; }

		
				std::stable_sort(
					binary_cvpoints_resize.begin(),
					binary_cvpoints_resize.end(), 
					sort_cvpoint_by_x
				);        
				cvpoint_t left_point = binary_cvpoints_resize[0];
				cvpoint_t right_point = binary_cvpoints_resize[n_binary-1];
				left_point.x = left_point.x * w_rad;
				right_point.x = right_point.x * w_rad;  
        
				// (9) get rotated middle mask
				// lane mask in origin size
				cv::Mat rotated_middle_mask(ORIGIN_HEIGHT, ORIGIN_WIDTH, CV_8UC1, cv::Scalar(0));
				for(auto& cvpoint: middle_cvpoints_orgsize){
					rotated_middle_mask.at<uchar>(cvpoint.y,cvpoint.x) = 255; // image[y,x] = 255
				} 
				// rotated origin image 
				cv::Mat image_with_mask = image.clone();
				cv::Mat rotated_image_with_mask = NumpyUtil::cv_rotate_image_keep_size(image_with_mask, degree);
                                       
				// (5.2) transform lane points from binary image to origin image 
        /***
				transform_binary_points_to_origin_points(middle_cvpoints, h_rad, w_rad);

				// (5.3) sort lane points by x-axis and get middle left/right point
				std::stable_sort(
					middle_cvpoints.begin(),
					middle_cvpoints.end(), 
					sort_cvpoint_by_x
				);
				cvpoint_t middle_left = middle_cvpoints[0];
				cvpoint_t middle_right = middle_cvpoints[n-1];

				// (5.4) get rotate angle 
				float degree = NumpyUtil::get_degree_angle(middle_left,middle_right);
				float rad = NumpyUtil::get_rad_angle(middle_left,middle_right);

#ifdef DEBUG_INFO 
				{
					std::cout<<"degree = "<< degree << std::endl;
				}
#endif

				// (6) get rotated middle mask
				// lane mask in origin size
				cv::Mat middle_mask(ORIGIN_HEIGHT, ORIGIN_WIDTH, CV_8UC1, cv::Scalar(0));
				for(auto& cvpoint: middle_cvpoints){
					middle_mask.at<uchar>(cvpoint.y,cvpoint.x) = 255; // image[y,x] = 255
				}
				// rotate middle mask 
				cv::Mat rotated_middle_mask = NumpyUtil::cv_rotate_image_keep_size(middle_mask, degree);
				
				// rotated origin image 
				cv::Mat image_with_mask = image.clone();
#ifdef DEBUG_IMAGE
				{
					for(auto& cvpoint: middle_cvpoints){
						image_with_mask.at<cv::Vec3b>(cvpoint.y,cvpoint.x) = cv::Vec3b(0,0,255); // image[y,x] = 255
					}

					cv::circle(image_with_mask, middle_left,  2, COLOR_YELLOW, 6);
					cv::circle(image_with_mask, middle_right, 2, COLOR_YELLOW, 6);
				}
#endif
				cv::Mat rotated_image_with_mask = NumpyUtil::cv_rotate_image_keep_size(image_with_mask, degree);


#ifdef DEBUG_IMAGE
				{
					cv::imwrite("6_1_middle_mask.png", middle_mask);
					cv::imwrite("6_2_rotated_middle_mask.png", rotated_middle_mask);
					cv::imwrite("6_3_image_with_mask.png", image_with_mask);
					cv::imwrite("6_4_rotated_image_with_mask.png", rotated_image_with_mask);
				}
#endif
				***/				
				cvpoint_t top_point;
				cvpoint_t bottom_point;  
				mosun_result.success = get_top_bottom_point(rotated_middle_mask,top_point,bottom_point,middle_left_org,middle_right_org);
				//std::cout<<" bottom_point = "<<bottom_point.x<<" "<<bottom_point.y << std::endl;            
#ifdef DEBUG_IMAGE
				{
					for(auto& cvpoint: middle_cvpoints_orgsize){
						rotated_image_with_mask.at<cv::Vec3b>(cvpoint.y,cvpoint.x) = cv::Vec3b(0,0,255); // image[y,x] = 255
					}

					cv::circle(rotated_image_with_mask, middle_left_org,  2, COLOR_YELLOW, 6);
					cv::circle(rotated_image_with_mask, middle_right_org, 2, COLOR_YELLOW, 6);                                 
          cv::circle(rotated_image_with_mask, bottom_point, 2, COLOR_BLUE, 6);                                                     
				}
#endif
  
#ifdef DEBUG_IMAGE
				{
					cv::imwrite("6_1_rotated_middle_mask_resize.png", rotated_middle_mask_resize);
          cv::imwrite("6_2_rotated_middle_mask.png", rotated_middle_mask);                                                 
					cv::imwrite("6_3_image_with_mask.png", rotated_image_with_mask);
					cv::imwrite("6_4_rotated_binary_resize.png", rotated_binary_resize);                                                 
				}
#endif 
				if (mosun_result.success){ 
					mosun_result.rotated = rotated_image_with_mask;
					mosun_result.top_point = top_point;
					mosun_result.bottom_point = bottom_point;
#ifdef DEBUG_INFO
          std::cout<<"bottom_point = "<< bottom_point.x << std::endl;
#endif        
					mosun_result.left_point = left_point;
					mosun_result.right_point = right_point;
					mosun_result.mosun_in_pixel = bottom_point.y - top_point.y;

					int y_delta = mosun_result.mosun_in_pixel;
					int x_delta = (right_point.x - left_point.x);
          int middle_distance = middle_right_org.x - middle_left_org.x;
          int x_delta_fix = x_delta;
          int point_correct = 0;
          float density = 1.0*mosun_result.middle_area/(middle_distance/w_rad);
#ifdef DEBUG_INFO
          std::cout<<"density = "<< density << std::endl;
#endif          
          if (mosun_result.left_area<50)
          {
              point_correct = mosun_result.left_point.x-(x_delta-middle_distance);
              mosun_result.left_point.x = std::max(0,point_correct);
          } 
          if (mosun_result.right_area<50)     
          {
              point_correct = mosun_result.right_point.x+(x_delta-middle_distance);
              mosun_result.right_point.x = std::min(ORIGIN_WIDTH-1,point_correct);
          }                 
					//int x_delta_fix = x_delta/ cos(rad); 
          x_delta_fix = mosun_result.right_point.x - mosun_result.left_point.x;
                    
          //if (density>3.7 && density>6.5)
          //{
              mosun_result.mosun_in_pixel = std::max(0,mosun_result.mosun_in_pixel-int(density));
          //}
          float pixel_meter_factor = 0.0;
            if (x_delta_fix>0.3*ORIGIN_WIDTH && BETWEEN(top_point.y,ORIGIN_HEIGHT*0.2,ORIGIN_HEIGHT*0.8)){   
					    pixel_meter_factor = params.real_width / (x_delta_fix*1.0);
          }
					if (!mosun_result.left_right_symmetrical){

#ifdef DEBUG_INFO
						printf("fix dynamic factor %f   to default %f \n",
							pixel_meter_factor,
							params.default_pixel_meter_factor
						);
#endif

						pixel_meter_factor = params.default_pixel_meter_factor;
					} 

					mosun_result.mosun_in_meter = mosun_result.mosun_in_pixel * pixel_meter_factor;
					if (mosun_result.mosun_in_meter>4.0)
             mosun_result.mosun_in_meter = 0.0;

#ifdef DEBUG_INFO
					std::cout<<" degree = "<<degree << std::endl;
					std::cout<<" x_delta = "<<x_delta << std::endl;
					std::cout<<" x_delta_fix = "<<x_delta << std::endl;
					std::cout<<" y_delta = "<<y_delta << std::endl;
					std::cout<<" pixel_meter_factor = "<<pixel_meter_factor << std::endl; // 0.0780069
					std::cout<<" mosun_in_meter = "<<mosun_result.mosun_in_meter << std::endl;
#endif
					// >=1.0 WARNING; <1.0 no.
				}
				return mosun_result;
			}

		}	
	}
}// end namespace
