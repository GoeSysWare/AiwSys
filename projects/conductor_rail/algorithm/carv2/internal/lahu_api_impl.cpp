#include "lahu_api_impl.h"
//#include "lahu_net.h"

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
      pt_module_t LahuApiImpl::lahu_net;
      
			void LahuApiImpl::init(
				int net_count,
				const LahuParam& lahu_param
			)
			{
				LahuApiImpl::m_param = lahu_param;
        

				torch::DeviceType device_type = torch::kCUDA;  //torch::kCUDA  and torch::kCPU
				torch::Device device(device_type, 0);
				lahu_net = torch::jit::load(lahu_param.model_path);
				//assert(lahu_net != nullptr);
 				lahu_net.to(device);
				std::cout<<" pytorch load module OK \n";
        
			}

			void LahuApiImpl::free()
			{
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
        int lahu_net =0; 
				return LahuApiImpl::detect(
          net_id,
					lahu_net,
					v_image,
					v_has_lahu,
					v_score1,
					v_score2,
					v_boxes
				);
			}

			bool LahuApiImpl::detect(
        int net_id,
				int net,       
				const std::vector<cv::Mat>& v_image,
				std::vector<bool>& v_has_lahu,
				std::vector<float>& v_score1,
				std::vector<float>& v_score2,
				std::vector<cv::Rect>& v_boxes
			)
			{
				m_counter++;
				pt_module_t& module = lahu_net;
				torch::DeviceType device_type = torch::kCUDA;  //torch::kCUDA  and torch::kCPU
				torch::Device device(device_type, 0);
        
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
				std::vector<bool> v_roi_success;
				v_roi_success.resize(batch_size, false);

				double normalize_value = 1;
        torch::Tensor tensor_image;
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
          cv::Mat resized_image;
	      	//cv::resize(roi_image,resized_image,cv::Size(INPUT_WIDTH,INPUT_HEIGHT),0,0,CV_INTER_NN);
	      	cv::resize(roi_image,resized_image,cv::Size(INPUT_WIDTH,INPUT_HEIGHT),0,0,CV_INTER_LINEAR);                                                                                                 
	
	      	at::TensorOptions options(at::ScalarType::Byte);
		      tensor_image = torch::from_blob(resized_image.data,{1,resized_image.rows,resized_image.cols,1},options).to(device);
		      tensor_image = tensor_image.permute({0,3,1,2});
	      	tensor_image = tensor_image.toType(torch::kFloat32);
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

   	    torch::Tensor output = module.forward({tensor_image}).toTensor();	

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
        output = output.cpu();
        float* data_score = (float *)output.data_ptr(); 
				for (size_t i = 0; i < batch_size; i++)
				{
					// 2,1,1 

#ifdef DEBUG_INFO
					//std::cout<<"channel_mat.size() = "<< channel_mat.size() << std::endl;
#endif

#ifdef DEBUG_INFO
					printf("roi_success = %d, score_0 = %f, score_1 = %f \n",
				int(v_roi_success[i]), data_score[0], data_score[1]);
#endif
	
          bool has_lahu = v_roi_success[i] && (data_score[0] >=m_param.score_threshold);
					v_score1.push_back(data_score[0]);
					v_score2.push_back(data_score[1]);
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


			bool LahuApiImpl::sort_rect_by_width(
				const cv::Rect& left_rect,
				const cv::Rect& right_rect
			)
			{
				return left_rect.width > right_rect.width;
			}

      double laplacian(cv::Mat &image){
      	//assert(image.empty());
      	cv::Mat lap_image;
      	
      	cv::Laplacian(image,lap_image,CV_32FC1);
      	cv::Scalar mean,dev;
      	cv::meanStdDev(lap_image,mean,dev);
      	
      	return dev.val[0];
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
        cv::Mat gray_hist;
				NumpyUtil::cv_cvtcolor_to_gray(image, gray);
        double score= laplacian(gray);
    		if (score <5.47)
    			return NULL;
    
    		if (score< 12.24){
    				cv::equalizeHist(gray,gray_hist);
    		}
    		else if(score<22.36){
    			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    			clahe->setClipLimit(4.);
    			//clahe->setTilesGridSize(Size(8,8));
    			clahe->apply(gray,gray_hist);
    		} 
        else{
          gray_hist = gray;
        }  
				float factor = 4.;
				int minVal=10;
				int maxVal=160;
				
				int h = image.rows;
				int w = image.cols; 
				int resized_w = int(w/factor);
				int resized_h = int(h/factor);

				cv::Mat resized;
				NumpyUtil::cv_resize(gray_hist, resized, cv::Size(resized_w,resized_h));

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