#include "yolo_net.h"

//#include "algorithm/core/caffe/internal/caffe_net_v2.h"

// caffe
#include <caffe/caffe.hpp>
using namespace caffe;

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region init and free for YoloNet
			// 	https://github.com/sfzhang15/RefineDet.git
			// public
			YoloNetConfig YoloNet::config_;
			std::vector<shared_caffe_net_t> YoloNet::v_net_;

			int YoloNet::input_height_ = 416;
			int YoloNet::input_width_ = 416;
			int YoloNet::output_height_ = 1; // N

			// private
			int YoloNet::m_counter = 0; // init
			int YoloNet::class_count_ = 80; // by default 80 (5 for autotrain)
			std::vector<std::string> YoloNet::class_labels_;
			cv::Size YoloNet::input_size_;
			cv::Mat YoloNet::mean_;

			void YoloNet::Init(
				const YoloNetConfig& config
			)
			{
				config_ = config;
				CaffeNetV2::Init(config_.net_count, config_.proto_filepath, config_.weight_filepath, v_net_);
				SetClassLabels(config_.label_filepath);
				input_size_ = config_.input_size; // size = (width,height)
				input_height_ = input_size_.height;
				input_width_ = input_size_.width;
				SetMean(config_.bgr_means);
			}

			void YoloNet::Free()
			{
				CaffeNetV2::Free(v_net_);
			}

			void YoloNet::SetClassLabels(const std::string& filepath)
			{
				std::ifstream infile (filepath);
				if (! infile.is_open())
				{
					std::cout << "Error opening file "<< filepath; 
					return;
				}
				
				char buffer[256];
				while (! infile.eof() ) {
					infile.getline (buffer,256);
		
					std::string label(buffer); // last label is empty 81 ===>80
					if (label!=""){
						//std::cout << label << std::endl;
						class_labels_.push_back(label);
					}
				}
				infile.close();

				class_count_ = (int)class_labels_.size();
				CHECK_GE(class_count_, 0) << "class_labels count must >= 0";

				//printf("[YOLO] class_labels_.size() = %d \n",class_count_);
			}

#pragma endregion 

#pragma region preprocess and postprocess

			void YoloNet::SetMean(const std::vector<float>& bgr_means) 
			{
				// set mean_ (bgr) mat
				const vector<float>& values = bgr_means;

				CHECK(values.size() == 1 || values.size() == input_channel_) <<
				"Specify either 1 mean_value or as many as channels: " << input_channel_;

				std::vector<cv::Mat> channels; // bgr channels
				for (int i = 0; i < input_channel_; ++i) 
				{
					/* Extract an individual channel. */
					//std::cout<<" i = "<< values[i] << std::endl;

					cv::Mat channel(input_size_.height, input_size_.width, CV_32FC1, cv::Scalar(values[i]));
					channels.push_back(channel);
				}
				cv::merge(channels, mean_); //hwc, bgr mean  416,416,3  

				//std::cout<<" mean_ = "<< mean_.channels() << std::endl;
			}

			/* Wrap the input layer of the network in separate cv::Mat objects
			* (one per channel). This way we save one memcpy operation and we
			* don't need to rely on cudaMemcpy2D. The last preprocessing
			* operation will write the separate channels directly to the input
			* layer. */
			void YoloNet::WrapInputLayer(
				shared_caffe_net_t net,
				std::vector<cv::Mat>* input_channels
			) 
			{
				Blob<float>* input_layer = net->input_blobs()[0];

				int width = input_layer->width();
				int height = input_layer->height();
				//std::cout<<"WrapInputLayer w:"<<width<<"  h:"<<height<<std::endl;
				//int width = input_width_;
				//int height = input_height_;
				float* input_data = input_layer->mutable_cpu_data();
				for (int i = 0; i < input_layer->channels(); ++i) {
					cv::Mat channel(height, width, CV_32FC1, input_data);
					input_channels->push_back(channel);
					input_data += width * height;
				}
			}

			/* Keep the original proportion of the image to resize. */
			void YoloNet::ResizeKP(const cv::Mat& img, cv::Mat& dst, const cv::Size& input_size) {
				int img_w = img.cols;
				int img_h = img.rows;
				int dst_w = input_size.width;
				int dst_h = input_size.height;
				float min_scale = std::min(dst_w * 1.0 / img_w, dst_h * 1.0 / img_h);
				int dst_w_ = int(img_w * min_scale);
				int dst_h_ = int(img_h * min_scale);

				int x = (dst_w - dst_w_) / 2;
				int y = (dst_h - dst_h_) / 2;

				int b = int(config_.bgr_means[0]);
				int g = int(config_.bgr_means[1]);
				int r = int(config_.bgr_means[2]);
				dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(b, g, r)); // 128,128,128
				cv::Mat imageKP;
				cv::Mat imageROI;

				cv::resize(img, imageKP, cv::Size(dst_w_, dst_h_), 0, 0, cv::INTER_CUBIC);
				imageROI = dst(cv::Rect(x, y, dst_w_, dst_h_));
				imageKP.copyTo(imageROI);  
			}


			/* Keep the original proportion of the image to resize. */
			void YoloNet::ResizeKPV2(const cv::Mat& img, cv::Mat& dst, const cv::Size& input_size) {
				int img_w = img.cols;
				int img_h = img.rows;
				int dst_w = input_size.width;
				int dst_h = input_size.height;
				float min_scale = std::min(dst_w * 1.0 / img_w, dst_h * 1.0 / img_h);
				//int dst_w_ = int(img_w * min_scale);
				//int dst_h_ = int(img_h * min_scale);

				// int x = (dst_w - dst_w_) / 2;
				// int y = (dst_h - dst_h_) / 2;

				int b = int(config_.bgr_means[0]);
				int g = int(config_.bgr_means[1]);
				int r = int(config_.bgr_means[2]);
				//dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(128, 128, 128));
				dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(b, g, r));

				cv::Mat imageKP;
				cv::Mat imageROI;

				cv::resize(img, imageKP, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_CUBIC);
				//imageROI = dst(cv::Rect(x, y, dst_w_, dst_h_));
				imageROI = dst(cv::Rect(0, 0, dst_w, dst_h));
				imageKP.copyTo(imageROI);  
			}


			void YoloNet::Preprocess(
				shared_caffe_net_t net,
				const cv::Mat& img,
				std::vector<cv::Mat>* input_channels, 
				double normalize_value
			) 
			{
				/* Convert the input image to the input image format of the network. */
				//std::cout<<"Preprocess c:"<<img.channels()<<" ic:"<<input_channel_<<std::endl;
				cv::Mat sample;
				if (img.channels() == 3 && input_channel_ == 1)
					cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
				else if (img.channels() == 4 && input_channel_ == 1)
					cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
				else if (img.channels() == 4 && input_channel_ == 3)
					cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
				else if (img.channels() == 1 && input_channel_ == 3)
					cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
				else
					sample = img; // bgr

				cv::Mat sample_resized; // 416,416
				if (sample.size() != input_size_)
					if (config_.resize_keep_flag){
						ResizeKP(sample, sample_resized, input_size_);
					} else {
						cv::resize(sample, sample_resized, input_size_);
					}
				else{
					sample_resized = sample;
				}

				/*
				void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;
				M[y,x] = m[y,x] * alpha + beta
				*/

				cv::Mat sample_float; // CV_8UC3 ===> CV_32FC3   0---255 ===> (0.0---255.0)*(1/255.0) = 0.0---1.0
				if (input_channel_ == 3) // alpa
					sample_resized.convertTo(sample_float, CV_32FC3, normalize_value);
				else
					sample_resized.convertTo(sample_float, CV_32FC1, normalize_value);

				cv::Mat sample_normalized; // bgr - bgr_mean  (0.0-1.0)
				cv::subtract(sample_float, mean_, sample_normalized);

				/* This operation will write the separate BGR planes directly to the
				* input layer of the network because it is wrapped by the cv::Mat
				* objects in input_channels. */
				cv::split(sample_normalized, *input_channels);

				CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
						== net->input_blobs()[0]->cpu_data())
					<< "Input channels are not wrapping the input layer of the network.";
			}


			void YoloNet::Postprocess(
				shared_caffe_net_t net,
				cv::Size origin_size,
				float confidence_threshold,
				detection_boxs_t& output
			)
			{
				int origin_width = origin_size.width;
				int origin_height = origin_size.height;

				/* Copy the output layer to a std::vector */
				Blob<float>* output_layer = net->output_blobs()[0];
				//std::cout<<"output_layer shape_string:" << output_layer->shape_string() << std::endl;
				// output_layer shape_string:1 1 N 7 (N*7)

				const float* p = output_layer->cpu_data();
				const int num_det = output_layer->height();
				
				//std::cout<<"origin num_det :" << num_det << std::endl;

				/*
				// p [0,1,2,3,4,5,6,7]  
				Detection format: (image_id, class_index, confidence, xmin,ymin,xmax,ymax) 
						0 3 0.998248 0.286413 0.528119 0.302957 0.579223 
				
				"__background__","car", "train", "person", "traffic light", "traffic sign"
				*/

				/*
				
				det_label = detections[0, 0, :, 1]
				det_conf = detections[0, 0, :, 2]
				det_xmin = detections[0, 0, :, 3] * image.shape[1]
				det_ymin = detections[0, 0, :, 4] * image.shape[0]
				det_xmax = detections[0, 0, :, 5] * image.shape[1]
				det_ymax = detections[0, 0, :, 6] * image.shape[0]
				 */

				for (int k = 0; k < num_det; ++k) {
					int image_id = int(p[0]);
					//int class_index = int(p[1]-1); // 0 background (1,2,3,4,5)
					int class_index = int(p[1]); // 0 background (1,2,3,4,5)
					float confidence = p[2]; // 0-1
					//std::cout<<"  class_index"<<class_index<<"  confidence"<<confidence<<std::endl;
					int xmin,ymin,xmax,ymax;

					if (!config_.resize_keep_flag){
						/* Parsing the result of cv::resize. */
						//xmin = int(p[3]*origin_width); // 0-1
						//ymin = int(p[4]*origin_height); // 0-1
						//xmax = int(p[5]*origin_width); // 0-1
						//ymax = int(p[6]*origin_height); // 0-1
						
						xmin = std::max(int(p[3]*origin_width), 0);
						ymin = std::max(int(p[4]*origin_height), 0);
						xmax = std::min(int(p[5]*origin_width), origin_width-1);
						ymax = std::min(int(p[6]*origin_height), origin_height-1);
					} else {
						/* Parsing the result of ResizeKP. */
						int img_w = origin_width;
						int img_h = origin_height;
						int dst_w = input_size_.width;
						int dst_h = input_size_.height;

						int dst_w_ = int(img_w * std::min(dst_w * 1.0 / img_w, dst_h * 1.0 / img_h));
						int dst_h_ = int(img_h * std::min(dst_w * 1.0 / img_w, dst_h * 1.0 / img_h));
						float shift_w = (dst_w - dst_w_) * 1.0 / 2 / dst_w;
						float shift_h = (dst_h - dst_h_) * 1.0 / 2 / dst_h;
						int max_wh = std::max(img_w, img_h);
						xmin = std::max(int(max_wh * (p[3] - shift_w)), 0);
						ymin = std::max(int(max_wh * (p[4] - shift_h)), 0);
						xmax = std::min(int(max_wh * (p[5] - shift_w)), img_w);
						ymax = std::min(int(max_wh * (p[6] - shift_h)), img_h);
					}
					
					if (image_id == -1 || class_index < 0 || confidence < confidence_threshold) {
						// Skip invalid detection or detection with low confidence
						p += 7;
						continue;
					}

					detection_box_t detection_box{ 
								xmin,
								ymin,
								xmax,
								ymax,
								confidence,
								class_index,
								class_labels_[class_index],
								false,
								0,
								0
							};
					output.push_back(detection_box);

		//#define DEBUG_INFO
#ifdef DEBUG_INFO
				if (false){
					printf("[YOLO] xmin,xmax,ymin,ymax=%d,%d,%d,%d, clas_label= %s, conf= %f \n",
						detection_box.xmin,
						detection_box.xmax,
						detection_box.ymin,
						detection_box.ymax,
						detection_box.class_name.c_str(),
						detection_box.confidence
					);
				}
#endif

					//vector<float> detection(result, result + 7);
					//detections.push_back(detection);
					p += 7;
				}

				//std::cout<<"final num_boxs =  :" << output.size() << std::endl;
			}

#pragma endregion


			bool YoloNet::Detect(
				int net_id,
				const cv::Mat& image,
				detection_boxs_t& output
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, v_net_.size()) << "net_id invalid";
				shared_caffe_net_t net = v_net_[net_id];

				cv::Size origin_size = image.size();
				
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
				
				//printf("num_inputs is %d\n",net->num_inputs()); // 1
				//printf("num_outputs is %d\n",net->num_outputs()); // 1
				CHECK_EQ(net->num_inputs(), 1) << "[YOLO] Network should have exactly one input.";
				CHECK_EQ(net->num_outputs(), 1) << "[YOLO] Network should have exactly one outputs.";

				Blob<float>* input_layer = net->input_blobs()[0];
				//std::cout<<"heigh: "<<input_size_.height<<"  width: "<<input_size_.width<<std::endl;
				input_layer->Reshape(1, input_channel_, input_size_.height, input_size_.width);
				/* Forward dimension change to all layers. */
				net->Reshape();

				// cpu_data(b,g,r) <--->  vector<cv::Mat> input_channels  (b,g,r) <---> cv::Mat img

				std::vector<cv::Mat> input_channels;
				WrapInputLayer(net, &input_channels);

				Preprocess(net, image, &input_channels, config_.normalize_value);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-YOLO] [1] pre-process data: cost=" << cost*1.0 << std::endl;
				//std::cout<< "[API-YOLO] [1] pre-process data: cost=" << cost*1.0 << std::endl;
				total_cost+=cost;
				pre_cost += cost;
				LOG(INFO) << "[API-YOLO] #counter ="<<m_counter<<" pre_cost=" << pre_cost/(m_counter*1.0) << std::endl;
				//std::cout<< "[API-YOLO] #counter ="<<m_counter<<" pre_cost=" << pre_cost/(m_counter*1.0) << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				net->Forward();

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-YOLO] [2] forward net: cost=" << cost*1.0 << std::endl;
				//std::cout << "[API-YOLO] [2] forward net: cost=" << cost*1.0 << std::endl;

				total_cost+=cost;
				forward_cost += cost;
				LOG(INFO) << "[API-YOLO] #counter ="<<m_counter<<" forward_cost=" << forward_cost/(m_counter*1.0) << std::endl;
				//std::cout << "[API-YOLO] #counter ="<<m_counter<<" forward_cost=" << forward_cost/(m_counter*1.0) << std::endl;
				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				/* Copy the output layer to a std::vector */
				Postprocess(net, origin_size, config_.confidence_threshold, output);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-YOLO] [3] post-process: cost=" << cost*1.0 << std::endl;
				//std::cout << "[API-YOLO] [3] post-process: cost=" << cost*1.0 << std::endl;
				pt1 = boost::posix_time::microsec_clock::local_time();

				total_cost+=cost;
				post_cost += cost;
				LOG(INFO) << "[API-YOLO] #counter ="<<m_counter<<" post_cost=" << post_cost/(m_counter*1.0) << std::endl;
				LOG(INFO) << "[API-YOLO] #counter ="<<m_counter<<" total_cost=" << total_cost/(m_counter*1.0) << std::endl;
				//std::cout<< "[API-YOLO] #counter ="<<m_counter<<" post_cost=" << post_cost/(m_counter*1.0) << std::endl;
				//std::cout<< "[API-YOLO] #counter ="<<m_counter<<" total_cost=" << total_cost/(m_counter*1.0) << std::endl;				
#endif // DEBUG_TIME

				return true;

			}

		}
	}
}