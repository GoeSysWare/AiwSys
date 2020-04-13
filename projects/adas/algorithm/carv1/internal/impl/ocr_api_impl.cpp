#include "ocr_api_impl.h"

#ifdef USE_DLIB 

#include "../net/ocr_net.h"
#include "../ocr_util.h"

#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"

// glog
#include <glog/logging.h>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

// dlib
#include <dlib/image_transforms.h> // extract_fhog_features  draw_fhog
#include <dlib/opencv.h> // cv_image
using namespace dlib;


namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region ocr net
			void OcrApiImpl::init(
				const caffe_net_file_t& detect_net_params
			)
			{
				OcrNet::init(detect_net_params);
			}

			void OcrApiImpl::free()
			{
				OcrNet::free();
			}

			void OcrApiImpl::detect(
				const int& net_id,
				const std::vector<OcrType::ocr_mat_pair_t>& v_pair_image,
				const std::vector<bool>& v_up,
				std::vector<bool>& v_has_roi,
				std::vector<cv::Rect>& v_box,
				std::vector<cv::Mat>& v_roi
			)
			{
				CHECK_GE(net_id, 0) << "net_id invalid";
				CHECK_LE(net_id, 3) << "net_id invalid";
				shared_caffe_net_t ocr_net = OcrNet::v_net[net_id];
				return OcrApiImpl::detect(
					ocr_net,
					v_pair_image,
					v_up,
					v_has_roi,
					v_box,
					v_roi
				);
			}

			void OcrApiImpl::detect(
				shared_caffe_net_t detect_net,
				const std::vector<OcrType::ocr_mat_pair_t>& v_pair_image,
				const std::vector<bool>& v_up,
				std::vector<bool>& v_has_roi,
				std::vector<cv::Rect>& v_box,
				std::vector<cv::Mat>& v_roi
			)
			{
#ifdef DEBUG_TIME
				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::ptime pt2;
				int64_t cost;
#endif // DEBUG_TIME

				int batch_size = v_pair_image.size();
				CHECK_LE(batch_size, m_max_batch_size) << "ocr batch_size must <" << m_max_batch_size;
				CHECK_GE(batch_size, 1) << "ocr batch_size must >=1";
				for (size_t i = 0; i < v_pair_image.size(); i++)
				{
					CHECK(!v_pair_image[i].first.empty()) << "invalid mat";
					CHECK(!v_pair_image[i].second.empty()) << "invalid mat";
				}

				int width = v_pair_image[0].first.cols; // 2048
				int height = v_pair_image[0].first.rows; // 1024
				int clip_start = int(width * ocr_param.clip_start_ratio);
				int clip_end = int(width * ocr_param.clip_end_ratio);

				int input_width = OcrNet::input_width;
				int input_height = OcrNet::input_height;
				cv::Size input_size(input_width, input_height);

				std::vector<cv::Mat> v_input;
				std::vector<cv::Mat> v_clip_input;
				std::vector<cv::Mat> v_rotate_input; // rotate 90

				int output_width = OcrNet::output_width;
				int output_height = OcrNet::output_height;

				bool rotate_image_flag = false;
				/*
				num_inputs()=1
				num_outputs()=1
				input_blob shape_string:1 1 1024 512 (524288)
				output_blob shape_string:1 1 1024 512 (524288)
				*/
				CaffeNet::caffe_net_n_inputs_t detect_n_inputs;
				CaffeNet::caffe_net_n_outputs_t detect_n_outputs;
				OcrNet::get_inputs_outputs(detect_n_inputs, detect_n_outputs); // 1-inputs, 1-outpus

				for (int i = 0; i < batch_size; i++)
				{
					const OcrType::ocr_mat_pair_t& ocr_pair = v_pair_image[i];

					const cv::Mat& first_image = ocr_pair.first;
					const cv::Mat& second_image = ocr_pair.second;

					cv::Mat full_image = OpencvUtil::concat_mat(first_image, second_image);

					if (!v_up[i]) { // flip image
						flip(full_image, full_image, 0);	//rotate 180 in the vertical direction.
					}
#ifdef shared_DEBUG
					LOG(INFO)<<"[API] full_image.type()=" << full_image.type() << std::endl; // CV_8UC1
					LOG(INFO)<<"[API] full_image.size()=" << full_image.size() << std::endl; //full_image.size()=[2048 x 2048]
#endif
					
					v_input.push_back(full_image);

					// input->clip_input->rotate_input--->resized_input=======> output_diff 
					// 2048,2048--->1024,2048--->2048,1024--->1024,512====> 1024,512
					cv::Rect rect(clip_start, 0, clip_end - clip_start, full_image.size().height);
					Mat clip_input2 = full_image(rect); // clip image 

					Mat clip_input = OpencvUtil::clip_mat(
						full_image, 
						ocr_param.clip_start_ratio, 
						ocr_param.clip_end_ratio
					);
					v_clip_input.push_back(clip_input);

					Mat rotate_input;
					if (rotate_image_flag) {
						rotate_input = OpencvUtil::rotate_mat(clip_input);
					}

#ifdef shared_DEBUG
					LOG(INFO)<<"[API] clip_input.size()=" << clip_input.size() << std::endl;
#endif // shared_DEBUG

					channel_mat_t channel_mat;
					cv::Mat resized_input;

					if (rotate_image_flag) {
						cv::resize(rotate_input, resized_input, input_size);
					}
					else {
						cv::resize(clip_input, resized_input, input_size);
					}
					
					channel_mat.push_back(resized_input); // 

#ifdef shared_DEBUG
					cv::imwrite("ocr/0_clip_input.jpg", clip_input);
					if (rotate_image_flag){
						cv::imwrite("ocr/1_rotate_input.jpg", rotate_input);
					}

					cv::imwrite("ocr/2_resized_input.jpg", resized_input);
					LOG(INFO)<<"[API] resized_input.size()=" << resized_input.size() << std::endl;
#endif // shared_DEBUG

					detect_n_inputs[0].blob_channel_mat.push_back(channel_mat); 
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-OCR] [1] before net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				OcrNet::forward(
					detect_net,
					detect_n_inputs,
					detect_n_outputs
				);

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-OCR] [2] forward net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

				blob_channel_mat_t& v_output_mat = detect_n_outputs[0].blob_channel_mat; // 1-outputs

#ifdef shared_DEBUG
				LOG(INFO)<<"[API] v_output_mat.size()=" << v_output_mat.size() << std::endl;
#endif // shared_DEBUG

				for (int i = 0; i < batch_size; i++)
				{
					cv::Mat output_diff;
					if (rotate_image_flag)
					{
						cv::Mat rotate_output_diff = v_output_mat[i][0]; // 1-channel
						output_diff = OpencvUtil::rotate_mat2(rotate_output_diff);

#ifdef shared_DEBUG
						cv::imwrite("ocr/3_rotate_output_diff.jpg", rotate_output_diff);
#endif // shared_DEBUG
					}
					else {
						output_diff = v_output_mat[i][0]; // 1-channel
					}

					cv::Mat input = v_input[i]; // 2048,2048
					cv::Mat clip_input = v_clip_input[i]; // 1024,2048

#ifdef shared_DEBUG
					cv::imwrite("ocr/4_output_diff.jpg", output_diff);
#endif // shared_DEBUG

					contours_t contours_in_diff;
					OpencvUtil::get_contours(
						output_diff,
						0.5f, 
						ocr_param.contour_min_area,
						contours_in_diff
					);

#ifdef shared_DEBUG
					LOG(INFO)<<"[API] contours_in_diff.size()=" << contours_in_diff.size() << std::endl; // 5
#endif // shared_DEBUG

					boxs_t boxs_in_diff;
					if (contours_in_diff.size() > 0) {

						//LOG(INFO)<<"[API] 4444444444444444444444444444444444444444\n";

						std::vector<int> bbox_x;
						std::vector<int> bbox_y;
						for (int i = 0; i < contours_in_diff.size(); i++) {
							cv::Rect rect = boundingRect(contours_in_diff[i]);
							bbox_x.push_back(int(rect.x + rect.width / 2));
							bbox_y.push_back(int(rect.y + rect.height / 2));

							boxs_in_diff.push_back(rect);
						}

						int median_x = int(bbox_x.size() / 2);
						int median_y = int(bbox_y.size() / 2);
						std::nth_element(bbox_x.begin(), bbox_x.begin() + median_x, bbox_x.end());
						std::nth_element(bbox_y.begin(), bbox_y.begin() + median_y, bbox_y.end());

						// 对N个boxs的面积进行sort，只使用面积最大的那个box
						//std::sort(boxs_in_clip.begin(), boxs_in_clip.end(), OpencvUtil::rect_compare);

						int W = output_diff.size().width;
						int H = output_diff.size().height;
						//roi_size 需要作为输入参数 200 for diff,400 for clip
						cv::Rect rect_in = cv::Rect(
							bbox_x[median_x] - int(ocr_param.roi_image_width / 2),
							bbox_y[median_y] - int(ocr_param.roi_image_height / 2),
							ocr_param.roi_image_width,
							ocr_param.roi_image_height
						);

						cv::Rect box_in_diff = OpencvUtil::boundary(rect_in, output_diff.size());

						cv::Rect box_in_clip = OpencvUtil::diff_box_to_origin_box(
							box_in_diff,
							output_diff.size(),
							clip_input.size(),
							0
							);

						cv::Rect box_in_input = box_in_clip;
						box_in_input.x += clip_start; // offset with 2048*0.4

						//LOG(INFO)<<"[API] 55555555555555555555555555555555555555555555555\n";

#ifdef shared_DEBUG
						LOG(INFO)<<"[API] rect_in=" << rect_in << std::endl;
						LOG(INFO)<<"[API] box_in_diff=" << box_in_diff << std::endl;
						LOG(INFO)<<"[API] box_in_clip=" << box_in_clip << std::endl;
						LOG(INFO)<<"[API] box_in_input=" << box_in_input << std::endl;
#endif // shared_DEBUG

						cv::Mat roi_image = clip_input(box_in_clip);

						bool has_roi;
						cv::Rect roi_box;
						cv::Mat roi;
						OcrApiImpl::roi_detect(
							roi_image, 
							v_up[0], 
							has_roi, 
							roi_box,
							roi
						);

						//LOG(INFO)<<"[API] 6666666666666666666666666666666666666666666\n";

						// roi_box --->  box in clip (x,y offset)
						cv::Rect roi_box_in_clip = roi_box;
						roi_box_in_clip.x += box_in_clip.x;
						roi_box_in_clip.y += box_in_clip.y;

						// roi box in input
						cv::Rect roi_box_in_input = roi_box_in_clip;
						roi_box_in_input.x += clip_start; // offset with 2048*0.4

#ifdef shared_DEBUG
						cv::Mat diff_with_boxs;
						DisplayUtil::draw_boxs(output_diff, boxs_in_diff, 2, diff_with_boxs);
						cv::imwrite("ocr/5_diff_with_boxs.jpg", diff_with_boxs);

						cv::Mat clip_with_2box;
						DisplayUtil::draw_box(clip_input, box_in_clip, 2, clip_with_2box);
						DisplayUtil::draw_box(clip_with_2box, roi_box_in_clip, 2, clip_with_2box);
						cv::imwrite("ocr/6_clip_with_2box.jpg", clip_with_2box);

						cv::Mat input_with_2box;
						DisplayUtil::draw_box(input, box_in_input, 2, input_with_2box);
						DisplayUtil::draw_box(input_with_2box, roi_box_in_input, 2, input_with_2box);
						cv::imwrite("ocr/7_input_with_2box.jpg", input_with_2box);
#endif // shared_DEBUG

						v_has_roi.push_back(has_roi);
						//v_box.push_back(box_in_input); // roi image box
						v_box.push_back(roi_box_in_input); // roi box
						v_roi.push_back(roi);
					}
					else {
						v_has_roi.push_back(false);
						cv::Rect box;
						v_box.push_back(box);
						cv::Mat roi;
						v_roi.push_back(roi);
					}
				}

#ifdef DEBUG_TIME
				pt2 = boost::posix_time::microsec_clock::local_time();

				cost = (pt2 - pt1).total_milliseconds();
				LOG(INFO) << "[API-OCR] [3] after net1: cost=" << cost*1.0 << std::endl;

				pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME
				
			}

#pragma endregion

#pragma region ocr non-net

			OcrType::ocr_param_t OcrApiImpl::ocr_param;

			void OcrApiImpl::init_ocr_param(const OcrType::ocr_param_t& param)
			{
				ocr_param = param;
			}

			void OcrApiImpl::roi_detect(
				const cv::Mat& roi_image,
				const bool up,
				bool& has_roi,
				cv::Rect& roi_box,
				cv::Mat& roi
			)
			{
				CHECK(!roi_image.empty()) << "invalid mat";
				/*
				roi_image with 400,400 size from OcrNet
				*/

				//LOG(INFO)<<"[API] 5aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n";


				cv::Mat dst;
				//int threshold = (int)(ocr_param.box_binary_threshold * 255);
				//cv::threshold(roi_image, dst, threshold, 255, CV_THRESH_BINARY);
				// 特例，此处使用阈值过滤，效果反而不好。

				cv::threshold(roi_image, dst, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

#ifdef shared_DEBUG
				{
					cv::imwrite("ocr/roi_0.jpg", roi_image);
					cv::imwrite("ocr/roi_1_threshold.jpg", dst);
				}
#endif

				Mat element = cv::getStructuringElement(MORPH_RECT, Size(10, 10));
				cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, element);
				
#ifdef shared_DEBUG
				{
					cv::imwrite("ocr/roi_2_close.jpg", dst);
				}
#endif

				cv::medianBlur(dst, dst, 5);
				
				cv::Mat blur_diff = dst.clone();
#ifdef shared_DEBUG
				{
					cv::imwrite("ocr/roi_3_medianBlur.jpg", dst);
				}
#endif

				std::vector<contour_t> contours;
				cv::findContours(dst, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

				const int roi_min_width = ocr_param.roi_min_width;
				const int roi_max_width = ocr_param.roi_max_width;
				const int roi_min_height = ocr_param.roi_min_height;
				const int roi_max_height = ocr_param.roi_max_height;
				const int roi_width_delta = ocr_param.roi_width_delta;
				const int roi_height_delta = ocr_param.roi_height_delta;
				const float height_width_min_ratio = ocr_param.height_width_min_ratio;
				const float height_width_max_ratio = ocr_param.height_width_max_ratio;

				//if (x_t > 70 && x_t < 120 && y_t > 100 && y_t < 180) {
				//if (x_t > 80 && x_t < 180 && y_t > 100 && y_t < 240 && y_t / x_t > 0.9 && y_t / x_t < 2.5) {

#ifdef shared_DEBUG
				LOG(INFO)<<"[API] contours.size()=" << contours.size() << std::endl;
#endif

				std::vector<cv::Rect> filtered_boxs; // filtered_boxs
				std::vector<contour_t> refine_contours;

				//LOG(INFO)<<"[API] 5bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n";

				for (int i = 0; i < contours.size(); i++) {
					double contour_area2 = contourArea(contours[i]);

					cv::Rect box = cv::boundingRect(contours[i]);

#ifdef shared_DEBUG
					LOG(INFO)<<"[API] [*] box = " << box << std::endl;   // 88 x 110
#endif

					int width = box.width;
					int height = box.height;
					float height_width_ratio = height / (width*1.0f);
					
					if (width > roi_min_width && width < roi_max_width &&
						height > roi_min_height && height < roi_max_height
						&& height_width_ratio > height_width_min_ratio 
						&& height_width_ratio < height_width_max_ratio
						)
					{

#ifdef shared_DEBUG
						LOG(INFO)<<"[API] box = " << box << std::endl;   // 88 x 110
						LOG(INFO)<<"[API] height_width_ratio = " << height_width_ratio << std::endl;   // 88 x 110
#endif

						refine_contours.push_back(contours[i]);
						filtered_boxs.push_back(box);
					}
				}

#ifdef shared_DEBUG
				LOG(INFO)<<"[API] filtered_boxs.size()="<< filtered_boxs.size() << std::endl;

				cv::Mat roi_with_contours;
				DisplayUtil::draw_contours(roi_image, contours, 1, roi_with_contours);
				cv::imwrite("ocr/roi_4_with_contours.jpg", roi_with_contours);

				cv::Mat roi_with_refine_contours;
				DisplayUtil::draw_contours(roi_image, refine_contours, 1, roi_with_refine_contours);
				imwrite("ocr/roi_5_with_refine_contours.jpg", roi_with_refine_contours);

				cv::Mat roi_with_filtered_boxs;
				DisplayUtil::draw_boxs(roi_image, filtered_boxs, 1, roi_with_filtered_boxs);
				imwrite("ocr/roi_6_with_filtered_boxs.jpg", roi_with_filtered_boxs);
#endif
				
				has_roi = false;
				if (filtered_boxs.size()>0)
				{
					// we get filtered_boxs, sort and get max box
					// 对N个boxs的面积进行sort，只使用面积最大的那个box
					std::sort(filtered_boxs.begin(), filtered_boxs.end(), OpencvUtil::rect_compare);

					cv::Rect max_box = filtered_boxs[0];

					cv::Rect result_box(
						(max_box.x + roi_width_delta),
						(max_box.y + roi_height_delta),
						max_box.width - 2 * roi_width_delta,
						max_box.height - 2 * roi_height_delta
					);

					//one_roi = image(cv::Range(y_min + 25, y_max - 25), cv::Range(x_min + 4, x_max - 4));
					
					roi_box = result_box;
					roi = roi_image(roi_box);
					
					//roi = blur_diff(roi_box);
					//has_roi = true;
					// 对roi的平均灰度在过滤一次
					float avg_pixel = OpencvUtil::get_average_pixel(roi);
					if (avg_pixel>ocr_param.roi_avg_pixel_min_threshold)
					{
						has_roi = true;
					} 

#ifdef shared_DEBUG
					LOG(INFO)<<"[API] roi_box = " << roi_box << std::endl;

					cv::Mat roi_with_ocr_box;
					DisplayUtil::draw_box(roi_image, roi_box, 1, roi_with_ocr_box);
					imwrite("ocr/roi_7_with_ocr_box.jpg", roi_with_ocr_box);

					cv::Mat diff_with_ocr_box;
					DisplayUtil::draw_box(blur_diff, roi_box, 1, diff_with_ocr_box);
					imwrite("ocr/roi_8_diff_with_ocr_box.jpg", diff_with_ocr_box);

					imwrite("ocr/roi_9.jpg", roi);
#endif	
				}
				else {
					// no roi box
					has_roi = false;
				}

				//LOG(INFO)<<"[API] 5cccccccccccccccccccccccccccccccccccccccccccccccccccc\n";
			}

			void OcrApiImpl::roi_detect(
				const std::vector<cv::Mat>& v_roi_image,
				const std::vector<bool>& v_up,
				std::vector<bool>& v_has_roi,
				std::vector<cv::Rect>& v_roi_box,
				std::vector<cv::Mat>& v_roi
			)
			{
				for (size_t i = 0; i < v_roi_image.size(); i++)
				{
					bool has_roi = false;
					cv::Rect roi_box;
					cv::Mat roi;
					OcrApiImpl::roi_detect(
						v_roi_image[i],
						v_up[i],
						has_roi,
						roi_box,
						roi
					);
					v_has_roi.push_back(has_roi);
					v_roi_box.push_back(roi_box);
					v_roi.push_back(roi);
				}
			}

			bool OcrApiImpl::get_feature(
				const cv::Mat& roi,
				OcrType::feature_t& feature
			) 
			{
				CHECK(!roi.empty()) << "invalid mat";

				// fhog feature 
				cv::Mat dst, gray, blur, bin_img;
				cv::resize(roi, dst, cv::Size(128, 128));
				cv::GaussianBlur(dst, blur, cv::Size(5, 5), 0);
				cv::adaptiveThreshold(blur, bin_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);

				dlib::cv_image<unsigned char> cimg(bin_img);
				dlib::array2d<dlib::matrix<float, 31, 1> > hog;
				dlib::extract_fhog_features(cimg, hog);  // 14*14 mat with 31-vector
				dlib::draw_fhog(hog);
				feature.clear();

				for (long i = 0; i < hog.nr(); i++) {
					for (long j = 0; j < hog.nc(); j++) {
						for (long z = 0; z < hog[i][j].size(); z++) {
							feature.push_back(hog[i][j](z));
						}
					}
				}
				return true;
			}


			void OcrApiImpl::roi_recognise(
				const cv::Mat& roi,
				const OcrType::features_t& v_features,
				const float min_similarity,
				bool& success,
				float& similarity,
				std::string& result
			)
			{
				success = false;

				for (size_t i = 0; i < v_features.size(); i++)
				{
					const OcrType::id_feature_t& id_feature = v_features[i];
					LOG(INFO) << "[API-OCR] #"<<i << " id_feature =  " << id_feature.m_id << std::endl;
				}
				LOG(INFO) << "[API-OCR] min_similarity =  " << min_similarity << std::endl;
				
				float max_similarity = min_similarity;
				auto temp_features = v_features;
				std::vector<OcrType::id_feature_t>::iterator max_iter = temp_features.end();

				std::vector<float> cur_feat;
				OcrApiImpl::get_feature(roi, cur_feat);

				std::vector<OcrType::id_feature_t>::iterator iter = temp_features.begin();
				for (; iter != temp_features.end(); iter++) {
					float cur_similarity = OcrUtil::cosine(cur_feat, iter->feature);

					//LOG(INFO) << "[API-OCR] roi_recognise similarity_threshold = "<< min_similarity << ", similarity=" << cur_similarity << ",iter->m_id=" << iter->m_id << std::endl;

					LOG(INFO) << "[API-OCR] roi_recognise similarity = " << cur_similarity <<",iter->m_id=" << iter->m_id << std::endl;

					if (cur_similarity > max_similarity) {
						max_similarity = cur_similarity;
						max_iter = iter;
					}
				}

				if (max_iter == temp_features.end()) {
					success = false;
					LOG(INFO) << "[API-OCR] roi_recognise failed. " << std::endl;
				}
				else {
					success = true;
					similarity = max_similarity;
					result = max_iter->m_id;
					LOG(INFO) << "[API-OCR] roi_recognise success. similarity =" << similarity << ",result =" << max_iter->m_id << std::endl;
				}
			}

			void OcrApiImpl::recognise(
				const int& net_id,
				const std::vector<OcrType::ocr_mat_pair_t>& v_pair_image,
				const std::vector<bool>& v_up,
				const OcrType::features_t& v_features,
				const float min_similarity,
				std::vector<bool>& v_has_roi,
				std::vector<cv::Rect>& v_box,
				std::vector<cv::Mat>& v_roi,
				std::vector<bool>& v_success,
				std::vector<float>& v_similarity,
				std::vector<std::string>& v_result
			)
			{
				OcrApiImpl::detect(
					net_id,
					v_pair_image,
					v_up,
					v_has_roi,
					v_box,
					v_roi
				);

				for (size_t i = 0; i < v_has_roi.size(); i++)
				{
					bool success = false;
					float similarity = 0.0f;
					std::string result;
					if (v_has_roi[i]) {
						OcrApiImpl::roi_recognise(
							v_roi[i], 
							v_features,
							min_similarity, 
							success, 
							similarity,
							result
						);
					}

					v_success.push_back(success);
					v_similarity.push_back(similarity);
					v_result.push_back(result);
				}
				
			}
#pragma endregion


		}
	}
}


#endif