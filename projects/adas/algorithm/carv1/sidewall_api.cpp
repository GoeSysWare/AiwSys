#include "sidewall_api.h"
#include "internal/impl/sidewall_api_impl.h"
#include "internal/sidewall_util.h"

#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"

// for opencv orb extracttor
#include <opencv2/core.hpp> // Mat
#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imdecode imshow
#include <opencv2/features2d.hpp> // KeyPoint
#include <opencv2/calib3d.hpp> //findHomography

namespace watrix {
	namespace algorithm {

		void SidewallApi::init(
			const caffe_net_file_t& caffe_net_params
		)
		{
			internal::SidewallApiImpl::init(caffe_net_params);
		}

		void SidewallApi::free()
		{
			internal::SidewallApiImpl::free();
		}

		void SidewallApi::set_image_size(int height, int width)
		{
			internal::SidewallApiImpl::set_image_size(height, width);
		}

		void SidewallApi::init_sidewall_param(const SidewallType::sidewall_param_t& param)
		{
			internal::SidewallUtil::init_sidewall_param(param);
		}

		void SidewallApi::detect_and_compute(
			const bool enable_gpu,
			const cv::Mat& image,
			keypoints_t& keypoints,
			cv::Mat& descriptor
		)
		{
			internal::SidewallUtil::detect_and_compute(
				enable_gpu,
				image, 
				keypoints, 
				descriptor
			);
		}

		bool SidewallApi::sidewall_match(
			const bool enable_gpu,
			const cv::Mat& image_object,
			const keypoints_t& keypoint_object,
			const cv::Mat& descriptor_object,
			const std::vector<cv::Mat>& images_history,
			const std::vector<keypoints_t>& keypoints_history,
			const std::vector<cv::Mat>& descriptors_history,
			bool& match_success,
			cv::Mat& best_roi,
			uint32_t& image_index,
			uint32_t& y_offset
		)
		{
			match_success =  internal::SidewallUtil::sidewall_match(
				enable_gpu,
				image_object, 
				keypoint_object, 
				descriptor_object,
				images_history, 
				keypoints_history, 
				descriptors_history,
				best_roi, 
				image_index, 
				y_offset
			);
			return match_success;
		}

		bool SidewallApi::sidewall_detect(
			const int& net_id,
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
			return internal::SidewallApiImpl::sidewall_detect(
				net_id,
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

		bool SidewallApi::sidewall_detect(
			const int& net_id,
			const cv::Mat& image,
			const cv::Mat& roi,
			const cv::Size& blur_size,
			const bool fix_distortion_flag,
			const float box_min_binary_threshold,
			const int box_min_width,
			const int box_min_height,
			bool& has_anomaly,
			boxs_t& anomaly_boxs
		)
		{
			std::vector<cv::Mat> v_image;
			std::vector<cv::Mat> v_roi;
			v_image.push_back(image);
			v_roi.push_back(roi);

			std::vector<bool> v_has_anomaly;
			std::vector<boxs_t> v_anomaly_boxs;

			internal::SidewallApiImpl::sidewall_detect(
				net_id,
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
			has_anomaly = v_has_anomaly[0];
			anomaly_boxs = v_anomaly_boxs[0];
			return has_anomaly;
		}

		void SidewallApi::sidewall_match_and_detect(
			const bool enable_gpu_match,
			const cv::Mat& image_object,
			const keypoints_t& keypoint_object,
			const cv::Mat& descriptor_object,
			const std::vector<cv::Mat>& images_history,
			const std::vector<keypoints_t>& keypoints_history,
			const std::vector<cv::Mat>& descriptors_history,

			const int& net_id,
			const cv::Size& blur_size,
			const bool fix_distortion_flag,
			const float box_min_binary_threshold,
			const int box_min_width,
			const int box_min_height,

			bool& match_success,
			cv::Mat& best_roi,
			uint32_t& image_index,
			uint32_t& y_offset,

			bool& has_anomaly,
			boxs_t& boxs
		)
		{
#ifdef DEBUG_TIME
			boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
			boost::posix_time::ptime pt2;
			int64_t cost;
#endif // DEBUG_TIME

			match_success = internal::SidewallUtil::sidewall_match(
				enable_gpu_match,
				image_object,
				keypoint_object,
				descriptor_object,
				images_history,
				keypoints_history,
				descriptors_history,
				best_roi,
				image_index,
				y_offset
			);

#ifdef DEBUG_TIME
			pt2 = boost::posix_time::microsec_clock::local_time();

			cost = (pt2 - pt1).total_milliseconds();
			LOG(INFO) << "[API-SIDEWALL] [1] match: cost=" << cost*1.0 << std::endl;

			pt1 = boost::posix_time::microsec_clock::local_time();
#endif // DEBUG_TIME

			if (match_success)
			{
				std::vector<cv::Mat> v_image;
				std::vector<cv::Mat> v_roi;
				v_image.push_back(image_object);
				v_roi.push_back(best_roi);

				std::vector<bool> v_has_anomaly;
				std::vector<boxs_t> v_anomaly_boxs;

				internal::SidewallApiImpl::sidewall_detect(
					net_id,
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
				has_anomaly = v_has_anomaly[0];
				boxs = v_anomaly_boxs[0];
			}
			else {
				has_anomaly = false;
			}

		}

	}
}// end namespace


/*
// Step 5: Do sidewall detect
	int batch_size = 3; //sidewall input_blob shape_string:n 2 1024 2048, which consumes large GPU memory. (batch_size<=3)

	std::vector<cv::Mat> v_image, v_roi;
	for (size_t i = 0; i < batch_size; i++)
	{
		v_image.push_back(image_object);
		v_roi.push_back(best_roi);
	}

	int box_min_height = 10;
	int box_min_width = 10;
	std::vector<bool> v_has_anomaly;
	std::vector<std::vector<cv::Rect>> v_anomaly_boxs;

	boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
	int test_count = 1;
	for (int t = 0; t < test_count; t++)
	{
		v_has_anomaly.clear();
		v_anomaly_boxs.clear();
		SidewallApi::sidewall_detect(
			int::NET_SIDEWALL_0, // change id for multiple threads (支持4个)
			v_image, v_roi, box_min_height, box_min_width,
			v_has_anomaly, v_anomaly_boxs
		);
	}

	boost::posix_time::ptime pt2 = boost::posix_time::microsec_clock::local_time();
	int64_t cost = (pt2 - pt1).total_milliseconds();
	int image_count = v_image.size()*test_count;
	LOG(INFO)<<"[API] cost=" << cost << " ms for #" << image_count << ",avg=" << cost*1.0 / image_count << std::endl;
	// sidewall cost=1048 ms,avg=524 ms for 2
	// cost=67704 ms for #200,avg=338.52
	// cost=335391 ms for #1000,avg=335.391

	bool display_image = true;
	if (display_image)
	{
		for (size_t i = 0; i < v_has_anomaly.size(); i++)
		{
			if (v_has_anomaly[i])
			{
				LOG(INFO)<<"[API] sidewall anomaly detected." << std::endl;
			}

			const std::vector<cv::Rect>& boxs = v_anomaly_boxs[i];

			cv::Mat mat_with_boxs;
			DisplayUtil::draw_boxs(v_image[i], boxs, 10, mat_with_boxs);
			cv::resize(
				mat_with_boxs,
				mat_with_boxs,
				cv::Size(mat_with_boxs.cols*0.5, mat_with_boxs.rows*0.5)
			);
			cv::imshow("mat_with_boxs", mat_with_boxs);
			cv::waitKey(0);

			cv::imwrite("sidewall/" + to_string(i) + "_mat_with_boxs.jpg", mat_with_boxs);
			
		}
	}
*/
