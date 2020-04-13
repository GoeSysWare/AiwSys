#include "railway_api.h"
#include "internal/impl/railway_api_impl.h"

namespace watrix {
	namespace algorithm {

		void RailwayApi::init(
			const caffe_net_file_t& crop_net_params,
			const caffe_net_file_t& detect_net_params
		)
		{
			internal::RailwayApiImpl::init(crop_net_params, detect_net_params);
		}

		void RailwayApi::free()
		{
			internal::RailwayApiImpl::free();
		}

		void RailwayApi::railway_detect(
			const int& net_id,
			const std::vector<cv::Mat>& v_image,
			const cv::Size& blur_size,
			const int dilate_size,
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
			internal::RailwayApiImpl::crop_and_detect(
				net_id, 
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

	}
}// end namespace

