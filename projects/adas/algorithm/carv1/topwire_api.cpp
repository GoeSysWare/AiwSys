#include "topwire_api.h"
#include "internal/impl/topwire_api_impl.h"

namespace watrix {
	namespace algorithm {

		void TopwireApi::init(
			const caffe_net_file_t& crop_net_params,
			const caffe_net_file_t& detect_net_params
		)
		{
			internal::TopwireApiImpl::init(crop_net_params, detect_net_params);
		}

		void TopwireApi::free()
		{
			internal::TopwireApiImpl::free();
		}

		void TopwireApi::topwire_detect(
			const int& net_id,
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
			internal::TopwireApiImpl::crop_and_detect(
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
				v_crop_success,
				v_has_hole,
				v_hole_boxs,
				v_has_anomaly,
				v_anomaly_boxs
			);
		}

	}
}// end namespace

