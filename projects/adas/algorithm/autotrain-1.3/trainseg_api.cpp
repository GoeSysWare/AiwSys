#include "trainseg_api.h"
#include "internal/caffe/trainseg_api_impl.h"

namespace watrix {
	namespace algorithm {

		void TrainSegApi::init(
			const caffe_net_file_t& detect_net_params,
			int net_count
		)
		{
			internal::TrainSegApiImpl::init(detect_net_params,net_count);
		}

		void TrainSegApi::free()
		{
			internal::TrainSegApiImpl::free();
		}

		void TrainSegApi::set_bgr_mean(
			const std::vector<float>& bgr_mean
		)
		{
			internal::TrainSegApiImpl::set_bgr_mean(bgr_mean);
		}

		bool TrainSegApi::train_seg(
			int net_id,
			const std::vector<cv::Mat>& v_image,
			std::vector<cv::Mat>& v_output
		)
		{
			return internal::TrainSegApiImpl::train_seg(net_id, v_image, v_output);
		}

		bool TrainSegApi::train_seg(
			int net_id,
			const cv::Mat& image,
			cv::Mat& output
		)
		{
			std::vector<cv::Mat> v_image;
			std::vector<cv::Mat> v_output;
			v_image.push_back(image);
			internal::TrainSegApiImpl::train_seg(net_id, v_image, v_output);
			output = v_output[0];
			return true;
		}

	}
}// end namespace

