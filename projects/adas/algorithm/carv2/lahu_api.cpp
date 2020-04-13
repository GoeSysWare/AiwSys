#include "lahu_api.h"
#include "internal/lahu_api_impl.h"

namespace watrix {
	namespace algorithm {

		void LahuApi::init(
			const caffe_net_file_t& detect_net_params,
			int net_count,
			const LahuParam& lahu_param
		)
		{
			internal::LahuApiImpl::init(detect_net_params,net_count,lahu_param);
		}

		void LahuApi::free()
		{
			internal::LahuApiImpl::free();
		}

		bool LahuApi::detect(
			int net_id,
			const std::vector<cv::Mat>& v_image,
			std::vector<bool>& v_has_lahu,
			std::vector<float>& v_score1,
			std::vector<float>& v_score2,
			std::vector<cv::Rect>& v_boxes
		)
		{
			return internal::LahuApiImpl::detect(
				net_id, 
				v_image, 
				v_has_lahu, 
				v_score1,
				v_score2,
				v_boxes
			);
		}

		bool LahuApi::detect(
			int net_id,
			const cv::Mat& image,
			float& score1,
			float& score2,
			cv::Rect& box
		)
		{
			std::vector<cv::Mat> v_image;
			v_image.push_back(image);

			std::vector<bool> v_has_lahu;
			std::vector<float> v_score1;
			std::vector<float> v_score2;
			std::vector<cv::Rect> v_boxes;

			internal::LahuApiImpl::detect(
				net_id, 
				v_image, 
				v_has_lahu, 
				v_score1, 
				v_score2, 
				v_boxes
			);
			bool has_lahu = v_has_lahu[0];
			score1 = v_score1[0];
			score2 = v_score2[0];
			box = v_boxes[0];
			return has_lahu;
		}

	}
}// end namespace

