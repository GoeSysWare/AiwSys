#include "ocr_api.h"

#ifdef USE_DLIB 
#include "internal/impl/ocr_api_impl.h"

namespace watrix {
	namespace algorithm {

		void OcrApi::init(
			const caffe_net_file_t& detect_net_params
		)
		{
			internal::OcrApiImpl::init(detect_net_params);
		}

		void OcrApi::free()
		{
			internal::OcrApiImpl::free();
		}

		void OcrApi::init_ocr_param(const OcrType::ocr_param_t& param)
		{
			return internal::OcrApiImpl::init_ocr_param(param);
		}

		void OcrApi::detect(
			const int& net_id,
			const std::vector<OcrType::ocr_mat_pair_t>& v_pair_image,
			const std::vector<bool>& v_up,
			std::vector<bool>& v_has_roi,
			std::vector<cv::Rect>& v_box,
			std::vector<cv::Mat>& v_roi
		)
		{
			internal::OcrApiImpl::detect(
				net_id,
				v_pair_image,
				v_up,
				v_has_roi,
				v_box,
				v_roi
			);
		}

		void OcrApi::detect(
			const int& net_id,
			const OcrType::ocr_mat_pair_t& pair_image,
			const bool& up,
			bool& has_roi,
			cv::Rect& box,
			cv::Mat& roi
		)
		{
			std::vector<OcrType::ocr_mat_pair_t> v_pair_image;
			std::vector<bool> v_up;

			std::vector<bool> v_has_roi;
			std::vector<cv::Rect> v_box;
			std::vector<cv::Mat> v_roi;

			v_pair_image.push_back(pair_image);
			v_up.push_back(up);

			internal::OcrApiImpl::detect(
				net_id,
				v_pair_image,
				v_up,
				v_has_roi,
				v_box,
				v_roi
			);

			has_roi = v_has_roi[0];
			box = v_box[0];
			roi = v_roi[0];
		}

		

		bool OcrApi::get_feature(
			const cv::Mat& roi,
			OcrType::feature_t& feature
		)
		{
			return internal::OcrApiImpl::get_feature(roi, feature);
		}

		void OcrApi::recognise(
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
			internal::OcrApiImpl::recognise(
				net_id,
				v_pair_image,
				v_up,
				v_features,
				min_similarity, 
				v_has_roi,
				v_box,
				v_roi,
				v_success,
				v_similarity,
				v_result
			);
		}

		void OcrApi::recognise(
			const int& net_id,
			const OcrType::ocr_mat_pair_t& pair_image,
			const bool& up,
			const OcrType::features_t& v_features,
			const float min_similarity,
			bool& has_roi,
			cv::Rect& box,
			cv::Mat& roi,
			bool& success,
			float& similarity,
			std::string& result
		)
		{
			std::vector<OcrType::ocr_mat_pair_t> v_pair_image;
			std::vector<bool> v_up;

			std::vector<bool> v_has_roi;
			std::vector<cv::Rect> v_box;
			std::vector<cv::Mat> v_roi;

			std::vector<bool> v_success;
			std::vector<float> v_similarity;
			std::vector<std::string> v_result;
			v_pair_image.push_back(pair_image);
			v_up.push_back(up);
			
			internal::OcrApiImpl::recognise(
				net_id,
				v_pair_image,
				v_up,
				v_features,
				min_similarity,
				v_has_roi,
				v_box,
				v_roi,
				v_success,
				v_similarity,
				v_result
			);

			has_roi = v_has_roi[0];
			box = v_box[0];
			roi = v_roi[0];
			success = v_success[0];
			similarity = v_similarity[0];
			result = v_result[0];
		}

		void OcrApi::roi_recognise(
			const cv::Mat& roi,
			const OcrType::features_t& v_features,
			const float min_similarity,
			bool& success,
			float& similarity,
			std::string& result
		)
		{
			internal::OcrApiImpl::roi_recognise(
				roi,
				v_features,
				min_similarity,
				success,
				similarity,
				result
			);
		}
	}
}

#endif