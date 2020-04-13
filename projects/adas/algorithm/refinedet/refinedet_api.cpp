#include "refinedet_api.h"
#include "internal/refinedet_api_impl.h"

namespace watrix {
	namespace algorithm {

		void RefineDetApi::init(
			const caffe_net_file_t& detect_net_params
		)
		{
			internal::RefineDetApiImpl::init(detect_net_params);
		}

		void RefineDetApi::free()
		{
			internal::RefineDetApiImpl::free();
		}

		void RefineDetApi::set_image_size(int height, int width)
		{
			internal::RefineDetApiImpl::set_image_size(height, width);
		}

		void RefineDetApi::set_bgr_mean(
			const std::vector<float>& bgr_mean
		)
		{
			internal::RefineDetApiImpl::set_bgr_mean(bgr_mean);
		}

		void RefineDetApi::detect(
			const int& net_id,
			const std::vector<cv::Mat>& v_image,
			float threshold,
			std::vector<detection_boxs_t>& v_output
		)
		{
			internal::RefineDetApiImpl::detect(net_id, v_image, threshold, v_output);
		}

		void RefineDetApi::detect(
			const int& net_id,
			const cv::Mat& image,
			float threshold,
			detection_boxs_t& output
		)
		{
			std::vector<cv::Mat> v_image;
			std::vector<detection_boxs_t> v_output;
			v_image.push_back(image);
			internal::RefineDetApiImpl::detect(net_id, v_image, threshold, v_output);
			output = v_output[0];
		}

		void RefineDetApi::detect_roi(
			const int& net_id, 
			const cv::Mat& image,
			const cv::Rect& roi,
			float threshold,
			detection_boxs_t& output
		)
		{
			cv::Mat roi_image = image(roi);
			detection_boxs_t roi_output;
			RefineDetApi::detect(
				(int)net_id, // for multiple threads (now only 1)
				roi_image,
				threshold,
				roi_output
			);

			int x_offset = roi.x;
			int y_offset = roi.y;

			// roi output ===> origin output
			for (size_t i = 0; i < roi_output.size(); i++)
			{
				detection_box_t& roi_detection_box = roi_output[i];

				roi_detection_box.xmin += x_offset;
				roi_detection_box.ymin += y_offset;

				roi_detection_box.xmax += x_offset;
				roi_detection_box.ymax += y_offset;
			}
			// roi_output becomes output
			output = roi_output;
		}


	}
}// end namespace

