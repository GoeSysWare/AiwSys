#include "yolo_api.h"

#include "internal/caffe/yolo_net.h"

namespace watrix {
	namespace algorithm {

		void YoloApi::Init(const YoloNetConfig& config)
		{
			internal::YoloNet::Init(config);
		}

		void YoloApi::Free()
		{
			internal::YoloNet::Free();
		}

		bool YoloApi::Detect(
			int net_id,
			const cv::Mat& image,
			detection_boxs_t& output
		)
		{
			return internal::YoloNet::Detect(net_id, image, output);
		}

		bool YoloApi::Detect(
			int net_id,
			const std::vector<cv::Mat>& v_image,
			std::vector<detection_boxs_t>& v_output
		)
		{
			bool success = false;
			for(int i=0;i<v_image.size();++i){
				detection_boxs_t output;
				success = internal::YoloNet::Detect(net_id, v_image[i], output);
				v_output.push_back(output);
			}
			return success;
		}
	}
}// end namespace

