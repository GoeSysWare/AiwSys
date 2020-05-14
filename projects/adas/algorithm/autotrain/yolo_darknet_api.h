#include "projects/adas/algorithm/algorithm_shared_export.h" 
#include "projects/adas/algorithm/algorithm_type.h" 

#include "darknet.h" // darknet

#include <vector>
#include <string>

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT YoloDarknetApi
		{
			public:
				static void Init(const DarknetYoloConfig& config);
				static void Free();

                static void SetClassLabels(const std::string& filepath);

                static void img2buffer(cv::Mat img, float* data);
                
                static image Mat2Image(cv::Mat mat);

                static void DealWithDetection(detection *dets, int nboxes, cv::Mat img, detection_boxs_t& output);

				static bool Detect(
					const std::vector<cv::Mat>& v_image,
					std::vector<detection_boxs_t>& v_output
				);

            public:
                static DarknetYoloConfig config_;
                static network *net_dk; // 

				static int class_count_; // 
				static std::vector<std::string> class_labels_; // class labels: person,car,...
                static int batchsize;
                static int CHANNELS; // 3
                static int INPUT_H; // 416
                static int INPUT_W; // 416
		};
	}
}// end namespace