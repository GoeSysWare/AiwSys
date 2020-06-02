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
				YoloDarknetApi(const DarknetYoloConfig& config){Init(config); }
				~YoloDarknetApi(){Free(); }				
				 void Init(const DarknetYoloConfig& config);
				 void Free();

                 void SetClassLabels(const std::string& filepath);

                 void img2buffer(cv::Mat img, float* data);
                
                 image Mat2Image(cv::Mat mat);

                 void DealWithDetection(detection *dets, int nboxes, cv::Mat img, detection_boxs_t& output);

				 bool Detect(
					const std::vector<cv::Mat>& v_image,
					std::vector<detection_boxs_t>& v_output
				);

            public:
           		network* net_dk; // darknet ģ���ļ�
                 DarknetYoloConfig config_;
				 int class_count_; // 
				 std::vector<std::string> class_labels_; // class labels: person,car,...
                 int batchsize;
                 int CHANNELS; // 3
                 int INPUT_H; // 416
                 int INPUT_W; // 416
		};
	}
}// end namespace