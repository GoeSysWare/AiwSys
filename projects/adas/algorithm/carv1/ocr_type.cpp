#include "ocr_type.h"

namespace watrix {
	namespace algorithm {

		std::ostream& operator<<(std::ostream& cout, const OcrType::ocr_param_t& ocr_param)
		{
			cout << ocr_param.clip_start_ratio << std::endl;
			cout << ocr_param.clip_end_ratio << std::endl;

			cout << ocr_param.roi_min_width << std::endl;
			cout << ocr_param.roi_max_width << std::endl;
			cout << ocr_param.roi_min_height << std::endl;
			cout << ocr_param.roi_max_height << std::endl;
			cout << ocr_param.roi_width_delta << std::endl;
			cout << ocr_param.roi_height_delta << std::endl;
			cout << ocr_param.height_width_min_ratio << std::endl;
			cout << ocr_param.height_width_max_ratio << std::endl;

			return cout;
		}

	}
}

