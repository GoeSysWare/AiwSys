#include "sidewall_type.h"

namespace watrix {
	namespace algorithm {

		int min_history_size = 3; // at lease 3 history images
		int min_good_match_size = 10; // good match的最小数量
		float min_nn_match_ratio = 0.7f; // Nearest-neighbour matching ratio
		int max_corner_angle = 10; // 变换后的box的4个corner的最大角度 

		int y_move_negative_max_offset = 5; // y_move<0的时候，能够容忍的最大像素偏移值
											//eg.如果 y_move = -0.5 强制y_move = 0
		int y_move_min_valid_pixel_count = 10; // 计算y_move所使用的有效像素数量

		std::ostream& operator<<(
			std::ostream& cout, 
			const SidewallType::sidewall_param_t& sidewall_param
		)
		{
			cout <<"sidewall_param.min_history_size="<< sidewall_param.min_history_size << std::endl;
			cout << "sidewall_param.min_good_match_size=" << sidewall_param.min_good_match_size << std::endl;

			cout << "sidewall_param.min_nn_match_ratio=" << sidewall_param.min_nn_match_ratio << std::endl;
			cout << "sidewall_param.max_corner_angle=" << sidewall_param.max_corner_angle << std::endl;

			cout << "sidewall_param.y_move_negative_max_offset=" << sidewall_param.y_move_negative_max_offset << std::endl;
			cout << "sidewall_param.y_move_min_valid_pixel_count=" << sidewall_param.y_move_min_valid_pixel_count << std::endl;

			return cout;
		}

	}
}

