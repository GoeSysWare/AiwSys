#include "distance_api.h"
#include "internal/impl/distance_api_impl.h"

namespace watrix {
	namespace algorithm {

		bool DistanceApi::get_distance(
			const cv::Mat& left,
			const cv::Mat& right,
			const int image_distance,
			int& distance
		)
		{
			return internal::DistanceApiImpl::get_distance(
				left, 
				right, 
				image_distance, 
				distance
			);
		}

	}
}// end namespace

