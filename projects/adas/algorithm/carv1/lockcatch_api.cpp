#include "lockcatch_api.h"
#include "internal/impl/lockcatch_api_impl.h"

namespace watrix {
	namespace algorithm {

#pragma region init and free 
		void LockcatchApi::init(
			const caffe_net_file_t& crop_net_params,
			const caffe_net_file_t& refine_net_params
		)
		{
			internal::LockcatchApiImpl::init(crop_net_params, refine_net_params);
		}

		void LockcatchApi::free()
		{
			internal::LockcatchApiImpl::free();
		}

#pragma endregion 

		bool LockcatchApi::lockcatch_detect(
			const int& net_id,
			const std::vector<LockcatchType::lockcatch_mat_pair_t>& v_lockcatch,
			const LockcatchType::lockcatch_threshold_t& net1_lockcatch_threshold,
			const LockcatchType::lockcatch_threshold_t& net2_lockcatch_threshold,
			const cv::Size& blur_size,
			std::vector<bool>& v_has_lockcatch,
			std::vector<LockcatchType::lockcatch_status_t>& v_status,
			std::vector<boxs_t>& v_roi_boxs
		)
		{
			return 	internal::LockcatchApiImpl::lockcatch_detect(
				net_id,
				v_lockcatch,
				net1_lockcatch_threshold,
				net2_lockcatch_threshold,
				blur_size,
				v_has_lockcatch,
				v_status,
				v_roi_boxs
			);
		}

		bool LockcatchApi::lockcatch_detect_v0(
			const int& net_id,
			const std::vector<LockcatchType::lockcatch_mat_pair_t>& v_lockcatch,
			const LockcatchType::lockcatch_threshold_t& lockcatch_threshold,
			const cv::Size& blur_size,
			std::vector<bool>& v_has_lockcatch,
			std::vector<LockcatchType::lockcatch_status_t>& v_status,
			std::vector<boxs_t>& v_roi_boxs
		)
		{
			return 	internal::LockcatchApiImpl::lockcatch_detect_v0(
				net_id,
				v_lockcatch,
				lockcatch_threshold,
				blur_size,
				v_has_lockcatch,
				v_status,
				v_roi_boxs
			);
		}

#pragma region print lockcatch status
		std::string LockcatchApi::get_lockcatch_status_string(const LockcatchType::lockcatch_status_t& lockcatch_status)
		{
			return 	internal::LockcatchApiImpl::get_lockcatch_status_string(lockcatch_status);
		}

		bool LockcatchApi::has_anomaly(const LockcatchType::lockcatch_status_t& lockcatch_status)
		{
			return 	internal::LockcatchApiImpl::has_anomaly(lockcatch_status);
		}
#pragma endregion

	}
}// end namespace

