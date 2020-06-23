#include "laneseg_api.h"
#include "internal/laneseg_util.h"

#include "internal/caffe/caffe_laneseg_api_impl.h"
#include "internal/pytorch/pt_simple_laneseg_api_impl.h"

namespace watrix {
	namespace algorithm {


		int LaneSegApi::lane_model_type = LANE_MODEL_TYPE::LANE_MODEL_CAFFE;

		void LaneSegApi::set_model_type(int lane_model_type)
		{
			LaneSegApi::lane_model_type = lane_model_type;
		}

		void LaneSegApi::init(
			const caffe_net_file_t& detect_net_params,
			int feature_dim,
			int net_count
		)
		{
			internal::CaffeLaneSegApiImpl::init(detect_net_params, feature_dim, net_count);
		}

		void LaneSegApi::init(
			const PtSimpleLaneSegNetParams& params,
			int net_count
		)
		{
			internal::PtSimpleLaneSegApiImpl::init(params, net_count);
		}

		void LaneSegApi::free()
		{
			switch (LaneSegApi::lane_model_type)
			{
			case LANE_MODEL_CAFFE:
				internal::CaffeLaneSegApiImpl::free();
				break;
			case LANE_MODEL_PT_SIMPLE:
				internal::PtSimpleLaneSegApiImpl::free();
				break;
			case LANE_MODEL_PT_COMPLEX:
				break;
			default:
				break;
			}
		}

		void LaneSegApi::set_bgr_mean(
			const std::vector<float>& bgr_mean
		)
		{
			switch (LaneSegApi::lane_model_type)
			{
			case LANE_MODEL_CAFFE:
				internal::CaffeLaneSegApiImpl::set_bgr_mean(bgr_mean);
				break;
			case LANE_MODEL_PT_SIMPLE:
				internal::PtSimpleLaneSegApiImpl::set_bgr_mean(bgr_mean);
				break;
			case LANE_MODEL_PT_COMPLEX:
				break;
			default:
				break;
			}
		}

		bool LaneSegApi::lane_seg(
			int net_id,
			const std::vector<cv::Mat>& v_image,
			int min_area_threshold, 
			std::vector<cv::Mat>& v_binary_mask, 
			std::vector<channel_mat_t>& v_instance_mask
		)
		{
			switch (LaneSegApi::lane_model_type)
			{
			case LANE_MODEL_CAFFE:
				return internal::CaffeLaneSegApiImpl::lane_seg(
					net_id, v_image, min_area_threshold, v_binary_mask, v_instance_mask
				);
				break;
			case LANE_MODEL_PT_SIMPLE:
				return internal::PtSimpleLaneSegApiImpl::lane_seg(
					net_id, v_image, min_area_threshold, v_binary_mask, v_instance_mask
				);
				break;
			case LANE_MODEL_PT_COMPLEX:
				break;
			default:
				break;
			}

			return true;			
		}


		bool LaneSegApi::lane_seg_sequence(
			int net_id,
			const std::vector<cv::Mat>& v_image_front_result,
			const std::vector<cv::Mat>& v_image_cur,
			int min_area_threshold, 
			std::vector<cv::Mat>& v_binary_mask, 
			std::vector<channel_mat_t>& v_instance_mask
		)
		{
			switch (LaneSegApi::lane_model_type)
			{
			case LANE_MODEL_CAFFE:
				break;
			case LANE_MODEL_PT_SIMPLE:
				return internal::PtSimpleLaneSegApiImpl::lane_seg_sequence(
					net_id, v_image_front_result, v_image_cur, min_area_threshold, v_binary_mask, v_instance_mask
				);
				break;
			case LANE_MODEL_PT_COMPLEX:
				break;
			default:
				break;
			}

			return true;			
		}


		bool LaneSegApi::lane_seg(
			int net_id,
			const cv::Mat& image,
			int min_area_threshold, 
			cv::Mat& binary_mask, 
			channel_mat_t& instance_mask
		)
		{
			std::vector<cv::Mat> v_image;
			std::vector<cv::Mat> v_binary_mask;
			std::vector<channel_mat_t> v_instance_mask;
			v_image.push_back(image);
			lane_seg(
				net_id, v_image, min_area_threshold, v_binary_mask, v_instance_mask
			);
			binary_mask = v_binary_mask[0];
			instance_mask = v_instance_mask[0];
			return true;
		}

		cv::Mat LaneSegApi::get_lane_full_binary_mask(const cv::Mat& binary_mask)
		{
			return internal::lanesegutil::get_lane_full_binary_mask(binary_mask);
		}

		bool LaneSegApi::lane_invasion_detect(
			CAMERA_TYPE camera_type, // long, short
			const cv::Mat& origin_image, 
			const cv::Mat& binary_mask, 
			const channel_mat_t& instance_mask,
			const detection_boxs_t& detection_boxs,
			//const std::vector<cvpoints_t>& trains_cvpoints,
			const cvpoints_t& lidar_cvpoints,
			// const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, // lidar pointscloud
			const std::vector<cv::Point3f> cloud, // lidar pointscloud
			const LaneInvasionConfig& config,
			cv::Mat& image_with_color_mask,
			int& lane_count,
			int& id_left,
			int& id_right,
			box_invasion_results_t& box_invasion_results,
			std::vector<int>& lidar_invasion_status,
			lane_safe_area_corner_t& lane_safe_area_corner,
			bool& is_open_long_camera, // 是否开远焦
			std::vector<lidar_invasion_cvbox>& cv_obstacle_box // lidar invasion object cv box
		)
		{
			std::vector<cvpoints_t> trains_cvpoints;
			return internal::lanesegutil::lane_invasion_detect(
				LaneSegApi::lane_model_type, // caffe / pt_simple / pt_complex
				camera_type,
				origin_image,
				binary_mask,
				instance_mask,
				detection_boxs,
				trains_cvpoints,
				lidar_cvpoints,
				cloud,
				config,
				image_with_color_mask,
				lane_count,
				id_left,
				id_right,
				box_invasion_results,
				lidar_invasion_status,
				lane_safe_area_corner,
				is_open_long_camera,
				cv_obstacle_box
			);
		}

	}
}// end namespace

