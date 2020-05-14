#pragma once

#include <stdint.h>
#include <atomic>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> //
#include <opencv2/highgui.hpp> // imwrite
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include "modules/drivers/proto/sensor_image.pb.h"
#include "projects/adas/proto/adas_detection.pb.h"
#include "projects/adas/proto/adas_camera.pb.h"
#include "projects/adas/proto/adas_perception.pb.h"
#include "projects/adas/proto/adas_config.pb.h"

#include "projects/adas/component/perception/adas_perception_component.h"

#include "projects/adas/algorithm/algorithm_type.h"
#include "projects/adas/algorithm/algorithm.h"


namespace watrix
{
namespace projects
{
namespace adas
{

using namespace watrix::algorithm;
using namespace cv;

using watrix::projects::adas::AdasPerceptionComponent;

class PerceptionTask {
 public:

  using PerceptionComponentPtr = std::shared_ptr<AdasPerceptionComponent>;

  PerceptionTask(PerceptionComponentPtr p);
  virtual ~PerceptionTask();

  void Excute();

  static std::atomic<uint64_t> taskd_excuted_num_;

 private:
  PerceptionComponentPtr perception_;
  bool if_use_simulator_ = false;
  bool if_save_image_result_ = false;
  uint32_t sequence_num_;
  std::string save_image_dir_;
  std::vector<cv::Mat> v_image_;
  std::vector<std::string> sim_image_files_;
  std::vector<cv::Mat>  v_image_lane_front_result_;
	 apollo::drivers::PointCloud lidar2image_paint_;
	 apollo::drivers::PointCloud lidar_safe_area_;
	std::vector<cv::Point3f> lidar_cloud_buf_;
  watrix::algorithm::LaneInvasionConfig lane_invasion_config_;
  std::string parameter_name_;

//算法需要的参数
private:


detection_boxs_t yolo_detection_boxs0_;
detection_boxs_t yolo_detection_boxs1_;
cv::Mat laneseg_binary_mask0_;
cv::Mat laneseg_binary_mask1_;
channel_mat_t v_instance_mask0_;
channel_mat_t v_instance_mask1_;


void DoDarknetDetectGPU();
void DoLaneSegSeqGPU();
void DoYoloDetectGPU();
void DoTrainSegGPU();
void SyncPerceptionResult();


 cvpoints_t GetTrainCVPoints(
   detection_boxs_t &detection_boxs,
  std::vector<cvpoints_t> &v_trains_cvpoint);
cvpoints_t GetLidarData();

};

}  // namespace record
}  // namespace cyber
}  // namespace apollo


