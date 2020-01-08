
#include "projects/conductor_rail/component/wear_component.h"

#include <yaml-cpp/yaml.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "modules/common/math/math_utils.h"
#include "modules/common/time/time.h"
#include "modules/common/time/time_util.h"

#include "modules/perception/camera/common/util.h"
#include "modules/perception/common/sensor_manager/sensor_manager.h"

#include "projects/conductor_rail/configs/config_gflags.h"

namespace aiwsys
{
namespace projects
{
namespace conductor_rail
{

using apollo::cyber::common::GetAbsolutePath;

using namespace apollo;
using namespace apollo::cyber;
using namespace apollo::perception;

static int GetGpuId(const WearDetectorInitOptions &options)
{
 
  detection::DetectorParam detection_param;
  std::string work_root = apollo::perception::camera::GetCyberWorkRoot();
  std::string config_file =
      GetAbsolutePath(options.root_dir, options.conf_file);
  config_file = GetAbsolutePath(work_root, config_file);
  if (!cyber::common::GetProtoFromFile(config_file, &detection_param))
  {
    AERROR << "Read config failed: " << config_file;
    return -1;
  }
  if (!detection_param.has_gpu_id())
  {
    AINFO << "gpu id not found.";
    return -1;
  }
  return detection_param.gpu_id();
}


WearComponent::~WearComponent() {}

bool WearComponent::Init()
{
  if (InitConfig() != cyber::SUCC)
  {
    AERROR << "InitConfig() failed.";
    return false;
  }
  //磨损的实际输出，内置定义格式
  sensorframe_writer_ =
      node_->CreateWriter<aiwsys::projects::common::SensorFrameMessage>(output_camera_channel_name_);
  //磨损的调试输出.proto 定义格式
  camera_debug_writer_ =
      node_->CreateWriter<detection::DetectionWearResult>(
          camera_debug_channel_name_);
  //初始化传感器
  if (InitSensorInfo() != cyber::SUCC)
  {
    AERROR << "InitSensorInfo() failed.";
    return false;
  }
  //初始化算法插件
  if (InitAlgorithmPlugin() != cyber::SUCC)
  {
    AERROR << "InitAlgorithmPlugin() failed.";
    return false;
  }
  //初始化图像帧
  if (InitCameraFrames() != cyber::SUCC)
  {
    AERROR << "InitCameraFrames() failed.";
    return false;
  }
  //初始化接收器
  if (InitCameraListeners() != cyber::SUCC)
  {
    AERROR << "InitCameraListeners() failed.";
    return false;
  }

  return true;
}

void WearComponent::OnReceiveImage(
    const std::shared_ptr<apollo::drivers::Image> &message,
    const std::string &camera_name)
{
  // std::lock_guard<std::mutex> lock(mutex_);
  // const double msg_timestamp = message->measurement_time() + timestamp_offset_;
  // AINFO << "Enter FusionCameraDetectionComponent::Proc(), "
  //       << " camera_name: " << camera_name
  //       << " image ts: " + std::to_string(msg_timestamp);
  // // timestamp should be almost monotonic
  // if (last_timestamp_ - msg_timestamp > ts_diff_)
  // {
  //   AINFO << "Received an old message. Last ts is " << std::setprecision(19)
  //         << last_timestamp_ << " current ts is " << msg_timestamp
  //         << " last - current is " << last_timestamp_ - msg_timestamp;
  //   return;
  // }
  // last_timestamp_ = msg_timestamp;
  // ++seq_num_;

  // // for e2e lantency statistics
  // {
  //   const double cur_time = apollo::common::time::Clock::NowInSeconds();
  //   const double start_latency = (cur_time - message->measurement_time()) * 1e3;
  //   AINFO << "FRAME_STATISTICS:Camera:Start:msg_time[" << camera_name << "-"
  //         << GLOG_TIMESTAMP(message->measurement_time()) << "]:cur_time["
  //         << GLOG_TIMESTAMP(cur_time) << "]:cur_latency[" << start_latency
  //         << "]";
  // }

  // // protobuf msg
  // std::shared_ptr<apollo::perception::PerceptionObstacles> out_message(
  //     new (std::nothrow) apollo::perception::PerceptionObstacles);
  // apollo::common::ErrorCode error_code = apollo::common::OK;

  // // prefused msg
  // std::shared_ptr<SensorFrameMessage> prefused_message(new (std::nothrow)
  //                                                          SensorFrameMessage);

  // if (InternalProc(message, camera_name, &error_code, prefused_message.get(),
  //                  out_message.get()) != cyber::SUCC)
  // {
  //   AERROR << "InternalProc failed, error_code: " << error_code;
  //   if (MakeProtobufMsg(msg_timestamp, seq_num_, {}, {}, error_code,
  //                       out_message.get()) != cyber::SUCC)
  //   {
  //     AERROR << "MakeProtobufMsg failed";
  //     return;
  //   }
  //   if (output_final_obstacles_)
  //   {
  //     writer_->Write(out_message);
  //   }
  //   return;
  // }

  // bool send_sensorframe_ret = sensorframe_writer_->Write(prefused_message);
  // AINFO << "send out prefused msg, ts: " << std::to_string(msg_timestamp)
  //       << "ret: " << send_sensorframe_ret;
  // // Send output msg
  // if (output_final_obstacles_)
  // {
  //   writer_->Write(out_message);
  // }
  // // for e2e lantency statistics
  // {
  //   const double end_timestamp = apollo::common::time::Clock::NowInSeconds();
  //   const double end_latency =
  //       (end_timestamp - message->measurement_time()) * 1e3;
  //   AINFO << "FRAME_STATISTICS:Camera:End:msg_time[" << camera_name << "-"
  //         << GLOG_TIMESTAMP(message->measurement_time()) << "]:cur_time["
  //         << GLOG_TIMESTAMP(end_timestamp) << "]:cur_latency[" << end_latency
  //         << "]";
  // }
}

int WearComponent::InitConfig()
{
  // the macro READ_CONF would return cyber::FAIL if config not exists
  aiwsys::projects::conductor_rail::RailCameraDetection
      rail_camera_detection_param;
  if (!GetProtoConfig(&rail_camera_detection_param))
  {
    AINFO << "load rail camera detection component proto param failed";
    return false;
  }
  // proto.1
  std::string camera_names_str = rail_camera_detection_param.camera_names();
  boost::algorithm::split(camera_names_, camera_names_str,
                          boost::algorithm::is_any_of(","));
  // 目前一个功能组件支持1个相机，软件内部可以支持多个
  if (camera_names_.size() != FLAGS_wear_camera_size)
  {
    AERROR << "Now RailCameraDetectionComponent only support " << FLAGS_wear_camera_size << " cameras";
    return cyber::FAIL;
  }
  // proto.2
  std::string input_camera_channel_names_str =
      rail_camera_detection_param.input_camera_channel_names();
  boost::algorithm::split(input_camera_channel_names_,
                          input_camera_channel_names_str,
                          boost::algorithm::is_any_of(","));
  if (input_camera_channel_names_.size() != camera_names_.size())
  {
    AERROR << "wrong input_camera_channel_names_.size(): "
           << input_camera_channel_names_.size();
    return cyber::FAIL;
  }
  // proto.4
  camera_perception_init_options_.root_dir =
      rail_camera_detection_param.camera_detection_conf_dir();
  // proto.5
  camera_perception_init_options_.conf_file =
      rail_camera_detection_param.camera_detection_conf_file();

  camera_perception_init_options_.use_cyber_work_root = true;
  // proto.6
  frame_capacity_ = rail_camera_detection_param.frame_capacity();
  // proto.7
  image_channel_num_ = rail_camera_detection_param.image_channel_num();
  // proto.8
  enable_visualization_ = rail_camera_detection_param.enable_visualization();
  // proto.9
  output_camera_channel_name_ =
      rail_camera_detection_param.output_camera_channel_name();
  // proto.10

  // proto.11

  // proto.13
  output_camera_debug_msg_ =
      rail_camera_detection_param.output_camera_debug_msg();
  // proto.14
  camera_debug_channel_name_ =
      rail_camera_detection_param.camera_debug_channel_name();
  // proto.15      
  ts_diff_ = rail_camera_detection_param.ts_diff();
  // proto.16
  visual_debug_folder_ = rail_camera_detection_param.visual_debug_folder();
  // proto.17
  visual_camera_ = rail_camera_detection_param.visual_camera();
  // proto.18
  // proto.19
  debug_level_ = static_cast<int>(rail_camera_detection_param.debug_level());


  std::string format_str = R"(
      FusionCameraDetectionComponent InitConfig success
      camera_names:    %s
      input_camera_channel_names:     %s
      camera_obstacle_perception_conf_dir:    %s
      camera_obstacle_perception_conf_file:    %s
      frame_capacity:    %d
      image_channel_num:    %d
      enable_visualization:    %d
      output_camera_channel_name:    %s
      visual_debug_folder_:     %s
      visual_camera_:     %s)";
  std::string config_info_str =
      str(boost::format(format_str.c_str()) % camera_names_[0] %
          input_camera_channel_names_[0] %
          camera_perception_init_options_.root_dir %
          camera_perception_init_options_.conf_file % 
          frame_capacity_ %
          image_channel_num_  % 
          enable_visualization_ %
          output_camera_channel_name_ %
          visual_debug_folder_ %
          visual_camera_ );
  AINFO << config_info_str;

  return cyber::SUCC;
}

int WearComponent::InitSensorInfo()
{
  if (camera_names_.size() != FLAGS_wear_camera_size)
  {
    AERROR << "invalid camera_names_.size(): " << camera_names_.size();
    return cyber::FAIL;
  }

  auto *sensor_manager = apollo::perception::common::SensorManager::Instance();
  for (size_t i = 0; i < camera_names_.size(); ++i)
  {
    if (!sensor_manager->IsSensorExist(camera_names_[i]))
    {
      AERROR << ("sensor_name: " + camera_names_[i] + " not exists.");
      return cyber::FAIL;
    }

    apollo::perception::base::SensorInfo sensor_info;
    if (!(sensor_manager->GetSensorInfo(camera_names_[i], &sensor_info)))
    {
      AERROR << "Failed to get sensor info, sensor name: " << camera_names_[i];
      return cyber::FAIL;
    }
    sensor_info_map_[camera_names_[i]] = sensor_info;
  }

  // assume all camera have same image size
  apollo::perception::base::BaseCameraModelPtr camera_model_ptr =
      sensor_manager->GetUndistortCameraModel(camera_names_[0]);
  image_width_ = static_cast<int>(camera_model_ptr->get_width());
  image_height_ = static_cast<int>(camera_model_ptr->get_height());

  std::string format_str = R"(
      camera_names: %s
      image_width: %d
      image_height: %d
      image_channel_num: %d)";
  std::string sensor_info_str =
      str(boost::format(format_str.c_str()) %
         camera_names_[0] %
          image_width_ %
          image_height_ % 
          image_channel_num_);
  AINFO << sensor_info_str;

  return cyber::SUCC;
}

int WearComponent::InitAlgorithmPlugin()
{
  // camera_obstacle_pipeline_.reset(new camera::ObstacleCameraPerception);
  // if (!camera_obstacle_pipeline_->Init(camera_perception_init_options_))
  // {
  //   AERROR << "camera_obstacle_pipeline_->Init() failed";
  //   return cyber::FAIL;
  // }
  // AINFO << "camera_obstacle_pipeline_->Init() succeed";
  return cyber::SUCC;
}

int WearComponent::InitCameraFrames()
{
  if (camera_names_.size() != FLAGS_wear_camera_size)
  {
    AERROR << "invalid camera_names_.size(): " << camera_names_.size();
    return cyber::FAIL;
  }
  // fixed size
  camera_frames_.resize(frame_capacity_);
  if (camera_frames_.empty())
  {
    AERROR << "frame_capacity_ must > 0";
    return cyber::FAIL;
  }

  // init data_providers for each camera
  for (const auto &camera_name : camera_names_)
  {
    camera::DataProvider::InitOptions data_provider_init_options;
    data_provider_init_options.image_height = image_height_;
    data_provider_init_options.image_width = image_width_;
    data_provider_init_options.do_undistortion = enable_undistortion_;
    data_provider_init_options.sensor_name = camera_name;
    int gpu_id = GetGpuId(camera_perception_init_options_);
    if (gpu_id == -1)
    {
      return cyber::FAIL;
    }
    data_provider_init_options.device_id = gpu_id;
    AINFO << "data_provider_init_options.device_id: "
          << data_provider_init_options.device_id
          << " camera_name: " << camera_name;

    std::shared_ptr<camera::DataProvider> data_provider(
        new camera::DataProvider);
    data_provider->Init(data_provider_init_options);
    data_providers_map_[camera_name] = data_provider;
  }

  for (auto &frame : camera_frames_)
  {
    frame.lane_detected_blob.reset(new apollo::perception::base::Blob<float>());
  }

  return cyber::SUCC;
}


int WearComponent::InitCameraListeners()
{
  for (size_t i = 0; i < camera_names_.size(); ++i)
  {
    const std::string &camera_name = camera_names_[i];
    const std::string &channel_name = input_camera_channel_names_[i];
    const std::string &listener_name = camera_name + "_fusion_camera_listener";
    AINFO << "listener name: " << listener_name;

    typedef std::shared_ptr<apollo::drivers::Image> ImageMsgType;
    std::function<void(const ImageMsgType &)> camera_callback =
        std::bind(&WearComponent::OnReceiveImage, this,
                  std::placeholders::_1, camera_name);
    auto camera_reader = node_->CreateReader(channel_name, camera_callback);
  }
  return cyber::SUCC;
}


// int WearDetectionComponent::InternalProc(
//     const std::shared_ptr<apollo::drivers::Image const> &in_message,
//     const std::string &camera_name, apollo::common::ErrorCode *error_code,
//     SensorFrameMessage *prefused_message,
//     apollo::perception::PerceptionObstacles *out_message)
// {
//   const double msg_timestamp =
//       in_message->measurement_time() + timestamp_offset_;
//   const int frame_size = static_cast<int>(camera_frames_.size());
//   camera::CameraFrame &camera_frame = camera_frames_[frame_id_ % frame_size];

//   prefused_message->timestamp_ = msg_timestamp;
//   prefused_message->seq_num_ = seq_num_;
//   prefused_message->process_stage_ = ProcessStage::MONOCULAR_CAMERA_DETECTION;
//   prefused_message->sensor_id_ = camera_name;
//   prefused_message->frame_ = base::FramePool::Instance().Get();
//   prefused_message->frame_->sensor_info = sensor_info_map_[camera_name];
//   prefused_message->frame_->timestamp = msg_timestamp;


//   prefused_message->frame_->sensor2world_pose = camera2world_trans;
//   // Fill camera frame
//   // frame_size != 0, see InitCameraFrames()
//   camera_frame.camera2world_pose = camera2world_trans;
//   camera_frame.data_provider = data_providers_map_[camera_name].get();
//   camera_frame.data_provider->FillImageData(
//       image_height_, image_width_,
//       reinterpret_cast<const uint8_t *>(in_message->data().data()),
//       in_message->encoding());

//   camera_frame.frame_id = frame_id_;
//   camera_frame.timestamp = msg_timestamp;
//   // get narrow to short projection matrix
//   if (camera_frame.data_provider->sensor_name() == camera_names_[1])
//   {
//     camera_frame.project_matrix = project_matrix_;
//   }
//   else
//   {
//     camera_frame.project_matrix.setIdentity();
//   }

//   ++frame_id_;
//   // Run camera perception pipeline
//   camera_obstacle_pipeline_->GetCalibrationService(
//       &camera_frame.calibration_service);

//   if (!camera_obstacle_pipeline_->Perception(camera_perception_options_,
//                                              &camera_frame))
//   {
//     AERROR << "camera_obstacle_pipeline_->Perception() failed"
//            << " msg_timestamp: " << std::to_string(msg_timestamp);
//     *error_code = apollo::common::ErrorCode::PERCEPTION_ERROR_PROCESS;
//     prefused_message->error_code_ = *error_code;
//     return cyber::FAIL;
//   }
//   AINFO << "##" << camera_name << ": pitch "
//         << camera_frame.calibration_service->QueryPitchAngle()
//         << " | camera_grond_height "
//         << camera_frame.calibration_service->QueryCameraToGroundHeight();
//   prefused_message->frame_->objects = camera_frame.tracked_objects;
//   // TODO(gaohan02, wanji): check the boxes with 0-width in perception-camera
//   prefused_message->frame_->objects.clear();
//   for (auto obj : camera_frame.tracked_objects)
//   {
//     auto &box = obj->camera_supplement.box;
//     if (box.xmin < box.xmax && box.ymin < box.ymax)
//     {
//       prefused_message->frame_->objects.push_back(obj);
//     }
//   }

//   // process success, make pb msg
//   if (output_final_obstacles_ &&
//       MakeProtobufMsg(msg_timestamp, seq_num_, camera_frame.tracked_objects,
//                       camera_frame.lane_objects, *error_code,
//                       out_message) != cyber::SUCC)
//   {
//     AERROR << "MakeProtobufMsg failed"
//            << " ts: " << std::to_string(msg_timestamp);
//     *error_code = apollo::common::ErrorCode::PERCEPTION_ERROR_UNKNOWN;
//     prefused_message->error_code_ = *error_code;
//     return cyber::FAIL;
//   }
//   *error_code = apollo::common::ErrorCode::OK;
//   prefused_message->error_code_ = *error_code;
//   prefused_message->frame_->camera_frame_supplement.on_use = true;
//   if (FLAGS_obs_enable_visualization)
//   {
//     camera::DataProvider::ImageOptions image_options;
//     image_options.target_color = base::Color::RGB;

//     // Use Blob to pass image data
//     prefused_message->frame_->camera_frame_supplement.image_blob.reset(
//         new base::Blob<uint8_t>);
//     camera_frame.data_provider->GetImageBlob(
//         image_options,
//         prefused_message->frame_->camera_frame_supplement.image_blob.get());
//   }


//   // Send msg for visualization
//   if (enable_visualization_)
//   {
//     camera::DataProvider::ImageOptions image_options;
//     image_options.target_color = base::Color::BGR;
//     std::shared_ptr<base::Blob<uint8_t>> image_blob(new base::Blob<uint8_t>);
//     camera_frame.data_provider->GetImageBlob(image_options, image_blob.get());
//     std::shared_ptr<CameraPerceptionVizMessage> viz_msg(
//         new (std::nothrow) CameraPerceptionVizMessage(
//             camera_name, msg_timestamp, camera2world_trans.matrix(), image_blob,
//             camera_frame.tracked_objects, camera_frame.lane_objects,
//             *error_code));
//     bool send_viz_ret = camera_viz_writer_->Write(viz_msg);
//     AINFO << "send out camera visualization msg, ts: "
//           << std::to_string(msg_timestamp) << " send_viz_ret: " << send_viz_ret;

//     // visualize right away
//     if (camera_name == visual_camera_)
//     {
//       cv::Mat output_image(image_height_, image_width_, CV_8UC3,
//                            cv::Scalar(0, 0, 0));
//       base::Image8U out_image(image_height_, image_width_, base::Color::RGB);
//       camera_frame.data_provider->GetImage(image_options, &out_image);
//       memcpy(output_image.data, out_image.cpu_data(),
//              out_image.total() * sizeof(uint8_t));
//       visualize_.ShowResult_all_info_single_camera(
//           output_image, camera_frame, motion_buffer_, world2camera);
//     }
//   }

//   // send out camera debug message
//   if (output_camera_debug_msg_)
//   {
//     // camera debug msg
//     std::shared_ptr<apollo::perception::camera::CameraDebug> camera_debug_msg(
//         new (std::nothrow) apollo::perception::camera::CameraDebug);
//     if (MakeCameraDebugMsg(msg_timestamp, camera_name, camera_frame,
//                            camera_debug_msg.get()) != cyber::SUCC)
//     {
//       AERROR << "make camera_debug_msg failed";
//       return cyber::FAIL;
//     }
//     camera_debug_writer_->Write(camera_debug_msg);
//   }

//   return cyber::SUCC;
// }

// int FusionCameraDetectionComponent::MakeProtobufMsg(
//     double msg_timestamp, int seq_num,
//     const std::vector<base::ObjectPtr> &objects,
//     const std::vector<base::LaneLine> &lane_objects,
//     const apollo::common::ErrorCode error_code,
//     apollo::perception::PerceptionObstacles *obstacles)
// {
//   double publish_time = apollo::common::time::Clock::NowInSeconds();
//   apollo::common::Header *header = obstacles->mutable_header();
//   header->set_timestamp_sec(publish_time);
//   header->set_module_name("perception_camera");
//   header->set_sequence_num(seq_num);
//   // in nanosecond
//   // PnC would use lidar timestamp to predict
//   header->set_lidar_timestamp(static_cast<uint64_t>(msg_timestamp * 1e9));
//   header->set_camera_timestamp(static_cast<uint64_t>(msg_timestamp * 1e9));
//   // header->set_radar_timestamp(0);

//   // write out obstacles in world coordinates
//   obstacles->set_error_code(error_code);
//   for (const auto &obj : objects)
//   {
//     apollo::perception::PerceptionObstacle *obstacle =
//         obstacles->add_perception_obstacle();
//     if (ConvertObjectToPb(obj, obstacle) != cyber::SUCC)
//     {
//       AERROR << "ConvertObjectToPb failed, Object:" << obj->ToString();
//       return cyber::FAIL;
//     }
//   }

//   // write out lanes in ego coordinates
//   apollo::perception::LaneMarkers *lane_markers =
//       obstacles->mutable_lane_marker();
//   apollo::perception::LaneMarker *lane_marker_l0 =
//       lane_markers->mutable_left_lane_marker();
//   apollo::perception::LaneMarker *lane_marker_r0 =
//       lane_markers->mutable_right_lane_marker();
//   apollo::perception::LaneMarker *lane_marker_l1 =
//       lane_markers->add_next_left_lane_marker();
//   apollo::perception::LaneMarker *lane_marker_r1 =
//       lane_markers->add_next_right_lane_marker();

//   for (const auto &lane : lane_objects)
//   {
//     base::LaneLineCubicCurve curve_coord = lane.curve_car_coord;

//     switch (lane.pos_type)
//     {
//     case base::LaneLinePositionType::EGO_LEFT:
//       fill_lane_msg(curve_coord, lane_marker_l0);
//       break;
//     case base::LaneLinePositionType::EGO_RIGHT:
//       fill_lane_msg(curve_coord, lane_marker_r0);
//       break;
//     case base::LaneLinePositionType::ADJACENT_LEFT:
//       fill_lane_msg(curve_coord, lane_marker_l1);
//       break;
//     case base::LaneLinePositionType::ADJACENT_RIGHT:
//       fill_lane_msg(curve_coord, lane_marker_r1);
//       break;
//     default:
//       break;
//     }
//   }

//   return cyber::SUCC;
// }

// int FusionCameraDetectionComponent::MakeCameraDebugMsg(
//     double msg_timestamp, const std::string &camera_name,
//     const camera::CameraFrame &camera_frame,
//     apollo::perception::camera::CameraDebug *camera_debug_msg) {
//   CHECK_NOTNULL(camera_debug_msg);
//   auto itr = std::find(camera_names_.begin(), camera_names_.end(), camera_name);
//   if (itr == camera_names_.end()) {
//     AERROR << "invalid camera_name: " << camera_name;
//     return cyber::FAIL;
//   }
//   int input_camera_channel_names_idx =
//       static_cast<int>(itr - camera_names_.begin());
//   int input_camera_channel_names_size =
//       static_cast<int>(input_camera_channel_names_.size());
//   if (input_camera_channel_names_idx < 0 ||
//       input_camera_channel_names_idx >= input_camera_channel_names_size) {
//     AERROR << "invalid input_camera_channel_names_idx: "
//            << input_camera_channel_names_idx
//            << " input_camera_channel_names_.size(): "
//            << input_camera_channel_names_.size();
//     return cyber::FAIL;
//   }
//   std::string source_channel_name =
//       input_camera_channel_names_[input_camera_channel_names_idx];
//   camera_debug_msg->set_source_topic(source_channel_name);

//   // Fill header info.
//   apollo::common::Header *header = camera_debug_msg->mutable_header();
//   header->set_camera_timestamp(static_cast<uint64_t>(msg_timestamp * 1.0e9));

//   // Fill the tracked objects
//   const std::vector<base::ObjectPtr> &objects = camera_frame.tracked_objects;
//   for (const auto &obj : objects) {
//     apollo::perception::camera::CameraObstacle *camera_obstacle =
//         camera_debug_msg->add_camera_obstacle();
//     ConvertObjectToCameraObstacle(obj, camera_obstacle);
//   }

//   // Fill the laneline objects
//   const std::vector<base::LaneLine> &lane_objects = camera_frame.lane_objects;
//   for (const auto &lane_obj : lane_objects) {
//     apollo::perception::camera::CameraLaneLine *camera_laneline =
//         camera_debug_msg->add_camera_laneline();
//     ConvertLaneToCameraLaneline(lane_obj, camera_laneline);
//   }

//   // Fill the calibrator information(pitch angle)
//   float pitch_angle = camera_frame.calibration_service->QueryPitchAngle();
//   camera_debug_msg->mutable_camera_calibrator()->set_pitch_angle(pitch_angle);
//   return cyber::SUCC;
// }


} // namespace onboard
} // namespace perception
} // namespace apollo
