
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cyber/component/component.h"
#include "modules/drivers/proto/sensor_image.pb.h"
#include "modules/perception/camera/common/camera_frame.h"
#include "projects/conductor_rail/proto/rail_detection.pb.h"
#include "projects/conductor_rail/proto/rail_camera.pb.h"
#include "projects/conductor_rail/detector/interface/base_wear_detector.h"
#include "projects/common/inner_component_messages/inner_component_messages.h"

namespace aiwsys
{
namespace projects
{
namespace adas
{

using apollo::cyber::Component;
using apollo::cyber::Reader;
using apollo::cyber::Writer;
using apollo::drivers::Image;

class AdasLidarComponent : public apollo::cyber::Component<>
{
public:
    AdasLidarComponent() : seq_num_(0) {}
    ~AdasLidarComponent();

    AdasLidarComponent(const AdasLidarComponent &) =
        delete;
    AdasLidarComponent &operator=(
        const AdasLidarComponent &) = delete;

    bool Init() override;

private:
    void OnReceiveImage(const std::shared_ptr<apollo::drivers::Image> &in_message,
                        const std::string &camera_name);
    int InitConfig();
    int InitSensorInfo();
    int InitAlgorithmPlugin();
    int InitCameraFrames();
    int InitCameraListeners();

    //   int InternalProc(
    //       const std::shared_ptr<apollo::drivers::Image const>& in_message,
    //       const std::string& camera_name, apollo::common::ErrorCode* error_code,
    //       SensorFrameMessage* prefused_message,
    //       apollo::perception::PerceptionObstacles* out_message);

private:
    std::mutex mutex_;
    uint32_t seq_num_;

    std::vector<std::shared_ptr<apollo::cyber::Node>> camera_listener_nodes_;

    std::vector<std::string> camera_names_; // camera sensor names
    std::vector<std::string> input_camera_channel_names_;

    // camera name -> SensorInfo
    std::map<std::string, apollo::perception::base::SensorInfo> sensor_info_map_;

    // pre-allocaated-mem data_provider;
    std::map<std::string, std::shared_ptr<apollo::perception::camera::DataProvider>>
        data_providers_map_;

    // camera obstacle pipeline
    WearDetectorInitOptions camera_perception_init_options_;
    WearDetectorOptions camera_perception_options_;

    // fixed size camera frames
    int frame_capacity_ = 20;
    int frame_id_ = 0;
    std::vector< apollo::perception::camera:: CameraFrame> camera_frames_;

    // image info.
    int image_width_ = 1920;
    int image_height_ = 1080;
    int image_channel_num_ = 3;
    int image_data_size_ = -1;

    // options for DataProvider
    bool enable_undistortion_ = false;
    double timestamp_offset_ = 0.0;

    bool enable_visualization_ = false;
    std::string camera_perception_viz_message_channel_name_;

    std::string visual_debug_folder_;
    std::string visual_camera_;

    bool output_final_obstacles_ = false;
    std::string output_camera_channel_name_;

    bool output_camera_debug_msg_ = false;
    std::string camera_debug_channel_name_;

    double pitch_diff_ = 0.0;

    double last_timestamp_ = 0.0;
    double ts_diff_ = 1.0;

    std::shared_ptr<apollo::cyber::Writer<aiwsys::projects::common::ProjectsSensorFrameMessage>>  sensorframe_writer_;

    std::shared_ptr<
        apollo::cyber::Writer<detection::DetectionWearResult>>
        camera_debug_writer_;

    int debug_level_ = 0;
};

CYBER_REGISTER_COMPONENT(AdasLidarComponent);

} // namespace adas
} // namespace projects
} // namespace aiwsys
