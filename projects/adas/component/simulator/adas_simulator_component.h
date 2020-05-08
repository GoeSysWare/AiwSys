/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include <memory>



#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"
#include "cyber/component/timer_component.h"

#include "modules/drivers/proto/sensor_image.pb.h"
#include "modules/drivers/proto/pointcloud.pb.h"

#include "projects/adas/proto/adas_simulator.pb.h"
#include "projects/adas/proto/adas_config.pb.h"

namespace watrix
{
namespace projects
{
namespace adas
{
using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::TimerComponent;
using apollo::cyber::Writer;
using apollo::drivers::Image;
using apollo::drivers::PointCloud;
using apollo::drivers::PointXYZIT;
using watrix::projects::adas::proto::SimulatorConfig;


class AdasSimulatorComponent : public TimerComponent
{
public:
  bool Init() override;
  bool Proc() override;
//同步计数 每个周期发出去的同一值
  static std::atomic<uint64_t> procs_num_;
//分别计数 每个通道分别计数
  static std::atomic<uint64_t> element_num_;

private:

  bool InitConfig();
  bool InitSimulatorFiles();
  void ParseCameraFiles(std::string file_6mm, std::string file_12mm);
  void ParseLidarFiles(std::string file_lidar);
  void SendSimulator();

  //根据dag文件中配置的pb.txt来填充此配置
  SimulatorConfig adas_simulator_param_;

  std::vector<std::vector<std::string>> filesList_;
  std::vector<std::vector<std::string>>::iterator cur_Iter_;
  bool is_Circle_ = false;
  //仿真文件存放目录
  std::string sim_files_dir_;
  //仿真文件"照片--Lidar"对齐的配置文件
  std::string sim_config_file_;
  int sim_interval_;
  std::shared_ptr<Writer<apollo::drivers::Image >> front_6mm_writer_ = nullptr;
  std::shared_ptr<Writer<apollo::drivers::Image >> front_12mm_writer_ = nullptr;
  std::shared_ptr<Writer<apollo::drivers::PointCloud>> lidar_writer_ = nullptr;
  std::vector<std::string> output_camera_channel_names_;
  std::vector<std::string> output_lidar_channel_names_;

  std::shared_ptr<apollo::drivers::Image>  front_6mm_image_;
  std::shared_ptr<apollo::drivers::Image>  front_12mm_image_;
  std::shared_ptr<apollo::drivers::PointCloud> lidar_pointcloud_;
};
CYBER_REGISTER_COMPONENT(AdasSimulatorComponent)

} // namespace adas
} // namespace projects
} // namespace watrix