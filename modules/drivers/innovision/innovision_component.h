
#pragma once

#include <memory>
#include <string>
#include <thread>

#include "cyber/cyber.h"
#include "cyber/base/concurrent_object_pool.h"


#include "modules/drivers/innovision/inno_lidar.h"
#include "modules/drivers/innovision/proto/config.pb.h"
#include "modules/drivers/innovision/proto/inno_lidar.pb.h"
#include "modules/drivers/proto/pointcloud.pb.h"


namespace apollo {
namespace drivers {
namespace innovision {

using apollo::cyber::Component;
using apollo::cyber::Reader;
using apollo::cyber::Writer;
using apollo::cyber::base::CCObjectPool;

using apollo::drivers::innovision::InnoLidarDriver;
using apollo::drivers::PointCloud;

/**
 * @brief  Lidar 驱动组件
 * 
 * 一个驱动组件只运行一个Lidar，如果需要多个Lidar之类，就在dag文件中配置
 * 
 */
class InnovisionDriverComponent : public Component<> {
 public:
  ~InnovisionDriverComponent() {
    if (device_thread_->joinable()) {
      device_thread_->join();
    }
  }
  bool Init() override;

 private:
  proto::Config inno_config_;

  volatile bool runing_;  ///< device thread is running
  uint32_t seq_ = 0;
  std::shared_ptr<std::thread> device_thread_;
  std::shared_ptr<InnoLidarDriver> dvr_;  ///< driver implementation class
  std::shared_ptr<Writer<PointCloud>> writer_;
  std::shared_ptr<CCObjectPool<PointCloud>> point_cloud_pool_ = nullptr;
  int pool_size_ = 8;
  void device_poll( struct inno_frame * frame);

};

CYBER_REGISTER_COMPONENT(InnovisionDriverComponent)

}  // namespace velodyne
}  // namespace drivers
}  // namespace apollo
