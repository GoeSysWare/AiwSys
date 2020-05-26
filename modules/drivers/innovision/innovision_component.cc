
#include <memory>
#include <string>
#include <thread>

#include "cyber/cyber.h"

#include "modules/common/util/message_util.h"
#include "modules/drivers/innovision/innovision_component.h"

namespace apollo
{
  namespace drivers
  {
    namespace innovision
    {

      bool InnovisionDriverComponent::Init()
      {
        AINFO << "Innovision driver component init";
        proto::Config inno_config_;
        if (!GetProtoConfig(&inno_config_))
        {
          return false;
        }
        AINFO << "Innovision config: " << inno_config_.DebugString();
        // 输出通道
        writer_ = node_->CreateWriter<PointCloud>(inno_config_.convert_channel_name());
        // 启动驱动
        InnoLidarDriver *driver = InnovisionDriverFactory::CreateDriver(inno_config_);
        if (driver == nullptr)
        {
          return false;
        }
        dvr_.reset(driver);
        if(dvr_->Init() ){
          dvr_->SetConnection(std::bind(&InnovisionDriverComponent::device_poll,this,std::placeholders::_1));
        }
        else
        {
          return false;
        }
        
        //启动内存池，为pointclound内置缓存，防止大内存下的反复开辟内存
        point_cloud_pool_.reset(new CCObjectPool<PointCloud>(pool_size_));
        point_cloud_pool_->ConstructAll();
        for (int i = 0; i < pool_size_; i++)
        {
          auto point_cloud = point_cloud_pool_->GetObject();
          if (point_cloud == nullptr)
          {
            AERROR << "fail to getobject, i: " << i;
            return false;
          }
          point_cloud->mutable_point()->Reserve(140000);
        }
        AINFO << "Point cloud comp convert init success";

        return true;
      }

      //发送pointcloud的线程函数
      void InnovisionDriverComponent::device_poll( struct inno_frame * frame)
      {
          static std::atomic<uint64_t> sequence_num = {0};

          std::shared_ptr<PointCloud> point_cloud_out = point_cloud_pool_->GetObject();
          //一般不会为nullptr
          if (point_cloud_out == nullptr)
          {
            AWARN << "poin cloud pool return nullptr, will be create new.";
            point_cloud_out = std::make_shared<PointCloud>();
            point_cloud_out->mutable_point()->Reserve(140000);
          }
          if (point_cloud_out == nullptr)
          {
            AWARN << "point cloud out is nullptr";
            return ;
          }
          point_cloud_out->Clear();
          //转换格式
            double timestamp = apollo::common::time::Clock::NowInSeconds();
          point_cloud_out->mutable_header()->set_module_name("Inno-Lidar");
          point_cloud_out->mutable_header()->set_timestamp_sec(timestamp);
          point_cloud_out->mutable_header()->set_sequence_num(static_cast<unsigned int>(sequence_num.fetch_add(1)));
          point_cloud_out->mutable_header()->set_frame_id(inno_config_.frame_id());
   

            for (unsigned int i = 0; i < frame->points_number; i++) {
              //过滤一些点
                struct inno_point *pt = &frame->points[i];
                if( (pt->x < -0.5) || (pt->x >3) ){
                  //x 高低 -0.5 ---3
                  continue;
                }
                if( pt->z > 80 ){
                  // y左右 -1.5--1 is fit, but here bigger
                  continue;
                }
              apollo::drivers::PointXYZIT* point_new = point_cloud_out->add_point();
              point_new->set_x(pt->x);
              point_new->set_y(pt->y);
              point_new->set_z(pt->z);
              //lidar点的时间戳ms
              point_new->set_timestamp(pt->ts_us);
              point_new->set_intensity(pt->ref);
            }
          if (point_cloud_out == nullptr || point_cloud_out->point_size() == 0)
          {
            AWARN << "point_cloud_out convert is empty.";
            return;
          }
            point_cloud_out->set_width(point_cloud_out->point_size());
	         point_cloud_out->set_height(1);
           point_cloud_out->set_measurement_time(frame->ts_us_start);
           point_cloud_out->set_is_dense(true);

   
          writer_->Write(point_cloud_out);
      }

      } //
    }   // namespace innovision
  }     // namespace drivers
