#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "cyber/component/component.h"
#include "cyber/common/file.h"
#include "cyber/parameter/parameter_client.h"
#include "cyber/parameter/parameter_server.h"
#include "cyber/base/thread_pool.h"

#include "modules/common/time/time.h"
#include "modules/common/time/time_util.h"

#include "modules/drivers/proto/sensor_image.pb.h"
#include "modules/drivers/proto/pointcloud.pb.h"

#include "projects/adas/proto/adas_detection.pb.h"
#include "projects/adas/proto/adas_camera.pb.h"
#include "projects/adas/proto/adas_perception.pb.h"
#include "projects/adas/proto/adas_simulator.pb.h"
#include "projects/adas/proto/adas_config.pb.h"

#include "projects/adas/configs/config_gflags.h"

//算法库加载
#include "projects/adas/algorithm/algorithm_type.h"
#include "projects/adas/algorithm/algorithm.h"

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // 
#include <opencv2/highgui.hpp> // imwrite
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


#define SYNC_BUF_SIZE 20
#define LIDAR_QUEUE_SIZE 20
typedef std::vector<int> point_status_t;
typedef std::vector<point_status_t> points_status_t;

namespace watrix
{
namespace projects
{
namespace adas
{

using apollo::cyber::Component;
using apollo::cyber::Reader;
using apollo::cyber::Writer;
using apollo::drivers::Image;
using apollo::drivers::PointCloud;
using cv::Mat;


using namespace std;
using namespace cv;
//公共数据结构定义namespace
using namespace watrix::projects::adas::proto;
using namespace watrix::projects::adas;
//算法SDK的namespace
using namespace watrix::algorithm;


class AdasPerceptionComponent : public apollo::cyber::Component<>
{
    using OutputWriters = std::map<std::string , std::shared_ptr<apollo::cyber::Writer<aiwsys::projects::adas::proto::SendResult>> >;
    using DebugWriters = std::map<std::string ,std::shared_ptr<apollo::cyber::Writer<aiwsys::projects::adas::proto::CameraImages>> >;

public:
    AdasPerceptionComponent() ;
    ~AdasPerceptionComponent();

    AdasPerceptionComponent(const AdasPerceptionComponent &) =
        delete;
    AdasPerceptionComponent &operator=(
        const AdasPerceptionComponent &) = delete;

    bool Init() override;

private:
    /**
     * @brief 处理接受到的照片
     * 
     * @param in_message  接受到的照片流
     * @param camera_name   相机名称
     */
    void OnReceiveImage(const std::shared_ptr<apollo::drivers::Image> &in_message,
                        const std::string &camera_name);
    void OnReceivePointCloud(const std::shared_ptr<apollo::drivers::PointCloud> &in_message,
                        const std::string &lidar_name);
    /**
     * @brief  仿真时触发
     * 
     * @param in_message 
     * @param camera_name 
     */
    void OnReceiveSimulatorImage(const std::shared_ptr<SimulatorImage> &in_message,
                        const std::string &camera_name);
    void OnReceiveSimulatorPointCloud(const std::shared_ptr<SimulatorPointCloud> &in_message,
                        const std::string &lidar_name);
    /**
     * @brief 初始化配置
     * 
     * @return bool 
     */
    bool InitConfig();
    /**
     * @brief 初始化算法模块
     * 
     * @return bool 
     */
    bool InitAlgorithmPlugin();
    /**
     * @brief 初始化接收器
     * 
     * @return bool 
     */
    bool InitListeners();
    /**
     * @brief  初始化发送器
     * 
     */
    bool InitWriters();

      int InternalProc(
      const std::shared_ptr<apollo::drivers::Image const>& in_message,
      const std::string& camera_name);

    //这是要传递到task中的参数
public:
    bool if_use_simulator_ = false;
     watrix::projects::adas::proto::PerceptionConfig   adas_perception_param_;
    LaneInvasionConfig lane_invasion_config;
    std::vector<cv::Mat> v_image_lane_front_result;
    //接收缓存
    std::vector<cv::Mat> images_;
    //仿真模式时的仿真文件名
    std::vector<std::string> sim_image_files_;

	 apollo::drivers::PointCloud lidar2image_paint_;
	 apollo::drivers::PointCloud lidar_safe_area_;
	std::vector<cv::Point3f> lidar_cloud_buf_;
private:
    //处理线程池
    std::shared_ptr<apollo::cyber::base::ThreadPool> task_processor_;
    //线程池的线程个数
    int  perception_tasks_num_;
    std::mutex camera_mutex_;
    std::mutex lidar_mutex_;

    uint32_t seq_num_;

   double last_timestamp_ = 0.0; 
   double timestamp_offset_ = 0.0;
   double ts_diff_ = 1.0;


    //相机名称
    std::vector<std::string> camera_names_; // camera sensor names
    //实际运行时的的输入通道名称
    std::vector<std::string> input_camera_channel_names_;
    //仿真运行时的的输入通道名称
    std::vector<std::string> sim_camera_channel_names_;

    std::vector<std::string> lidar_names_; // lidar sensor names
    std::vector<std::string> input_lidar_channel_names_;

    std::vector<std::string> output_camera_channel_names_;
    std::vector<std::string> debug_camera_channel_names_;

    //输出发送器
    OutputWriters camera_out_writers_;
    //调试发送器
    DebugWriters camera_debug_writers_;
    //参数服务
    std::shared_ptr<apollo::cyber::ParameterServer> param_server_ = nullptr;


    //
    cv::Mat a_matrix_;
	cv::Mat r_matrix_;
	cv::Mat t_matrix_;
	cv::Mat mat_rt_;
	cv::Mat camera_matrix_long_;
	cv::Mat camera_distCoeffs_long_;
	cv::Mat camera_matrix_short_;
	cv::Mat camera_distCoeffs_short_;
	std::vector<std::vector<std::pair<int, int>>> distortTable_;

void load_lidar_map_parameter(void);
void load_calibrator_parameter();

void doPerceptionTask();


};

CYBER_REGISTER_COMPONENT(AdasPerceptionComponent);

} // namespace conductor_rail
} // namespace projects
} // namespace aiwsys
