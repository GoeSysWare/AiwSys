#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "cyber/cyber.h"
#include "cyber/component/component.h"
#include "cyber/common/file.h"
#include "cyber/parameter/parameter_client.h"
#include "cyber/parameter/parameter_server.h"
#include "cyber/common/time_conversion.h"
#include "cyber/time/time.h"

#include "modules/common/time/time.h"
#include "modules/common/time/time_util.h"
#include "modules/drivers/proto/sensor_image.pb.h"
#include "modules/drivers/proto/pointcloud.pb.h"

#include "projects/adas/proto/adas_detection.pb.h"
#include "projects/adas/proto/adas_camera.pb.h"
#include "projects/adas/proto/adas_perception.pb.h"
#include "projects/adas/proto/adas_simulator.pb.h"
#include "projects/adas/proto/adas_config.pb.h"
#include "projects/adas/component/common/threadpool.h"

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



// typedef std::vector<int> point_status_t;
// typedef std::vector<point_status_t> points_status_t;

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
    using OutputWriters = std::map<std::string , std::shared_ptr<apollo::cyber::Writer<watrix::projects::adas::proto::SendResult>> >;
    using DebugWriters = std::map<std::string ,std::shared_ptr<apollo::cyber::Writer<watrix::projects::adas::proto::CameraImages>> >;

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
    //运行模式 
    watrix::projects::adas::proto::ModelType model_type_;
    //是否保存图片结果
    bool if_save_image_result_ = false;
    //是否保存侵界检测后的log结果
    bool if_save_log_result_ = false;

    //保存图片的路径
     std::string save_image_dir_;
     //感知模块的配置
     watrix::projects::adas::proto::PerceptionConfig   adas_perception_param_;
     //算法模块里需要的侵界检测配置
    LaneInvasionConfig lane_invasion_config_;

    std::vector<cv::Mat> v_image_lane_front_result_;
    //接收缓存
    std::vector<cv::Mat> images_;
    //仿真模式时的仿真文件名
    std::vector<std::string> sim_image_files_;

    //侵界检测结果记录的文件名
    std::string result_check_file_;
    std::string result_log_file_;

    //全局的序列号
     uint32_t sequence_num_;

	 apollo::drivers::PointCloud lidar2image_paint_;
	 apollo::drivers::PointCloud lidar_safe_area_;
	std::vector<cv::Point3f> lidar_cloud_buf_;
      std::shared_ptr<apollo::cyber::Node> param_node_ = nullptr;

    //参数服务
    std::shared_ptr<apollo::cyber::ParameterServer> param_server_ = nullptr;
    std::string record_para_name_;
        //相机名称
    std::vector<std::string> camera_names_; // camera sensor names
        //输出发送器
    OutputWriters camera_out_writers_;
    //调试发送器
    DebugWriters camera_debug_writers_;
private:
    //处理线程池
    std::shared_ptr< watrix::projects::adas::ThreadPool> task_processor_;
    //线程池的线程个数
    int  perception_tasks_num_;
    std::mutex camera_mutex_;
    std::mutex lidar_mutex_;

   double last_camera_timestamp_ = 0.0; 

   double last_lidar_timestamp_ = 0.0; 
   double timestamp_offset_ = 0.0;
   double ts_diff_ = 1.0;



    //实际运行时的的输入通道名称
    std::vector<std::string> input_camera_channel_names_;
    //仿真运行时的的输入通道名称
    std::vector<std::string> sim_camera_channel_names_;

    std::vector<std::string> lidar_names_; // lidar sensor names
    std::vector<std::string> input_lidar_channel_names_;

    std::vector<std::string> output_camera_channel_names_;
    std::vector<std::string> debug_camera_channel_names_;





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
