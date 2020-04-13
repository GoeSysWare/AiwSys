#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cyber/component/component.h"
#include "modules/drivers/proto/sensor_image.pb.h"
#include "modules/drivers/proto/pointcloud.pb.h"
#include "modules/perception/camera/common/camera_frame.h"
#include "projects/adas/proto/adas_detection.pb.h"
#include "projects/adas/proto/adas_camera.pb.h"
#include "projects/adas/proto/node_config.pb.h"
#include "projects/adas/detector/interface/base_adas_detector.h"
#include "projects/common/inner_component_messages/inner_component_messages.h"

#include "projects/adas/algorithm/algorithm_type.h"
#include "projects/adas/algorithm/algorithm.h"

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // 
#include <opencv2/highgui.hpp> // imwrite
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>

#define SYNC_BUF_SIZE 20
#define LIDAR_QUEUE_SIZE 20
typedef std::vector<int> point_status_t;
typedef std::vector<point_status_t> points_status_t;

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
using cv::Mat;

using namespace watrix::algorithm;
using namespace std;
using namespace cv;
using namespace aiwsys::projects::adas::proto;
using namespace aiwsys::projects::adas;

class AdasCameraComponent : public apollo::cyber::Component<>
{
public:
    AdasCameraComponent() : seq_num_(0) {}
    ~AdasCameraComponent();

    AdasCameraComponent(const AdasCameraComponent &) =
        delete;
    AdasCameraComponent &operator=(
        const AdasCameraComponent &) = delete;

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

    AdasDetectorInitOptions camera_perception_init_options_;
    AdasDetectorOptions camera_perception_options_;

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

    std::shared_ptr<apollo::cyber::Writer<aiwsys::projects::adas::proto::SyncPerceptionResult>> 
     front_6mm_writer_;

    std::shared_ptr<apollo::cyber::Writer<aiwsys::projects::adas::proto::SyncPerceptionResult>> 
     front_12mm_writer_;

    std::shared_ptr<
        apollo::cyber::Writer<aiwsys::projects::adas::proto::SyncPerceptionResult>>
        camera_debug_writer_;

    int debug_level_ = 0;



// 旧版保留的成员
		proto::NodeConfig node_config_;
		void InitConfig(proto::NodeConfig& node_config);

		void OnCameraImage(const apollo::drivers::Image& data);
		void OnSyncCameraResult(const proto::SyncCameraResult& data);
		void OnPointCloud(const apollo::drivers::PointCloud& data);
		void OnInt(const int& data);

		void GetCameraImage(const cv::Mat& image, int image_type, int id,  apollo::drivers::Image* out);
		void PublishCameraImage(const cv::Mat& image, int image_type, int id);

		// run in worker thread

		void DoYoloDetectGPU(const std::vector<cv::Mat>& image, long index, int net_id, int gpu_id, int thread_id);
		void DoTrainSegGPU(const std::vector<cv::Mat>& image, long index, int net_id, int gpu_id, int thread_id);
		void DoLaneSegGPU(const std::vector<cv::Mat>& v_image, long index, int net_id, int gpu_id, int thread_id);
		void DoLaneSegSeqGPU(const std::vector<cv::Mat>& v_image, long index, int net_id, int gpu_id, int thread_id);
		void SyncPerceptionResult(const std::vector<cv::Mat>& v_image, long image_time, int index, int net_id, int gpu_id, int thread_id);
		//watrix::util::thread::ThreadPool* thread_pool_;

		boost::mutex yolo_mutex_;
		boost::mutex train_seg_mutex_;
		boost::mutex lane_seg_mutex_;

		proto::PerceptionConfig perception_config_;
		bool lane_invasion_save = false;
		std::string lane_invasion_result_folder_ ;


		int detect_counter_ = 0;
		int detect_total_cost_ = 0;

		int trainseg_counter_ = 0;
		int trainseg_total_cost_ = 0;

		int laneseg_counter_ = 0;
		int laneseg_total_cost_ = 0;

		int lidar_queue_stamp_ = 0;
	
		void load_lidar_map_parameter(void);
		apollo::drivers::PointCloud lidar2image_check_[LIDAR_QUEUE_SIZE];
		apollo::drivers::PointCloud lidar2image_paint_[LIDAR_QUEUE_SIZE];  
		apollo::drivers::PointCloud lidar_safe_area_[LIDAR_QUEUE_SIZE]; 
		std::vector<cv::Point3f> lidar_cloud_buf[LIDAR_QUEUE_SIZE];
		std::vector<std::vector<std::pair<int, int>>> distortTable;
		int lidar_buf_index_=0;

		cv::Mat	 a_matrix_;
		cv::Mat  r_matrix_;
		cv::Mat  t_matrix_;
		cv::Mat  mat_rt_;



		watrix::algorithm::cvpoints_t GetLidarData(long image_time, int image_index, int &match_index);

		watrix::algorithm::cvpoints_t GetTrainCVPoints(watrix::algorithm::detection_boxs_t & detection_boxs, int queue_index, std::vector<watrix::algorithm::cvpoints_t> &v_trains_cvpoint);
		
        std::vector<int> lidar_object_distance(apollo::drivers::PointCloud data);
		
        void get_demo_lidar_points_v1(watrix::algorithm::cvpoints_t& lidar_cvpoints);
		
        void PainPoint2Image(cv::Mat mat, int x, int y, cv::Vec3b p_color);

		void LidarP2CVp(apollo::drivers::PointXYZIT lidar_point, watrix::algorithm::cvpoint_t & cv_point);

		apollo::drivers::PointXYZIT PointCloud2ImagePoint(int imagexy[][1920], double px,  double py, double pz);

		void calibrator_image(int camear_id, cv::Mat &src_image, cv::Mat &out_image);

		cv::Point2f calibrator_lidar_point(cv::Point2f inputPoint, int whice_cam);

		void load_calibrator_parameter(void);

		void FillDetectBox(const cv::Mat &input_mat, watrix::algorithm::detection_boxs_t & detection_boxs, watrix::algorithm::box_invasion_results_t box_invasion_cell, watrix::proto::DetectionBoxs *pb_mutable_detection_boxs);
		
        watrix::algorithm::detection_box_t GetLidarPointBox(apollo::drivers::PointCloud&point_stack);

		void GetWorldPlace(int cy, int cx, float& safe_x, float& safe_y, int whice_table);

		cv::Mat camera_matrix_long_;
		cv::Mat camera_distCoeffs_long_;
		cv::Mat camera_matrix_short_;
		cv::Mat camera_distCoeffs_short_;	

		std::vector<double> mlpack_meanshift_radiuss_;
		std::vector<double> mlpack_meanshift_max_iterationss_;
		std::vector<double> mlpack_meanshift_bandwidths_;
		void ClusterEpsilonTimes(void);
		std::string RemoveTrim(std::string& str);
		std::string mlpack_config_file_;
		long save_frame_index=0;
		long process_frame_counts=0;
		long process_frame_alltime=0;
	


};

CYBER_REGISTER_COMPONENT(AdasCameraComponent);

} // namespace conductor_rail
} // namespace projects
} // namespace aiwsys
