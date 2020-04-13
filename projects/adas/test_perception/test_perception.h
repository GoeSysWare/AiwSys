
#pragma once
#include <thread>


#include "cyber/cyber.h"
#include "cyber/parameter/parameter_client.h"
#include "cyber/parameter/parameter_server.h"

#include "modules/common/status/status.h"
#include "modules/drivers/proto/sensor_image.pb.h"
// protobuf message to publish and subscribe
#include "projects/adas/proto/point_cloud.pb.h"
#include "projects/adas/proto/camera_image.pb.h"
#include "projects/adas/proto/test_config.pb.h" // NodeConfig
#include "projects/adas/proto/node_config.pb.h" // NodeConfig
#include "projects/adas/algorithm/algorithm_type.h"
#include "projects/adas/algorithm/algorithm.h"




#define USE_GPU

namespace watrix
{


using apollo::cyber::Node;
using namespace apollo::common;

//为了与运行态的定义格式兼容，不修改运行态定义
// 附加了线下测试的文件名
typedef struct __test_image
{
	// watrix::proto::CameraImages img;
	cv::Mat img;
	std::string filename;
} TestCameraImage;
typedef struct __test_pointcloud
{
	watrix::proto::PointCloud points;
	std::string filename;
} TestPointCloud;


//#define USE_GPU
class Test_Perception 
{
public:
	Test_Perception();
	virtual ~Test_Perception();

public:
	virtual std::string Name() const;
	virtual apollo::common::Status Init(char *app_name);
	virtual apollo::common::Status Start();
	virtual void Stop();

protected:
	void Run();

	apollo::common::Status  InitCyber(char *app_name);
std::shared_ptr<Node> talker_node_ = nullptr;
std::shared_ptr<apollo::cyber::Writer<apollo::drivers::Image>> 
    front_6mm_writer_ = nullptr;

std::shared_ptr<apollo::cyber::Writer<apollo::drivers::Image>> 
     front_12mm_writer_ = nullptr ;

	 std::shared_ptr<apollo::cyber::Writer<watrix::proto::SendResult>> 
    front_6mm_writer_result_ = nullptr;

std::shared_ptr<apollo::cyber::Writer<watrix::proto::SendResult>> 
     front_12mm_writer_result_ = nullptr ;

std::shared_ptr<apollo::cyber::ParameterServer> 
param_server_ = nullptr;


	 void SendCyberImg();


private:
	std::thread server_thread_;
	bool started_ = false;
	bool init_finished_ = false;
	bool is_circled_ = false;
	//===================================================================
	// Put your customized code here
	//===================================================================
protected:
	watrix::proto::TestConfig test_config;
	watrix::proto::NodeConfig node_config;

	apollo::common::Status InitMediaFiles();
	apollo::common::Status InitNodeParas();

private:
	void InitConfig(watrix::proto::NodeConfig &node_config);
	void init_algorithm_api(watrix::proto::PerceptionConfig perception_config);
	void init_distance_api();
	void init_trainseg_api(watrix::proto::PerceptionConfig perception_config);
	void init_yolo_api(watrix::proto::PerceptionConfig perception_config);
	void init_laneseg_api(watrix::proto::PerceptionConfig perception_config);
	void load_calibrator_parameter(void);
	void load_lidar_map_parameter(void);

	watrix::proto::PerceptionConfig perception_config;

	bool lane_invasion_save = false;
	std::string lane_invasion_result_folder_;

	bool if_use_detect_model_ = true;
	bool if_use_train_seg_model_ = true;
	bool if_use_lane_seg_model_ = true;
	bool save_image_result_ = false; // save detect results to disk

	int detect_counter_ = 0;
	int detect_total_cost_ = 0;

	int trainseg_counter_ = 0;
	int trainseg_total_cost_ = 0;

	int laneseg_counter_ = 0;
	int laneseg_total_cost_ = 0;

	int lidar_queue_stamp_ = 0;

	 watrix::proto::PointCloud lidar2image_check_;
	watrix::proto::PointCloud lidar2image_paint_;
	watrix::proto::PointCloud lidar_safe_area_;
	std::vector<cv::Point3f> lidar_cloud_buf;

	std::vector<std::vector<std::pair<int, int>>> distortTable;
	cv::Mat a_matrix_;
	cv::Mat r_matrix_;
	cv::Mat t_matrix_;
	cv::Mat mat_rt_;
	cv::Mat camera_matrix_long_;
	cv::Mat camera_distCoeffs_long_;
	cv::Mat camera_matrix_short_;
	cv::Mat camera_distCoeffs_short_;

	cv::Mat alarm_mask_;

	std::vector<std::vector<std::string>> filesList_;
	std::string save_dir_;
	std::string sdk_version_;
	std::string lidar_version_;	
	//线下处理仿真文件只需要以下三个参数，不需要对齐机制和缓冲机制
	TestCameraImage imgShort_;
	TestCameraImage imgLong_;
	TestPointCloud lidarpoints_;
	watrix::algorithm::cvpoints_t  cvpoints_;

	//网络发送
	// watrix::network::NetworkTransfer *networkTransfer_;
	bool net_connect_flag_ = false;

	void OnPointCloud(const watrix::proto::PointCloud& data);
	void OnSyncCameraResult(const watrix::proto::SyncCameraResult &data);
	void SyncPerceptionResult(const std::vector<watrix::TestCameraImage> &test_image, long image_time, int index, int net_id, int gpu_id, int thread_id);

	void DoYoloDetectGPU(const std::vector<watrix::TestCameraImage> &test_image, long index, int net_id, int gpu_id, int thread_id);
	void DoTrainSegGPU(const std::vector<watrix::TestCameraImage> &test_image, long index, int net_id, int gpu_id, int thread_id);
	void DoLaneSegSeqGPU(const std::vector<watrix::TestCameraImage> &test_image, long index, int net_id, int gpu_id, int thread_id);
	void DoLaneSegGPU(const std::vector<watrix::TestCameraImage> &test_image, long index, int net_id, int gpu_id, int thread_id);

	watrix::algorithm::cvpoints_t GetLidarData(long image_time, int image_index, int &match_index);
	void GetWorldPlace(int cy, int cx, float &safe_x, float &safe_y, int whice_table);
	void GetCameraImage(const cv::Mat &image, int image_type, int id, watrix::proto::CameraImage *out);
	void FillDetectBox(const cv::Mat &input_mat, watrix::algorithm::detection_boxs_t &detection_boxs, watrix::algorithm::box_invasion_results_t box_invasion_cell, watrix::proto::DetectionBoxs *pb_mutable_detection_boxs);
	void PainPoint2Image(cv::Mat mat, int x, int y, cv::Vec3b p_color);


	void  ParseLidarFiles(std::string file);
	void  ParseCameraFiles(std::string file_short,std::string file_long);
	watrix::algorithm::cvpoints_t  GetTrainCVPoints(watrix::algorithm::detection_boxs_t & detection_boxs, int queue_index, std::vector<watrix::algorithm::cvpoints_t> &v_trains_cvpoint);

	void CreateNetwork(void);
	void OnSendResult(const watrix::proto::SendResult& data);
	void DoSendoutThread(char* buffer, int size);
};
} // namespace watrix