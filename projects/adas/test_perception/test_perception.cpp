
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>  // One-stop header.

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> //
#include <opencv2/highgui.hpp> // imwrite
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include "test_perception.h"
#include "gflags/test_perception_flags.h"
#include "test_uitls.h"
#include  "datetime_util.h"
#include "FindContours_v2.h"


using namespace watrix::algorithm;
using namespace watrix;
using namespace watrix::proto;
using namespace cv;

std::vector<cv::Mat> v_image_lane_front_result;

LaneInvasionConfig lane_invasion_config;

detection_boxs_t yolo_detection_boxs0;
detection_boxs_t yolo_detection_boxs1;
channel_mat_t laneseg_binary_mask0(1);
channel_mat_t laneseg_binary_mask1(1);
blob_channel_mat_t v_instance_mask0(1);
blob_channel_mat_t v_instance_mask1(1);

int seed = 1234;
int gpu_id = 0;
int number_camera_;
int g_index = 0;
FindContours_v2 findContours_v2;
#define RED_POINT Vec3b(0, 0, 255)
#define YELLOW_POINT Vec3b(0, 255, 255)
#define GREEN_POINT Vec3b(0, 255, 0)
#define BLUE_POINT Vec3b(255, 0, 0)

//全绿的框
static void here_draw_detection_boxs(
	const cv::Mat &image,
	const detection_boxs_t &boxs,
	const unsigned int thickness,
	cv::Mat &image_with_boxs)
{
	image_with_boxs = image.clone();
	for (size_t i = 0; i < boxs.size(); i++)
	{
		const detection_box_t detection_box = boxs[i];
		int xmin = detection_box.xmin;
		int xmax = detection_box.xmax;
		int ymin = detection_box.ymin;
		int ymax = detection_box.ymax;
		cv::Rect box(xmin, ymin, xmax - xmin, ymax - ymin);
		// for distance
		bool valid_dist = detection_box.valid_dist;
		float x = detection_box.dist_x;
		float y = detection_box.dist_y;

		//stringstream ss;
		std::string display_text = "";

		display_text = detection_box.class_name + " x=" + watrix::util::DatetimeUtil::GetFloatRound(x, 2) + " y=" + watrix::util::DatetimeUtil::GetFloatRound(y, 2);

		cv::Point2i origin(xmin, ymin - 10);
		cv::putText(image_with_boxs, display_text, origin,
					cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 2);

		cv::rectangle(image_with_boxs, box.tl(), box.br(), COLOR_GREEN, thickness, 8, 0);
	}
}
//分色彩的框
static void here_draw_detection_boxs_ex(
	const cv::Mat &image,
	const detection_boxs_t &boxs,
	box_invasion_results_t box_invasion_cell,
	const unsigned int thickness,
	cv::Mat &image_with_boxs)
{
	image_with_boxs = image.clone();
	for (size_t i = 0; i < boxs.size(); i++)
	{
		const detection_box_t detection_box = boxs[i];
		int xmin = detection_box.xmin;
		int xmax = detection_box.xmax;
		int ymin = detection_box.ymin;
		int ymax = detection_box.ymax;
		cv::Rect box(xmin, ymin, xmax - xmin, ymax - ymin);
		// for distance
		bool valid_dist = detection_box.valid_dist;
		float x = detection_box.dist_x;
		float y = detection_box.dist_y;

		//stringstream ss;
		std::string display_text = "";

		display_text = detection_box.class_name + " x=" + watrix::util::DatetimeUtil::GetFloatRound(x, 2) + " y=" + watrix::util::DatetimeUtil::GetFloatRound(y, 2);

		cv::Point2i origin(xmin, ymin - 10);
		cv::putText(image_with_boxs, display_text, origin,
					cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 2);

		int invasion_status = box_invasion_cell[i].invasion_status;

		if (invasion_status == 1)
		{
			//sound_tools_.set_pcm_play(WARRING_SOUND);
			//DisplayWarning();
			rectangle(image_with_boxs, cvPoint(xmin, ymin), cvPoint(xmax, ymax),
					  cvScalar(0, 0, 255), 2, 4, 0); //red
													 //invasion_object++;
		}
		else if (invasion_status == 0)
		{
			rectangle(image_with_boxs, cvPoint(xmin, ymin), cvPoint(xmax, ymax),
					  cvScalar(0, 255, 0), 2, 4, 0); //green
		}
		else if (invasion_status == -1)
		{
			rectangle(image_with_boxs, cvPoint(xmin, ymin), cvPoint(xmax, ymax),
					  cvScalar(0, 255, 255), 2, 4, 0); //yellow
		}
	}
}
static void addTwoImg(cv::Mat src, cv::Mat mask, cv::Mat &output)
{

	float dis = mask.rows - 1;
	float step = 1.5;
	cv::Scalar end(255, 255, 255);
	cv::Scalar start(0, 0, 0);
	end[0] = (float)mask.at<cv::Vec3b>(0, 0)[0];
	end[1] = (float)mask.at<cv::Vec3b>(0, 0)[1];
	end[2] = (float)mask.at<cv::Vec3b>(0, 0)[2];
	float weightB = (end[0] - start[0]) / dis;
	float weightG = (end[1] - start[1]) / dis;
	float weightR = (end[2] - start[2]) / dis;
	float valB = 0;
	float valG = 0;
	float valR = 0;
	for (int i = 0; i < mask.cols; i++)
	{
		for (int j = 0; j < mask.rows; j++)
		{
			valB = mask.at<cv::Vec3b>(j, i)[0] - weightB * j * step;
			valG = mask.at<cv::Vec3b>(j, i)[1] - weightG * j * step;
			valR = mask.at<cv::Vec3b>(j, i)[2] - weightR * j * step;

			mask.at<cv::Vec3b>(j, i)[0] = valB < 0 ? 0 : valB;
			mask.at<cv::Vec3b>(j, i)[1] = valG < 0 ? 0 : valG;
			mask.at<cv::Vec3b>(j, i)[2] = valR < 0 ? 0 : valR;
		}
	}
	cv::addWeighted(mask, 0.4, src, 1, 0, output);
}

Test_Perception::Test_Perception()
{
}

Test_Perception::~Test_Perception()
{

	AERROR << ("Test_Perception::~Test_Perception  2");
}

std::string Test_Perception::Name() const
{

	return "Test_Perception";
}

apollo::common::Status Test_Perception::Init(char *app_name)
{

	// AdapterManager::Init(FLAGS_perception_adapter_filepath);
	InitCyber(app_name);
	InitMediaFiles();
	InitNodeParas();
	CreateNetwork();
	load_lidar_map_parameter();

	init_algorithm_api(this->perception_config);

	load_calibrator_parameter();

	init_finished_ = true;
	return Status::OK();
}

apollo::common::Status Test_Perception::Start()
{
	if (started_)
	{
		return Status::OK();
	}

	started_ = true;
	// server_thread_ = std::thread([this]() {
	// 	this->Run();
	// });
	this->Run();
	return Status::OK();
}

void Test_Perception::Stop()
{
	started_ = false;
	return;
}
uint32_t counter = 0;

void Test_Perception::Run()
{
	watrix::proto::SyncCameraResult unuse_data;
	CaffeApi::set_mode(true, gpu_id, seed);

	while (started_)
	{

		//等到初始化结束了再干活
		if (init_finished_)
		{
			std::string text = std::to_string(counter);
			//线程用来同步读取长短焦图像和雷达信息
			for (auto &files : filesList_)
			{
				counter++;
				boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
				ParseCameraFiles(files[0], files[1]);
				SendCyberImg();
				ParseLidarFiles(files[2]);
				OnPointCloud(this->lidarpoints_.points);

				OnSyncCameraResult(unuse_data);
				boost::posix_time::ptime pt2 = boost::posix_time::microsec_clock::local_time();
				int64_t cost = (pt2 - pt1).total_milliseconds();
				std::cout << "++++++++Test_Perception:Run Cost :" << cost << "ms" << std::endl;
			}
			std::cout << "++++++++Test_Perception:Run Finished :" << counter << std::endl;
			//如果不循环
			if (!is_circled_)
				return;
		}
		std::chrono::milliseconds dura(10);
		std::this_thread::sleep_for(dura);
	}
}
apollo::common::Status Test_Perception::InitCyber(char *app_name)
{
  apollo::cyber::Init(app_name);
  // autocreate talker node
   talker_node_ = apollo::cyber::CreateNode("adas_perception");
  // create talker
   front_6mm_writer_ = talker_node_->CreateWriter<apollo::drivers::Image>("adas/camera/front_6mm/image");
   front_12mm_writer_ = talker_node_->CreateWriter<apollo::drivers::Image>("adas/camera/front_12mm/image");

   //
   front_6mm_writer_result_ = talker_node_->CreateWriter<watrix::proto::SendResult>("adas/camera/result_6mm");
   front_12mm_writer_result_ = talker_node_->CreateWriter<watrix::proto::SendResult>("adas/camera/result_12mm");


   param_server_ = std::make_shared<apollo::cyber::ParameterServer>(talker_node_);

	apollo::cyber::Parameter  record_parameter("is_record",false);

    param_server_->SetParameter(record_parameter);

	return Status::OK();

}
void Test_Perception::SendCyberImg()
{
	 //当没有侵界时，不发送
	  	apollo::cyber::Parameter  parameter;
  		param_server_->GetParameter("is_record", &parameter);

  		// if(parameter.AsBool()== false) return ;

	    auto pb_img_short= std::make_shared<apollo::drivers::Image >();
	    auto pb_img_long= std::make_shared<apollo::drivers::Image >();

    	pb_img_short->mutable_header()->set_frame_id(std::to_string(counter));
 		pb_img_short->set_width(1920);
    	pb_img_short->set_height(1080);
	   pb_img_short->mutable_data()->reserve(1920*1080*	imgShort_.img.channels());
       pb_img_short->set_encoding("bgr8");
	    pb_img_short->set_step(3* imgShort_.img.cols);
       pb_img_short->mutable_header()->set_timestamp_sec(apollo::cyber::Time::Now().ToSecond());
       pb_img_short->set_measurement_time(apollo::cyber::Time::Now().ToSecond());
      pb_img_short->set_data(	imgShort_.img.data, 1920*1080*	imgShort_.img.channels());
      front_6mm_writer_->Write(pb_img_short);

    	pb_img_long->mutable_header()->set_frame_id(std::to_string(counter));
 		pb_img_long->set_width(1920);
    	pb_img_long->set_height(1080);
	   pb_img_long->mutable_data()->reserve(1920*1080*	imgLong_.img.channels());
       pb_img_long->set_encoding("bgr8");
	   	pb_img_long->set_step(3 * imgLong_.img.cols);
       pb_img_long->mutable_header()->set_timestamp_sec(apollo::cyber::Time::Now().ToSecond());
       pb_img_long->set_measurement_time(apollo::cyber::Time::Now().ToSecond());
      pb_img_long->set_data(	imgLong_.img.data, 1920*1080*	imgLong_.img.channels());
      front_12mm_writer_->Write(pb_img_long);



 }

apollo::common::Status Test_Perception::InitMediaFiles()
{
	bool load_success = apollo::cyber::common::GetProtoFromFile(FLAGS_test_config_filepath, &test_config);
	if (!load_success)
	{
		return Status(CONTROL_INIT_ERROR, "fail load camera config file ");
	}
	save_dir_ = test_config.test_save_dir();
	sdk_version_ = test_config.sdk_version();
	lidar_version_ = test_config.lidar_version();

	filesList_ = Test_ParseFiles(test_config.test_config_filename(), test_config.test_file_dir());
	std::cout << "Read config images & lidar files array:" << filesList_.size() << std::endl;
	std::cout << "sdk_version:" << sdk_version_ << std::endl;
	std::cout << "lidar_version:" << lidar_version_<< std::endl;
	alarm_mask_ = cv::imread("warn.png");

	return Status::OK();
}

apollo::common::Status Test_Perception::InitNodeParas()
{

	bool load_success = apollo::cyber::common::GetProtoFromFile(FLAGS_node_config_filepath, &node_config); // Load config
	if (!load_success)
	{
		return Status(CONTROL_INIT_ERROR, "fail load camera config file ");
	}
	InitConfig(node_config);
	return Status::OK();
}

void Test_Perception::InitConfig(watrix::proto::NodeConfig &node_config)
{
	perception_config = node_config.perception();
	number_camera_ = node_config.sync().camera_count();

	if_use_detect_model_ = perception_config.if_use_detect_model();
	if_use_train_seg_model_ = perception_config.if_use_train_seg_model();
	if_use_lane_seg_model_ = perception_config.if_use_lane_seg_model();
	save_image_result_ = perception_config.save_image_result();
	lidar_queue_stamp_ = perception_config.lidar_queue_stamp();

	lane_invasion_config.use_tr34 = perception_config.laneinvasion().use_tr34(); //use_tr34;// true/false
		// cluster params
	dpoints_t tr33;
	dpoints_t tr34_long_b;
	dpoints_t tr34_short_b;
	if (lane_invasion_config.use_tr34)
	{
		tr34_long_b.push_back(dpoint_t{perception_config.laneinvasion().tr34_long_b().dpoints_1(), perception_config.laneinvasion().tr34_long_b().dpoints_2(), perception_config.laneinvasion().tr34_long_b().dpoints_3(), perception_config.laneinvasion().tr34_long_b().dpoints_4()});
		tr34_long_b.push_back(dpoint_t{perception_config.laneinvasion().tr34_long_b().dpoints_5(), perception_config.laneinvasion().tr34_long_b().dpoints_6(), perception_config.laneinvasion().tr34_long_b().dpoints_7(), perception_config.laneinvasion().tr34_long_b().dpoints_8()});
		tr34_long_b.push_back(dpoint_t{perception_config.laneinvasion().tr34_long_b().dpoints_9(), perception_config.laneinvasion().tr34_long_b().dpoints_10(), perception_config.laneinvasion().tr34_long_b().dpoints_11(), perception_config.laneinvasion().tr34_long_b().dpoints_12()});

		tr34_short_b.push_back(dpoint_t{perception_config.laneinvasion().tr34_short_b().dpoints_1(), perception_config.laneinvasion().tr34_short_b().dpoints_2(), perception_config.laneinvasion().tr34_short_b().dpoints_3(), perception_config.laneinvasion().tr34_short_b().dpoints_4()});
		tr34_short_b.push_back(dpoint_t{perception_config.laneinvasion().tr34_short_b().dpoints_5(), perception_config.laneinvasion().tr34_short_b().dpoints_6(), perception_config.laneinvasion().tr34_short_b().dpoints_7(), perception_config.laneinvasion().tr34_short_b().dpoints_8()});
		tr34_short_b.push_back(dpoint_t{perception_config.laneinvasion().tr34_short_b().dpoints_9(), perception_config.laneinvasion().tr34_short_b().dpoints_10(), perception_config.laneinvasion().tr34_short_b().dpoints_11(), perception_config.laneinvasion().tr34_short_b().dpoints_12()});
		lane_invasion_config.z_height = 1.0;
	}
	else
	{
		tr33.push_back(dpoint_t{perception_config.laneinvasion().tr33().dpoints_1(), perception_config.laneinvasion().tr33().dpoints_2(), perception_config.laneinvasion().tr33().dpoints_3()});
		tr33.push_back(dpoint_t{perception_config.laneinvasion().tr33().dpoints_4(), perception_config.laneinvasion().tr33().dpoints_5(), perception_config.laneinvasion().tr33().dpoints_6()});
		tr33.push_back(dpoint_t{perception_config.laneinvasion().tr33().dpoints_7(), perception_config.laneinvasion().tr33().dpoints_8(), perception_config.laneinvasion().tr33().dpoints_9()});
		lane_invasion_config.z_height = perception_config.laneinvasion().z_height();
	}
	lane_invasion_config.tr33 = tr33;
	lane_invasion_config.tr34_long_b = tr34_long_b;
	lane_invasion_config.tr34_short_b = tr34_short_b;

	lane_invasion_config.output_dir = perception_config.laneinvasion().output_dir();
	lane_invasion_config.b_save_temp_images = perception_config.laneinvasion().b_save_temp_images(); // save temp image results

	lane_invasion_config.b_draw_lane_surface = perception_config.laneinvasion().b_draw_lane_surface();
	lane_invasion_config.b_draw_boxs = perception_config.laneinvasion().b_draw_boxs();
	lane_invasion_config.b_draw_left_right_lane = perception_config.laneinvasion().b_draw_left_right_lane();
	lane_invasion_config.b_draw_other_lane = perception_config.laneinvasion().b_draw_other_lane();
	lane_invasion_config.b_draw_left_right_fitted_lane = perception_config.laneinvasion().b_draw_left_right_fitted_lane();
	lane_invasion_config.b_draw_other_fitted_lane = perception_config.laneinvasion().b_draw_other_fitted_lane();
	lane_invasion_config.b_draw_expand_left_right_lane = perception_config.laneinvasion().b_draw_expand_left_right_lane();
	lane_invasion_config.b_draw_lane_keypoint = perception_config.laneinvasion().b_draw_lane_keypoint();
	lane_invasion_config.b_draw_safe_area = perception_config.laneinvasion().b_draw_safe_area();
	lane_invasion_config.b_draw_safe_area_corner = perception_config.laneinvasion().b_draw_safe_area_corner();
	lane_invasion_config.b_draw_train_cvpoints = perception_config.laneinvasion().b_draw_train_cvpoints();
	lane_invasion_config.b_draw_stats = perception_config.laneinvasion().b_draw_stats();

	//lane_invasion_config.safe_area_y_step 			= perception_config.laneinvasion().safe_area_y_step();// y step for drawing safe area  >=1
	//lane_invasion_config.safe_area_alpha 			= perception_config.laneinvasion().safe_area_alpha(); // overlay aplp
	lane_invasion_config.grid_size = perception_config.laneinvasion().grid_size();								   // cluster grid related paramsdefault 8
	lane_invasion_config.min_grid_count_in_cluster = perception_config.laneinvasion().min_grid_count_in_cluster(); // if grid_count <=10 then filter out this cluste
	//lane_invasion_config.cluster_type = MLPACK_MEANSHIFT; // // (1 USER_MEANSHIFT,2 MLPACK_MEANSHIFT, 3 MLPACK_DBSCAN)
	lane_invasion_config.cluster_type = perception_config.laneinvasion().cluster_type(); // cluster algorithm params
	lane_invasion_config.user_meanshift_kernel_bandwidth = perception_config.laneinvasion().user_meanshift_kernel_bandwidth();
	lane_invasion_config.user_meanshift_cluster_epsilon = perception_config.laneinvasion().user_meanshift_cluster_epsilon();

	lane_invasion_config.mlpack_meanshift_radius = perception_config.laneinvasion().mlpack_meanshift_radius();
	lane_invasion_config.mlpack_meanshift_max_iterations = perception_config.laneinvasion().mlpack_meanshift_max_iterations(); // max iterations
	lane_invasion_config.mlpack_meanshift_bandwidth = perception_config.laneinvasion().mlpack_meanshift_bandwidth();		   //  0.50, 0.51, 0.52, ...0.

	lane_invasion_config.mlpack_dbscan_cluster_epsilon = perception_config.laneinvasion().mlpack_dbscan_cluster_epsilon(); // not same
	lane_invasion_config.mlpack_dbscan_min_pts = perception_config.laneinvasion().mlpack_dbscan_min_pts();				   // cluster at least >=3 pt

	lane_invasion_config.filter_out_lane_noise = perception_config.laneinvasion().filter_out_lane_noise(); // filter out lane noise
	lane_invasion_config.min_area_threshold = perception_config.laneinvasion().min_area_threshold();	   // min area for filter lane noise
	lane_invasion_config.min_lane_pts = perception_config.laneinvasion().min_lane_pts();				   // at least >=10 points for one lan

	lane_invasion_config.polyfit_order = perception_config.laneinvasion().polyfit_order(); // bottom_y default 4;  value range = 1,2,...9
	lane_invasion_config.reverse_xy = perception_config.laneinvasion().reverse_xy();
	; //
	lane_invasion_config.x_range_min = perception_config.laneinvasion().x_range_min();
	lane_invasion_config.x_range_max = perception_config.laneinvasion().x_range_max();
	lane_invasion_config.y_range_min = perception_config.laneinvasion().y_range_min();
	lane_invasion_config.y_range_max = perception_config.laneinvasion().y_range_max();
	// standard limit params (m)
	lane_invasion_config.railway_standard_width = perception_config.laneinvasion().railway_standard_width();
	lane_invasion_config.railway_half_width = perception_config.laneinvasion().railway_half_width();
	lane_invasion_config.railway_limit_width = perception_config.laneinvasion().railway_limit_width();
	lane_invasion_config.railway_delta_width = perception_config.laneinvasion().railway_delta_width();

	lane_invasion_config.case1_x_threshold = perception_config.laneinvasion().case1_x_threshold();
	lane_invasion_config.case1_y_threshold = perception_config.laneinvasion().case1_y_threshold();

	lane_invasion_config.use_lane_status = perception_config.laneinvasion().use_lane_status();
	lane_invasion_config.use_lidar_pointcloud_smallobj_invasion_detect = perception_config.laneinvasion().use_lidar_pointcloud_smallobj_invasion_detect();
	//mlpack_config_file_ = perception_config.laneinvasion().mlpack_config_file();
	//ClusterEpsilonTimes();
	lane_invasion_save = perception_config.laneinvasion().save_image_result();
	lane_invasion_result_folder_ = save_dir_ + "/data/autotrain/lane_invasion_results/src_input_image/" + watrix::util::DatetimeUtil::GetDateTime() + "/";
}

void Test_Perception::load_lidar_map_parameter(void)
{
	cv::FileStorage fs_lidar(FLAGS_lidar_map_parameter, cv::FileStorage::READ);
	fs_lidar["lidar_a_matrix"] >> a_matrix_;
	AERROR << "\n a_matrix_:" << a_matrix_;

	fs_lidar["lidar_r_matrix"] >> r_matrix_;
	AERROR << "\n r_matrix_:" << r_matrix_;

	fs_lidar["lidar_t_matrix"] >> t_matrix_;
	AERROR << "\n t_matrix_:" << t_matrix_;
}

void Test_Perception::init_algorithm_api(watrix::proto::PerceptionConfig perception_config)
{

	if (perception_config.if_use_detect_model())
	{
		init_yolo_api(perception_config);
		//init_monocular_distance_api();
	}

	if (perception_config.if_use_train_seg_model())
	{
		init_trainseg_api(perception_config);
	}

	if (perception_config.if_use_lane_seg_model())
	{
		init_laneseg_api(perception_config);

	if (v_image_lane_front_result.size() == 0)
	{
		// ?????:0.2?w:272 h:480 c:5
		// cv::Mat image_0(272, 480, CV_32FC(5), Scalar::all(0.2f)); error cn<=4
		cv::Mat image_0 = cv::Mat::zeros(272, 480, CV_32FC(5));
		for (int h = 0; h < 272; h++)
		{
			for (int w = 0; w < 480; w++)
			{
				for (int idx_c = 0; idx_c < 5; idx_c++)
				{
					image_0.at<cv::Vec<float, 5>>(h, w)[idx_c] = 0.2;
				}
			}
		}
		for (int i = 0; i < 2; i++)
		{
			v_image_lane_front_result.push_back(image_0);
		}
	}
		findContours_v2.load_params();
	}
	init_distance_api();
}

void Test_Perception::init_yolo_api(watrix::proto::PerceptionConfig perception_config)
{

	YoloNetConfig cfg;
	cfg.net_count = perception_config.yolo().net().net_count();
	cfg.proto_filepath = perception_config.yolo().net().proto_filepath();
	cfg.weight_filepath = perception_config.yolo().net().weight_filepath();
	cfg.label_filepath = perception_config.yolo().label_filepath();
	cfg.input_size = cv::Size(512, 512);
	float b = perception_config.yolo().mean().b();
	float g = perception_config.yolo().mean().g();
	float r = perception_config.yolo().mean().r();
	cfg.bgr_means = {104, 117, 123}; // 104,117,123 ERROR
	cfg.normalize_value = 1.0;		 //perception_config.yolo().normalize_value();; // 1/255.0
	cfg.confidence_threshold = 0.50; //perception_config.yolo().confidence_threshold(); // for filter out box
	cfg.resize_keep_flag = false;	//perception_config.yolo().resize_keep_flag(); // true, use ResizeKP; false use cv::resize
	YoloApi::Init(cfg);
}

void Test_Perception::init_trainseg_api(watrix::proto::PerceptionConfig perception_config)
{

	int net_count = perception_config.trainseg().net().net_count();
	std::string proto_filepath = perception_config.trainseg().net().proto_filepath();
	std::string weight_filepath = perception_config.trainseg().net().weight_filepath();

	caffe_net_file_t net_params = {proto_filepath, weight_filepath};
	TrainSegApi::init(net_params, net_count);

	float b = perception_config.trainseg().mean().b();
	float g = perception_config.trainseg().mean().g();
	float r = perception_config.trainseg().mean().r();

	// set bgr mean
	std::vector<float> bgr_mean{b, g, r};
	TrainSegApi::set_bgr_mean(bgr_mean);
}

void Test_Perception::init_laneseg_api(watrix::proto::PerceptionConfig perception_config)
{
	std::cout << "init_laneseg_api 1 \n";
	int mode = 2;
	LaneSegApi::set_model_type(mode);
	//if(perception_config.model_type()==LANE_MODEL_TYPE::LANE_MODEL_CAFFE){
	if (mode == LANE_MODEL_TYPE::LANE_MODEL_CAFFE)
	{
		int net_count = perception_config.laneseg().net().net_count();
		std::string proto_filepath = perception_config.laneseg().net().proto_filepath();
		std::string weight_filepath = perception_config.laneseg().net().weight_filepath();

		caffe_net_file_t net_params = {proto_filepath, weight_filepath};
		int feature_dim = perception_config.laneseg().feature_dim(); // 8;//16; // for v1,v2,v3, use 8; for v4, use 16
		LaneSegApi::init(net_params, feature_dim, net_count);
		float b = perception_config.laneseg().mean().b();
		float g = perception_config.laneseg().mean().g();
		float r = perception_config.laneseg().mean().r();
		// set bgr mean
		std::vector<float> bgr_mean{b, g, r};
		LaneSegApi::set_bgr_mean(bgr_mean);
	}
	else if (mode == 2)
	{ //LANE_MODEL_TYPE::LANE_MODEL_PT_SIMPLE

		//std::string model_file =  perception_config.laneseg().net().weight_filepath();
		PtSimpleLaneSegNetParams params;
		params.model_path = perception_config.laneseg().net().weight_filepath();
		params.surface_id = 0;
		params.left_id = 1;
		params.right_id = 2;

		int net_count = 2;
		LaneSegApi::init(params, net_count);
	}
	else if (mode == LANE_MODEL_TYPE::LANE_MODEL_PT_COMPLEX)
	{
		//} else if (perception_config.model_type() == LANE_MODEL_TYPE::LANE_MODEL_PT_COMPLEX) {
		// pt complex model
	}
	std::cout << "init_laneseg_api 2 \n";
}

void Test_Perception::init_distance_api()
{
	std::cout << "init_distance_api 1 \n";

	table_param_t params;
	params.long_a = FLAGS_distance_cfg_long_a;
	params.long_b =FLAGS_distance_cfg_long_b;
	params.short_a = FLAGS_distance_cfg_short_a;
	params.short_b =FLAGS_distance_cfg_short_b;
	MonocularDistanceApi::init(params);

}

void Test_Perception::load_calibrator_parameter(void)
{
	FileStorage fs_short(FLAGS_calibrator_cfg_short, FileStorage::READ);
	fs_short["camera_matrix"] >> camera_matrix_short_;
	fs_short["distortion_coefficients"] >> camera_distCoeffs_short_;
	AERROR << "matrix short:" << camera_matrix_short_ << "\ncoefficients:" << camera_distCoeffs_short_;

	FileStorage fs_long(FLAGS_calibrator_cfg_long, FileStorage::READ);
	fs_long["camera_matrix"] >> camera_matrix_long_;
	fs_long["distortion_coefficients"] >> camera_distCoeffs_long_;
	AERROR << "matrix long:" << camera_matrix_long_ << "\ncoefficients:" << camera_distCoeffs_long_;

	const int size = 1920 * 1080 * 2;
	int *memblock = new int[size];
	boost::filesystem::ifstream file(FLAGS_calibrator_cfg_distortTable, std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open())
	{
		file.seekg(0, std::ios::beg);
		file.read((char *)memblock, sizeof(float) * size);
		file.close();

		std::vector<std::pair<int, int>> temp;
		for (int i = 0; i < size; i += 2)
		{
			cv::Point2f pt;
			pt.x = memblock[i];
			pt.y = memblock[i + 1];
			temp.push_back(std::make_pair(memblock[i], memblock[i + 1]));
			if (temp.size() == 1920)
			{
				distortTable.push_back(temp);
				temp.clear();
			}
		}

		delete[] memblock;
	}
	file.close();
}

void Test_Perception::DoYoloDetectGPU(const std::vector<watrix::TestCameraImage> &test_image, long index, int net_id, int gpu_id, int thread_id)
{

	detect_counter_++;
	//CaffeApi::set_mode(true, gpu_id, seed); // set in worker thread-1, use GPU-0

	std::vector<cv::Mat> image = {test_image[0].img,
								  test_image[1].img};

	std::vector<boost::filesystem::path> files_fullname = {test_image[0].filename,
														   test_image[1].filename};

	yolo_detection_boxs0.clear();
	yolo_detection_boxs1.clear();

	std::vector<detection_boxs_t> v_output;

	boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
	YoloApi::Detect(
		1,
		image,
		v_output);

	boost::posix_time::ptime pt2 = boost::posix_time::microsec_clock::local_time();
	int64_t cost = (pt2 - pt1).total_milliseconds();
	detect_total_cost_ += cost;
	AERROR << index << "------- [detect_cost] = " << detect_total_cost_ / (detect_counter_ * 1.0) << " ms" << std::endl;

	for (int j = 0; j < v_output.size(); j++)
	{

		if (j == 0)
		{
			yolo_detection_boxs0 = v_output[j];
			AINFO << " cam0 yolo detection---" << yolo_detection_boxs0.size();
			// set dist x and y
			for (int i = 0; i < yolo_detection_boxs0.size(); ++i)
			{
				detection_box_t &box = yolo_detection_boxs0[i];
				int cx = (box.xmin + box.xmax) / 2;
				int cy = box.ymax;
				AINFO << "image :" << index << "  [detect] box piexl pose x:" << cx << "  y:" << cy;
				float dist_x, dist_y;
				box.valid_dist = MonocularDistanceApi::get_distance(TABLE_SHORT_A, cy, cx, box.dist_x, box.dist_y);
				if (box.valid_dist)
				{
					yolo_detection_boxs0[i].dist_x = std::atof(watrix::util::DatetimeUtil::GetFloatRound(box.dist_x, 2).c_str());
					yolo_detection_boxs0[i].dist_y = std::atof(watrix::util::DatetimeUtil::GetFloatRound(box.dist_y, 2).c_str());
				}
			}
		}
		else
		{
			yolo_detection_boxs1 = v_output[j];
			AINFO << " cam1 yolo detection---" << yolo_detection_boxs0.size();
			// set dist x and y
			for (int i = 0; i < yolo_detection_boxs1.size(); ++i)
			{
				detection_box_t &box = yolo_detection_boxs1[i];
				int cx = (box.xmin + box.xmax) / 2;
				int cy = box.ymax;
				float dist_x, dist_y;
				box.valid_dist = MonocularDistanceApi::get_distance(TABLE_LONG_A, cy, cx, box.dist_x, box.dist_y);
				if (box.valid_dist)
				{
					yolo_detection_boxs1[i].dist_x = std::atof(watrix::util::DatetimeUtil::GetFloatRound(box.dist_x, 2).c_str());
					yolo_detection_boxs1[i].dist_y = std::atof(watrix::util::DatetimeUtil::GetFloatRound(box.dist_y, 2).c_str());
				}
			}
		}

		detection_boxs_t detection_boxs = v_output[j];
		if (this->save_image_result_)
		{
			std::string result_folder = save_dir_ + "/data/autotrain/det_results/" + std::to_string(j) + "/";
			FilesystemUtil::mkdir(result_folder);
			cv::Mat image_with_boxs;
			std::string image_with_box_path;
			if (j == 0)
			{
				here_draw_detection_boxs(image[j], yolo_detection_boxs0, 5, image_with_boxs);
				image_with_box_path = result_folder + files_fullname[0].stem().string() + "_0_boxs.jpg";
			}
			else
			{
				here_draw_detection_boxs(image[j], yolo_detection_boxs1, 5, image_with_boxs);
				image_with_box_path = result_folder + files_fullname[0].stem().string() + "_1_boxs.jpg";
			}
			cv::imwrite(image_with_box_path, image_with_boxs);
			std::cout << "Saved to " << image_with_box_path << std::endl;
		}
	}
}

void Test_Perception::DoTrainSegGPU(const std::vector<watrix::TestCameraImage> &test_image, long index, int net_id, int gpu_id, int thread_id)
{

	trainseg_counter_++;
	std::vector<cv::Mat> image = {test_image[0].img,
								  test_image[1].img};

	std::vector<boost::filesystem::path> files_fullname = {test_image[0].filename,
														   test_image[1].filename};

	std::string result_folder = save_dir_ + "/data/autotrain/seg_results_small/" + std::to_string(0) + "/";
	FilesystemUtil::mkdir(result_folder);
	result_folder = save_dir_ + "/data/autotrain/seg_results_small/" + std::to_string(1) + "/";
	FilesystemUtil::mkdir(result_folder);
	//	CaffeApi::set_mode(true, gpu_id, seed); // set in worker thread-1, use GPU-0

	//cv::Mat output;
	std::vector<cv::Mat> output;
	boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
	TrainSegApi::train_seg(
		//net_id,
		0, //NET_TRAINSEG_0, // only 1 net
		image,
		output);

	boost::posix_time::ptime pt2 = boost::posix_time::microsec_clock::local_time();
	int64_t cost = (pt2 - pt1).total_milliseconds();

	trainseg_total_cost_ += cost;

	AERROR << index << "------- [trainseg_cost] = " << trainseg_total_cost_ / (trainseg_counter_ * 1.0) << " ms" << std::endl;

	if (this->save_image_result_)
	{
		// get output
		for (int j = 0; j < image.size(); j++)
		{
			cv::Mat binary_mask = output[j];
			cv::Mat image_with_mask = OpencvUtil::merge_mask(image[j], binary_mask, 0, 0, 255);
			std::string binary_mask_path = save_dir_ + "/data/autotrain/seg_results_small/" + files_fullname[j].stem().string() + "_binary_mask.jpg";
			cv::imwrite(binary_mask_path, binary_mask);
			binary_mask_path = save_dir_ + "/data/autotrain/seg_results_small/" + files_fullname[j].stem().string() + "_src.jpg";
			cv::imwrite(binary_mask_path, image[j]);
			std::cout << " Saved to " << binary_mask_path << std::endl;
		}
	}
}

//两个image，0 是短焦，1 是长焦
void Test_Perception::DoLaneSegSeqGPU(const std::vector<watrix::TestCameraImage> &test_image, long index, int net_id, int gpu_id, int thread_id)
{

	laneseg_counter_++;
	std::vector<cv::Mat> v_image = {test_image[0].img,
									test_image[1].img};
	std::vector<boost::filesystem::path> files_fullname = {test_image[0].filename,
														   test_image[1].filename};
	// CaffeApi::set_mode(true, gpu_id, seed); // set in worker thread-1, use GPU-0
	std::vector<cv::Mat> v_binary_mask;
	std::vector<channel_mat_t> v_instance_mask;
	int min_area_threshold = 200; // connect components min area

	if (v_image_lane_front_result.size() == 0)
	{
		// ?????:0.2?w:272 h:480 c:5
		// cv::Mat image_0(272, 480, CV_32FC(5), Scalar::all(0.2f)); error cn<=4
		cv::Mat image_0 = cv::Mat::zeros(272, 480, CV_32FC(5));
		for (int h = 0; h < 272; h++)
		{
			for (int w = 0; w < 480; w++)
			{
				for (int idx_c = 0; idx_c < 5; idx_c++)
				{
					image_0.at<cv::Vec<float, 5>>(h, w)[idx_c] = 0.2;
				}
			}
		}
		for (int i = 0; i < 2; i++)
		{
			v_image_lane_front_result.push_back(image_0);
		}
	}
	boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
	LaneSegApi::lane_seg_sequence(
		//net_id,
		1, //NET_LANESEG_0, // only 1 net
		v_image_lane_front_result,
		v_image,
		min_area_threshold,
		v_binary_mask,
		v_instance_mask);

	boost::posix_time::ptime pt2 = boost::posix_time::microsec_clock::local_time();
	int64_t cost = (pt2 - pt1).total_milliseconds();
	laneseg_total_cost_ += cost;
	AERROR << index << "------- [laneseg_cost] = " << cost << "  average: " << laneseg_total_cost_ / (laneseg_counter_ * 1.0) << " ms" << std::endl;
	if (v_binary_mask.size() > 0)
	{
		v_image_lane_front_result.clear();
	}

	for (int j = 0; j < v_binary_mask.size(); j++)
	{
		if (j == 0)
		{
			laneseg_binary_mask0[0] = v_binary_mask[j];
			v_instance_mask0[0] = v_instance_mask[j];
			v_image_lane_front_result.push_back(v_binary_mask[j]);
		}
		else if (j == 1)
		{
			laneseg_binary_mask1[0] = v_binary_mask[j];
			v_instance_mask1[0] = v_instance_mask[j];
			v_image_lane_front_result.push_back(v_binary_mask[j]);
		}
	}

	if (this->save_image_result_)
	{
		for (int j = 0; j < v_binary_mask.size(); j++)
		{
			std::string result_folder = save_dir_ + "/data/autotrain/lanenet_results/" + std::to_string(j) + "/";
			FilesystemUtil::mkdir(result_folder);
			cv::Mat binary_mask = v_binary_mask[j];
			std::string binary_mask_path = result_folder + files_fullname[j].stem().string() + "_binary_mask.jpg";
			cv::imwrite(binary_mask_path, binary_mask);
			binary_mask_path = result_folder + files_fullname[j].stem().string() + "_src.jpg";
			cv::imwrite(binary_mask_path, v_image[j]);
			std::cout << " Saved to " << binary_mask_path << std::endl;
		}
	}
}

void Test_Perception::DoLaneSegGPU(const std::vector<watrix::TestCameraImage> &test_image, long index, int net_id, int gpu_id, int thread_id)
{

	laneseg_counter_++;

	std::vector<cv::Mat> v_image = {test_image[0].img,
									test_image[1].img};
	std::vector<boost::filesystem::path> files_fullname = {test_image[0].filename,
														   test_image[1].filename};
	//CaffeApi::set_mode(true, gpu_id, seed); // set in worker thread-1, use GPU-0

	std::vector<cv::Mat> v_binary_mask;
	std::vector<channel_mat_t> v_instance_mask;
	int min_area_threshold = 200; // connect components min area

	boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
	LaneSegApi::lane_seg(
		//net_id,
		1, //NET_LANESEG_0, // only 1 net
		v_image,
		min_area_threshold,
		v_binary_mask,
		v_instance_mask);

	boost::posix_time::ptime pt2 = boost::posix_time::microsec_clock::local_time();
	int64_t cost = (pt2 - pt1).total_milliseconds();
	laneseg_total_cost_ += cost;
	AERROR << index << "------- [laneseg_cost] = " << cost << "  average: " << laneseg_total_cost_ / (laneseg_counter_ * 1.0) << " ms" << std::endl;

	for (int j = 0; j < v_binary_mask.size(); j++)
	{
		if (j == 0)
		{
			laneseg_binary_mask0[0] = v_binary_mask[j];
			v_instance_mask0[0] = v_instance_mask[j];
		}
		else if (j == 1)
		{
			laneseg_binary_mask1[0] = v_binary_mask[j];
			v_instance_mask1[0] = v_instance_mask[j];
		}
	}
	if (this->save_image_result_)
	{
		for (int j = 0; j < v_binary_mask.size(); j++)
		{
			std::string result_folder = save_dir_ + "/data/autotrain/lanenet_results/" + std::to_string(j) + "/";
			FilesystemUtil::mkdir(result_folder);
			cv::Mat binary_mask = v_binary_mask[j];
			std::string binary_mask_path = result_folder + files_fullname[j].stem().string() + "_binary_mask.jpg";
			cv::imwrite(binary_mask_path, binary_mask);
			binary_mask_path = result_folder + files_fullname[j].stem().string() + "_src.jpg";
			cv::imwrite(binary_mask_path, v_image[j]);
			std::cout << " Saved to " << binary_mask_path << std::endl;
		}
	}
}

void Test_Perception::SyncPerceptionResult(const std::vector<watrix::TestCameraImage> &test_image, long image_time, int index, int net_id, int gpu_id, int thread_id)
{


	boost::posix_time::ptime pt0 = boost::posix_time::microsec_clock::local_time();

	std::vector<cv::Mat> v_image = {test_image[0].img,
									test_image[1].img};
	std::vector<boost::filesystem::path> files_fullname = {test_image[0].filename,
														   test_image[1].filename};
	boost::posix_time::ptime pt1, pt2, pt3;
	pt1 = boost::posix_time::microsec_clock::local_time();
	std::vector<cv::Mat> v_image_with_color_mask(2);
	std::vector<box_invasion_results_t> v_box_invasion_results(2);
	std::vector<lane_safe_area_corner_t> v_lane_safe_area_corner(2); // 4 cornet point
	std::vector<int> lidar_invasion_status0;						 // status:  -1 UNKNOW, 0 NOT Invasion, 1 Yes Invasion
	std::vector<int> lidar_invasion_status1;
	int lidar2img = -2;
	cvpoints_t lidar_cvpoints;
	cvpoints_t train_cvpoints;
	lidar_cvpoints.clear();
	train_cvpoints.clear();
	std::vector<cvpoints_t> v_trains_cvpoint;
	if (perception_config.if_use_detect_model())
	{
		GetLidarData(image_time, thread_id, lidar2img);
		if (lidar2img >= 0)
		{
			train_cvpoints = GetTrainCVPoints(yolo_detection_boxs0, lidar2img, v_trains_cvpoint);
		}
	}

	bool long_camera_open_status = false;
	bool short_camera_open_status = false;
	std::vector<lidar_invasion_cvbox> long_cv_obstacle_box;  // lidar invasion object cv box
	std::vector<lidar_invasion_cvbox> short_cv_obstacle_box; // lidar invasion object cv box

	v_box_invasion_results[0].clear();
	v_box_invasion_results[1].clear();
	
	for (int v = 0; v < v_image.size(); v++)
	{
		AERROR << "go in lane_invasion_detect " << v;
		cv::Mat origin_image = v_image[v];
		cv::Mat binary_mask;
		channel_mat_t instance_mask;
		detection_boxs_t detection_boxs;
		std::vector<int> lidar_invasion_status;

		int lane_count; // total lane count >=0
		int id_left;	// -1 invalid; >=0 valid
		int id_right;   // -1 invalid; >=0 valid
		if (v == 0)
		{
			binary_mask = laneseg_binary_mask0[0];
			instance_mask = v_instance_mask0[0];
			detection_boxs = yolo_detection_boxs0;

			bool instance_success = LaneSegApi::lane_invasion_detect(
				CAMERA_SHORT,
				origin_image,  // 1080,1920
				binary_mask,   // 256,1024
				instance_mask, // 8,256,1024
				detection_boxs,
				//train_cvpoints, // for train points
				cvpoints_,
				lidar_cloud_buf,
				lane_invasion_config,
				v_image_with_color_mask[v],
				lane_count,
				id_left,
				id_right,
				v_box_invasion_results[v],
				lidar_invasion_status0,
				v_lane_safe_area_corner[v],
				short_camera_open_status,
				short_cv_obstacle_box);
		}
		else
		{
			binary_mask = laneseg_binary_mask1[0];
			instance_mask = v_instance_mask1[0];
			detection_boxs = yolo_detection_boxs1;
			std::vector<cvpoints_t> train_cvpoints; // for train points
			for (int t = 0; t < detection_boxs.size(); t++)
			{
				cvpoints_t t_point;
				train_cvpoints.push_back(t_point);
			}
			std::vector<cv::Point3f> v_lidar_points; // for train points
			cvpoints_.clear();
			// do invasion detect
			bool instance_success = LaneSegApi::lane_invasion_detect(
				CAMERA_LONG,
				origin_image,  // 1080,1920
				binary_mask,   // 256,1024
				instance_mask, // 8,256,1024
				detection_boxs,
				//train_cvpoints, // for train points
				cvpoints_,
				v_lidar_points,
				lane_invasion_config,
				v_image_with_color_mask[v],
				lane_count,
				id_left,
				id_right,
				v_box_invasion_results[v],
				lidar_invasion_status1,
				v_lane_safe_area_corner[v],
				long_camera_open_status,
				long_cv_obstacle_box);
		}

		if (lane_invasion_save)
		{
			FilesystemUtil::mkdir(lane_invasion_result_folder_);
			std::string color_mask_filepath = lane_invasion_result_folder_ + files_fullname[1].stem().string() + "_src.jpg";
			// cv::imwrite(color_mask_filepath, v_image[1]);
		}
	}
	pt2 = boost::posix_time::microsec_clock::local_time();
	int64_t cost = (pt2 - pt1).total_milliseconds();
	AERROR << " [lane_invasion_detect]  times cost :" << cost << "  corner: " << v_lane_safe_area_corner.size();
	//出来的 v_image_with_color_mask v_box_invasion_results v_lane_safe_area_corner
	//进行处理

	float short_safe_x = 0;
	float short_safe_y = 0;
	float long_safe_x = 0;
	float long_safe_y = 0;

	for (int v_d = 0; v_d < v_lane_safe_area_corner.size(); v_d++)
	{
		int cx = (v_lane_safe_area_corner[v_d].left_upper.x + v_lane_safe_area_corner[v_d].right_upper.x) / 2;
		int cy = (v_lane_safe_area_corner[v_d].left_upper.y + v_lane_safe_area_corner[v_d].right_upper.y) / 2;
		std::cout << "left_upper x: " << v_lane_safe_area_corner[v_d].left_upper.x << "   right_upper  x:" << v_lane_safe_area_corner[v_d].left_upper.x << std::endl;
		std::cout << "left_upper y: " << v_lane_safe_area_corner[v_d].left_upper.x << "   right_upper  y:" << v_lane_safe_area_corner[v_d].left_upper.x << std::endl;
		bool get_fg;
		if (v_d == 0)
		{
			//20--80
			GetWorldPlace(cy, cx, short_safe_x, short_safe_y, 1);
		}
		else
		{
			//30--350
			GetWorldPlace(cy, cx, long_safe_x, long_safe_y, 2);
		}
	}

	//这里对短焦的雷达区域进行障碍物，距离计算
	int lidar_obj_distance = 0;
	watrix::proto::PointCloud lidar_object; //限界内的雷达点
	std::vector<int> lidar_point_status_t;
	lidar_point_status_t = lidar_invasion_status0;
	box_invasion_results_t short_box_invasion_status = v_box_invasion_results[0];
	int short_lidar_safe = 0;
	if ((lidar_point_status_t.size() > 0) && (lidar2img >= 0))
	{
		//v_trains_cvpoint
		std::vector<int> v_train_invasion_status;
		//count of train
		int train_points_all_index = 0;
		AERROR << "-----train_invasion_status= " << lidar_invasion_status0.size() << "   there is " << v_trains_cvpoint.size();
		for (int ts = 0; ts < v_trains_cvpoint.size(); ts++)
		{
			std::vector<float> limit_width;
			int train_invasion_tc = 0;
			//points of train
			for (int j = train_points_all_index; j < train_cvpoints.size(); j++)
			{
				if (lidar_point_status_t[j] == 1)
				{
					limit_width.push_back(train_cvpoints[j].x);
					train_invasion_tc++;
				}
			}
			if (train_invasion_tc > 3)
			{
				//invasion
				v_train_invasion_status.push_back(1);
				//std::cout<<"there  "<<ts<<" train is invasion!!"<<std::endl;
			}
			else
			{
				//no
				v_train_invasion_status.push_back(0);
				//std::cout<<"there  "<<ts<<" train is ok!!"<<std::endl;
			}
		}
		//change train status
		detection_boxs_t detection_boxs = yolo_detection_boxs0;

		int train_index = 0;
		for (int box = 0; box < detection_boxs.size(); box++)
		{
			std::string cname = detection_boxs[box].class_name;
			if (std::strcmp(cname.c_str(), "train") == 0)
			{
				if (v_train_invasion_status[train_index])
				{
					short_box_invasion_status[box].invasion_status = 1;
				}
				else
				{
					short_box_invasion_status[box].invasion_status = 0;
				}
			}
		}
	}
	//跟据最远安全距离进行，long,short的帧选择
	//当目标在短焦镜头，同时在远焦镜出现，远焦镜头的最短安全距离是30，可能会大于近焦的安全距离
	int select_cam = 0;
	if (short_camera_open_status)
	{
		if ((0 <= short_safe_y) && (short_safe_y <= 40))
		{
			select_cam = 0;
		}
		else if (long_safe_y > short_safe_y)
		{
			select_cam = 1;
		}
		// // 设置一个预值，长焦大于短焦 10米
		// if((long_safe_y-short_safe_y) > 10){
		// 	select_cam=1;
		// }else{
		// 	select_cam=0;
		// }
	}
	else
	{
		select_cam = 0;
	}
	AERROR << "safe area select at cam  " << select_cam;

	//打包成SendResult
	auto sendData_0= std::make_shared<watrix::proto::SendResult >();
	auto sendData_1= std::make_shared<watrix::proto::SendResult >();

	//向protobuf填入detection_boxs
	watrix::proto::CameraImage *seg_binary_mask = sendData_0->mutable_seg_binary_mask();
	watrix::proto::MaxSafeDistance *max_safe_distance = sendData_1->mutable_max_safe_distance();
	std::vector<int> lidar_point_status;

	//cv::Mat	tmp_mat;
	// 将雷达点画到图片上
	lidar_point_status = lidar_invasion_status0;
	//std::cout<<"------------------"<<lidar_point_status.size()<<std::endl;
	cv::Mat short_mat;
	if ((lidar_point_status.size() > 0) && (lidar2img >= 0))
	{
		//if((lidar2img>=0)){
		AERROR << "reback invasion check points=   " << lidar_point_status.size() << "------::" << std::endl;
		short_mat = v_image_with_color_mask[0];
		std::cout << train_cvpoints.size() << "  paint: " << std::endl;
		for (int j = 0; j < train_cvpoints.size(); j++)
		{

			uint pos_y = (uint)(train_cvpoints[j].y);
			uint pos_x = (uint)(train_cvpoints[j].x);
			//cout<<"----"<<pos_x<<"  "<<pos_y<<endl;
			//PainPoint2Image(short_mat, pos_x, pos_y, GREEN_POINT);

			if (lidar_point_status[j] == 1)
			{
				//从3D-2D再从新计算一边x,y
				PainPoint2Image(short_mat, pos_x, pos_y, RED_POINT);
			}
			else if (lidar_point_status[j] == 0)
			{
				PainPoint2Image(short_mat, pos_x, pos_y, GREEN_POINT);
			}
			else if (lidar_point_status[j] == -1)
			{
				PainPoint2Image(short_mat, pos_x, pos_y, YELLOW_POINT);
			}
		}
	}
	else
	{
		short_mat = v_image_with_color_mask[0];
	}

	if (lane_invasion_save)
	{
		cv::Mat combine;
		cv::Mat s_image_boxs;
		cv::Point2i origin(900, 50);
		std::string display_text;
		int if_combine = 0;
		if (v_image_with_color_mask[0].empty())
		{
			std::cout << "cam1 is null" << std::endl;
		}
		else
		{
			display_text = "short safe distance" + std::to_string(short_safe_y);
			here_draw_detection_boxs_ex(v_image_with_color_mask[0], yolo_detection_boxs0, v_box_invasion_results[0], 5, s_image_boxs);
			cv::putText(s_image_boxs, display_text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2);
			if_combine++;
		}

		cv::Mat l_image_boxs;
		if (v_image_with_color_mask[1].empty())
		{
			std::cout << "cam2 is null" << std::endl;
		}
		else
		{
			here_draw_detection_boxs_ex(v_image_with_color_mask[1], yolo_detection_boxs1, v_box_invasion_results[1], 5, l_image_boxs);
			display_text = "long safe distance" + std::to_string(long_safe_y);
			cv::putText(l_image_boxs, display_text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2);
			if_combine++;
		}

		if (if_combine == 2)
		{
			cv::vconcat(l_image_boxs, s_image_boxs, combine);
			std::string save_filepath = lane_invasion_result_folder_ + files_fullname[1].stem().string() + " _combine.jpg";
			cv::imwrite(save_filepath, combine);
		}
	}

	//向protobuf填入source_image
	watrix::proto::CameraImage *source_image_0 = sendData_0->mutable_source_image();
	watrix::proto::DetectionBoxs *pb_mutable_detection_boxs_0 = sendData_0->mutable_detection_boxs();
	watrix::proto::CameraImage *source_image_1 = sendData_1->mutable_source_image();
	watrix::proto::DetectionBoxs *pb_mutable_detection_boxs_1 = sendData_1->mutable_detection_boxs();
	//resize to hdmi need screen
	cv::Mat out_mat_0;
	cv::Mat out_mat_1;

		FillDetectBox(v_image_with_color_mask[0], yolo_detection_boxs0, v_box_invasion_results[0], pb_mutable_detection_boxs_0);
		max_safe_distance->set_image_distance(short_safe_y);
		cv::resize(short_mat, out_mat_0, cv::Size(1280, 800));

		//mat image
		FillDetectBox(v_image_with_color_mask[1], yolo_detection_boxs1, v_box_invasion_results[1], pb_mutable_detection_boxs_1);
		max_safe_distance->set_image_distance(long_safe_y);
		//tmp_mat = v_image_with_color_mask[select_cam];
		cv::resize(v_image_with_color_mask[1], out_mat_1, cv::Size(1280, 800));


	//侵界判断
	bool invasion_flag = false;

	// 长较焦相机有一个东西侵界就算侵界
	for (auto &it : v_box_invasion_results[0])
	{
		if (it.invasion_status == YES_INVASION)
		{
			invasion_flag = true;
			break;
		}
	}
	for (auto &it : v_box_invasion_results[1])
	{
		if (it.invasion_status == YES_INVASION)
		{
			invasion_flag = true;
			break;

		}
	}
	//侵界时需要存视频
	if (invasion_flag)
	{
		AERROR << "adas_perception"<< std::endl;
		apollo::cyber::Parameter  parameter;
  		param_server_->GetParameter("is_record", &parameter);
  		if(parameter.AsBool()== false) 
		  {
			apollo::cyber::Parameter  record_parameter("is_record",true);
    		param_server_->SetParameter(record_parameter);
		  }

	}


	GetCameraImage(out_mat_0, CameraImage::ORIGIN, 0, source_image_0);
	source_image_0->set_timestamp_msec(image_time);
	source_image_0->set_frame_count(g_index);
	pt3 = boost::posix_time::microsec_clock::local_time();
	int64_t cost1 = (pt3 - pt1).total_milliseconds();
	int64_t cost2 = (pt3 - pt0).total_milliseconds();
	AERROR << "------PublishSendResult  after get obj+lane [" << cost1 << "]  all thread spent [" << cost2;

	front_6mm_writer_result_->Write(sendData_0);
	// OnSendResult(sendData_0);

	GetCameraImage(out_mat_1, CameraImage::ORIGIN, 1, source_image_1);
	source_image_1->set_timestamp_msec(image_time);
	source_image_1->set_frame_count(g_index++);
	pt3 = boost::posix_time::microsec_clock::local_time();
	int64_t cost3 = (pt3 - pt1).total_milliseconds();
	int64_t cost4 = (pt3 - pt0).total_milliseconds();
	AERROR << "------PublishSendResult  after get obj+lane [" << cost3<< "]  all thread spent [" << cost4;
	front_12mm_writer_result_->Write(sendData_1);
	// OnSendResult(sendData_1);
}

void Test_Perception::OnSendResult(const watrix::proto::SendResult &data)
{

	long time_p1 = watrix::util::DatetimeUtil::GetMillisec();
	long image_time = data.source_image().timestamp_msec();
	AINFO << "OnSyncPerceptionResult source_image id= " << data.source_image().camera_id() << "  box===" << data.detection_boxs().boxs_size();
	int size = data.ByteSize();
	char *send_buffer = (char *)malloc(size);
	data.SerializeToArray(send_buffer, size);
	DoSendoutThread(send_buffer, size);

	free(send_buffer);
}

//这个在测试里不用回调，已经是主动调用了，为了保持原理一致，名字不改。SyncCameraResult参数没用上
void Test_Perception::OnSyncCameraResult(const watrix::proto::SyncCameraResult &data)
{
	//无效参数，纯粹了接口兼容
	int unuse_para = -1;

	std::vector<TestCameraImage> v_image(2);
	std::vector<TestCameraImage> v_image2(2);
	imgShort_.img.copyTo(v_image[0].img);
	imgShort_.img.copyTo(v_image2[0].img);
	imgLong_.img.copyTo(v_image[1].img);
	imgLong_.img.copyTo(v_image2[1].img);
	v_image[0].filename = imgShort_.filename;
	v_image2[0].filename = imgShort_.filename;
	v_image[1].filename = imgLong_.filename;
	v_image2[1].filename = imgLong_.filename;
	if (this->if_use_lane_seg_model_)
	{
		DoLaneSegSeqGPU(v_image, unuse_para, unuse_para, gpu_id, unuse_para);
	}
	if (this->if_use_detect_model_)
	{
		DoYoloDetectGPU(v_image2, unuse_para, unuse_para, gpu_id, unuse_para);
	}
	if (this->if_use_train_seg_model_)
	{
		DoTrainSegGPU(v_image, unuse_para, unuse_para, gpu_id, unuse_para);
	}
	if (this->if_use_lane_seg_model_ && this->if_use_detect_model_)
	{
		SyncPerceptionResult(v_image, unuse_para, unuse_para, unuse_para, gpu_id, unuse_para);
	}
}
void Test_Perception::OnPointCloud(const watrix::proto::PointCloud &data)
{
	int image_height = 1080;
	int image_width = 1920;
	//int * imagexy= (int *)malloc(1080*1920*sizeof(int));
	int imagexy_check[1080][1920] = {0, 0};
	int effect_point = 0;
	boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();

	lidar2image_check_.clear_points();
	lidar2image_check_.set_timestamp_msec(data.timestamp_msec());

	lidar2image_paint_.clear_points();
	lidar2image_paint_.set_timestamp_msec(data.timestamp_msec());

	lidar_safe_area_.clear_points();
	lidar_safe_area_.set_timestamp_msec(data.timestamp_msec());

	lidar_cloud_buf.clear();
	int p1_ep = 0;
	int p2_ep = 0;

	findContours_v2.OnPointCloud(data, lidar2image_paint_, lidar_safe_area_, lidar_cloud_buf, effect_point);
}
//简化运行态的处理过程，测试时只需要一个格式转化
cvpoints_t Test_Perception::GetLidarData(long image_time, int image_index, int &match_index)
{
	int all_points = lidarpoints_.points.points_size();
	int sel_points = (all_points / 3) + 1;
	cvpoints_.resize(all_points, cv::Point2i(0, 0));
	int point_i = 0;
	for (int i = 0; i < all_points; i++)
	{
		cvpoints_[i].x = lidarpoints_.points.points(i).x();
		cvpoints_[i].y = lidarpoints_.points.points(i).y();
	}
	AERROR << " lidar index: "
				 << "  size:  " << all_points;
	return cvpoints_;
}

void Test_Perception::GetWorldPlace(int cy, int cx, float &safe_x, float &safe_y, int whice_table)
{
	bool get_fg;
	if (whice_table == 1)
	{
		get_fg = MonocularDistanceApi::get_distance(TABLE_SHORT_A, cy, cx, safe_x, safe_y);
	}
	else
	{
		get_fg = MonocularDistanceApi::get_distance(TABLE_LONG_A, cy, cx, safe_x, safe_y);
	}
	if (!get_fg)
	{
		safe_x = 0;
		safe_y = 0;
		AERROR << "cam " << whice_table << " :safe distance unknow---------";
	}

	if (safe_y == 1000)
	{
		safe_y = 0;
		safe_x = 0;
	}
	AERROR << "cam " << whice_table << "   x:" << safe_x << "   y:" << safe_y;
}

void Test_Perception::GetCameraImage(const cv::Mat &image, int image_type, int id, watrix::proto::CameraImage *out)
{
	uint height = image.rows;
	uint width = image.cols;
	uint channel = image.channels();
	uint length = height * width * channel;

	AINFO << "image id= " << id << "  hwc = " << height << "," << width << "," << channel << ",length=" << length;

	out->set_camera_id(id);
	out->set_height(height);
	out->set_width(width);
	out->set_channel(channel);
	out->set_type((CameraImage::ImageType)image_type);
	void *pData = (void *)image.ptr();
	out->set_data(pData, length);
}

void Test_Perception::FillDetectBox(const cv::Mat &input_mat, detection_boxs_t &detection_boxs, box_invasion_results_t box_invasion_cell, watrix::proto::DetectionBoxs *pb_mutable_detection_boxs)
{
	int train_count = 0;
	for (int i = 0; i < detection_boxs.size(); ++i)
	{
		detection_box_t &box = detection_boxs[i];
		// create pb box
		watrix::proto::DetectionBox *pb_box = pb_mutable_detection_boxs->add_boxs();
		pb_box->set_xmin(box.xmin);
		pb_box->set_ymin(box.ymin);
		pb_box->set_xmax(box.xmax);
		pb_box->set_ymax(box.ymax);
		pb_box->set_confidence(box.confidence);
		pb_box->set_class_index(box.class_index);
		pb_box->set_class_name(box.class_name);
		// status:  -1 UNKNOW, 0 NOT Invasion, 1 Yes Invasion
		pb_box->set_invasion_status(box_invasion_cell[i].invasion_status);
		pb_box->set_invasion_distance(box_invasion_cell[i].invasion_distance);
		// pb_box->set_invasion_distance(100);
		// pb_box->set_invasion_status(-1);
		// for image distance
		watrix::proto::Point *distance = pb_box->mutable_distance();
		distance->set_x(box.dist_x);
		distance->set_y(box.dist_y);
		distance->set_z(0);

		if (1)
		{
			int x_start = box.xmin;
			int x_end = box.xmax;
			int y_start = box.ymin;
			int y_end = box.ymax;
			int centre_point = (x_end - x_start) / 2 + x_start;
			int invasion_status = box_invasion_cell[i].invasion_status;
			//AERROR<<"中心点   x:"<<centre_point<<"   y: "<<y_end;
			std::string pose_text;
			float text_x = box.dist_x;
			float text_y = box.dist_y;

			cv::Point2i center(centre_point, y_end);
			//distance_box_centers.push_back(center);  watrix::util::DatetimeUtil::GetMillisec()
			pose_text = "x:" + watrix::util::DatetimeUtil::GetFloatRound(text_x, 2) + " y:" + watrix::util::DatetimeUtil::GetFloatRound(text_y, 2);
			if (invasion_status == 1)
			{
				//sound_tools_.set_pcm_play(WARRING_SOUND);
				//DisplayWarning();
				rectangle(input_mat, cvPoint(x_start, y_start), cvPoint(x_end, y_end),
						  cvScalar(0, 0, 255), 2, 4, 0); //red
														 //invasion_object++;
			}
			else if (invasion_status == 0)
			{
				rectangle(input_mat, cvPoint(x_start, y_start), cvPoint(x_end, y_end),
						  cvScalar(0, 255, 0), 2, 4, 0); //green
			}
			else if (invasion_status == -1)
			{
				rectangle(input_mat, cvPoint(x_start, y_start), cvPoint(x_end, y_end),
						  cvScalar(0, 255, 255), 2, 4, 0); //yellow
			}

			std::string confidence = watrix::util::DatetimeUtil::GetFloatRound(box.confidence, 2);
			std::string invasion_dis = "100"; //DatetimeUtil::GetFloatRound(box_invasion_cell[i].invasion_distance, 2);

			std::string class_name = box.class_name;
			std::string display_text = class_name + confidence + "  in_dis:" + invasion_dis;

			uint32_t x_top = x_start;
			cv::Point2i origin(x_top, y_start - 10);
			cv::putText(input_mat, display_text, origin,
						cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(0, 255, 255), 2, 8, 0);

			uint32_t y_bottom = y_end + 20;
			if (y_bottom >= 1060)
			{
				y_bottom = 1060;
			}
			cv::Point2i origin1(x_top, y_bottom);
			cv::putText(input_mat, pose_text, origin1,
						cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(0, 255, 255), 2, 8, 0);
		}
	}
}
void Test_Perception::PainPoint2Image(cv::Mat mat, int x, int y, cv::Vec3b p_color)
{
	mat.at<Vec3b>(y - 1, x + 1) = p_color;
	mat.at<Vec3b>(y - 1, x) = p_color;
	mat.at<Vec3b>(y - 1, x - 1) = p_color;
	mat.at<Vec3b>(y, x + 1) = p_color;
	mat.at<Vec3b>(y, x) = p_color;
	mat.at<Vec3b>(y, x - 1) = p_color;
	mat.at<Vec3b>(y + 1, x + 1) = p_color;
	mat.at<Vec3b>(y + 1, x) = p_color;
	mat.at<Vec3b>(y + 1, x - 1) = p_color;
}

///解析雷达文件的
void Test_Perception::ParseLidarFiles(std::string file)
{

	boost::filesystem::ifstream pcd_file;
	lidarpoints_.points.clear_points();
	lidarpoints_.filename = file;
	pcd_file.open(file); //将文件流对象与文件连接起来
	std::string str_line;
	int hang_t = 0;
	if (!pcd_file.is_open())
		return;
	while (getline(pcd_file, str_line))
	{
		if (hang_t++ < 11)
		{
			continue;
		}

		std::vector<std::string> v_str;
		watrix::util::DatetimeUtil::SplitString(str_line, v_str, " ");
		float x;
		float y;
		float z;
		int ix = 0;
		for (int i = 0; i < v_str.size(); i++)
		{
			if (ix == 0)
			{
				x = std::atof(v_str[i].c_str());
			}
			else if (ix == 1)
			{
				y = std::atof(v_str[i].c_str());
			}
			else if (ix == 2)
			{
				z = std::atof(v_str[i].c_str());
			}
			ix++;
			//cout << v[i] << " ";
		}
		LidarPoint *pt = lidarpoints_.points.add_points();
		pt->set_x(x);
		pt->set_y(y);
		pt->set_z(z);
	}
}

watrix::algorithm::cvpoints_t Test_Perception::GetTrainCVPoints(watrix::algorithm::detection_boxs_t &detection_boxs, int queue_index, std::vector<watrix::algorithm::cvpoints_t> &v_trains_cvpoint)
{
	watrix::algorithm::cvpoints_t v_train_cvpoint;

	//std::vector<LidarPoint> train_invasion_points[5];
	int train_invasion_flag[5];
	int train_count = 0;
	float bottom_x, top_x, bottom_y, top_y;
	boost::posix_time::ptime ptr1, ptr2, ptr3;
	ptr1 = boost::posix_time::microsec_clock::local_time();
	//获取train，box，的 vector xyz wayne
	for (int box = 0; box < detection_boxs.size(); box++)
	{
		std::vector<cv::Point2d> train_bottom_points;
		std::string cname = detection_boxs[box].class_name;
		//std::cout<<"class_name:     "<<cname<<std::endl;
		int xmin = 0, xmax = 0, ymin = 0, ymax = 0;
		if (std::strcmp(cname.c_str(), "train") == 0)
		{
			watrix::algorithm::cvpoints_t train_cvpoint_tmp;
			watrix::proto::LidarPoint train_lidar_points;
			xmin = detection_boxs[box].xmin;
			xmax = detection_boxs[box].xmax;
			ymin = detection_boxs[box].ymin;
			ymax = detection_boxs[box].ymax;
			std::vector<LidarPoint> train_points;
			for (int j = 0; j < lidar2image_paint_.points_size(); j++)
			{ //lidar2image_check_
				uint pos_y = (uint)(lidar2image_paint_.points(j).y());
				uint pos_x = (uint)(lidar2image_paint_.points(j).x());
				if ((pos_x > xmin) && (pos_x < xmax))
				{
					if ((pos_y > ymin) && (pos_y < ymax))
					{
						LidarPoint tp;
						tp.set_x(lidar_safe_area_.points(j).x());
						tp.set_y(lidar_safe_area_.points(j).y());
						tp.set_z(lidar_safe_area_.points(j).z());
						train_points.push_back(tp);
					}
				}
			}

			train_bottom_points = findContours_v2.start_contours(train_points);

			if (train_bottom_points.size() < 1)
			{
				AERROR << " get train lidar point too small##################### ";
				return v_train_cvpoint;
			}
			cvpoints_t train_cvpoints;
			for (int p = 0; p < train_bottom_points.size(); p++)
			{
				//在lidar_invasion_points中进行查找
				// LidarPoint train_lidar_point;
				// //float bx = train_bottom_points.points(p).x();
				// train_lidar_point.set_x(train_bottom_points.points(p).x());
				// train_lidar_point.set_y(train_bottom_points.points(p).y());
				// train_lidar_point.set_z(train_bottom_points.points(p).z());

				cvpoint_t train_cvpoint;
				//LidarP2CVp(train_lidar_point, train_cvpoint);
				train_cvpoint.x = (int)train_bottom_points[p].x;
				train_cvpoint.y = (int)train_bottom_points[p].y;
				if (train_cvpoint.x < 2)
					continue;
				//下底边的cvpoint输出进行检测
				v_train_cvpoint.push_back(train_cvpoint);
				train_cvpoint_tmp.push_back(train_cvpoint);
				///AERROR<<" get train  ---";
			}

			v_trains_cvpoint.push_back(train_cvpoint_tmp);
			train_count++;
		}
	}
	ptr2 = boost::posix_time::microsec_clock::local_time();
	int64_t cost1 = (ptr2 - ptr1).total_milliseconds();
	AERROR << "------  FindContours::start_contours cost :" << cost1;

	return v_train_cvpoint;
}

void Test_Perception::ParseCameraFiles(std::string file_short, std::string file_long)
{
	imgShort_.img = cv::imread(file_short); // bgr, 0-255
	imgShort_.filename = file_short;
	imgLong_.img = cv::imread(file_long); // bgr, 0-255
	imgLong_.filename = file_long;
}

void Test_Perception::CreateNetwork(void)
{

	// networkTransfer_ = new watrix::network::NetworkTransfer();
	// net_connect_flag_ = networkTransfer_->Connect(node_config.network().ipconfig().ip(), node_config.network().ipconfig().port());
	// if (!net_connect_flag_)
	// {
	// 	AERROR << "cant connect to server Please Open UI first !!";
	// }
	// else
	// {
	// 	AERROR << "connect to server success!";
	// }
}

void Test_Perception::DoSendoutThread(char *buffer, int size)
{
	// long camera_current_timestamp_ = 0;
	// try
	// {
	// 	PACKET_TYPE ptype = PACKET_TYPE_(SEND_RESULT);

	// 	if (net_connect_flag_)
	// 	{
	// 		int rsize = networkTransfer_->SendPacket(ptype, camera_current_timestamp_, buffer, size, 0, 0);
	// 		if (rsize <= 0)
	// 		{
	// 			AERROR << "client_sender send DoSendoutThread fail!!!!!";
	// 		}
	// 	}
	// 	else
	// 	{
	// 		AERROR << "client_sender send DoSendoutThread fail  Open UI first!!!!!";
	// 	}
	// }
	// catch (std::out_of_range &exc)
	// {
	// 	AERROR << "client_sender send DoSendoutThread error error!" << exc.what();
	// }
}