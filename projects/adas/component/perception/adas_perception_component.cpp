
#include "projects/adas/component/perception/adas_perception_component.h"

#include <yaml-cpp/yaml.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include "projects/adas/component/common/util.h"
#include "projects/adas/configs/config_gflags.h"
#include "projects/adas/component/perception/FindContours_v2.h"
#include "projects/adas/component/perception/perception_task.h"

namespace watrix
{
  namespace projects
  {
    namespace adas
    {

      using apollo::cyber::common::GetAbsolutePath;

      using namespace std;
      using namespace cv;

      //初始化yolo参数
      void init_yolo_api(watrix::projects::adas::proto::YoloConfig perception_config)
      {
        watrix::algorithm::YoloNetConfig cfg;

        cfg.net_count = perception_config.net().net_count();
        cfg.proto_filepath = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            perception_config.net().proto_filepath());

        cfg.weight_filepath = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            perception_config.net().weight_filepath());

        cfg.label_filepath = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            perception_config.label_filepath());

        cfg.input_size = cv::Size(512, 512);
        cfg.bgr_means = {104, 117, 123}; // 104,117,123 ERROR
        cfg.normalize_value = 1.0;       //perception_config.normalize_value();; // 1/255.0
        cfg.confidence_threshold = 0.50; //perception_config.confidence_threshold(); // for filter out box
        cfg.resize_keep_flag = false;    //perception_config.resize_keep_flag(); // true, use ResizeKP; false use cv::resize

        watrix::algorithm::YoloApi::Init(cfg);
      }

      static  YoloDarknetApi *  init_darknet_api(watrix::projects::adas::proto::DarkNetConfig config)
      {

        DarknetYoloConfig cfg;
        // cfg.batchsize = 2;
        cfg.cfg_filepath = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            config.cfg_filepath());

        cfg.weight_filepath = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            config.weight_filepath());
        cfg.label_filepath = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            config.label_filepath());

        cfg.confidence_threshold = config.confidence_threshold();
        cfg.hier_thresh = config.hier_thresh();
        cfg.iou_thresh = config.iou_thresh();

        YoloDarknetApi*  darknet_ptr= new YoloDarknetApi(cfg);

        return darknet_ptr;
      }

      void init_trainseg_api(watrix::projects::adas::proto::TrainSegConfig perception_config)
      {

        int net_count = perception_config.net().net_count();
        std::string proto_filepath =
            apollo::cyber::common::GetAbsolutePath(
                watrix::projects::adas::GetAdasWorkRoot(), perception_config.net().proto_filepath());
        std::string weight_filepath = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(), perception_config.net().weight_filepath());

        watrix::algorithm::caffe_net_file_t net_params = {proto_filepath, weight_filepath};
        watrix::algorithm::TrainSegApi::init(net_params, net_count);

        float b = perception_config.mean().b();
        float g = perception_config.mean().g();
        float r = perception_config.mean().r();

        // set bgr mean
        std::vector<float> bgr_mean{b, g, r};
        watrix::algorithm::TrainSegApi::set_bgr_mean(bgr_mean);
      }

      void init_laneseg_api(watrix::projects::adas::proto::LaneSegConfig perception_config)
      {
        int mode = 2;
        watrix::algorithm::LaneSegApi::set_model_type(mode);
        //if(perception_config.model_type()==LANE_MODEL_TYPE::LANE_MODEL_CAFFE){
        if (mode == LANE_MODEL_TYPE::LANE_MODEL_CAFFE)
        {
          int net_count = perception_config.net().net_count();
          std::string proto_filepath = apollo::cyber::common::GetAbsolutePath(
              watrix::projects::adas::GetAdasWorkRoot(), perception_config.net().proto_filepath());
          std::string weight_filepath = apollo::cyber::common::GetAbsolutePath(
              watrix::projects::adas::GetAdasWorkRoot(), perception_config.net().weight_filepath());

          watrix::algorithm::caffe_net_file_t net_params = {proto_filepath, weight_filepath};
          int feature_dim = perception_config.feature_dim(); // 8;//16; // for v1,v2,v3, use 8; for v4, use 16
          watrix::algorithm::LaneSegApi::init(net_params, feature_dim, net_count);
          float b = perception_config.mean().b();
          float g = perception_config.mean().g();
          float r = perception_config.mean().r();
          // set bgr mean
          std::vector<float> bgr_mean{b, g, r};
          watrix::algorithm::LaneSegApi::set_bgr_mean(bgr_mean);
        }
        else if (mode == 2)
        {
          watrix::algorithm::PtSimpleLaneSegNetParams params;
          params.model_path = apollo::cyber::common::GetAbsolutePath(
              watrix::projects::adas::GetAdasWorkRoot(), perception_config.net().weight_filepath());
          params.surface_id = 0;
          params.left_id = 1;
          params.right_id = 2;

          int net_count = 2;
          watrix::algorithm::LaneSegApi::init(params, net_count);
        }
        else if (mode == LANE_MODEL_TYPE::LANE_MODEL_PT_COMPLEX)
        {
          //} else if (perception_config.model_type() == LANE_MODEL_TYPE::LANE_MODEL_PT_COMPLEX) {
          // pt complex model
        }
      }

      void init_distance_api()
      {
        table_param_t params;

        params.long_a = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            FLAGS_distance_cfg_long_a);
        params.long_b = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            FLAGS_distance_cfg_long_b);
        params.short_a = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            FLAGS_distance_cfg_short_a);
        params.short_b = apollo::cyber::common::GetAbsolutePath(
            watrix::projects::adas::GetAdasWorkRoot(),
            FLAGS_distance_cfg_short_b);

        MonocularDistanceApi::init(params);
      }

      //这个 unused
      void AdasPerceptionComponent::load_lidar_map_parameter(void)
      {
        cv::FileStorage fs_lidar(
            apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(),
                                                   FLAGS_lidar_map_parameter),
            cv::FileStorage::READ);
        fs_lidar["lidar_a_matrix"] >> a_matrix_;
        AINFO << "\n a_matrix_:" << a_matrix_;

        fs_lidar["lidar_r_matrix"] >> r_matrix_;
        AINFO << "\n r_matrix_:" << r_matrix_;

        fs_lidar["lidar_t_matrix"] >> t_matrix_;
        AINFO << "\n t_matrix_:" << t_matrix_;
      }
      //这个 unused
      void AdasPerceptionComponent::load_calibrator_parameter(void)
      {

        cv::FileStorage fs_short(apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(),
                                                                        FLAGS_calibrator_cfg_short),
                                 cv::FileStorage::READ);
        fs_short["camera_matrix"] >> camera_matrix_short_;
        fs_short["distortion_coefficients"] >> camera_distCoeffs_short_;
        AINFO << "matrix short:" << camera_matrix_short_ << "\ncoefficients:" << camera_distCoeffs_short_;

        cv::FileStorage fs_long(apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(),
                                                                       FLAGS_calibrator_cfg_long),
                                cv::FileStorage::READ);

        fs_long["camera_matrix"] >> camera_matrix_long_;
        fs_long["distortion_coefficients"] >> camera_distCoeffs_long_;
        AINFO << "matrix long:" << camera_matrix_long_ << "\ncoefficients:" << camera_distCoeffs_long_;

        const int size = 1920 * 1080 * 2;
        int *memblock = new int[size];
        std::ifstream file(
            apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(),
                                                   FLAGS_calibrator_cfg_distortTable),
            std::ios::in | std::ios::binary | std::ios::ate);
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
              distortTable_.push_back(temp);
              temp.clear();
            }
          }

          delete[] memblock;
        }
        file.close();
      }
      AdasPerceptionComponent::AdasPerceptionComponent()
      {
      }

      AdasPerceptionComponent::~AdasPerceptionComponent()
      {
        //是否用检测功能，是否采用yolo算法
        if (adas_perception_param_.if_use_detect_model() && adas_perception_param_.has_yolo())
        {
          YoloApi::Free();
        }
        // //检测时是否采用darknet算法
        // if (!adas_perception_param_.has_yolo() && adas_perception_param_.if_use_detect_model() && adas_perception_param_.has_darknet() )
        // {
        //   YoloDarknetApi::Free();
        // }

        //是否用列车分割功能
        if (adas_perception_param_.if_use_train_seg_model() && adas_perception_param_.has_trainseg())
        {
        }
        //是否用轨道分割功能
        if (adas_perception_param_.if_use_lane_seg_model() && adas_perception_param_.has_laneseg())
        {
          LaneSegApi::free();
        }
      }

      bool AdasPerceptionComponent::Init()
      {

        if (!InitConfig())
        {
          AERROR << "InitConfig() failed.";
          return false;
        }

        //初始化线程池
        task_processor_.reset(new watrix::projects::adas::ThreadPool(perception_tasks_num_,
                                                                     [this]() -> void* { 
                                                                       //caffe初始化,一个线程初始化一次
                                                                       watrix::algorithm::CaffeApi::set_mode(true, 0, 1234);
                                                                       YoloDarknetApi * darknet_ptr =0;
                                                                        //检测时是否采用darknet算法,如果有yolo，优先用yolo，屏蔽darknet
                                                                                
                                                                      if (!this->adas_perception_param_.has_yolo() && this->adas_perception_param_.if_use_detect_model() && this->adas_perception_param_.has_darknet() )
                                                                      {
                                                                        darknet_ptr =init_darknet_api(this->adas_perception_param_.darknet());
                                                                      }
                                                                      return darknet_ptr;
                                                                     }));

        //初始化算法SDK
        if (!InitAlgorithmPlugin())
        {
          AERROR << "InitAlgorithmPlugin() failed.";
          return false;
        }
        //初始化接收器
        if (!InitListeners())
        {
          AERROR << "InitCameraListeners() failed.";
          return false;
        }
        //初始化发送器
        if (!InitWriters())
        {
          AERROR << "InitWriters() failed.";
          return false;
        }
        return true;
      }

      bool AdasPerceptionComponent::InitConfig()
      {
        //从模块配置中取得配置信息
        bool ret = GetProtoConfig(&adas_perception_param_);
        if (!ret)
        {
          return false;
        }
        //把proto格式的配置转换为C结构体格式
        //其实不应该把计算参数放在proto里，应该定义一个yaml文件，直接赋值给算法SDK结构体
        lane_invasion_config_.use_tr34 = adas_perception_param_.laneinvasion().use_tr34(); //use_tr34;// true/false
        // cluster params
        dpoints_t tr33;
        dpoints_t tr34_long_b;
        dpoints_t tr34_short_b;
        if (lane_invasion_config_.use_tr34)
        {
          tr34_long_b.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_1(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_2(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_3(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_4()});
          tr34_long_b.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_5(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_6(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_7(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_8()});
          tr34_long_b.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_9(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_10(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_11(),
              adas_perception_param_.laneinvasion().tr34_long_b().dpoints_12()});

          tr34_short_b.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_1(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_2(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_3(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_4()});
          tr34_short_b.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_5(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_6(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_7(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_8()});
          tr34_short_b.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_9(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_10(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_11(),
              adas_perception_param_.laneinvasion().tr34_short_b().dpoints_12()});
          lane_invasion_config_.z_height = 1.0;
        }
        else
        {
          tr33.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr33().dpoints_1(),
              adas_perception_param_.laneinvasion().tr33().dpoints_2(),
              adas_perception_param_.laneinvasion().tr33().dpoints_3()});
          tr33.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr33().dpoints_4(),
              adas_perception_param_.laneinvasion().tr33().dpoints_5(),
              adas_perception_param_.laneinvasion().tr33().dpoints_6()});
          tr33.push_back(dpoint_t{
              adas_perception_param_.laneinvasion().tr33().dpoints_7(),
              adas_perception_param_.laneinvasion().tr33().dpoints_8(),
              adas_perception_param_.laneinvasion().tr33().dpoints_9()});
          lane_invasion_config_.z_height = adas_perception_param_.laneinvasion().z_height();
        }
        lane_invasion_config_.tr33 = tr33;
        lane_invasion_config_.tr34_long_b = tr34_long_b;
        lane_invasion_config_.tr34_short_b = tr34_short_b;
        lane_invasion_config_.output_dir = adas_perception_param_.laneinvasion().output_dir();
        lane_invasion_config_.b_save_temp_images = adas_perception_param_.laneinvasion().b_save_temp_images(); // save temp image results

        lane_invasion_config_.b_draw_lane_surface = adas_perception_param_.laneinvasion().b_draw_lane_surface();
        lane_invasion_config_.b_draw_boxs = adas_perception_param_.laneinvasion().b_draw_boxs();
        lane_invasion_config_.b_draw_left_right_lane = adas_perception_param_.laneinvasion().b_draw_left_right_lane();
        lane_invasion_config_.b_draw_other_lane = adas_perception_param_.laneinvasion().b_draw_other_lane();
        lane_invasion_config_.b_draw_left_right_fitted_lane = adas_perception_param_.laneinvasion().b_draw_left_right_fitted_lane();
        lane_invasion_config_.b_draw_other_fitted_lane = adas_perception_param_.laneinvasion().b_draw_other_fitted_lane();
        lane_invasion_config_.b_draw_expand_left_right_lane = adas_perception_param_.laneinvasion().b_draw_expand_left_right_lane();
        lane_invasion_config_.b_draw_lane_keypoint = adas_perception_param_.laneinvasion().b_draw_lane_keypoint();
        lane_invasion_config_.b_draw_safe_area = adas_perception_param_.laneinvasion().b_draw_safe_area();
        lane_invasion_config_.b_draw_safe_area_corner = adas_perception_param_.laneinvasion().b_draw_safe_area_corner();
        lane_invasion_config_.b_draw_train_cvpoints = adas_perception_param_.laneinvasion().b_draw_train_cvpoints();
        lane_invasion_config_.b_draw_stats = adas_perception_param_.laneinvasion().b_draw_stats();

        //lane_invasion_config_.safe_area_y_step 			= perception_config.laneinvasion().safe_area_y_step();// y step for drawing safe area  >=1
        //lane_invasion_config_.safe_area_alpha 			= perception_config.laneinvasion().safe_area_alpha(); // overlay aplp
        lane_invasion_config_.grid_size = adas_perception_param_.laneinvasion().grid_size();                                 // cluster grid related paramsdefault 8
        lane_invasion_config_.min_grid_count_in_cluster = adas_perception_param_.laneinvasion().min_grid_count_in_cluster(); // if grid_count <=10 then filter out this cluste
        //lane_invasion_config_.cluster_type = MLPACK_MEANSHIFT; // // (1 USER_MEANSHIFT,2 MLPACK_MEANSHIFT, 3 MLPACK_DBSCAN)
        lane_invasion_config_.cluster_type = adas_perception_param_.laneinvasion().cluster_type(); // cluster algorithm params
        lane_invasion_config_.user_meanshift_kernel_bandwidth = adas_perception_param_.laneinvasion().user_meanshift_kernel_bandwidth();
        lane_invasion_config_.user_meanshift_cluster_epsilon = adas_perception_param_.laneinvasion().user_meanshift_cluster_epsilon();

        lane_invasion_config_.mlpack_meanshift_radius = adas_perception_param_.laneinvasion().mlpack_meanshift_radius();
        lane_invasion_config_.mlpack_meanshift_max_iterations = adas_perception_param_.laneinvasion().mlpack_meanshift_max_iterations(); // max iterations
        lane_invasion_config_.mlpack_meanshift_bandwidth = adas_perception_param_.laneinvasion().mlpack_meanshift_bandwidth();           //  0.50, 0.51, 0.52, ...0.

        lane_invasion_config_.mlpack_dbscan_cluster_epsilon = adas_perception_param_.laneinvasion().mlpack_dbscan_cluster_epsilon(); // not same
        lane_invasion_config_.mlpack_dbscan_min_pts = adas_perception_param_.laneinvasion().mlpack_dbscan_min_pts();                 // cluster at least >=3 pt

        lane_invasion_config_.filter_out_lane_noise = adas_perception_param_.laneinvasion().filter_out_lane_noise(); // filter out lane noise
        lane_invasion_config_.min_area_threshold = adas_perception_param_.laneinvasion().min_area_threshold();       // min area for filter lane noise
        lane_invasion_config_.min_lane_pts = adas_perception_param_.laneinvasion().min_lane_pts();                   // at least >=10 points for one lan

        lane_invasion_config_.polyfit_order = adas_perception_param_.laneinvasion().polyfit_order(); // bottom_y default 4;  value range = 1,2,...9
        lane_invasion_config_.reverse_xy = adas_perception_param_.laneinvasion().reverse_xy();
        lane_invasion_config_.x_range_min = adas_perception_param_.laneinvasion().x_range_min();
        lane_invasion_config_.x_range_max = adas_perception_param_.laneinvasion().x_range_max();
        lane_invasion_config_.y_range_min = adas_perception_param_.laneinvasion().y_range_min();
        lane_invasion_config_.y_range_max = adas_perception_param_.laneinvasion().y_range_max();
        // standard limit params (m)
        lane_invasion_config_.railway_standard_width = adas_perception_param_.laneinvasion().railway_standard_width();
        lane_invasion_config_.railway_half_width = adas_perception_param_.laneinvasion().railway_half_width();
        lane_invasion_config_.railway_limit_width = adas_perception_param_.laneinvasion().railway_limit_width();
        lane_invasion_config_.railway_delta_width = adas_perception_param_.laneinvasion().railway_delta_width();

        lane_invasion_config_.case1_x_threshold = adas_perception_param_.laneinvasion().case1_x_threshold();
        lane_invasion_config_.case1_y_threshold = adas_perception_param_.laneinvasion().case1_y_threshold();

        lane_invasion_config_.use_lane_status = adas_perception_param_.laneinvasion().use_lane_status();
        lane_invasion_config_.use_lidar_pointcloud_smallobj_invasion_detect = adas_perception_param_.laneinvasion().use_lidar_pointcloud_smallobj_invasion_detect();

        //得到运行模式，很重要的一个参数
        ///////////!!!!!!!!
        //  ONLINE 或者 0; //正常在线运行
        // SIM 或者 1;//离线仿真运行
        // ONLINE_ACQ 或者 2; //在线采集运行
        std::string  type = adas_perception_param_.model_type();
        //配置文件里不区分大小写
        type = boost::algorithm::to_lower_copy(type);

        if(type == "online" || type == "0")
        {
            model_type_ =   watrix::projects::adas::proto::ONLINE;
        }else if(type == "sim" || type == "1")
        {
            model_type_ =   watrix::projects::adas::proto::SIM;
        }else if(type == "online_acq" || type == "2")
        {
              model_type_ =   watrix::projects::adas::proto::ONLINE_ACQ;
        }
        else
        {
                AERROR << "UNKNOWN  Perception Model Type: " << type << " Please Use:  online  || sim || online_acq";
                return false;
        }
          ///////////!!!!!!!!
        //是否保存图片
        if_save_image_result_ = adas_perception_param_.if_save_image_result();
        save_image_dir_ = adas_perception_param_.save_image_dir();
        if_save_log_result_ = adas_perception_param_.if_save_log_result();


        //取得全局参数设定
        watrix::projects::adas::proto::InterfaceServiceConfig interface_config;
        apollo::cyber::common::GetProtoFromFile(
            apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(), FLAGS_adas_cfg_interface_file),
            &interface_config);

        // 取得线程池个数
        perception_tasks_num_ = interface_config.perception_tasks_num();
        //取得相机名
        std::string camera_names_str = interface_config.camera_names();
        boost::algorithm::split(camera_names_, camera_names_str, boost::algorithm::is_any_of(","));
        // 目前一个功能组件支持2个相机，软件内部可以支持多个
        if (camera_names_.size() != FLAGS_adas_camera_size)
        {
          AERROR << "Now Perception Component only support " << FLAGS_adas_camera_size << " cameras";
          return false;
        }
        //取得相机通道名
        std::string input_camera_channel_names_str =
            interface_config.camera_channels();
        boost::algorithm::split(input_camera_channel_names_,
                                input_camera_channel_names_str,
                                boost::algorithm::is_any_of(","));
        if (input_camera_channel_names_.size() != camera_names_.size())
        {
          AERROR << "wrong input_camera_channel_names_.size(): "
                 << input_camera_channel_names_.size();
          return false;
        }

        //取得lidar名
        std::string lidar_names_str = interface_config.lidar_names();
        boost::algorithm::split(lidar_names_, lidar_names_str, boost::algorithm::is_any_of(","));
        //软件内部可以支持多个
        if (lidar_names_.size() != FLAGS_adas_lidar_size)
        {
          AERROR << "Now Perception Component only support " << FLAGS_adas_lidar_size << " cameras";
          return false;
        }
        //取得lidar通道名
        std::string input_lidar_channel_names_str =
            interface_config.lidar_channels();
        boost::algorithm::split(input_lidar_channel_names_,
                                input_lidar_channel_names_str,
                                boost::algorithm::is_any_of(","));
        if (input_lidar_channel_names_.size() != lidar_names_.size())
        {
          AERROR << "wrong input_lidar_channel_names_.size(): "
                 << input_lidar_channel_names_.size();
          return false;
        }

        std::string format_str = R"(
      AdasPerceptionComponent InitConfig success
      camera_names:    %s,%s
      input_camera_channel_names:     %s,%s
      lidar_names:    %s
      input_lidar_channel_names:     %s
      perception  pool size: %d )";
        std::string config_info_str =
            boost::str(boost::format(format_str.c_str()) % camera_names_[0] % camera_names_[1] %
                       input_camera_channel_names_[0] % input_camera_channel_names_[1] %
                       lidar_names_[0] %
                       input_lidar_channel_names_[0] %
                       perception_tasks_num_);
        AINFO << config_info_str;

        //初始化空间
        images_.resize(FLAGS_adas_camera_size);
        sim_image_files_.resize(FLAGS_adas_camera_size);



       //取得算法模块的配置参数设定
        watrix::projects::adas::proto::AlgorithmConfig algorithm_config;
        apollo::cyber::common::GetProtoFromFile(
            apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(), FLAGS_adas_cfg_algorithm_file),
            &algorithm_config);


          result_check_file_ = "result-" + algorithm_config.sdk_version() + "-" + algorithm_config.lidar_version() + ".log";
          result_log_file_ =  "log-" + algorithm_config.sdk_version() + "-" + algorithm_config.lidar_version() + ".log";
          //放在存储的全路径上
          result_check_file_ = apollo::cyber::common::GetAbsolutePath(save_image_dir_, result_check_file_);
          result_log_file_ =  apollo::cyber::common::GetAbsolutePath(save_image_dir_, result_log_file_);

        return true;
      }

      bool AdasPerceptionComponent::InitAlgorithmPlugin()
      {
        //lidar参数初始化
        load_lidar_map_parameter();
        //如果配置文件不存在的，则不会被加载

        //是否用检测功能，是否采用yolo算法
        if (adas_perception_param_.if_use_detect_model() && adas_perception_param_.has_yolo())
        {
          init_yolo_api(adas_perception_param_.yolo());
        }
        //检测时是否采用darknet算法,如果有yolo，优先用yolo，屏蔽darknet
        // if (!adas_perception_param_.has_yolo() && adas_perception_param_.if_use_detect_model() && adas_perception_param_.has_darknet() )
        // {
        //   init_darknet_api(adas_perception_param_.darknet());
        // }

        //是否用列车分割功能
        if (adas_perception_param_.if_use_train_seg_model() && adas_perception_param_.has_trainseg())
        {
          init_trainseg_api(adas_perception_param_.trainseg());
        }
        //是否用轨道分割功能
        if (adas_perception_param_.if_use_lane_seg_model() && adas_perception_param_.has_laneseg())
        {
          init_laneseg_api(adas_perception_param_.laneseg());

          if (v_image_lane_front_result_.size() == 0)
          {
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
              v_image_lane_front_result_.push_back(image_0);
            }
          }
          FindContours_v2::load_params();
        }

        init_distance_api();

        load_calibrator_parameter();

        return true;
      }

      bool AdasPerceptionComponent::InitWriters()
      {
        // 根据配置获得内置的接口配置信息
        watrix::projects::adas::proto::InterfaceServiceConfig interface_config;
        apollo::cyber::common::GetProtoFromFile(
            apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(), FLAGS_adas_cfg_interface_file),
            &interface_config);
        //取得输出通道
        std::string output_channels_str = interface_config.camera_output_channels();
        boost::algorithm::split(output_camera_channel_names_, output_channels_str, boost::algorithm::is_any_of(","));
        // 目前一个功能组件支持2个相机，软件内部可以支持多个
        if (output_camera_channel_names_.size() != FLAGS_adas_camera_size)
        {
          AERROR << "Now Perception Component only support " << FLAGS_adas_camera_size << " cameras output";
          return false;
        }

        //取得调试通道
        std::string debug_channels_str = interface_config.camera_debug_channels();
        boost::algorithm::split(debug_camera_channel_names_, debug_channels_str, boost::algorithm::is_any_of(","));

        //实际输出，
        camera_out_writers_[camera_names_[0]] =
            node_->CreateWriter<watrix::projects::adas::proto::SendResult>(output_camera_channel_names_[0]);
        camera_out_writers_[camera_names_[1]] =
            node_->CreateWriter<watrix::projects::adas::proto::SendResult>(output_camera_channel_names_[1]);

        //可以支持多个调试通道
        camera_debug_writers_[camera_names_[0]] =
            node_->CreateWriter<watrix::projects::adas::proto::CameraImages>(debug_camera_channel_names_[0]);
        camera_debug_writers_[camera_names_[1]] =
            node_->CreateWriter<watrix::projects::adas::proto::CameraImages>(debug_camera_channel_names_[1]);

        //参数服务,触发记录存档用
        std::string node_name = interface_config.records_parameter_servicename();
        param_node_ = apollo::cyber::CreateNode(node_name);

        param_server_ = std::make_shared<apollo::cyber::ParameterServer>(param_node_);
        record_para_name_ = interface_config.record_parameter_name();
        // 初始化为false
        apollo::cyber::Parameter record_parameter(record_para_name_, false);
        param_server_->SetParameter(record_parameter);

        return true;
      }
      bool AdasPerceptionComponent::InitListeners()
      {
        //仿真模式下
        // 相同的通道名称，不同的接收类型，不同的处理回调

        //初始化图像接收器
        for (size_t i = 0; i < camera_names_.size(); ++i)
        {
          const std::string &camera_name = camera_names_[i];
          const std::string &channel_name = input_camera_channel_names_[i];

          typedef std::shared_ptr<apollo::drivers::Image> ImageMsgType;
          std::function<void(const ImageMsgType &)> camera_callback =
              std::bind(&AdasPerceptionComponent::OnReceiveImage, this,
                        std::placeholders::_1, camera_name);

          auto camera_reader = node_->CreateReader(channel_name, camera_callback);
        }
        //初始化lidar接收器
        for (size_t i = 0; i < lidar_names_.size(); ++i)
        {
          const std::string &lidar_name = lidar_names_[i];
          const std::string &lidar_channel_name = input_lidar_channel_names_[i];

          typedef std::shared_ptr<apollo::drivers::PointCloud> PointCloudMsgType;
          std::function<void(const PointCloudMsgType &)> lidar_callback =
              std::bind(&AdasPerceptionComponent::OnReceivePointCloud, this,
                        std::placeholders::_1, lidar_name);

          auto lidar_reader = node_->CreateReader(lidar_channel_name, lidar_callback);
        }

        return true;
      }

      void AdasPerceptionComponent::OnReceiveImage(
          const std::shared_ptr<apollo::drivers::Image> &message,
          const std::string &camera_name)
      {
        std::lock_guard<std::mutex> lock(camera_mutex_);

        //测量时间是一个相对时间，由驱动决定
        const double msg_timestamp = message->measurement_time() + timestamp_offset_;

        AERROR << "OnReceiveImage(), "
               << " FrameID: " << message->header().frame_id()
               << " Seq: " << message->header().sequence_num()
               << " image ts: " + std::to_string(msg_timestamp)
               << " ms";

        // timestamp should be almost monotonic
        if (last_camera_timestamp_ - msg_timestamp > ts_diff_)
        {
          AINFO << "Received an old message. Last ts is " << std::setprecision(19)
                << last_camera_timestamp_ << " current ts is " << msg_timestamp
                << " last - current is " << last_camera_timestamp_ - msg_timestamp;
          return;
        }
        last_camera_timestamp_ = msg_timestamp;
        //内部处理
        if (InternalProc(message, camera_name) != apollo::cyber::SUCC)
        {
          AERROR << "InternalProc failed ";
        }
      }

      int AdasPerceptionComponent::InternalProc(
          const std::shared_ptr<apollo::drivers::Image const> &in_message,
          const std::string &camera_name)
      {
        //过滤长宽不对的img
        if (in_message->height() <= 0 || in_message->width() <= 0)
          return apollo::cyber::FAIL;
        //时间采用发包时间，而不是measure_time，因为每种设备在measure_time里自定义测量时间
        apollo::cyber::Time timestamp(in_message->header().timestamp_sec());

        for (auto i = 0; i < camera_names_.size(); i++)
        {
          //根据camera_name 把 image赋值给对应的缓存
          if (camera_names_[i] == camera_name)
          {

            //如果是仿真的，就需要 仿真文件信息,不是仿真则用采集时间作为文件名
            sim_image_files_[i] = (model_type_ == SIM ) ?  in_message->frame_id() : timestamp.ToString("%Y-%m-%d-%H-%M-%S") + ".jpg";

            int image_size = in_message->height() * in_message->step();

            cv::Mat tmp = cv::Mat::zeros(in_message->height(), in_message->width(), CV_8UC3);

            memcpy(tmp.data, in_message->data().c_str(), image_size);

            cv::cvtColor(tmp, tmp, CV_RGB2BGR); //将RGB图像转换为BGR

            //填充本地cv::Mat
            images_[i] = tmp;
          }
        }
        //暂时只有一路相机出发计算线程
        if (camera_names_[0] == camera_name)
          return apollo::cyber::SUCC;
        //序列号
        this->sequence_num_ = in_message->header().sequence_num();

        //进入线程池处理
        task_processor_->Enqueue(std::bind(&AdasPerceptionComponent::doPerceptionTask, this));

        return apollo::cyber::SUCC;
      }
      //核心处理任务
      void AdasPerceptionComponent::doPerceptionTask( )
      {
        std::shared_ptr<AdasPerceptionComponent> share_this =
            std::dynamic_pointer_cast<AdasPerceptionComponent>(shared_from_this());
        //新建PerceptionTask 任务


        YoloDarknetApi* darknet_ptr = static_cast<YoloDarknetApi*>(this->task_processor_->ptr_map_[std::this_thread::get_id()]);

      
        std::shared_ptr<PerceptionTask> task = make_shared<PerceptionTask>(share_this,darknet_ptr);

        task->Excute();

      }

      void AdasPerceptionComponent::OnReceivePointCloud(const std::shared_ptr<apollo::drivers::PointCloud> &in_message,
                                                        const std::string &lidar_name)
      {
         std::lock_guard<std::mutex> lock(lidar_mutex_);

        const double msg_timestamp = in_message->measurement_time() + timestamp_offset_;
        AERROR << "OnReceivePointCloud(), "
               << " FrameID: " << in_message->header().frame_id()
               << " Seq: " << in_message->header().sequence_num()
               << " image ts: " + std::to_string(msg_timestamp)
               << " ms";

        if (last_lidar_timestamp_ - msg_timestamp > ts_diff_)
        {
          AINFO << "Received an old Lidar message. Last ts is " << std::setprecision(19)
                << last_lidar_timestamp_ << " current ts is " << msg_timestamp
                << " last - current is " << last_lidar_timestamp_ - msg_timestamp;
          return;
        }
        last_lidar_timestamp_ = msg_timestamp;
        //核心处理
        int effect_point = 0;
        FindContours_v2::OnPointCloud(
            *(in_message.get()),
            lidar2image_paint_,
            lidar_safe_area_,
            lidar_cloud_buf_,
            effect_point);
      }

    } // namespace adas
  }   // namespace projects
} // namespace watrix
