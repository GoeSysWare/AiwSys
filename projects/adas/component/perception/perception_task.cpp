
#include "cyber/cyber.h"
#include "projects/adas/component/perception/perception_task.h"
#include "projects/adas/component/common/util.h"
#include "projects/adas/component/common/timer.h"
#include "projects/adas/component/perception/FindContours_v2.h"

#define RED_POINT Vec3b(0, 0, 255)
#define YELLOW_POINT Vec3b(0, 255, 255)
#define GREEN_POINT Vec3b(0, 255, 0)
#define BLUE_POINT Vec3b(255, 0, 0)

namespace watrix
{
namespace projects
{
namespace adas
{

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

    display_text = detection_box.class_name + " x=" + GetFloatRound(x, 2) + " y=" + GetFloatRound(y, 2);

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

static void MakeCvImageToProtoMsg(const cv::Mat &image, apollo::drivers::Image *out_msg)
{
  uint height = image.rows;
  uint width = image.cols;
  uint channel = image.channels();
  uint length = height * width * channel;

  out_msg->mutable_header()->set_frame_id(std::to_string(PerceptionTask::taskd_excuted_num_));
  out_msg->set_width(width);
  out_msg->set_height(height);
  out_msg->mutable_data()->reserve(width * width * channel);
  out_msg->set_encoding("bgr8");
  out_msg->set_step(channel * width);
  out_msg->mutable_header()->set_timestamp_sec(apollo::cyber::Time::Now().ToSecond());
  out_msg->set_measurement_time(apollo::cyber::Time::Now().ToSecond());
  out_msg->set_data(image.data, width * width * channel);
}
static void GetWorldPlace(int cy, int cx, float &safe_x, float &safe_y, int whice_table)
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
  }

  if (safe_y == 1000)
  {
    safe_y = 0;
    safe_x = 0;
  }
}

static void PainPoint2Image(cv::Mat mat, int x, int y, cv::Vec3b p_color)
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

std::atomic<uint64_t> PerceptionTask::taskd_excuted_num_ = {0};

PerceptionTask::PerceptionTask(PerceptionComponentPtr p) : perception_(p)
{
  this->v_image_= perception_->images_;
  for(auto i = 0; i < perception_->images_.size(); i++  ) this->v_image_[i] = perception_->images_[i].clone();
  this->v_image_lane_front_result_ = perception_->v_image_lane_front_result;
  this->lidar_cloud_buf_ = perception_->lidar_cloud_buf_;
  this->lidar_safe_area_ = perception_->lidar_safe_area_;
  this->lidar2image_paint_ = perception_->lidar2image_paint_;
  this->lane_invasion_config_ = perception_->lane_invasion_config;
}
PerceptionTask::~PerceptionTask()
{
}

void PerceptionTask::Excute()
{

    CaffeApi::set_mode(true, 0, 1234);

  watrix::projects::adas::Timer timer;
  if (perception_->adas_perception_param_.if_use_detect_model())
  {
    DoYoloDetectGPU();
  }
  if (perception_->adas_perception_param_.if_use_lane_seg_model())
  {
    DoLaneSegSeqGPU();
  }

  if (perception_->adas_perception_param_.if_use_train_seg_model())
  {
    DoTrainSegGPU();
  }
  if (perception_->adas_perception_param_.if_use_lane_seg_model() && perception_->adas_perception_param_.if_use_detect_model())
  {
    SyncPerceptionResult();
  }

  AERROR <<"[ thread : "<<std::this_thread::get_id() <<"] PerceptionTask::Excute " << static_cast<double>(timer.Toc()) * 0.001 << "ms";
  taskd_excuted_num_.fetch_add(1);
}

void PerceptionTask::DoLaneSegSeqGPU()
{
    AERROR <<"[ thread : "<<std::this_thread::get_id() <<"] 3. DoLaneSegSeqGPU Enter";

  std::vector<cv::Mat> v_binary_mask;
  std::vector<channel_mat_t> v_instance_mask;
  int min_area_threshold = 200; // connect components min area

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

  LaneSegApi::lane_seg_sequence(
      //net_id,
      1, //NET_LANESEG_0, // only 1 net
      v_image_lane_front_result_,
      v_image_,
      min_area_threshold,
      v_binary_mask,
      v_instance_mask);

  if (v_binary_mask.size() > 0)
  {
    v_image_lane_front_result_.clear();
  }

  for (int j = 0; j < v_binary_mask.size(); j++)
  {
    if (j == 0)
    {
      laneseg_binary_mask0_ = v_binary_mask[j];
      v_instance_mask0_ = v_instance_mask[j];
      v_image_lane_front_result_.push_back(v_binary_mask[j]);
    }
    else if (j == 1)
    {
      laneseg_binary_mask1_ = v_binary_mask[j];
      v_instance_mask1_ = v_instance_mask[j];
      v_image_lane_front_result_.push_back(v_binary_mask[j]);
    }
  }
}

void PerceptionTask::DoYoloDetectGPU()
{
  AERROR <<"[ thread : "<<std::this_thread::get_id() <<"] 1. DoYoloDetectGPU Enter";

  yolo_detection_boxs0_.clear();
  yolo_detection_boxs1_.clear();

  std::vector<detection_boxs_t> v_output;

  YoloApi::Detect(
      1,
      v_image_,
      v_output);

  for (int j = 0; j < v_output.size(); j++)
  {
    if (j == 0)
    {
      yolo_detection_boxs0_ = v_output[j];
      // set dist x and y
      for (int i = 0; i < yolo_detection_boxs0_.size(); ++i)
      {
        detection_box_t &box = yolo_detection_boxs0_[i];
        int cx = (box.xmin + box.xmax) / 2;
        int cy = box.ymax;
        box.valid_dist = MonocularDistanceApi::get_distance(TABLE_SHORT_A, cy, cx, box.dist_x, box.dist_y);
        if (box.valid_dist)
        {
          yolo_detection_boxs0_[i].dist_x = std::atof(GetFloatRound(box.dist_x, 2).c_str());
          yolo_detection_boxs0_[i].dist_y = std::atof(GetFloatRound(box.dist_y, 2).c_str());
        }
      }
    }
    else
    {
      yolo_detection_boxs1_ = v_output[j];
      // set dist x and y
      for (int i = 0; i < yolo_detection_boxs1_.size(); ++i)
      {
        detection_box_t &box = yolo_detection_boxs1_[i];
        int cx = (box.xmin + box.xmax) / 2;
        int cy = box.ymax;
        box.valid_dist = MonocularDistanceApi::get_distance(TABLE_LONG_A, cy, cx, box.dist_x, box.dist_y);
        if (box.valid_dist)
        {
          yolo_detection_boxs1_[i].dist_x = std::atof(GetFloatRound(box.dist_x, 2).c_str());
          yolo_detection_boxs1_[i].dist_y = std::atof(GetFloatRound(box.dist_y, 2).c_str());
        }
      }
    }
  }
}
void PerceptionTask::DoTrainSegGPU()
{
  AERROR <<"[ thread : "<<std::this_thread::get_id() <<"] 2. DoTrainSegGPU Enter";
  std::vector<cv::Mat> output;
  boost::posix_time::ptime pt1 = boost::posix_time::microsec_clock::local_time();
  TrainSegApi::train_seg(
      //net_id,
      0, //NET_TRAINSEG_0, // only 1 net
      v_image_,
      output);
}

void PerceptionTask::SyncPerceptionResult()
{
  AERROR <<"[ thread : "<<std::this_thread::get_id() <<"] 4.SyncPerceptionResult Enter";

  std::vector<cv::Mat> v_image_with_color_mask(2);
  std::vector<box_invasion_results_t> v_box_invasion_results(2);
  std::vector<lane_safe_area_corner_t> v_lane_safe_area_corner(2); // 4 cornet point
  std::vector<int> lidar_invasion_status0;                         // status:  -1 UNKNOW, 0 NOT Invasion, 1 Yes Invasion
  std::vector<int> lidar_invasion_status1;

  cvpoints_t lidar_cvpoints;
  cvpoints_t train_cvpoints;
  lidar_cvpoints.clear();
  train_cvpoints.clear();
  std::vector<cvpoints_t> v_trains_cvpoint;

  int lidar2img = -2;

  if (perception_->adas_perception_param_.if_use_detect_model())
  {
    if (lidar2img >= 0)
    {
      train_cvpoints = GetTrainCVPoints(yolo_detection_boxs0_, v_trains_cvpoint);
    }
  }
  bool long_camera_open_status = false;
  bool short_camera_open_status = false;
  std::vector<lidar_invasion_cvbox> long_cv_obstacle_box;  // lidar invasion object cv box
  std::vector<lidar_invasion_cvbox> short_cv_obstacle_box; // lidar invasion object cv box

  v_box_invasion_results[0].clear();
  v_box_invasion_results[1].clear();

  for (int v = 0; v < v_image_.size(); v++)
  {
    cv::Mat origin_image = v_image_[v];
    cv::Mat binary_mask;
    channel_mat_t instance_mask;
    detection_boxs_t detection_boxs;
    std::vector<int> lidar_invasion_status;

    int lane_count; // total lane count >=0
    int id_left;    // -1 invalid; >=0 valid
    int id_right;   // -1 invalid; >=0 valid
    if (v == 0)
    {
      binary_mask = laneseg_binary_mask0_[0];
      instance_mask = v_instance_mask0_;
      detection_boxs = yolo_detection_boxs0_;

      bool instance_success = LaneSegApi::lane_invasion_detect(
          CAMERA_SHORT,
          origin_image,  // 1080,1920
          binary_mask,   // 256,1024
          instance_mask, // 8,256,1024
          detection_boxs,
          train_cvpoints, // for train points
          lidar_cloud_buf_,
          lane_invasion_config_,
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
      binary_mask = laneseg_binary_mask1_[0];
      instance_mask = v_instance_mask1_;
      detection_boxs = yolo_detection_boxs1_;
      std::vector<cvpoints_t> train_cvpoints; // for train points
      for (int t = 0; t < detection_boxs.size(); t++)
      {
        cvpoints_t t_point;
        train_cvpoints.push_back(t_point);
      }
      std::vector<cv::Point3f> v_lidar_points; // for train points
      lidar_cvpoints.clear();
      // do invasion detect
      bool instance_success = LaneSegApi::lane_invasion_detect(
          CAMERA_LONG,
          origin_image,  // 1080,1920
          binary_mask,   // 256,1024
          instance_mask, // 8,256,1024
          detection_boxs,
          lidar_cvpoints,
          v_lidar_points,
          lane_invasion_config_,
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
  }

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
        v_train_invasion_status.push_back(1);
      }
      else
      {
        v_train_invasion_status.push_back(0);
      }
    }
    //change train status
    detection_boxs_t detection_boxs = yolo_detection_boxs0_;

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
  }
  else
  {
    select_cam = 0;
  }

  std::vector<int> lidar_point_status;
  // 将雷达点画到图片上
  lidar_point_status = lidar_invasion_status0;
  //std::cout<<"------------------"<<lidar_point_status.size()<<std::endl;
  cv::Mat short_mat;

  if ((lidar_point_status.size() > 0) && (lidar2img >= 0))
  {
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

  // if (lane_invasion_save)
  // {
  //   cv::Mat combine;
  //   cv::Mat s_image_boxs;
  //   cv::Point2i origin(900, 50);
  //   std::string display_text;
  //   int if_combine = 0;
  //   if (v_image_with_color_mask[0].empty())
  //   {
  //     std::cout << "cam1 is null" << std::endl;
  //   }
  //   else
  //   {
  //     display_text = "short safe distance" + std::to_string(short_safe_y);
  //     here_draw_detection_boxs_ex(v_image_with_color_mask[0], yolo_detection_boxs0_, v_box_invasion_results[0], 5, s_image_boxs);
  //     cv::putText(s_image_boxs, display_text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2);
  //     if_combine++;
  //   }

  //   cv::Mat l_image_boxs;
  //   if (v_image_with_color_mask[1].empty())
  //   {
  //     std::cout << "cam2 is null" << std::endl;
  //   }
  //   else
  //   {
  //     here_draw_detection_boxs_ex(v_image_with_color_mask[1], yolo_detection_boxs1_, v_box_invasion_results[1], 5, l_image_boxs);
  //     display_text = "long safe distance" + std::to_string(long_safe_y);
  //     cv::putText(l_image_boxs, display_text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2);
  //     if_combine++;
  //   }

  //   if (if_combine == 2)
  //   {
  //     cv::vconcat(l_image_boxs, s_image_boxs, combine);
  //     std::string save_filepath = lane_invasion_result_folder_ + files_fullname[1].stem().string() + " _combine.jpg";
  //     cv::imwrite(save_filepath, combine);
  //   }
  // }

  //   //打包成SendResult
  // auto sendData_0 = std::make_shared<watrix::projects::adas::proto::SendResult>();
  // auto sendData_1 = std::make_shared<watrix::projects::adas::proto::SendResult>();

  // //向protobuf填入detection_boxs
  // apollo::drivers::Image *seg_binary_mask = sendData_0->mutable_seg_binary_mask();
  // watrix::projects::adas::proto::MaxSafeDistance *max_safe_distance = sendData_1->mutable_max_safe_distance();

  // //向protobuf填入source_image
  // apollo::drivers::Image *source_image_0 = sendData_0->mutable_source_image();
  // watrix::projects::adas::proto::DetectionBoxs *pb_mutable_detection_boxs_0 = sendData_0->mutable_detection_boxs();
  // watrix::projects::adas::proto::CameraImage *source_image_1 = sendData_1->mutable_source_image();
  // watrix::projects::adas::proto::DetectionBoxs *pb_mutable_detection_boxs_1 = sendData_1->mutable_detection_boxs();
  // //resize to hdmi need screen
  // cv::Mat out_mat_0;
  // cv::Mat out_mat_1;

  // FillDetectBox(v_image_with_color_mask[0], yolo_detection_boxs0_, v_box_invasion_results[0], pb_mutable_detection_boxs_0);
  // max_safe_distance->set_image_distance(short_safe_y);
  // cv::resize(short_mat, out_mat_0, cv::Size(1280, 800));

  // //mat image
  // FillDetectBox(v_image_with_color_mask[1], yolo_detection_boxs1_, v_box_invasion_results[1], pb_mutable_detection_boxs_1);
  // max_safe_distance->set_image_distance(long_safe_y);
  // //tmp_mat = v_image_with_color_mask[select_cam];
  // cv::resize(v_image_with_color_mask[1], out_mat_1, cv::Size(1280, 800));

  // //侵界判断
  // bool invasion_flag = false;

  // // 长较焦相机有一个东西侵界就算侵界
  // for (auto &it : v_box_invasion_results[0])
  // {
  //   if (it.invasion_status == YES_INVASION)
  //   {
  //     invasion_flag = true;
  //     break;
  //   }
  // }
  // for (auto &it : v_box_invasion_results[1])
  // {
  //   if (it.invasion_status == YES_INVASION)
  //   {
  //     invasion_flag = true;
  //     break;
  //   }
  // }
  // //侵界时需要存视频
  // if (invasion_flag)
  // {
  //   AERROR << "adas_perception" << std::endl;
  //   apollo::cyber::Parameter parameter;
  //   param_server_->GetParameter("is_record", &parameter);
  //   if (parameter.AsBool() == false)
  //   {
  //     apollo::cyber::Parameter record_parameter("is_record", true);
  //     param_server_->SetParameter(record_parameter);
  //   }
  // }

  // GetCameraImage(out_mat_0, CameraImage::ORIGIN, 0, source_image_0);
  // source_image_0->set_timestamp_msec(image_time);
  // source_image_0->set_frame_count(g_index);
  // pt3 = boost::posix_time::microsec_clock::local_time();
  // int64_t cost1 = (pt3 - pt1).total_milliseconds();
  // int64_t cost2 = (pt3 - pt0).total_milliseconds();

  // front_6mm_writer_result_->Write(sendData_0);
  // // OnSendResult(sendData_0);

  // GetCameraImage(out_mat_1, CameraImage::ORIGIN, 1, source_image_1);
  // source_image_1->set_timestamp_msec(image_time);
  // source_image_1->set_frame_count(g_index++);
  // pt3 = boost::posix_time::microsec_clock::local_time();
  // int64_t cost3 = (pt3 - pt1).total_milliseconds();
  // int64_t cost4 = (pt3 - pt0).total_milliseconds();
  // AERROR << "------PublishSendResult  after get obj+lane [" << cost3 << "]  all thread spent [" << cost4;
  // front_12mm_writer_result_->Write(sendData_1);
  // // OnSendResult(sendData_1);
}

cvpoints_t PerceptionTask::GetTrainCVPoints(detection_boxs_t &detection_boxs, std::vector<cvpoints_t> &v_trains_cvpoint)
{
  cvpoints_t v_train_cvpoint;

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
    int xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    if (std::strcmp(cname.c_str(), "train") == 0)
    {
      watrix::algorithm::cvpoints_t train_cvpoint_tmp;
      apollo::drivers::PointXYZIT train_lidar_points;
      xmin = detection_boxs[box].xmin;
      xmax = detection_boxs[box].xmax;
      ymin = detection_boxs[box].ymin;
      ymax = detection_boxs[box].ymax;
      std::vector<apollo::drivers::PointXYZIT> train_points;
      for (int j = 0; j < lidar2image_paint_.point_size(); j++)
      { //lidar2image_check_
        uint pos_y = (uint)(lidar2image_paint_.point(j).y());
        uint pos_x = (uint)(lidar2image_paint_.point(j).x());
        if ((pos_x > xmin) && (pos_x < xmax))
        {
          if ((pos_y > ymin) && (pos_y < ymax))
          {
            apollo::drivers::PointXYZIT tp;
            tp.set_x(lidar_safe_area_.point(j).x());
            tp.set_y(lidar_safe_area_.point(j).y());
            tp.set_z(lidar_safe_area_.point(j).z());
            train_points.push_back(tp);
          }
        }
      }
      train_bottom_points = FindContours_v2::start_contours(train_points);
      if (train_bottom_points.size() < 1)
      {
        return v_train_cvpoint;
      }
      cvpoints_t train_cvpoints;
      for (int p = 0; p < train_bottom_points.size(); p++)
      {

        cvpoint_t train_cvpoint;
        train_cvpoint.x = (int)train_bottom_points[p].x;
        train_cvpoint.y = (int)train_bottom_points[p].y;
        if (train_cvpoint.x < 2)
          continue;
        //下底边的cvpoint输出进行检测
        v_train_cvpoint.push_back(train_cvpoint);
      }

      v_trains_cvpoint.push_back(train_cvpoint_tmp);
      train_count++;
    }
  }

  return v_train_cvpoint;
}

} // namespace adas
} // namespace projects
} // namespace watrix
