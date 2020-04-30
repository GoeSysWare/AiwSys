
#include "projects/adas/component/simulator/adas_simulator_component.h"

#include "cyber/class_loader/class_loader.h"
#include "cyber/component/component.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> //
#include <opencv2/highgui.hpp> // imwrite
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include "projects/adas/component/common/util.h"
#include "projects/adas/component/common/timer.h"
#include "projects/adas/configs/config_gflags.h"

namespace watrix
{
namespace projects
{
namespace adas
{

//在文件路径中的索引号
#define INDEX_6MM 0
#define INDEX_12MM 1
#define INDEX_LIDAR 2

//解析配置文件，返回一个路径数组
static std::vector<std::vector<std::string>> ParseFiles(std::string &filename, std::string &dir)
{
  std::vector<std::vector<std::string>> file_list;

  std::fstream pfstream(apollo::cyber::common::GetAbsolutePath(dir, filename), std::ios_base::in);
  if (!pfstream)
    return file_list;

  while (!pfstream.eof() && pfstream.good())
  {
    char line[2048] = {0};
    std::vector<std::string> string_list;
    pfstream.getline(line, 2048);

    boost::split(string_list, line, boost::is_any_of(",; "));
    if (string_list.size() != 3)
      continue;
    for (auto &name : string_list)
    {
      //处理以"./" 开头的路径
      if (name.front() == '.')
      {
        name = name.substr(2);
      }
      //取得绝对路径，如果是相对路径,则补全
      name = apollo::cyber::common::GetAbsolutePath(dir, name);
    }
    file_list.push_back(string_list);
  }
  pfstream.close();
  return file_list;
}
//封装image
static void makeProtoImageFormMat(cv::Mat &in_Mat, apollo::drivers::Image &out_Image)
{
  int width = in_Mat.cols;
  int height = in_Mat.rows;
  int channel = in_Mat.channels();
  int step = in_Mat.step;
  int img_size = width * height * channel;
  //计数器+1
  AdasSimulatorComponent::element_num_.fetch_add(1);
  
  out_Image.mutable_header()->set_frame_id(std::to_string(AdasSimulatorComponent::procs_num_.load()));
  out_Image.set_frame_id(std::to_string(AdasSimulatorComponent::element_num_.load()));
  out_Image.set_width(width);
  out_Image.set_height(height);
  out_Image.mutable_data()->reserve(img_size);
  out_Image.set_encoding("bgr8");
  out_Image.set_step(step);
  out_Image.mutable_header()->set_timestamp_sec(apollo::cyber::Time::Now().ToSecond());
  out_Image.set_measurement_time(apollo::cyber::Time::Now().ToSecond());
  // out_Image.mutable_data()->copy(in_Mat.data,img_size);
   out_Image.set_data(in_Mat.data, img_size);
}

std::atomic<uint64_t> AdasSimulatorComponent::procs_num_ = {0};
std::atomic<uint64_t> AdasSimulatorComponent::element_num_ = {0};

bool AdasSimulatorComponent::InitConfig()
{
  //从模块配置中取得配置信息
  bool ret = GetProtoConfig(&adas_simulator_param_);
  if (!ret)
  {
    return false;
  }
  // 根据配置获得内置的接口配置信息
  watrix::projects::adas::proto::InterfaceServiceConfig interface_config;
  apollo::cyber::common::GetProtoFromFile(FLAGS_adas_cfg_interface_file, &interface_config);

  //取得图像输出通道名
  std::string output_image_channels_str = interface_config.camera_channels();
  boost::algorithm::split(output_camera_channel_names_, output_image_channels_str, boost::algorithm::is_any_of(","));
  // 目前一个功能组件支持2个相机，软件内部可以支持多个
  if (output_camera_channel_names_.size() != FLAGS_adas_camera_size)
  {
    AERROR << "Now Simulator Component only support " << FLAGS_adas_camera_size << " cameras output";
    return false;
  }
  //取得Lidar输出通道名
  std::string output_lidar_channels_str = interface_config.lidar_channels();
  boost::algorithm::split(output_lidar_channel_names_, output_lidar_channels_str, boost::algorithm::is_any_of(","));
  // 目前一个功能组件支持1个Lidar
  if (output_lidar_channel_names_.size() != FLAGS_adas_lidar_size)
  {
    AERROR << "Now Simulator Component only support " << FLAGS_adas_lidar_size << " cameras output";
    return false;
  }
  sim_files_dir_ = adas_simulator_param_.simulator_files_dir();
  sim_config_file_ = adas_simulator_param_.config_file();
  sim_interval_ = adas_simulator_param_.sim_interval();
  is_Circle_ = adas_simulator_param_.is_circle();


  std::string format_str = R"(
      Simulator Component Init Successed
      Simulator dir:    %s
      Simulator config file:    %s
      Simulator Interval:  %ld ms
      input_camera_channel_names:     %s,%s
      input_lidar_channel_names:   %s)";
  std::string config_info_str =
      boost::str(boost::format(format_str.c_str()) 
      % sim_files_dir_
      % sim_config_file_
      % this->GetInterval() 
      % output_camera_channel_names_[0] % output_camera_channel_names_[1] 
      % output_lidar_channel_names_[0] );
  AINFO << config_info_str;

  return true;
}

bool AdasSimulatorComponent::Init()
{

  if (!InitConfig())
  {
    AERROR << "AdasSimulatorComponent InitConfig  failed. Please check pb.txt  file";
    return false;
  }
  //得到的时全路径= sim_files_dir_ + filename
  filesList_ = ParseFiles(sim_config_file_, sim_files_dir_);
  if (filesList_.size() <= 0)
  {
    AERROR << "AdasSimulatorComponent InitConfig  failed.Please check simulator config file";
  }
  //初始化当前的iter
  cur_Iter_ = filesList_.begin();
  //实际输出
  front_6mm_writer_ = node_->CreateWriter<SimulatorImage>(output_camera_channel_names_[0]);
  front_12mm_writer_ = node_->CreateWriter<SimulatorImage>(output_camera_channel_names_[1]);
  lidar_writer_ = node_->CreateWriter<SimulatorPointCloud>(output_lidar_channel_names_[0]);

  return true;
}

bool AdasSimulatorComponent::Proc()
{

  if (cur_Iter_ == filesList_.end())
  {
    //如果循环
    if (is_Circle_)
      cur_Iter_ = filesList_.begin();
    else
      return true;
  }
    watrix::projects::adas::Timer timer;
  ParseCameraFiles((*cur_Iter_)[INDEX_6MM], (*cur_Iter_)[INDEX_12MM]);

  AINFO <<" ParseCameraFiles " << static_cast<double>(timer.Toc()) * 0.001 << "ms";

  ParseLidarFiles((*cur_Iter_)[INDEX_LIDAR]);
  AINFO <<" ParseLidarFiles " << static_cast<double>(timer.Toc()) * 0.001 << "ms";

  SendSimulator();
  AINFO <<" SendSimulator " << static_cast<double>(timer.Toc()) * 0.001 << "ms";

 std::string format_str = R"(
      Simulator Proc Successed
      procs_num: %ld
      front_6mm file:    %s
      front_6mm file:    %s
      lidar file:  %s)";
  std::string config_info_str =
      boost::str(boost::format(format_str.c_str()) 
      %  procs_num_.load()
      % (*cur_Iter_)[INDEX_6MM]
      % (*cur_Iter_)[INDEX_12MM] 
      % (*cur_Iter_)[INDEX_LIDAR]);
  AINFO << config_info_str;

  cur_Iter_++;

  //同步计数+1
  procs_num_.fetch_add(1);
  return true;
}
void AdasSimulatorComponent::ParseCameraFiles(std::string file_6mm, std::string file_12mm)
{

  cv::Mat mat_6mm = cv::imread(file_6mm); // bgr, 0-255

  makeProtoImageFormMat(mat_6mm, *front_6mm_image_.mutable_image());
  front_6mm_image_.set_sim_img_file(file_6mm);

  cv::Mat mat_12mm = cv::imread(file_12mm); // bgr, 0-255
  makeProtoImageFormMat(mat_12mm, *front_12mm_image_.mutable_image());
  front_12mm_image_.set_sim_img_file(file_12mm);
}
///解析雷达文件的
void AdasSimulatorComponent::ParseLidarFiles(std::string file_lidar)
{

  std::ifstream pcd_file;
  lidar_pointcloud_.mutable_piontclound()->clear_point();
  lidar_pointcloud_.set_sim_lidar_file(file_lidar);
  pcd_file.open(file_lidar); //将文件流对象与文件连接起来
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
    boost::algorithm::split(v_str, str_line, boost::algorithm::is_any_of(" "));
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
    }
    PointXYZIT *pt = lidar_pointcloud_.mutable_piontclound()->add_point();
    pt->set_x(x);
    pt->set_y(y);
    pt->set_z(z);
  }
  //一些辅助信息也要添加
  lidar_pointcloud_.mutable_piontclound()->set_measurement_time(apollo::cyber::Time::Now().ToSecond());
  lidar_pointcloud_.mutable_piontclound()->mutable_header()->set_frame_id(std::to_string(AdasSimulatorComponent::procs_num_.load()));
  //计数器+1
  AdasSimulatorComponent::element_num_.fetch_add(1);
  lidar_pointcloud_.mutable_piontclound()->set_frame_id(std::to_string(AdasSimulatorComponent::element_num_.load()));
}

void AdasSimulatorComponent::SendSimulator()
{
  front_6mm_writer_->Write(front_6mm_image_);
  front_12mm_writer_->Write(front_12mm_image_);
  lidar_writer_->Write(lidar_pointcloud_);
}

} // namespace adas
} // namespace projects
} // namespace watrix
