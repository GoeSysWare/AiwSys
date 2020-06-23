#include "widget.h"
#include "algorithm/monocular_distance_api.h"
#include <QApplication>
#include <QScreen>
#include <QPushButton>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
// std
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include "widget.h"
#include "record_widget.h"
#include "projects/adas/component/hmi/ui_widget.h"
#include "algorithm/monocular_distance_api.h"
#include "projects/adas/algorithm/core/util/display_util.h"

#define MIN_SCALED_WIDTH (320 - 6)
#define MIN_SCALED_HEIGHT (200 - 6) //(150-6)
#define MAX_SCALED_WIDTH (1280)     //(780-6)
#define MAX_SCALED_HEIGHT (780 - 6) //(590-6)

using namespace watrix::algorithm;
using namespace cv;
using namespace std;

bool display_style = true;

Widget::Widget(QWidget *parent) : QWidget(parent),
                                  ui(new Ui::Widget),
                                  form(new RecordForm(parent))
{

    this->setWindowTitle("自动驾驶显示终端");
    this->setWindowIcon(QIcon(":/image/Icons/app_logo.png"));
    ui->setupUi(this);

    //this->setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint);
    this->setWindowFlags(Qt::Dialog);
    //this->setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    //this->showFullScreen();

    form->close();

    //初始化网络
    initNetwork();

    // 安装事件过滤器
    ui->label_sub->installEventFilter(this);
    ui->label_record->installEventFilter(this);

    show_index_ = 0;
    //初始化信号槽
    initConnect();
    //初始化定时器
    timer_id_ = startTimer(1000);
}

void Widget::initNetwork()
{
    // 根据配置获得内置的记录的通道名
  watrix::projects::adas::proto::InterfaceServiceConfig interface_config;
  apollo::cyber::common::GetProtoFromFile(
    apollo::cyber::common::GetAbsolutePath(watrix::projects::adas::GetAdasWorkRoot(), FLAGS_adas_cfg_interface_file),
        &interface_config);

    std::string record_channelname_prefix= interface_config.record_channelname_prefix();

    //取得相机名
    std::string camera_names_str = interface_config.camera_names();
    boost::algorithm::split(camera_names_, camera_names_str, boost::algorithm::is_any_of(","));
    // 目前一个功能组件支持2个相机，软件内部可以支持多个
    if (camera_names_.size() != FLAGS_adas_camera_size)
    {
        AERROR << "Now HMI Component only support " << FLAGS_adas_camera_size << " cameras";
        return ;
    }

    //取得perception的输出通道,即是HMI的正常显示输入通道
    std::string input_channels_str = interface_config.camera_output_channels();
    boost::algorithm::split(input_camera_channel_names_, input_channels_str, boost::algorithm::is_any_of(","));
    // 目前一个功能组件支持2个相机，软件内部可以支持多个
    if (input_camera_channel_names_.size() != FLAGS_adas_camera_size)
    {
        AERROR << "Now HMI Component only support " << FLAGS_adas_camera_size << " cameras output";
        return ;
    }

    //取得recorder的输出通道,即是HMI的回放输入通道
    std::string record_channels_str = interface_config.camera_channels();
    boost::algorithm::split(record_camera_channel_names_, record_channels_str, boost::algorithm::is_any_of(","));
    // 目前一个功能组件支持2个相机，软件内部可以支持多个
    if (record_camera_channel_names_.size() != FLAGS_adas_camera_size)
    {
        AERROR << "Now HMI Component only support " << FLAGS_adas_camera_size << " cameras output";
        return ;
    }
    //节点名这么设计，可以支持多HMI运行
    reader_node_ = apollo::cyber::CreateNode(GlobalData::Instance()->HostName()+ std::to_string(GlobalData::Instance()->ProcessId()) );

    //接收算法处理后的result数据
    typedef std::shared_ptr<watrix::projects::adas::proto::SendResult> ImageResultType;
    std::function<void(const ImageResultType &)> camera_6mm_callback =
        std::bind(&Widget::YoloResultDisplay, this, std::placeholders::_1,camera_names_[0]);

    front_6mm_result_reader_ = reader_node_->CreateReader(     
        input_camera_channel_names_[0],
        camera_6mm_callback);

    std::function<void(const ImageResultType &)> camera_12mm_callback =
        std::bind(&Widget::YoloResultDisplay, this, std::placeholders::_1,camera_names_[1]);

    front_12mm_result_reader_ = reader_node_->CreateReader(
        input_camera_channel_names_[1],
        camera_12mm_callback);

    //接收回放记录处理后的数据
    typedef std::shared_ptr<apollo::drivers::Image> ImageRecordType;
    std::function<void(const ImageRecordType &)> record_6mm_callback =
        std::bind(&Widget::RecordDisplay, this, std::placeholders::_1, camera_names_[0]);

    //回放里面是对通道名做了附加处理的,为了区分真实的通道与回放的通道
    front_6mm_record_reader_ = reader_node_->CreateReader(
        record_channelname_prefix +record_camera_channel_names_[0],
        record_6mm_callback);

    std::function<void(const ImageRecordType &)> record_12mm_callback =
        std::bind(&Widget::RecordDisplay, this, std::placeholders::_1, camera_names_[1]);

    front_12mm_record_reader_ = reader_node_->CreateReader(
        record_channelname_prefix + record_camera_channel_names_[1],
        record_12mm_callback);

    //建立处理回放的客户端
    file_q_client_ = reader_node_->CreateClient<FilesQueryParam, FilesAnswerParam>(
        interface_config.records_file_servicename());

    record_q_client_ = reader_node_->CreateClient<RecordQueryParam, RecordAnswerParam>(
        interface_config.records_play_servicename());


}

void Widget::initConnect()
{
    connect(form, &RecordForm::selectedRocordSignal, this, &Widget::playRecordSlot);
    connect(this, &Widget::signalDisplayImage, this,&Widget::DisplayInLabel);

}
//由于QLabel跟Qbutton不同，没有clicked信号，需要自己写事件机制
bool Widget::eventFilter(QObject *obj, QEvent *event)
{
    QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event); // 事件转换
    //点击子窗口
    if (obj == ui->label_sub) //指定某个QLabel
    {
        if (mouseEvent->button() == Qt::LeftButton)
        {
            if (mouseEvent->type() == QEvent::MouseButtonPress) //鼠标点击
            {
                mouseMovePosition_ = mouseEvent->globalPos() - ui->label_sub->pos();

                return true;
            }
            else if (mouseEvent->type() == QEvent::MouseButtonDblClick)
            {
                ChangeShow();
                return true;
            }
            else if (mouseEvent->type() == QEvent::MouseMove)
            {
                ui->label_sub->move(mouseEvent->globalPos() - mouseMovePosition_);
                return true;
            }
            else if (mouseEvent->type() == QEvent::MouseButtonRelease)
            {
                ui->label_sub->move(mouseEvent->globalPos() - mouseMovePosition_);
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    //点击回放按钮
    else if (obj == ui->label_record)
    {
        if (mouseEvent->button() == Qt::LeftButton)
        {
            if (mouseEvent->type() == QEvent::MouseButtonPress) //鼠标点击
            {
                ShowRecordWidget();
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        // pass the event on to the parent class
        return QWidget::eventFilter(obj, event);
    }
    return false;
}
void Widget::ShowRecordWidget()
{
    display_record_flag_ = false;
    //查询后台的回放文件
    auto files_query_msg = std::make_shared<FilesQueryParam>();
    files_query_msg->set_cmd_type(RecordCmdType::CMD_FILE_QUERY);
    files_query_msg->set_client_name("anonymous");
    std::chrono::seconds timeout(1);
    // apollo::cyber::Client<FilesQueryParam, FilesAnswerParam>::SharedResponse res =
    auto res =
        file_q_client_->SendRequest(files_query_msg, timeout);
    if (res == nullptr || res->files_size() == 0)
    {
        QMessageBox::warning(NULL,
                             "提示",
                             "后台无回放记录或回放服务未启动",
                             QMessageBox::Yes, QMessageBox::Yes);
    }
    else
    {
        //设置为modal窗口
        form->setWindowModality(Qt::ApplicationModal);
        //永远弹出在中间
        int diff_with = this->geometry().width() - form->geometry().width();
        int diff_height = this->geometry().height() - form->geometry().height();
        form->move(this->pos().x() + diff_with / 2, this->pos().y() + diff_height / 2);
        //根据实际文件个数添加显示区的按钮
        for (auto i = 0; i < res->files_size(); i++)
        {
            //已经完整有效的文件才显示出来
            if (res->files(i).is_completed() == false)
                continue;
            //创建按钮
            QString btnname = QString::fromStdString(res->files(i).start_time());
            QString filename = QString::fromStdString(res->files(i).file_name());
            //按回放记录的时间来设定按钮标题
            form->createPushbutton(btnname, filename);
        }
        form->show();
    }
}
void Widget::HideRecordWidget()
{
    form->close();
}

void Widget::playRecordSlot(const QString &filename)
{
    auto records_query_msg = std::make_shared<RecordQueryParam>();
    records_query_msg->set_cmd_type(RecordCmdType::CMD_RECORD_PLAY);
    records_query_msg->set_file_name(filename.toStdString());
    //以下播放参数保留，暂时不用
    records_query_msg->set_play_rate(1.0);
    records_query_msg->set_is_loop_playback(false);

    std::chrono::seconds timeout(1);
    auto res =
        record_q_client_->SendRequest(records_query_msg, timeout);
    //如果播放命名返回成功，就在切换主画面准备播放回放
    if (res->status() == true)
    {
        display_record_flag_ = true;
    }

    return;
}

void Widget::onStateChanged()
{

    display_style = !display_style;
}

Widget::~Widget()
{
    killTimer(timer_id_);
    delete form;
    delete ui;
}

void Widget::ChangeShow()
{
    // AERROR << "ChangeShow" << show_index_;
    show_index_++;
    if (show_index_ == 2)
        show_index_ = 0;
}

void Widget::DisplayWarning(void)
{

    //以最后出现的障碍物的时间开始，显示3秒
    warning_time_ = warning_time_ + 3;
    if (warning_time_ > 3)
    {
        warning_time_ = 3;
    }

    ui->label_warning->setVisible(true);
    display_warring_status_ = true;
}

void Widget::timerEvent(QTimerEvent *t)
{
    if (t->timerId() > 0)
    {
        warning_time_--;
        if (warning_time_ <= 0)
        {
            if (display_warring_status_)
            {
                ui->label_warning->setStyleSheet("QLabel{image: url(:/image/Icons/alarm_system.png); background-color: rgba(200, 200, 200,20%);}"); //图片在资源文件中
            }
            display_warring_status_ = false;
            warning_time_ = 0;
        }
    }

    if (display_warring_status_)
    {
        if (display_warring_flag_)
        {
            ui->label_warning->setStyleSheet("QLabel{image: url(:/image/Icons/alarm_system.png); background-color: rgba(200, 200, 200,20%);}"); //图片在资源文件中
        }
        else
        {
            ui->label_warning->setStyleSheet("QLabel{image: url(:/image/Icons/alarm_system_w.png); background-color: rgba(200, 200, 200,20%);}"); //图片在资源文件中
        }
        display_warring_flag_ = !display_warring_flag_;
    }
    //显示当前时间
    QDateTime time = QDateTime::currentDateTime();
    QString str = time.toString("yyyy-MM-dd  hh:mm:ss  dddd");
    ui->label_time->setText(str);
}

void Widget::YoloResultDisplay(const std::shared_ptr<watrix::projects::adas::proto::SendResult> &sync_result,const string &channel_name)
{

    std::lock_guard<std::mutex> lock(mutex_);
    // 不是播放回放，就正常播放
    if (display_record_flag_)
    {
        return;
    }
    uint screen_index = -1;
    for(auto i = 0; i <camera_names_.size();i++ )
    {
        if (channel_name == camera_names_[i])
        {
            screen_index = i;
        }
    }
    if(screen_index == -1) return;

    cv::Mat image_display_lane;
    cv::Mat image_display_seg;
    cv::Mat image_display;

    //安全线
	watrix::algorithm::cvpoint_t touch_point;
    //左轨道点
	watrix::algorithm::cvpoints_t left_fitted_lane_cvpoints;
    //右轨道点
	watrix::algorithm::cvpoints_t right_fitted_lane_cvpoints;

    touch_point.x = sync_result->perception_result().touch_point().x();
    touch_point.y = sync_result->perception_result().touch_point().y();
  
    left_fitted_lane_cvpoints.resize(sync_result->perception_result().left_fitted_lane_cvpoints_size());
    right_fitted_lane_cvpoints.resize(sync_result->perception_result().right_fitted_lane_cvpoints_size());
   //长焦的左轨道线
    for(auto  i =0; i<  sync_result->perception_result().left_fitted_lane_cvpoints_size();i++ )
    {
        auto pb_point_left = sync_result->perception_result().left_fitted_lane_cvpoints(i);
        left_fitted_lane_cvpoints[i].x = pb_point_left.x();
        left_fitted_lane_cvpoints[i].y = pb_point_left.y();
    }
    for(auto  i =0; i<  sync_result->perception_result().right_fitted_lane_cvpoints_size();i++ )
    {
        auto pb_point_right = sync_result->perception_result().right_fitted_lane_cvpoints(i);
        right_fitted_lane_cvpoints[i].x = pb_point_right.x();
        right_fitted_lane_cvpoints[i].y = pb_point_right.y();
    }

  


    uint net_time = image_tools_.GetMillisec();
    uint source_image_height = sync_result->source_image().height();
    uint source_image_width = sync_result->source_image().width();
    uint source_image_step = sync_result->source_image().step();

    //root_image 初始化为一个黑色分辨率的底图
    image_display = Mat::zeros(source_image_height, source_image_width, CV_8UC3);
    if ((source_image_height > 0) && (source_image_width > 0))
    {
        long source_img_length = source_image_height * source_image_step;
        memcpy(image_display.data, (void *)sync_result->source_image().data().c_str(), source_img_length);
    }

    // AERROR << "ResaultDisplay.....get image id==" << screen_index;
    uint seg_image_height = sync_result->seg_binary_mask().height();
    uint seg_image_width = sync_result->seg_binary_mask().width();
    uint seg_image_step = sync_result->seg_binary_mask().step();

    //copy seg data and convers to 3ch
    if ((seg_image_height > 0) && (seg_image_width > 0))
    {
        long seg_img_length = seg_image_height *seg_image_step;
        Mat seg_binary_image = Mat(seg_image_height, seg_image_width, CV_8UC1);
        memcpy(seg_binary_image.data, (void *)sync_result->seg_binary_mask().data().c_str(), seg_img_length);
        image_display_seg = Mat::zeros(seg_image_height, seg_image_width, CV_8UC3);
        image_display_seg = ConvertTo3Channels(seg_binary_image); 

        cv::addWeighted(image_display_seg, 0.2, image_display, 0.8, -1, image_display);
    }

      DisplayUtil::show_lane(image_display, 
      left_fitted_lane_cvpoints, 
      right_fitted_lane_cvpoints,
	  touch_point, 
      true, 
      true);


    QImage img = image_tools_.cvMat2QImage(image_display);

    emit signalDisplayImage(img,screen_index);

    watrix::projects::adas::proto::MaxSafeDistance max_safe_ditance_ = sync_result->max_safe_distance();

    int dis = max_safe_ditance_.image_distance();

    std::string mix_dis = std::to_string(dis);

    //d = V * (T1 + T2) + V*V/2a
    //d max_distance T1为警告后司机的最大制动反应时间 0.8
    //T2为司机制动后的列车中延迟 0.7
    //紧急制动最小减速度a为 1.2m/s2
    float sqrt1 = 2.25 + 9.6 * dis;
    float s1 = std::sqrt(sqrt1) / 4.8 - 0.3125;
    float s2 = -std::sqrt(sqrt1) / 4.8 - 0.3125;
    float max_speed = std::max(s1, s2);
    if (dis > 0)
    {
        std::string speed = image_tools_.GetFloatRound(max_speed, 2) + "  m/s";
        ui->label_speed->setText(speed.c_str());
        ui->label_max_dis->setText(mix_dis.c_str());
    }
    receive_image_counter_++;

}


void Widget::RecordDisplay(const std::shared_ptr<apollo::drivers::Image> &record, const string &channel_name)
{
    std::lock_guard<std::mutex> lock(record_mutex_);
    if (!display_record_flag_)
    {
        return;
    }
    uint screen_index = -1;
    for(auto i = 0; i <camera_names_.size();i++ )
    {
        if (channel_name == camera_names_[i])
        {
            screen_index = i;
        }
    }
    if(screen_index == -1) return;

   
    //root_image 初始化为一个黑色分辨率的底图、
    cv::Mat image_display;

    uint source_image_height = record->height();
    uint source_image_width = record->width();
    image_display = Mat::zeros(source_image_height, source_image_width, CV_8UC3);
    if ((source_image_height > 0) && (source_image_width > 0))
    {
        memcpy(image_display.data, 
                         (void *)record->data().c_str(), 
                        source_image_height * source_image_width * 3);
    } 

    //将RGB图像转换为BGR
    cv::cvtColor(image_display, image_display, CV_RGB2BGR); 

    //图像要缩小
    cv::resize(image_display, image_display, cv::Size(1280, 800));

    //把时间放到回放的图像上
    apollo::cyber::Time time_stamp(record->header().timestamp_sec());

    std::string format_str = R"(
      camera name:    %s
      frame id:    %s
      recorder time : %s )";

        std::string display_text =
            boost::str(boost::format(format_str.c_str()) 
            % camera_names_[screen_index]
             % record->header().frame_id()
             %  time_stamp.ToString());

    

    cv::putText(image_display, display_text, cv::Point2i (10, 50),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 0), 2);

    QImage img = image_tools_.cvMat2QImage(image_display);
    emit signalDisplayImage(img,screen_index);
}

void Widget::DisplayInLabel(const QImage play_img, int index)
{

    // QImage img = image_tools_.cvMat2QImage(play_img);
    QPixmap pixmap = QPixmap::fromImage(play_img);
    if (!pixmap.isNull())
    {
        if (show_index_ == 0)
        {
            if (index == 0)
            {
                pixmap = pixmap.scaled(MAX_SCALED_WIDTH, MAX_SCALED_HEIGHT, Qt::IgnoreAspectRatio, Qt::FastTransformation); //设置图  FastTransformation SmoothTransformation
                ui->label_main->setPixmap(pixmap);
            }
            else
            {
                pixmap = pixmap.scaled(ui->label_sub->width() - 2, ui->label_sub->height() - 2, Qt::IgnoreAspectRatio, Qt::FastTransformation); //设置图  FastTransformation SmoothTransformation
                ui->label_sub->setPixmap(pixmap);
            }
        }
        else
        {
            if (index == 0)
            {
                pixmap = pixmap.scaled(ui->label_sub->width() - 2, ui->label_sub->height() - 2, Qt::IgnoreAspectRatio, Qt::FastTransformation); //设置图  FastTransformation SmoothTransformation
                ui->label_sub->setPixmap(pixmap);
            }
            else
            {
                pixmap = pixmap.scaled(MAX_SCALED_WIDTH, MAX_SCALED_HEIGHT, Qt::IgnoreAspectRatio, Qt::FastTransformation); //设置图  FastTransformation SmoothTransformation
                ui->label_main->setPixmap(pixmap);
            }
        }
    }
}

cv::Mat Widget::ConvertTo3Channels(const cv::Mat &binImg)
{
    Mat three_channel = Mat::zeros(binImg.rows, binImg.cols, CV_8UC3);
    vector<Mat> channels;
    for (int i = 0; i < 3; i++)
    {
        channels.push_back(binImg);
    }
    merge(channels, three_channel);
    return three_channel;
}
