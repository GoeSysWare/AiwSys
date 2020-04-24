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

#include "widget.h"
#include "projects/adas/hmi/record_widget.h"

#include "algorithm/monocular_distance_api.h"

#include "projects/adas/hmi/ui_widget.h"

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

    reader_node_ = apollo::cyber::CreateNode("adas_perception_hmi_2");

    //接收算法处理后的result数据
    typedef std::shared_ptr<watrix::proto::SendResult> ImageResultType;
    std::function<void(const ImageResultType &)> camera_callback =
        std::bind(&Widget::YoloResaultDisplay, this, std::placeholders::_1);
    front_6mm_result_reader_ = reader_node_->CreateReader(
        "adas/camera/result_6mm",
        camera_callback);
    front_12mm_result_reader_ = reader_node_->CreateReader(
        "adas/camera/result_12mm",
        camera_callback);

    //接收回放记录处理后的数据
    typedef std::shared_ptr<apollo::drivers::Image> ImageRecordType;
    std::function<void(const ImageRecordType &)> record_6mm_callback =
        std::bind(&Widget::RecordDisplay, this, std::placeholders::_1, std::string("front_6mm"));

    front_6mm_record_reader_ = reader_node_->CreateReader(
        "records/adas/camera/front_6mm/image",
        record_6mm_callback);

    std::function<void(const ImageRecordType &)> record_12mm_callback =
        std::bind(&Widget::RecordDisplay, this, std::placeholders::_1, std::string("front_12mm"));

    front_12mm_record_reader_ = reader_node_->CreateReader(
        "records/adas/camera/front_12mm/image",
        record_12mm_callback);

    //建立处理回放的客户端
    file_q_client_ = reader_node_->CreateClient<FilesQueryParam, FilesAnswerParam>(
        "record_service_file");

    record_q_client_ = reader_node_->CreateClient<RecordQueryParam, RecordAnswerParam>(
        "record_service_record");

    // 安装事件过滤器
    ui->label_sub->installEventFilter(this);
    ui->label_record->installEventFilter(this);

    show_index_ = 0;
    //初始化信号槽
    initConnect();
}
void Widget::initConnect()
{
    connect(form, &RecordForm::selectedRocordSignal, this, &Widget::playRecordSlot);
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

void Widget::YoloResaultDisplay(const std::shared_ptr<watrix::proto::SendResult> &sync_result)
{

    std::lock_guard<std::mutex> lock(mutex_);
    // 不是播放回放，就正常播放
    if (display_record_flag_)
    {
        return;
    }

    cv::Mat image_display_lane_;
    cv::Mat image_display_seg_;
    cv::Mat image_display_;
    int display_who_ = 1;

    std::vector<cv::Point2i> distance_box_centers_;
    uint screen_index = 0;
    uint net_time = image_tools_.GetMillisec();

    uint source_image_height = sync_result->source_image().height();
    uint source_image_width = sync_result->source_image().width();
    uint source_image_channel = sync_result->source_image().channel();
    screen_index = sync_result->source_image().camera_id();
    int frame_count = sync_result->source_image().frame_count();
    int frame_time = sync_result->source_image().timestamp_msec();
    //root_image 初始化为一个黑色分辨率的底图
    image_display_ = Mat::zeros(source_image_height, source_image_width, CV_8UC3);
    if ((source_image_height > 0) && (source_image_width > 0))
    {
        long source_img_length = source_image_height * source_image_width * source_image_channel;
        memcpy(image_display_.data, (void *)sync_result->source_image().data().c_str(), source_img_length);
    }

    // AERROR << "ResaultDisplay.....get image id==" << screen_index;
    uint seg_image_height = sync_result->seg_binary_mask().height();
    uint seg_image_width = sync_result->seg_binary_mask().width();
    uint seg_image_channel = sync_result->seg_binary_mask().channel();
    //copy seg data and convers to 3ch
    if ((seg_image_height > 0) && (seg_image_width > 0))
    {
        long seg_img_length = seg_image_height * seg_image_width * seg_image_channel;
        Mat seg_binary_image = Mat(seg_image_height, seg_image_width, CV_8UC1);
        memcpy(seg_binary_image.data, (void *)sync_result->seg_binary_mask().data().c_str(), seg_img_length);
        image_display_seg_ = Mat::zeros(seg_image_height, seg_image_width, CV_8UC3);
        image_display_seg_ = ConvertTo3Channels(seg_binary_image); //image_display_seg_ = OpencvUtil::merge_mask(image_display_seg_, seg_binary_image, 255, 255, 0);

        cv::addWeighted(image_display_seg_, 0.2, image_display_, 0.8, -1, image_display_);
    }

    DisplayInLabel(image_display_, screen_index);
    // ui->label_speed->setVisible(true);
    // ui->label_max_dis->setVisible(true);
    watrix::proto::MaxSafeDistance max_safe_ditance_ = sync_result->max_safe_distance();

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
    uint screen_index = 0;
    if (channel_name == "front_6mm")
    {
        screen_index = 0;
    }
    else if (channel_name == "front_12mm")
    {
        screen_index = 1;
    }
    else
    {
        return;
    }
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
    DisplayInLabel(image_display, screen_index); 
}

void Widget::DisplayInLabel(const cv::Mat &play_img, int index)
{

    QImage img = image_tools_.cvMat2QImage(play_img);
    QPixmap pixmap = QPixmap::fromImage(img);
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
