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
#include <glog/logging.h>

#include "projects/adas/hmi/ui_widget.h"



using namespace watrix::algorithm;
using namespace cv;
using namespace std;

#define MIN_SCALED_WIDTH (320 - 6)
#define MIN_SCALED_HEIGHT (200 - 6) //(150-6)
#define MAX_SCALED_WIDTH (1280)     //(780-6)
#define MAX_SCALED_HEIGHT (780 - 6) //(590-6)
#define WARRING_SOUND "../data/warring.wav"
#define SAVE_FILE_PATH "../data/save/image/"
#define LOG_FILE_PATH "./log"

bool display_style = true;
#define LABEL_LONG "long camera"
#define LABEL_SHORT "short camera"

int current_screen_width = 0;
int current_screen_height = 0;

Widget::Widget(QWidget *parent) : QWidget(parent),
                                  ui(new Ui::Widget)
{
    // google::InitGoogleLogging("hdmi");
    FLAGS_log_dir = LOG_FILE_PATH;
    boost::filesystem::create_directories(LOG_FILE_PATH);

    this->setWindowTitle("自动驾驶显示终端");
    this->setWindowIcon(QIcon(":/image/Icons/app_logo.png"));
    ui->setupUi(this);
    //this->setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint);
    this->setWindowFlags(Qt::Dialog);
    //this->setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    //this->showFullScreen();

    //网络链接
    // autocreate talker node
    reader_node_ = apollo::cyber::CreateNode("adas_perception_hmi");
    // create talker
    typedef std::shared_ptr<watrix::proto::SendResult> ImageResultType;

    std::function<void(const ImageResultType &)> camera_callback =
        std::bind(&Widget::YoloResaultDisplay, this,std::placeholders::_1);

    front_6mm_result_reader_ = reader_node_->CreateReader(
        "adas/camera/result_6mm",
        camera_callback);
    front_12mm_result_reader_ = reader_node_->CreateReader(
        "adas/camera/result_12mm",
         camera_callback);


    ui->label_sub->installEventFilter(this); // 安装事件过滤器

    qsrand(time(0));
    QtConcurrent::run(this, &Widget::PlaySoundThread);
    startTimer(1000);

    save_image_flags_ = true; //该版本直接保存所有收到的结果
    QDateTime dir_cre_time = QDateTime::currentDateTime();
    QString str_time = dir_cre_time.toString("yyyy-MM-dd-hh.mm.ss");
    save_image_path_ = SAVE_FILE_PATH + str_time.toStdString() + "/";
    std::cout << save_image_path_ << std::endl;
    boost::filesystem::create_directories(save_image_path_.c_str());

    {
        //std::string s_path =  "/home/wayne/project/trainpilot-hdmi/bin/warring.wav";
        std::string s_path = ":/image/sound/warring.wav";
        sound_filePath_ = QDir::currentPath() + "/warring.wav";
        std::cout << sound_filePath_.toStdString() << std::endl;
        QFile file(sound_filePath_);
        if (!file.open(QIODevice::ReadOnly))
        {
            // AERROR << "Could not open file";
            sound_play_flag = false;
        }
        else
        {
            sound_player_ = new QMediaPlayer;
            sound_player_->setMedia(QUrl::fromLocalFile(sound_filePath_));
            sound_player_->setVolume(90);
            sound_player_->play();
            sound_play_flag = true;
        }
    }
    camera_1.camera_index = 0;
    camera_1.detect_status = 0;
    camera_2.camera_index = 0;
    camera_2.detect_status = 0;
    show_index_ = 0;

    max_distance_.distance = 0;
    max_distance_.confidence = 0;
}

bool Widget::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == ui->label_sub) //指定某个QLabel
    {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event); // 事件转换
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
    else
    {
        // pass the event on to the parent class
        return QWidget::eventFilter(obj, event);
    }
    return false;
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

void Widget::PlaySound(void)
{
    if (sound_play_flag)
    {

        if (sound_player_->mediaStatus() == QMediaPlayer::NoMedia)
        {
            sound_player_->setMedia(QUrl::fromLocalFile(sound_filePath_));
            sound_player_->play();
        }
        else if (sound_player_->state() == QMediaPlayer::PausedState)
        {
            sound_player_->play();
        }
        else if (sound_player_->state() == QMediaPlayer::PlayingState)
        {
            // AINFO << " is playing";
        }
        sound_player_->setPosition(1);
        sound_player_->play();
    }
}

bool play_flag_ = true;
void Widget::PlaySoundThread(void)
{
    int play = 0;
    while (1)
    {
        if (warning_time_ > 1)
        {
            //if(play_flag_){
            //play = 0;
            // AINFO << "PlaySound----" << warning_time_ << endl;
            PlaySound();
            //std::string s_path =  "../data/warring.wav";
            //play = QtConcurrent::run(set_pcm_play, s_path);//WARRING_SOUND);
            //play_flag_ = false;
            //}
        }
        sleep(1);
        // if(play == 1){
        //     play_flag_ = true;
        // }
    }
}

void Widget::DisplayWarning(void)
{
    //if(warning_time_==0){
    //    startTimer(3000);
    //}
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
    //cout<<"----"<<t->timerId()<<endl;
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
            //cout<<"警告结束----"<<warning_time_<<endl;
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

    QDateTime time = QDateTime::currentDateTime();
    QString str = time.toString("yyyy-MM-dd  hh:mm:ss  dddd");
    ui->label_time->setText(str);
}

int Widget::DrawObjectBox(const watrix::proto::DetectionBoxs &detection_boxs, const cv::Mat &input_mat, int camera_tpye, std::vector<cv::Point2i> &distance_box_centers)
{
    //水平有效-5,5 前方有效15-60米，无效返回1000,展示时需要避免
    int invasion_object = 0;
    uint objects;
    bool success = false;
    for (objects = 0; objects < detection_boxs.boxs_size(); objects++)
    {
        int invasion_status = detection_boxs.boxs(objects).invasion_status();

        if (invasion_status == 1)
        {
            DisplayWarning();
            invasion_object++;
        }
        continue;
        int x_start = detection_boxs.boxs(objects).xmin() * 0.67;
        int x_end = detection_boxs.boxs(objects).xmax() * 0.67;
        int y_start = detection_boxs.boxs(objects).ymin() * 0.74;
        int y_end = detection_boxs.boxs(objects).ymax() * 0.74;
        int centre_point = (x_end - x_start) / 2 + x_start;

        // AINFO << "中心点   x:" << centre_point << "   y: " << y_end;
        std::string pose_text;
        float text_x = detection_boxs.boxs(objects).distance().x();
        float text_y = detection_boxs.boxs(objects).distance().y();

        cv::Point2i center(centre_point, y_end);
        distance_box_centers.push_back(center);
        pose_text = "x:" + image_tools_.GetFloatRound(text_x, 2) + " y:" + image_tools_.GetFloatRound(text_y, 2);

        if (invasion_status == 1)
        {
            //sound_tools_.set_pcm_play(WARRING_SOUND);
            DisplayWarning();
            rectangle(input_mat, cvPoint(x_start, y_start), cvPoint(x_end, y_end),
                      cvScalar(0, 0, 255), 2, 4, 0); //red
            invasion_object++;
        }
        else if (invasion_status == 0)
        {
            rectangle(input_mat, cvPoint(x_start, y_start), cvPoint(x_end, y_end),
                      cvScalar(0, 255, 0), 2, 4, 0); //green
        }
        else if (invasion_status == -1)
        {
            // rectangle(input_mat, cvPoint(x_start, y_start), cvPoint(x_end, y_end),
            //           cvScalar(0, 255, 255), 2, 4, 0); //yellow
        }

        std::string confidence = image_tools_.GetFloatRound(detection_boxs.boxs(objects).confidence(), 2);
        std::string invasion_dis = image_tools_.GetFloatRound(detection_boxs.boxs(objects).invasion_distance(), 2);
        std::string class_name = detection_boxs.boxs(objects).class_name();
        std::string display_text = class_name + confidence + "  in_dis:" + invasion_dis;

        uint32_t x_top = x_start;
        //   if(x_top >= 1220){
        //       x_top = 1220;
        //   }
        cv::Point2i origin(x_top, y_start - 10);
        cv::putText(input_mat, display_text, origin,
                    cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(0, 255, 255), 2, 8, 0);

        uint32_t y_bottom = y_end + 20;
        if (y_bottom >= 780)
        {
            y_bottom = 780;
        }
        cv::Point2i origin1(x_top, y_bottom);
        cv::putText(input_mat, pose_text, origin1,
                    cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(0, 255, 255), 2, 8, 0);
    }

    return invasion_object;
}

void Widget::YoloResaultDisplay(const std::shared_ptr<watrix::proto::SendResult> &sync_result)
{

      std::lock_guard<std::mutex> lock(mutex_);
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

    //当一个相机中出现障碍物，需要停止1秒，或者5帧，避免反复切换，程序down
    int invasion_object = DrawObjectBox(sync_result->detection_boxs(), image_display_, screen_index, distance_box_centers_);

    DisplayInLabel(image_display_, screen_index);
    if (save_image_flags_)
    {
        std::string image_path = save_image_path_ + std::to_string(screen_index) + "_" + std::to_string(10000000 + frame_count) + "_" + std::to_string(frame_time) + ".jpg";
        cv::imwrite(image_path, image_display_);
    }

    watrix::proto::MaxSafeDistance max_safe_ditance_ = sync_result->max_safe_distance();
    // AERROR << "max_safe_ditance_     : " << max_safe_ditance_.image_distance();
    int dis = 0;

    dis = max_safe_ditance_.image_distance();
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

    sync_display_timer_--;
    receive_image_counter_++;

    // AERROR << sync_display_timer_ << "<<<<<<<<<  " << receive_image_counter_ << "  显示结束,网络接收=" << (net_time - start_time_) << "  总共耗时=" << image_tools_.GetMillisec() - start_time_;
}

void Widget::DisplayInLabel(const cv::Mat &play_img, int index)
{
    // AERROR << "DisplayInLabel.....get image id==" << index << endl;
    // AERROR << "DisplayInLabel.....show index id==" << show_index_ << endl;
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
    // 设置图  FastTransformation SmoothTransformation  IgnoreAspectRatio  KeepAspectRatio
    // pixmap2 = pixmap2.scaled(MIN_SCALED_WIDTH, MIN_SCALED_HEIGHT, Qt::IgnoreAspectRatio,Qt::FastTransformation);//设置图
    // ui->label_top->setPixmap(pixmap2);
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

void Widget::on_pushButton_save_clicked()
{
    if (!save_image_flags_)
    {
        save_image_path_ = save_path_edit_->text().toStdString();
        save_path_btn_->setText("存储中");
        //FilesystemUtil::mkdir(save_image_path_);
        boost::filesystem::create_directories(save_image_path_);
        std::cout << save_image_path_ << std::endl;
    }
    else
    {
        save_path_btn_->setText("保存");
    }
    save_image_flags_ = !save_image_flags_;
}
