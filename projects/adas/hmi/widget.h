#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QFileDialog>
#include <QList>
#include <QSpinBox>
#include <QToolButton>
#include <QToolBar>
#include <QCheckBox>
#include <QLabel>
#include <QGridLayout>
#include <QDebug>
#include <QMessageBox>
#include <QDockWidget>
#include <QTextEdit>
#include <QLineEdit>
#include <QMenu>
#include <QTimerEvent>
#include <QTime>
#include <QtMultimedia>
#include <QMouseEvent>
#include <QPoint>
// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// std
#include <vector>

//#include "threadpool.h"

#include "util/imagetools.h"
#include "util/soundtools.h"

#include "cyber/cyber.h"
#include "modules/drivers/proto/sensor_image.pb.h"
// protobuf message to publish and subscribe
#include "projects/adas/proto/point_cloud.pb.h"
#include "projects/adas/proto/camera_image.pb.h"


// #include "../build/src/ui_widget.h"

#define PACKAGE_USED 1
#define PACKAGE_UNUSED 0
#define PACKAGE_RECEIVE_BEGIN 0
#define PACKAGE_RECEIVE_ING 1
#define PACKAGE_RECEIVE_END 2
#define ONE_SCREEN 1
#define TWO_SCREEN 2
#define THREE_SCREEN 3
#define FOUR_SCREEN 4
#define CAMERA_TYPE_SHORT 1
#define CAMERA_TYPE_LONG 2

using namespace watrix::proto;
using namespace watrix::util;
using apollo::cyber::Node;


typedef struct{
    int camera_index;
    int detect_status;
}whice2display;

typedef struct{
    int distance;
    int confidence;
}maxDistance;

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();
public:
    bool eventFilter(QObject *obj, QEvent *event);	// 添加时间过滤器声明
private:
    Ui::Widget *ui;

 
private slots:
    void onStateChanged();

/////////////////////////////////
/////////////////////////////////
private:
    ImageTools image_tools_;

    //网络socket
    std::shared_ptr<Node> reader_node_ = nullptr;
    std::shared_ptr<apollo::cyber::Reader<watrix::proto::SendResult>> 
    front_6mm_result_reader_ = nullptr;

std::shared_ptr<apollo::cyber::Reader<watrix::proto::SendResult>> 
     front_12mm_result_reader_ = nullptr ;

    std::mutex mutex_;

    QLineEdit *ip_edit_;
    QPushButton *connect_btn_;
    QLineEdit *save_path_edit_;
    QPushButton *save_path_btn_; 
    std::string save_image_path_; 
    bool save_image_flags_ = false;  
    //网络数据包相关结构
    uint32_t transfer_time_=0;//计算发送端，到接收到的时间    
    uint32_t img_size_;//信息的实际有效长度
    char * get_info_buf_;//有效信息的buffer存储
    long get_info_buf_index_=0;
    uint32_t packet_size_;
    uint32_t curr_posize_;//当前收到的有效信息的位置 （一个有效信息可能由几个数据包组成）
    uint8_t  packet_type_;//信息的类型 POINT_CLOUD  CAMERA_IMAGE  TRAIN_SEG_RESULT  YOLO_DETECTION_RESULT
    
    //显示结果控制
    long start_time_=0;
    bool display_warring_flag_=false;
    bool display_warring_status_=false;
    uint receive_image_counter_=0;
    int sync_display_timer_=0;
    
    QPoint  mouseMovePosition_;
    void setup_connect();
    void DisplayWarning();
    void DisplayInLabel(const cv::Mat& play_img, int index);
    void timerEvent(QTimerEvent *);
    int warning_time_ = 0;;
    cv::Mat ConvertTo3Channels(const cv::Mat& binImg);

    void YoloResaultDisplay(const std::shared_ptr<watrix::proto::SendResult> &sync_result);
    void LidarResaultDisplay(char *get_img_buf , uint32_t img_size);
    int DrawObjectBox(const watrix::proto::DetectionBoxs &detection_boxs, const cv::Mat &input_mat, int camera_tpye, 
                            std::vector<cv::Point2i> &distance_box_centers);
    void PlaySound( );
    void PlaySoundThread();
    QMediaPlayer * sound_player_;
    QString sound_filePath_;
    bool sound_play_flag=false;

    whice2display  camera_1;
    whice2display  camera_2;
    maxDistance max_distance_;
    int show_index_;
private slots:
    void ChangeShow();

////////////////////////////////

    void on_pushButton_save_clicked();

};


#endif // WIDGET_H
