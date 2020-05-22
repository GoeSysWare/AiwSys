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
#include <QAtomicInt>
// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// std
#include <vector>

//#include "threadpool.h"

#include "util/imagetools.h"

#include "cyber/cyber.h"
#include "modules/drivers/proto/sensor_image.pb.h"
// protobuf message to publish and subscribe

#include "projects/adas/proto/adas_record.pb.h"
#include "projects/adas/proto/adas_detection.pb.h"
#include "projects/adas/proto/adas_camera.pb.h"
#include "projects/adas/proto/adas_perception.pb.h"
#include "projects/adas/proto/adas_simulator.pb.h"
#include "projects/adas/proto/adas_config.pb.h"
#include "projects/adas/configs/config_gflags.h"
#include "projects/adas/configs/config_gflags.h"
#include "projects/adas/component/common/util.h"

#include "record_widget.h"

using namespace watrix::projects::adas::proto;
using namespace watrix::util;
using apollo::cyber::Node;

namespace Ui
{
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();

public:
    bool eventFilter(QObject *obj, QEvent *event); // 添加时间过滤器声明
    void initConnect();
    void initNetwork();
private:
    Ui::Widget *ui;

    RecordForm *form;
public slots:
    void playRecordSlot(const QString &filename);
    void DisplayInLabel(const QImage play_img, int index);

private slots:
    void onStateChanged();
    void ChangeShow();
signals:
    void signalDisplayImage(const QImage  play_img, int index);
    /////////////////////////////////
    /////////////////////////////////
private:
    ImageTools image_tools_;
    std::vector<std::string> input_camera_channel_names_;
    std::vector<std::string> record_camera_channel_names_;
    std::vector<std::string> camera_names_;
    std::shared_ptr<apollo::cyber::Node> reader_node_ = nullptr;

    std::shared_ptr<apollo::cyber::Reader<watrix::projects::adas::proto::SendResult>>
        front_6mm_result_reader_ = nullptr;

    std::shared_ptr<apollo::cyber::Reader<watrix::projects::adas::proto::SendResult>>
        front_12mm_result_reader_ = nullptr;

    std::shared_ptr<apollo::cyber::Reader<apollo::drivers::Image>>
        front_6mm_record_reader_ = nullptr;
    std::shared_ptr<apollo::cyber::Reader<apollo::drivers::Image>>
        front_12mm_record_reader_ = nullptr;

    std::shared_ptr<apollo::cyber::Client<FilesQueryParam, FilesAnswerParam>> file_q_client_ = nullptr;
    std::shared_ptr<apollo::cyber::Client<RecordQueryParam, RecordAnswerParam>> record_q_client_ = nullptr;
    std::mutex mutex_;
    std::mutex record_mutex_;

    //显示结果控制
    bool display_record_flag_ = false;
    bool display_warring_flag_ = false;
    bool display_warring_status_ = false;
    uint receive_image_counter_ = 0;

    QPoint mouseMovePosition_;
    void DisplayWarning();
    void timerEvent(QTimerEvent *);

    void ShowRecordWidget();

    void HideRecordWidget();

    int warning_time_ = 0;
    int timer_id_;
    cv::Mat ConvertTo3Channels(const cv::Mat &binImg);
    void YoloResaultDisplay(const std::shared_ptr<watrix::projects::adas::proto::SendResult> &sync_result,const string &channel_name);

    void RecordDisplay(const std::shared_ptr<apollo::drivers::Image> &record,const string & channel_name);


    int show_index_;
};

#endif // WIDGET_H
