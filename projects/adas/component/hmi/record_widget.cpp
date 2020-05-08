
#include "record_widget.h"
#include "projects/adas/component/hmi/ui_recordlist.h"
#include "ui_util.h"

//根据现有的个数计算新按钮的布局
static const int btn_num_per_row = 5;
 QRect RecordForm::getNewButtonGeometry()
{
    int btn_close_height = ui->btn_close->geometry().height();
    int  paint_height =  this->geometry().height() - btn_close_height;
    int  paint_width = this->geometry().width();
    int  btn_height = paint_height/btn_num_per_row;
    int  btn_width = paint_width/btn_num_per_row;

    int exist_btn_size = recordButtons_.size();

    QRect rect;
    rect.setHeight(btn_height);
     rect.setWidth(btn_width);  
     rect.setTop(this->geometry().top()+ btn_close_height+exist_btn_size/btn_num_per_row );
    rect.setLeft(this->geometry().left() +(exist_btn_size % btn_num_per_row) * btn_width ); 
    return rect;
}

RecordForm::RecordForm(QWidget *parent) : QWidget(parent),
                                          ui(new Ui::RecordForm)
{
    this->setWindowFlags(Qt::FramelessWindowHint);
    ui->setupUi(this);
    ui->btn_close->setText(tr("返回"));
    ui->btn_close->setFlat(true);


    //返回按钮来关闭
    connect(ui->btn_close, &QPushButton::clicked, this, &QWidget::close);

    signalMapper_ = new QSignalMapper(this);


    connect(signalMapper_, SIGNAL(mapped(QWidget *)), this, SLOT(slotSignalMap(QWidget *)));

    // this->setAttribute(Qt::WA_TranslucentBackground, true);
    // this->setStyleSheet("RecordForm{ background-color: transparent;}");
}
 void RecordForm::showEvent(QShowEvent *e)
{
    return QWidget::showEvent(e);
}
//动态生成的按钮，需要在关闭前做一些处理
 void RecordForm::closeEvent(QCloseEvent * e)
{

    recordButtons_.clear();
    recordFiles_.clear();
    return QWidget::closeEvent(e);
}
//将所有的按钮点击事件关连到一个槽函数上
void RecordForm::slotSignalMap(QWidget *widget)
{
    QPushButton * btn = qobject_cast<QPushButton *>(widget);

    if(recordFiles_.find(btn->text()) != recordFiles_.end())
    {
        //发送按钮关连的文件名
        emit      selectedRocordSignal(recordFiles_[btn->text()]);
    }

    //关闭自己
    this->close();

    return;
 
}

//创建按钮
void RecordForm::createPushbutton(QString btnname,QString filename)
 {
      QPushButton *btn  = new QPushButton();

     if(btn != nullptr) 
     {
        btn->setGeometry(getNewButtonGeometry());

         //多信号连接
        connect(btn, SIGNAL(clicked()), signalMapper_, SLOT(map()));
         signalMapper_->setMapping(btn,btn);
        //添加按钮
        ui->layout_grid_records->addWidget(btn,recordButtons_.size()/btn_num_per_row,recordButtons_.size()%btn_num_per_row);
        btn->setText(btnname);
        btn->setFlat(true);
        recordButtons_.push_back(QSharedPointer<QPushButton>(btn));
        //设置按钮与文件名的Map
        recordFiles_[btnname] = filename;
     }
 }

RecordForm::~RecordForm()
{
    recordButtons_.clear();
    delete ui;
}