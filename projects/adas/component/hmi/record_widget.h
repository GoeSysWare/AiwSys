#ifndef RECORD_H
#define RECORD_H
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
#include <QPushButton>
#include <QVector>
#include <QMap>
#include <memory>
#include <memory>
namespace Ui
{
class RecordForm;
}

class RecordForm : public QWidget
{
    Q_OBJECT
public:
    explicit RecordForm(QWidget *parent = 0);
    ~RecordForm();
    void createPushbutton(QString btnname,QString filename);

    QMap<QString,QString> recordFiles_;
private:
    Ui::RecordForm *ui;
    QSignalMapper *signalMapper_;
    QRect getNewButtonGeometry();
    //QSharedPointer 智能指针就不需要再遍历二次delete了，只需要clear
    QVector<QSharedPointer<QPushButton>> recordButtons_;

    virtual void showEvent(QShowEvent *e);
    virtual void closeEvent(QCloseEvent *e);
private Q_SLOTS:
    void slotSignalMap(QWidget *widget);

//释放信号播放某个回放文件
// Q_SIGNALS:
signals:
    void selectedRocordSignal(const QString &title);
};

#endif