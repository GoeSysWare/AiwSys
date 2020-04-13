#include "widget.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
      apollo::cyber::Init(argv[0]);
    Widget w;
    w.show();

    return a.exec();
}
