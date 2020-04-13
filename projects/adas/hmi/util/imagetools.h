#pragma once


#include <QImageReader>
// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// Boost
#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <QDebug>
#include <iostream>
#include <cassert>
using namespace  std;
namespace watrix {
	
	namespace util{

        class  ImageTools
        {
        public:
            ImageTools();

            QImage cvMat2QImage(const cv::Mat& mat);
            long GetMillisec();
            int addAlpha(cv::Mat& src, cv::Mat& dst, cv::Mat& alpha);
            std::string GetFloatRound(double fValue, int bits);
        };
    }
}


