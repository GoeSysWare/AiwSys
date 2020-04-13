#include "imagetools.h"
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;
namespace watrix {
	
	namespace util{

        ImageTools::ImageTools()
        {

        }

        long ImageTools::GetMillisec()
        {
            boost::posix_time::ptime start_time =boost::posix_time::microsec_clock::local_time();
            const boost::posix_time::time_duration td = start_time.time_of_day();
            long millisecond = td.total_milliseconds();// - ((td.hours() * 3600 + td.minutes() * 60 + td.seconds()) * 1000) + td.seconds()*1000;
            return millisecond;
        }

        int ImageTools::addAlpha(cv::Mat& src, cv::Mat& dst, cv::Mat& alpha)
        {
            if (src.channels() == 4)
            {
                return -1;
            }
            else if (src.channels() == 1)
            {
                cv::cvtColor(src, src, cv::COLOR_GRAY2RGB);
            }

            dst = cv::Mat(src.rows, src.cols, CV_8UC4);

            std::vector<cv::Mat> srcChannels;
            std::vector<cv::Mat> dstChannels;
            //分离通道
            cv::split(src, srcChannels);

            dstChannels.push_back(srcChannels[0]);
            dstChannels.push_back(srcChannels[1]);
            dstChannels.push_back(srcChannels[2]);
            //添加透明度通道
            dstChannels.push_back(alpha);
            //合并通道
            cv::merge(dstChannels, dst);

            return 0;
        }

        QImage ImageTools::cvMat2QImage(const cv::Mat& mat)
        {
            // 8-bits unsigned, NO. OF CHANNELS = 1
            if(mat.type() == CV_8UC1)
            {
                QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
                // Set the color table (used to translate colour indexes to qRgb values)
                image.setColorCount(256);
                for(int i = 0; i < 256; i++)
                {
                    image.setColor(i, qRgb(i, i, i));
                }
                // Copy input Mat
                uchar *pSrc = mat.data;
                for(int row = 0; row < mat.rows; row ++)
                {
                    uchar *pDest = image.scanLine(row);
                    memcpy(pDest, pSrc, mat.cols);
                    pSrc += mat.step;
                }
                return image;
            }
            // 8-bits unsigned, NO. OF CHANNELS = 3
            else if(mat.type() == CV_8UC3)
            {
                // Copy input Mat
                const uchar *pSrc = (const uchar*)mat.data;
                // Create QImage with same dimensions as input Mat
                QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
                return image.rgbSwapped();
            }
            else if(mat.type() == CV_8UC4)
            {
                qDebug() << "CV_8UC4";
                // Copy input Mat
                const uchar *pSrc = (const uchar*)mat.data;
                // Create QImage with same dimensions as input Mat
                QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
                return image.copy();
            }
            else
            {
                qDebug() << "ERROR: Mat could not be converted to QImage.";
                return QImage();
            }
        }

        std::string ImageTools::GetFloatRound(double fValue, int bits)
        {
            stringstream sStream;
            string out;
            sStream << fixed << setprecision(bits) << fValue;
            sStream >> out;
            return out;
        }
    
    }
}
