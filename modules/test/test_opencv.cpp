

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
// opencv
#include "opencv2/opencv.hpp"

int addAlpha(cv::Mat& src, cv::Mat& dst, cv::Mat& alpha)
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


cv::Mat createAlpha(cv::Mat& src)
{
	cv::Mat alpha = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	cv::Mat gray = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
 
	cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);
 
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
		//	alpha.at<uchar>(i, j) = 255 - gray.at<uchar>(i, j);
        	alpha.at<uchar>(i, j) = 255;
		}
	}
 
	return alpha;
}

void jianbian(cv::Mat & Img)
{

    cv::Scalar a(255, 255, 255);
    cv::Scalar b(0.0, 0.0, 0.0);
 
    Img.setTo(a);
 
    int width, height;
    width=Img.cols;
    height=Img.rows;
 
    cv::Point2f origin(0.0, 0.0);
    cv::Point2f Cen(Img.cols/2.0, Img.rows/2.0);
 
    float dis;
 
    if (origin.x<=Cen.x && origin.y<=Cen.y)
    {
        dis=sqrt((width-1-origin.x)*(width-1-origin.x)+
                        (height-1-origin.y)*(height-1-origin.y));
    }
    else if (origin.x<=Cen.x && origin.y>Cen.y)
    {
        dis=sqrt((width-1-origin.x)*(width-1-origin.x)+
                        origin.y*origin.y);
 
    }
    else if (origin.x>Cen.x && origin.y>Cen.y)
    {
        dis=sqrt(origin.x*origin.x+(origin.y)*(origin.y));
    }
    else
    {
       dis=sqrt(origin.x*origin.x+
                        (height-1-origin.y)*(height-1-origin.y));
    }
 
    float weightB=(b[0]-a[0])/dis;
    float weightG=(b[1]-a[1])/dis;
    float weightR=(b[2]-a[2])/dis;
 
    float dis2;
    for (int i=0; i<Img.rows; i++)
    {
        for (int j=0; j<Img.cols; j++)
        {
            dis2=sqrt((i-origin.x)*(i-origin.x)+(j-origin.y)*(j-origin.y));
            Img.at<cv::Vec3f>(i,j)[0]=Img.at<cv::Vec3f>(i,j)[0]+weightB*dis2;
            Img.at<cv::Vec3f>(i,j)[1]=Img.at<cv::Vec3f>(i,j)[1]+weightG*dis2;
            Img.at<cv::Vec3f>(i,j)[2]=Img.at<cv::Vec3f>(i,j)[2]+weightR*dis2;
        }
    }
 
     Img=Img/255.0;
}




// int main()
// {
// 	cv::Mat mask = cv::imread("/home/shuimujie/02.github/AiwSys/warn.png",cv::  IMREAD_UNCHANGED);
//     cv::Mat src = cv::imread("/home/shuimujie/02.github/AiwSys/test.png",cv::  IMREAD_UNCHANGED);
//     std::cout<< src.channels() <<std::endl;
//     std::cout<< mask.channels() <<std::endl;
    
//     float dis  = mask.rows-1;
//     float step = 1.5;
//      cv::Scalar end(255, 255, 255);
//     cv::Scalar start(0, 0, 0);
//     end[0] = (float)mask.at<cv::Vec4b>(0,0)[0];
//     end[1] = (float)mask.at<cv::Vec4b>(0,0)[1];
//     end[2] = (float)mask.at<cv::Vec4b>(0,0)[2];
//     float weightB=(end[0]-start[0])/dis;
//     float weightG=(end[1]-start[1])/dis;
//     float weightR=(end[2]-start[2])/dis;
//     float valB =0;
//     float valG = 0;
//    float valR   = 0;
// 	for (int i = 0; i < mask.cols; i++)
// 	{
// 		for (int j = 0; j < mask.rows; j++)
// 		{
//              valB  = mask.at<cv::Vec4b>(j,i)[0]  - weightB*j*step;
//              valG  = mask.at<cv::Vec4b>(j,i)[1]  - weightG*j*step;           
//              valR  = mask.at<cv::Vec4b>(j,i)[2]   - weightR*j*step;  

// 	         mask.at<cv::Vec4b>(j,i)[0] =  valB < 0 ? 0:valB  ;
//              mask.at<cv::Vec4b>(j,i)[1] =   valG < 0 ? 0:valG  ;
//              mask.at<cv::Vec4b>(j,i)[2] =   valR  < 0 ? 0:valR  ;
//          //   std::cout<< (float)mask.at<cv::Vec4b>(j,j)[0] <<","<<(float)mask.at<cv::Vec4b>(i,j)[1] <<","<<(float)mask.at<cv::Vec4b>(i,j)[2]<<std::endl;
// 		}
// 	}
//     //std::cout << "【默认风格】" << std::endl << mask << std::endl << std::endl;
//    // std::cout << "Python风格】" << std::endl << cv::format(mask, cv::Formatter::FMT_PYTHON) << std::endl<< std::endl;

//     //std::cout<< src.channels() <<std::endl;
// 	cv::Mat dst;
// 	cv::Mat alpha = createAlpha(src);
 
// 	addAlpha(src, dst, alpha);


//     cv::Mat mask_dst;
//  	// addAlpha(mask, mask_dst, alpha);
// 	cv::imshow("1", mask);
// 	//cv::imshow("2", dst);
//     std::cout<< dst.channels() <<std::endl;


//     std::vector<int> compression_params;
//     compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
//     compression_params.push_back(9);
//  	cv::imwrite("/home/shuimujie/02.github/AiwSys/warn0.png", dst,compression_params);
//    cv::addWeighted(mask, 0.3, dst, 1, 0, dst); 


//    	cv::imshow("3", dst);
//  	cv::imwrite("/home/shuimujie/02.github/AiwSys/warn2.png", mask,compression_params);
// 	cv::imwrite("/home/shuimujie/02.github/AiwSys/warn1.png", dst);
// 	cv::waitKey(0);
// 	return 0;
// }

int main()
{
	cv::Mat mask = cv::imread("/home/shuimujie/02.github/AiwSys/warn.png");
    cv::Mat src = cv::imread("/home/shuimujie/02.github/AiwSys/test.png");
    std::cout<< src.channels() <<std::endl;
    std::cout<< mask.channels() <<std::endl;
    
    float dis  = mask.rows-1;
    float step = 1.5;
     cv::Scalar end(255, 255, 255);
    cv::Scalar start(0, 0, 0);
    end[0] = (float)mask.at<cv::Vec3b>(0,0)[0];
    end[1] = (float)mask.at<cv::Vec3b>(0,0)[1];
    end[2] = (float)mask.at<cv::Vec3b>(0,0)[2];
    float weightB=(end[0]-start[0])/dis;
    float weightG=(end[1]-start[1])/dis;
    float weightR=(end[2]-start[2])/dis;
    float valB =0;
    float valG = 0;
   float valR   = 0;
	for (int i = 0; i < mask.cols; i++)
	{
		for (int j = 0; j < mask.rows; j++)
		{
             valB  = mask.at<cv::Vec3b>(j,i)[0]  - weightB*j*step;
             valG  = mask.at<cv::Vec3b>(j,i)[1]  - weightG*j*step;           
             valR  = mask.at<cv::Vec3b>(j,i)[2]   - weightR*j*step;  

	         mask.at<cv::Vec3b>(j,i)[0] =  valB < 0 ? 0:valB  ;
             mask.at<cv::Vec3b>(j,i)[1] =   valG < 0 ? 0:valG  ;
             mask.at<cv::Vec3b>(j,i)[2] =   valR  < 0 ? 0:valR  ;
         //   std::cout<< (float)mask.at<cv::Vec4b>(j,j)[0] <<","<<(float)mask.at<cv::Vec4b>(i,j)[1] <<","<<(float)mask.at<cv::Vec4b>(i,j)[2]<<std::endl;
		}
	}

	cv::imshow("1", mask);
    std::vector<int> compression_params;

    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
   cv::addWeighted(mask, 0.3, src, 1, 0, src); 


   	cv::imshow("3", src);
 	cv::imwrite("/home/shuimujie/02.github/AiwSys/warn2.png", mask,compression_params);
	cv::imwrite("/home/shuimujie/02.github/AiwSys/warn1.png", src);
	cv::waitKey(0);
	return 0;
}


// int main()
// {
//     cv::Mat  abcimage=cv::imread("warn.png",cv:: IMREAD_UNCHANGED);
//     cv::Mat src = cv::imread("test.png",cv::  IMREAD_UNCHANGED);
//     std::cout<< abcimage.channels() <<std::endl;
//     std::cout<< src.channels() <<std::endl;
//     //display test
// //     //cv::cvtColor(src, src, 24);
// //      cv::Mat out;
// //    char *alphvalue = (char *)malloc(src.rows* src.cols);
// //    memset(alphvalue, 0xff , src.rows* src.cols);
// //    cv::Mat    alfdst = cv::Mat(src.rows, src.cols, CV_8UC1,alphvalue);
// //    image_tools_.addAlpha(src , out, alfdst);
// //     cv::addWeighted(abcimage, 0.5, out, 0.5, 0, out);   
//     return 0;
// }