#include "sensor_api.h"

#include <iostream>
#include <fstream>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imwrite imdecode imshow
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


namespace watrix {
	namespace algorithm {

		lidar_camera_param_t SensorApi::params;
		
		cv::Mat SensorApi::camera_matrix_short_; // 内参矩阵
		cv::Mat SensorApi::camera_distCoeffs_short_; // 畸变矩阵

		cv::Mat SensorApi::camera_matrix_long_;
		cv::Mat SensorApi::camera_distCoeffs_long_;

		cv::Mat SensorApi::a_matrix_;
		cv::Mat SensorApi::r_matrix_;
		cv::Mat SensorApi::t_matrix_;

		void SensorApi::init(lidar_camera_param_t& params)
		{
			SensorApi::params = params;
			load_params();
		}

		void SensorApi::load_params(void)
		{
			cv::FileStorage fs_short( params.camera_short, cv::FileStorage::READ);
			fs_short ["camera_matrix"] >> camera_matrix_short_;
			fs_short ["distortion_coefficients"] >> camera_distCoeffs_short_;
			//std::cout<<"matrix short:"<<camera_matrix_short_<<"\ncoefficients:"<<camera_distCoeffs_short_;

			cv::FileStorage fs_long( params.camera_long, cv::FileStorage::READ);
			fs_long ["camera_matrix"] >> camera_matrix_long_;
			fs_long ["distortion_coefficients"] >> camera_distCoeffs_long_;
			//std::cout<<"matrix long:"<<camera_matrix_long_<<"\ncoefficients:"<<camera_distCoeffs_long_;

			cv::FileStorage fs_lidar( params.lidar_short, cv::FileStorage::READ);
			fs_lidar ["lidar_a_matrix"] >> a_matrix_;
			fs_lidar ["lidar_r_matrix"] >> r_matrix_;
			fs_lidar ["lidar_t_matrix"] >> t_matrix_;	
		}

		cvpoint_t SensorApi::image_cvpoint_a2b(
			CAMERA_TYPE camera_type,
			cvpoint_t point_a
		)
		{
			// image: b ---> a  (short/long)
			cv::Point2f pa(point_a);
			cv::Point2f pb(1,1);

			std::vector<cv::Point2f> inputDistortedPoints;
			std::vector<cv::Point2f> outputDistortedPoints;
			inputDistortedPoints.push_back(pa);
			
			if(CAMERA_LONG == camera_type){ // long 
				cv::undistortPoints(
					inputDistortedPoints, outputDistortedPoints, 
					camera_matrix_long_, camera_distCoeffs_long_, 
					cv::noArray(), camera_matrix_long_
					);		
			}else{ // short
				cv::undistortPoints(
				 inputDistortedPoints, outputDistortedPoints,
				 camera_matrix_short_, camera_distCoeffs_short_, 
				 cv::noArray(), camera_matrix_short_);		
			}
			
			pb.x = outputDistortedPoints.at(0).x;
			pb.y = outputDistortedPoints.at(0).y;
			cvpoint_t point_b(pb.x, pb.y);
			return point_b;
		}

		cv::Mat SensorApi::image_a2b(
			CAMERA_TYPE camera_type, 
			const cv::Mat& image_a
		)
		{
			cv::Mat image_b; // result 		
			cv::Mat  map1, map2;
			cv::Size imageSize = image_a.size();

			if(CAMERA_SHORT == camera_type){ // short
				initUndistortRectifyMap(
					camera_matrix_short_, camera_distCoeffs_short_, cv::Mat(),
					getOptimalNewCameraMatrix(
						camera_matrix_short_, camera_distCoeffs_short_, imageSize, 1, imageSize, 0
					),
					imageSize, CV_16SC2, map1, map2
				);
			}else{ // long
				initUndistortRectifyMap(
					camera_matrix_long_, camera_distCoeffs_long_, cv::Mat(),
					getOptimalNewCameraMatrix(
						camera_matrix_long_, camera_distCoeffs_long_, imageSize, 1, imageSize, 0
					),
					imageSize, CV_16SC2, map1, map2);			
			}

			cv::remap(image_a, image_b, map1, map2, cv::INTER_LINEAR);
			return image_b;
		}
		
		/*
		cv::Point3d lidar_point3;
		std::vector<double> extra_params{0.5, -18, 131};
		std::vector<double> extra_params2{-lidar_point3.x, -18, 131};
		*/

		bool SensorApi::lidar_3d_to_2d(
			cv::Point3d lidar_point3, std::vector<double> extra_params, cv::Point2i& point_a
		)
		{
			// for short camera:  3d ---> B(2d)--->A(2d)
			cv::Mat current_point = cv::Mat::ones(3,1,CV_64FC1); // double
			current_point.at<double>(0,0)= lidar_point3.y;
			current_point.at<double>(1,0)= extra_params[0]; //限界匹配左右宽度
			current_point.at<double>(2,0)= lidar_point3.z;

			cv::Mat mat_rt = r_matrix_* current_point  + t_matrix_;
			double a1 = mat_rt.at<double>(0,0);
			double a2 = mat_rt.at<double>(1,0);
			double a3 = mat_rt.at<double>(2,0);
			double pos_x = a1/(a3);
			double pos_y = a2/(a3);
			//std:cout<<pos_x<<"   y:"<<pos_y<<std::endl;
			pos_x = a_matrix_.at<double>(0,0) * pos_x + a_matrix_.at<double>(0,2) + extra_params[1];   
			pos_y = a_matrix_.at<double>(1,1) * pos_y + a_matrix_.at<double>(1,2) + extra_params[2];   

			// b图的坐标转换为a图
			cv::Point2f input_pos(pos_x, pos_y);
			point_a = image_cvpoint_a2b(CAMERA_SHORT, input_pos); // only for short
			int ax = point_a.x;
			int ay = point_a.y;			
			if (ay < 1 || ay +1 >= IMAGE_HEIGHT || ax < 1 || ax+1 >= IMAGE_WIDTH) {			
				return false;
			}
			return true;
			/* 
			//计算后对应的图像坐标点，重复点，直接返回
			if(imagexy_check[(int)pos_y][(int)pos_x] == 1){
				continue;
			}
            //存在对应的2D图像点，设置1
			imagexy_check[(int)pos_y][(int)pos_x] = 1;
			*/
		}

	}
}// end namespace


/*
void init_sensor_api()
{
	std::cout << "init_sensor_api 1 \n";

	lidar_camera_param_t params; 
	params.camera_long = "../../cfg/autotrain_models/sensor/camera_long.yaml";
	params.camera_short = "../../cfg/autotrain_models/sensor/camera_short.yaml";
	params.lidar_short = "../../cfg/autotrain_models/sensor/lidar_map_image.yaml";	

	SensorApi::init(params);

	std::cout << "init_sensor_api 2 \n";
}


void test_sensor_api()
{
	init_sensor_api();

	cv::Mat long_a = cv::imread("./long.png");
	cv::Mat long_b = SensorApi::image_a2b(CAMERA_LONG, long_a);
	cv::imwrite("long_b.jpg",long_b);

	cv::Mat short_a = cv::imread("./short.png");
	cv::Mat short_b = SensorApi::image_a2b(CAMERA_LONG, short_a);
	cv::imwrite("short_b.jpg",short_b);
}
 */