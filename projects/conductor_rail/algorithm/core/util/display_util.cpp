#include "display_util.h"

// std
#include <iostream>
#include <map>

// glog
#include <glog/logging.h>

// opencv
#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imwrite imdecode imshow

// boost
#include <boost/date_time/posix_time/posix_time.hpp>  // boost::make_iterator_range
#include <boost/filesystem.hpp> // boost::filesystem

using namespace std;
using namespace cv;


namespace watrix {
	namespace algorithm {
		
#pragma region draw 
		void DisplayUtil::draw_box(
			const cv::Mat& image,
			const cv::Rect& box,
			const unsigned int thickness,
			cv::Mat& image_with_boxs
		)
		{
			if (image.channels() == 1)
			{
				cv::cvtColor(image, image_with_boxs, CV_GRAY2BGR);
			} else 
			{
				image_with_boxs = image.clone();
			}

			cv::rectangle(image_with_boxs, box.tl(), box.br(), CV_RGB(255, 0, 0), thickness, 8, 0);
		}

		void DisplayUtil::draw_boxs(
			const cv::Mat& image,
			const std::vector<cv::Rect>& boxs,
			const unsigned int thickness,
			cv::Mat& image_with_boxs
		)
		{
			if (image.channels() == 1)
			{
				cv::cvtColor(image, image_with_boxs, CV_GRAY2BGR);
			} 
			else 
			{
				image_with_boxs = image.clone();
			}

			for (size_t i = 0; i < boxs.size(); i++)
			{
				cv::rectangle(image_with_boxs, boxs[i].tl(), boxs[i].br(), CV_RGB(255, 0, 0), thickness, 8, 0);
			}
		}

		void DisplayUtil::draw_contour(
			const cv::Mat& image,
			const std::vector<cv::Point>& contour,
			const unsigned int thickness,
			cv::Mat& image_with_contours
		)
		{
			if (image.channels() == 1)
			{
				cv::cvtColor(image, image_with_contours, CV_GRAY2BGR);
			}
			else 
			{
				image_with_contours = image.clone();
			}
			vector<vector<cv::Point> > contours;
			contours.push_back(contour);

			cv::drawContours(image_with_contours, contours, 0, CV_RGB(255, 0, 0), thickness, 8);
		}

		void DisplayUtil::draw_contours(
			const cv::Mat& image,
			const std::vector<std::vector<cv::Point> >& contours,
			const unsigned int thickness,
			cv::Mat& image_with_contours
		)
		{
			if (image.channels() == 1)
			{
				cv::cvtColor(image, image_with_contours, CV_GRAY2BGR);
			}
			else 
			{
				image_with_contours = image.clone();
			}

			for (size_t i = 0; i < contours.size(); i++)
			{
				cv::drawContours(image_with_contours, contours, i, CV_RGB(255, 0, 0), thickness, 8);
			}
		}

#pragma endregion

		void DisplayUtil::draw_detection_boxs(
			const cv::Mat& image,
			const detection_boxs_t& boxs,
			const unsigned int thickness,
			cv::Mat& image_with_boxs
		)
		{
			image_with_boxs = image.clone();
			for (size_t i = 0; i < boxs.size(); i++)
			{
				const detection_box_t detection_box = boxs[i];
				int xmin = detection_box.xmin;
				int xmax = detection_box.xmax;
				int ymin = detection_box.ymin;
				int ymax = detection_box.ymax;
				cv::Rect box(xmin,ymin, xmax-xmin,ymax-ymin);

				// for distance
				bool valid_dist = detection_box.valid_dist;
				float x = detection_box.dist_x;
				float y = detection_box.dist_y;
				
				stringstream ss;
				ss << std::setprecision(2) << detection_box.confidence;
				ss <<"x=" <<std::setprecision(2) << x;
				ss <<",y=" << std::setprecision(2) << y;
				std::string display_text = detection_box.class_name+" "+ ss.str();

				int pt_x = xmin;
				int pt_y = ymin-10;
				if (pt_y<20){
					pt_y = 20;
				};
				cv::Point2i origin(pt_x,pt_y);
				cv::putText(image_with_boxs, display_text, origin, 
					cv::FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2
				);
	
				cv::rectangle(image_with_boxs, box.tl(), box.br(), COLOR_GREEN, thickness, 8, 0);
			}
		}


		void DisplayUtil::draw_mosun_result(
			const MosunResult& mosun_result,
			cv::Mat& image_with_result
		)
		{
			image_with_result = mosun_result.rotated;
			cv::circle(image_with_result, mosun_result.top_point,    2, CV_BGR(0, 0, 255), 6);
			cv::circle(image_with_result, mosun_result.bottom_point, 2, CV_BGR(255, 0, 0), 6);

			cvpoint_t left_point1 =  mosun_result.left_point;
			cvpoint_t left_point2 = left_point1;
			left_point2.y = 900;

			cvpoint_t right_point1 =  mosun_result.right_point;
			cvpoint_t right_point2 = right_point1;
			right_point2.y = 900;

			cv::line(image_with_result, left_point1,  left_point2,  CV_BGR(0, 255, 0), 6);
			cv::line(image_with_result, right_point1, right_point2, CV_BGR(0, 255, 0), 6);

			stringstream ss;
			if (!mosun_result.left_right_symmetrical){
				ss <<"YES FIX ";
			}
			else {
				ss <<"NO  FIX ";
			}
			ss <<", left_area =  " << mosun_result.left_area;
			ss <<", right_area =  " << mosun_result.right_area;
			ss <<", area_ratio =  " << mosun_result.left_right_area_ratio;
			ss <<", height(pixel): " <<std::setprecision(2) << mosun_result.mosun_in_pixel;
			ss << ", meter(mm): "<<std::setprecision(2) << mosun_result.mosun_in_meter;
			std::string display_text = ss.str();

			cv::Point2i origin(20,20);
			cv::putText(image_with_result, display_text, origin, 
				cv::FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1
			);
		}



#pragma region draw point/circles
			void DisplayUtil::draw_circle_point(
				cv::Mat& image, int x, int y, cv::Scalar color, int radius
			){
				cv::circle(image, cv::Point2i(x,y), radius,  color, -1);
			}

			void DisplayUtil::draw_circle_point(
				cv::Mat& image, cv::Point2i pt, cv::Scalar color, int radius
			){
				cv::circle(image, pt, radius,  color, -1);
			}

			void DisplayUtil::draw_circle_point_with_text(
				cv::Mat& image, cv::Point2i pt, cv::Scalar color, const std::string& text, int radius
			){
				cv::circle(image, pt, radius,  color, -1);
				cv::putText(image, text, pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, COLOR_RED, 2);
			}

			void DisplayUtil::draw_circle_point_with_text(
				cv::Mat& image, cv::Point2i pt, cv::Scalar color, int value, int radius
			){
				std::string text = std::to_string(value);
				draw_circle_point_with_text(image, pt, color, text, radius);
			}

			void DisplayUtil::draw_lane(
				cv::Mat& out, const dpoints_t& one_lane_dpoints, cv::Scalar color, int radius
			){
				/*
				cvpoints_t cvpoints = dpoints_to_cvpoints(one_lane_dpoints);
				cv::polylines(out, cvpoints, closed, color, thick);
				*/
				for(auto& dpoint: one_lane_dpoints){
					int x = dpoint[0];
					int y = dpoint[1];
					draw_circle_point(out, cv::Point2i(x,y), color, radius);
				}
			}

			void DisplayUtil::draw_lane(
				cv::Mat& out, const cvpoints_t& one_lane_cvpoints, cv::Scalar color, int radius
			){
				for(auto& cvpoint: one_lane_cvpoints){
					draw_circle_point(out, cvpoint, color, radius);
				}
			}

#pragma endregion


	}
}// end namespace