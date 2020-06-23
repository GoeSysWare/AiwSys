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

#include "numpy_util.h"

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
			const box_invasion_results_t& box_invasion_results,
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
				ss <<"x=" << fixed << std::setprecision(2) << x;
				ss <<",y=" << fixed << std::setprecision(2) << y;
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

				if (box_invasion_results[i].invasion_status == INVASION_STATUS::YES_INVASION){
					cv::rectangle(image_with_boxs, box.tl(), box.br(), COLOR_RED, thickness, 8, 0);
				}
				else{
					cv::rectangle(image_with_boxs, box.tl(), box.br(), COLOR_GREEN, thickness, 8, 0);
				}
				
			}
		}

		void DisplayUtil::draw_detection_boxs(
			cv::Mat& image_with_boxs,
			const detection_boxs_t& boxs,
			const box_invasion_results_t& box_invasion_results,
			const unsigned int thickness
		)
		{
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
				ss <<"x=" << fixed << std::setprecision(2) << x;
				ss <<",y=" << fixed << std::setprecision(2) << y;
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

				if (box_invasion_results[i].invasion_status == INVASION_STATUS::YES_INVASION){
					cv::rectangle(image_with_boxs, box.tl(), box.br(), COLOR_RED, thickness, 8, 0);
				}
				else{
					cv::rectangle(image_with_boxs, box.tl(), box.br(), COLOR_GREEN, thickness, 8, 0);
				}
				
			}
		}

		void DisplayUtil::draw_lidar_boxs(
			cv::Mat& image_with_boxs,
			const std::vector<lidar_invasion_cvbox> cv_obstacle_box,
			const unsigned int thickness
		)
		{
			for (size_t i = 0; i < cv_obstacle_box.size(); i++)
			{
				const lidar_invasion_cvbox cv_box = cv_obstacle_box[i];
				int xmin = cv_box.xmin;
				int xmax = cv_box.xmax;
				int ymin = cv_box.ymin;
				int ymax = cv_box.ymax;
				cv::Rect box(xmin,ymin, xmax-xmin,ymax-ymin);
				std::stringstream ss;
				ss << fixed << setprecision(2) << cv_box.dist;
				std::string str = "Dis: " + ss.str() + "m";
				cv::putText(image_with_boxs, str, cv::Point(xmin+(xmax-xmin)/2, ymin-10), cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(0, 128, 255), 1);
				cv::rectangle(image_with_boxs, box.tl(), box.br(), cv::Scalar(0,128,255), thickness, 8, 0);
			}
			// cv::putText(image_with_boxs, std::to_string(cv_obstacle_box.size()), cv::Point(100, 100), cv::FONT_HERSHEY_TRIPLEX, 2.0, cv::Scalar(0, 128, 255), 5);
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
			void DisplayUtil::draw_lane_line(
				cv::Mat& out, const dpoints_t& one_lane_dpoints, cv::Scalar color
			){
				for (int idx=0; idx < one_lane_dpoints.size()-1; idx++){
					int x = int(one_lane_dpoints[idx][0]);
					int y = int(one_lane_dpoints[idx][1]);
					int x1 = int(one_lane_dpoints[idx+1][0]);
					int y1 = int(one_lane_dpoints[idx+1][1]);
					if (x>=0 && x<out.cols && y>=0 && y <out.rows && x1>=0 && x1<out.cols && y1>=0 && y1<out.rows){
						// std::cout << x << " " << y << " " << x1 << " " << y1 << std::endl;
						cv::line(out, cv::Point2i(x,y), cv::Point2i(x1,y1), color, 3, 4);
					}
				}
			}

			void DisplayUtil::draw_lane_line(
				cv::Mat& out, const cvpoints_t& one_lane_cvpoints, cv::Scalar color
			){
				// cout<< "&&&&&&&one_lane_cvpoints:"<<one_lane_cvpoints.size()-1 <<endl;
				//ryh
				if(one_lane_cvpoints.size() == 0)
					return;
				for (int idx=0; idx < one_lane_cvpoints.size()-1; idx++){
					int x = one_lane_cvpoints[idx].x;
					int y = one_lane_cvpoints[idx].y;
					int x1 = one_lane_cvpoints[idx+1].x;
					int y1 = one_lane_cvpoints[idx+1].y;
					if (x>=0 && x<out.cols && y>=0 && y <out.rows && x1>=0 && x1<out.cols && y1>=0 && y1<out.rows){
						// std::cout << x << " " << y << " " << x1 << " " << y1 << std::endl;
						cv::line(out, cv::Point2i(x,y), cv::Point2i(x1,y1), color, 3, 4);
					}
				}
				// 	cv::line(out, one_lane_cvpoints[idx], one_lane_cvpoints[idx+1], color, 3, 4);
			}

			cv::Mat DisplayUtil::draw_lane_safe_area(
				const cv::Mat& image_,
				const cvpoints_t& left_lane_cvpoints,
				const cvpoints_t& right_lane_cvpoints
			){
				cv::Mat image = image_.clone(); // key step
				cv::Scalar green(0, 255, 0);
				int points_size = left_lane_cvpoints.size() + right_lane_cvpoints.size();
				if(points_size>3){
					cv::Point pt[1][points_size];
					int idx=0;
					for (int i=0; i < left_lane_cvpoints.size(); i++){
						pt[0][idx] = left_lane_cvpoints[i];
						idx++;
					}
					for (int i = right_lane_cvpoints.size()-1; i >= 0; i--){
						pt[0][idx] = right_lane_cvpoints[i];
						idx++;
					}
					const cv::Point* ppt[1] = {pt[0]};
					int npt[] = {points_size};
					cv::fillPoly(image, ppt, npt, 1, green, 0);
				}
				return image;
			}

			void DisplayUtil::show_lane(
				cv::Mat& image,
				const cvpoints_t& left_lane,
				const cvpoints_t& right_lane,
				const cvpoint_t& touch_point,
				const bool& is_clip_lane,
				const bool& is_draw_safe_area
			){
				// cvpoints_t y max->min
				std::vector<cvpoints_t> v_lane;

				cv::Scalar blue(255, 0, 0);
				cv::Scalar red(0, 0, 255);

				// 根据touch_point将轨道截断
				if(is_clip_lane){
					int near_y = touch_point.y;
					cvpoints_t left_clipped_lane;
					for(auto& point: left_lane){
						if (point.y>=near_y){
							left_clipped_lane.push_back(point);
						}
						else{
							break;
						}
					}
					cvpoints_t right_clipped_lane;
					for(auto& point: right_lane){
						if (point.y>=near_y){
							right_clipped_lane.push_back(point);
						}
						else{
							break;
						}
					}
					v_lane.push_back(left_clipped_lane);
					v_lane.push_back(right_clipped_lane);
				}
				else{
					v_lane.push_back(left_lane);
					v_lane.push_back(right_lane);
				}

				// 判断每条轨道至少2个点
				if(v_lane[0].size() >= 2 && v_lane[1].size() >= 2){
					// 以长度短的轨为基准，缩短长的轨，使两轨等长
					bool is_left = v_lane[0].back().y > v_lane[1].back().y ? 1 : 0;

					if(is_left){
						cvpoints_t right_clipped_lane;
						for(auto& point: right_lane){
							if (point.y>=v_lane[0].back().y){
								right_clipped_lane.push_back(point);
							}
							else{
								break;
							}
						}
						v_lane[1] = right_clipped_lane;
					}
					else{
						cvpoints_t left_clipped_lane;
						for(auto& point: left_lane){
							if (point.y>=v_lane[1].back().y){
								left_clipped_lane.push_back(point);
							}
							else{
								break;
							}
						}
						v_lane[0] = left_clipped_lane;
					}
					
					// 判断每条轨道至少2个点
					if(v_lane[0].size() >= 2 && v_lane[1].size() >= 2){
						draw_lane_line(image, v_lane[0], blue);
						draw_lane_line(image, v_lane[1], red);

						// 是否绘制半透明轨面
						if(is_draw_safe_area){
							cv::Mat image_mask = draw_lane_safe_area(image, v_lane[0], v_lane[1]);
							NumpyUtil::cv_add_weighted(image_mask, image, 0.5);
						}
					}
				}
			}

			// void DisplayUtil::draw_lane_line(
			// 	cv::Mat& out, const dpoints_t& one_lane_dpoints, cv::Scalar color
			// ){
			// 	//printf("draw_lane_line = %d  0: %f %f  last: %f %f\n",one_lane_dpoints.size(),one_lane_dpoints[0][0], one_lane_dpoints[1][0],one_lane_dpoints[one_lane_dpoints.size()-2][0], one_lane_dpoints[one_lane_dpoints.size()-1][0]);
			// 	for (int idx=0; idx < one_lane_dpoints.size()-1; idx++){

			// 		int x = (int)one_lane_dpoints[idx][0];
			// 		int y = (int)one_lane_dpoints[idx][1];
			// 		int x1 = (int)one_lane_dpoints[idx+1][0];
			// 		int y1 = (int)one_lane_dpoints[idx+1][1];
			// 		bool left = (x>=0) & (x<1920) & (y>=0) & (y<1080);
			// 		bool right = (x1>=0) & (x1<1920) & (y1>=0) & (y1<1080);
			// 		if( left & right){
			// 			cv::line(out, cv::Point2i(x,y), cv::Point2i(x1,y1), color, 3, 4);
			// 		}
						
					
			// 	}
			// 	printf("draw_lane__end \n");
			// }

			// void DisplayUtil::draw_lane_line(
			// 	cv::Mat& out, const cvpoints_t& one_lane_cvpoints, cv::Scalar color
			// ){
			// 	for (int idx=0; idx < one_lane_cvpoints.size()-1; idx++){
			// 		bool left = (one_lane_cvpoints[idx].x>=0) & (one_lane_cvpoints[idx].x<1920) & (one_lane_cvpoints[idx].y>=0) & (one_lane_cvpoints[idx].y<1080);
			// 		bool right = (one_lane_cvpoints[idx+1].x>=0) & (one_lane_cvpoints[idx+1].x<1920) & (one_lane_cvpoints[idx+1].y>=0) & (one_lane_cvpoints[idx+1].y<1080);
			// 		if( left & right){					
			// 		cv::line(out, one_lane_cvpoints[idx], one_lane_cvpoints[idx+1], color, 3, 4);
			// 		}
			// 	}
			// 	printf("draw_lane__vector_______end \n");
			// }


	}
}// end namespace