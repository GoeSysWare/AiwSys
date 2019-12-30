#include "topwire_distance_api.h"

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

#include <iostream>
#include <cmath>
using namespace std;

// boost
#include <boost/date_time/posix_time/posix_time.hpp>  


namespace watrix {
	namespace algorithm {


		int TopwireDistanceApi::counter = 0;

/* 
#pragma region helper for class 
		float TopwireDistance(float y1,float x1,float y2,float x2,float y3,float x3 )
		{
			std::vector<float> a{y3-y1, x3-x1};
			std::vector<float> b{y2-y1, x2-x1};
			float factor = vector_dot_vector(a,b) / (vector_dot_vector(b,b)*1.0);
			std::vector<float> array_temp = vector_dot_by_factor(b,factor);

			std::vector<float> a2; 
			for(int i=0;i<a.size();++i){
				a2.push_back(a[i]-array_temp[i]);
			}

			float dist = vector_dot_vector(a2,a2);
			return sqrt(dist);
		}

		
		float GetLinesAverageAngle(const std::vector<cv::Vec4i>& lines)
		{
			float avg_angle = 0;
			if (lines.size()<1){
				return avg_angle;
			}

			float sum_angle = 0;
			for (int i = 0;i < lines.size();++i)
			{
				Vec4i line = lines[i]; // 4-tuple
				int x1 = line[0];
				int y1 = line[1];
				int x2 = line[2];
				int y2 = line[3];

				sum_angle += get_angle(x1,y1,x2,y2); 
			}
			avg_angle = sum_angle/ (lines.size()*1.0);

			return rad_to_degree(avg_angle); // rad ---> degree
		}

		cv::Mat DrawImageWithLines(const cv::Mat& image, const std::vector<cv::Vec4i>& lines)
		{
			cv::Mat image_with_lines;
			cv_cvtcolor_to_bgr(image, image_with_lines);
			for (int i = 0;i < lines.size();++i)
			{
				Vec4i line = lines[i]; // 4-tuple
				int x1 = line[0];
				int y1 = line[1];
				int x2 = line[2];
				int y2 = line[3];

				cv::Point2i left_point(x1,y1);
				cv::Point2i right_point(x2,y2);
				cv::line(image_with_lines, left_point, right_point, CV_BGR(0, 0, 255),3);
			}
			return image_with_lines;
		}

		void GetLinesYMinMax(const std::vector<cv::Vec4i>& lines, int& ymin, int& ymax)
		{
			std::vector<int> line_ys;
			for (int i = 0;i < lines.size();++i)
			{
				Vec4i line = lines[i]; // 4-tuple
				//int x1 = line[0];
				int y1 = line[1];
				//int x2 = line[2];
				int y2 = line[3];

				// line y
				int line_y = (y1+y2)/2;
				line_ys.push_back(line_y);
			}
			max_min_of_vector(line_ys, ymax, ymin);
		}

		float GetLeftRightPointAngle(
			const cv::Point2i& left_point, 
			const cv::Point2i& right_point)
		{
			float left_right_angle = get_angle(left_point.x, left_point.y, right_point.x, right_point.y);
			return rad_to_degree(left_right_angle);
		}

		bool TopwireDistanceApi::GetLeftRightMaxPoint(
			const cv::Mat& image,
			float scale,
			std::vector<cv::Vec4i>& lines,
			int& line_ymin,
			int& line_ymax,
			cv::Mat& width_mask,
			cv::Point2i& leftp,
			cv::Point2i& rightp,
			cv::Point2i& maxp,
			float& maxd
		)
		{

#ifdef DEBUG_TIME
			int64_t cost = 0;
			boost::posix_time::ptime pt1, pt2;
			boost::posix_time::ptime start, end;
			pt1 = boost::posix_time::microsec_clock::local_time();
			start = pt1;
#endif

			// (1) get gray and threshold mask
			cv::Mat gray; // gray
			cv_cvtcolor_to_gray(image, gray);
			int height = gray.rows;
			int width = gray.cols; 

			cv::Mat mask;  // mask: binary mask with 0 and 255
			cv_threshold(gray, 128, mask); // 0,255

			// (2) get hough lines 
			cv_houghlinesp(mask, lines);

			if (lines.size()<1){
				return false;
			}

#ifdef DEBUG_INFO
			printf("lines.size = %d \n", (int)lines.size());
#endif

#ifdef DEBUG_INFO
			std::string result_dir = "./result/";
			cv::imwrite(result_dir+to_string(counter)+"_1_mask.jpg",mask);

			cv::Mat mask_with_lines = DrawImageWithLines(mask, lines);
			cv::imwrite(result_dir+to_string(counter)+"_2_0_mask_with_lines.jpg",mask_with_lines);
#endif 
	
			// (3) erode and dilate to remove noise
			cv::Mat eroded_mask = cv_erode_mat(mask, 3);
			cv::Mat dilated_mask = cv_dilate_mat(eroded_mask, 3);
			mask = dilated_mask; // 0,255
			
#ifdef DEBUG_INFO
			cv::imwrite(result_dir+to_string(counter)+"_2_1_eroded_mask.jpg",eroded_mask); 
			cv::imwrite(result_dir+to_string(counter)+"_2_2_dilated_mask.jpg",dilated_mask); 
#endif 

			//# (4) get mask line with 0/1
			cv::Mat mask_line(height,width, CV_8UC1, cv::Scalar(0)); // 0/1
			//std::cout<<"width="<< width<<std::endl; // # 1288
			//std::cout<<"height="<< height<<std::endl; // # 964

			std::vector<int> height_index(height);
			for(int h=0;h<height;h++){
				height_index[h] = h; // # 0,1,2,...963
			}

			for(int w=0;w<width;++w){ // # by col

				std::vector<int> coloum_w(height);
				int num_non_zeros = 0;
				for(int h=0;h<height;++h){
					coloum_w[h] = mask.at<uchar>(h,w)/255; // 0,1 
					if (coloum_w[h] >0){
						num_non_zeros ++; // 
					}
				}

				if (num_non_zeros>0){
					int mean_height = int(vector_dot_vector(coloum_w, height_index)/num_non_zeros);
					mask_line.at<uchar>(mean_height,w) = 1; //# image[y,x] = [h,w]
					//printf("col w =%d, num_non_zeros=%d, mean_height=%d \n",w,num_non_zeros,mean_height);
				}
			}

#ifdef DEBUG_INFO
			cv::imwrite(result_dir+to_string(counter)+"_3_mask_line.jpg",mask_line*255); //# mask line with 0 and 1
#endif 

			//#====================================================================================
			width_mask = mask; // for calculating width
			
			GetLinesYMinMax(lines, line_ymin, line_ymax);
#ifdef DEBUG_INFO
			printf("line_ymin = %d, line_ymax = %d \n",line_ymin,line_ymax); // 278,311
#endif 

			// mask_line[0:ymin]=0
			// mask_line[ymax:h]=0
			int y_upper_delta = 20;
			int y_lower_delta = 50;
			for(int y=0; y<line_ymin-y_upper_delta; y++){
				for(int x=0;x<width;x++){
					mask_line.at<uchar>(y,x) = 0;
					width_mask.at<uchar>(y,x) = 0;
				}
			}

			for(int y=line_ymax+y_lower_delta; y<height; y++){
				for(int x=0;x<width;x++){
					mask_line.at<uchar>(y,x) = 0;
					width_mask.at<uchar>(y,x) = 0;
				}
			}

#ifdef DEBUG_INFO
			cv::imwrite(result_dir+to_string(counter)+"_4_1_mask_line_REMOVE_NOISE.jpg",mask_line*255); //# mask line with 0 and 1 
			cv::imwrite(result_dir+to_string(counter)+"_4_2_width_mask_REMOVE_NOISE.jpg",width_mask*255); //# mask line with 0 and 1 
#endif
			//#====================================================================================

			//# (5) get roi from mask_line
			std::vector<int> y,x;
			np_where_g(mask_line,0,y,x);

			if (y.size()<1){ // invalid image
				return false;
			}

			int y1,y2;
			int x1,x2;
			max_min_of_vector(y,y2,y1);
			max_min_of_vector(x,x2,x1);

#ifdef DEBUG_INFO			
			printf("%d, %d, %d, %d \n",y1,y2,x1,x2); // 312,392,48,1207
#endif 
			cv::Mat roi = np_get_roi(mask_line,y1,y2,x1,x2);
			int roi_width = roi.cols;
			int half_roi_width = int(roi_width/2);
			cv::Mat roi_left = np_get_roi(roi,0,roi.rows,0,half_roi_width);
			cv::Mat roi_right = np_get_roi(roi,0,roi.rows,half_roi_width,roi.cols);

#ifdef DEBUG_INFO
			cv::imwrite(result_dir+to_string(counter)+"_5_0_roi.jpg",roi*255);
    		cv::imwrite(result_dir+to_string(counter)+"_5_1_roi_left.jpg",roi_left*255);
    		cv::imwrite(result_dir+to_string(counter)+"_5_2_roi_right.jpg",roi_right*255);
#endif 

			// # (6) get min height point of left roi/right roi
   			// # (6.1) for left roi
			np_where_g(roi_left,0,y,x);
			int y_min_left = y[0];
			int x_min_left = x[0];
			for(int i=0; i< y.size(); ++i){
				if (y[i] < y_min_left){
					y_min_left = y[i];
            		x_min_left = x[i];
				}
			}
#ifdef DEBUG_INFO
			printf("y_min_left= %d \n",y_min_left);
    		printf("x_min_left= %d \n",x_min_left);
#endif 

			// # (6.2) for right roi
			np_where_g(roi_right,0,y,x);
			int y_min_right = y[0];
			int x_min_right = x[0];
			for(int i=0; i< y.size(); ++i){
				if (y[i] < y_min_right){
					y_min_right = y[i];
            		x_min_right = x[i];
				}
			}
			x_min_right=x_min_right+half_roi_width;  // # move left coord to full roi coord  

#ifdef DEBUG_INFO
			printf("y_min_right= %d \n",y_min_right);
    		printf("x_min_right= %d \n",x_min_right);
#endif 

			//# roi coord ===> mask/image coord
			int y_min_left_g  = y_min_left  + y1;
			int y_min_right_g = y_min_right + y1;
			
			int x_min_left_g  = x_min_left  + x1;
			int x_min_right_g = x_min_right + x1;

			cv::Point2i left_point(x_min_left_g, y_min_left_g);
			cv::Point2i right_point (x_min_right_g, y_min_right_g);

			// pass out
			leftp = left_point;
			rightp = right_point;

			// # (7) get new_roi from left/right roi and cal distance and get max dist
			// # (7.1) get new_roi
			cv::Mat new_roi = np_get_roi(roi,0,roi.rows,x_min_left,x_min_right);

#ifdef DEBUG_INFO
			cv::imwrite(result_dir+to_string(counter)+"_6_new_roi.jpg",new_roi*255);
#endif 

			// # (7.2) cal distance and get max dist
			np_where_g(new_roi,0,y,x);
			int y_max = y[0];
			int x_max = x[0];
			float max_dist = 0;

			//printf("y_max=%d,x_max=%d,max_dist=%f \n",y_max,x_max,max_dist);

			for(int i=0; i< y.size(); ++i){
				float dist = TopwireDistance(y_min_left,0,y_min_right,x_min_right-x_min_left,y[i],x[i]);
				if (dist > max_dist){
					y_max=y[i];
					x_max=x[i];
					max_dist=dist;
				}
			}

#ifdef DEBUG_INFO
			printf("y_max=%d,x_max=%d,max_dist=%f \n",y_max,x_max,max_dist);
			// y_max=33,x_max=709,max_dist=31.192307 
#endif 
			// (8) get distance result( 3 points + max_dist)
			// # x_max,ymax(new_roi) ===> roi ===> mask/image
			int y_max_g = y_max + y1;
			int x_max_g = (x_min_left+x_max) + x1; //# new_roi
			cv::Point2i max_point(x_max_g, y_max_g);

			// pass out 
			maxp =  max_point;
			//maxd =  max_dist/scale; // scale if rotated
			maxd =  max_dist; 

#ifdef DEBUG_TIME
			end = boost::posix_time::microsec_clock::local_time();
			cost = (end - start).total_milliseconds();
			std::cout<<"[cost-1] total = "<< cost << std::endl;
#endif 
		
			return true;
		}

		bool TopwireDistanceApi::GetRailWidth(
			const cv::Mat& width_mask,
			cv::Point2i& left_point,
			cv::Point2i& right_point,
			int& width
		)
		{
			bool success = false;

			// 20190509: give up cv_connected_component method !!!
			//int min_area_threshold = 2000; // area   4400- 7900  can not >=2500
			//mask = cv_connected_component(mask,min_area_threshold);
			//mask = cv_largest_connected_component(mask);  // may ERROR

#ifdef DEBUG_INFO
			//std::string result_dir = "../../data/image_results3/";
			//cv::imwrite(result_dir+to_string(counter)+"_1_gray.jpg",gray); 
			//cv::imwrite(result_dir+to_string(counter)+"_2_mask.jpg",mask); 
#endif 

			std::vector<int> y,x;
			np_where_g(width_mask,0,y,x);

			if (x.size()<1){ // invalid image
				return success;
			}

			int xmax = *max_element(x.begin(),x.end());
			int xmin = *min_element(x.begin(),x.end());

			int max_index = get_index(x,xmax);
    		int min_index = get_index(x,xmin);

			if (max_index >=0 && min_index>=0){
				int y2 = y[max_index];
				int y1 = y[min_index];
				int ymax = std::max(y1,y2);

				printf("x_min=%d,y1=%d \n",xmin,y1);
				printf("x_max=%d,y2=%d \n",xmax,y2);
				
				// pass out result
				left_point = cv::Point2i(xmin,ymax);
				right_point = cv::Point2i(xmax,ymax);
				width = xmax - xmin;
				
				printf("xmax=%d,xmin=%d,width=%d \n",xmax,xmin,width);
			}

			return success;
		}

		bool IsFilterOutDistanceResult(const DistanceResult& distance_result, int image_height)
		{
			int upper_height = 50;
			int lower_height = image_height - 100; // 50 ===> 100
			
			cv::Point2i left_point = distance_result.left_point;
			cv::Point2i right_point = distance_result.right_point;
			int y_left = left_point.y;
			int y_right = right_point.y;
			printf(" y_left = %d, y_right = %d, (image_height = %d, upper_height = %d, lower_height = %d) \n",
				y_left, y_right, image_height, upper_height, lower_height
			);

			if (y_left >= lower_height || y_right>= lower_height){
				printf("[FILTER-OUT] lower \n");
				return true;
			} else if(y_left <= upper_height || y_right <= upper_height) {
				printf("[FILTER-OUT] upper \n");
				return true;
			} else {
				return false;
			}
		}

#pragma endregion


		DistanceResult TopwireDistanceApi::GetDistanceResult(
			const cv::Mat& image
		)
		{
#ifdef DEBUG_TIME
			int64_t cost = 0;
			boost::posix_time::ptime pt1, pt2;
			boost::posix_time::ptime start, end;
			pt1 = boost::posix_time::microsec_clock::local_time();
			start = pt1;
#endif

			printf("***************************************** \n");
			DistanceResult distance_result; 
			distance_result.success = false;
			distance_result.image = image; // no rotated
			distance_result.rotated = false; // rotated or not
			distance_result.left_point = cv::Point2i(0,0);
			distance_result.right_point = cv::Point2i(0,0);
			distance_result.max_point = cv::Point2i(0,0);
			distance_result.max_dist = 0;
			distance_result.w_left_point = cv::Point2i(0,0);
			distance_result.w_right_point = cv::Point2i(0,0);
			distance_result.width = 0;
			
			float scale = 1.0;

			std::vector<cv::Vec4i> lines;
			int line_ymin;
			int line_ymax;
			cv::Mat width_mask; 
			cv::Point2i left_point;
			cv::Point2i right_point;
			cv::Point2i max_point;
			float max_dist=0;

			// (1) get left and right point for the 1st time
			bool success = GetLeftRightMaxPoint(
				image,scale,
				lines,line_ymin,line_ymax,width_mask, 
				left_point,right_point,max_point,max_dist
			);
			if (!success){
				printf("[ERROR-1] GetLeftRightMaxPoint Failed \n");
				return distance_result;
			}

			// (2) choose smaller angle
			float line_angle = GetLinesAverageAngle(lines);
			float left_right_angle = GetLeftRightPointAngle(left_point,right_point);

			counter ++;

			float degree;
			printf("two angle = %f, %f \n",line_angle,left_right_angle);
			if ( abs(left_right_angle) <= abs(line_angle) ) {
				printf(" use left_right_angle %f \n", left_right_angle);
				degree = left_right_angle;
			}
			else{
				printf(" use line_angle %f \n", line_angle);
				degree = line_angle;
			}
			
			// (3) rotate image if angle is big
			float rotate_min_threshold = 0.4; //# in degree 
			if (abs(degree)>rotate_min_threshold) 
			{
				printf("*********************ROTATE************** \n");
				printf("[CASE-1] Rotate image with angle = %f \n",degree);

				cv::Mat rotated_image = cv_rotate_image(image, degree); // be rotated
				distance_result.image = rotated_image;
				distance_result.rotated = true; // rotated or not

				scale = image.cols*1.0/(rotated_image.cols*1.0);
				printf(" scale =  %f \n", scale);

				success = GetLeftRightMaxPoint(
					distance_result.image,scale,
					lines,line_ymin,line_ymax,width_mask, 
					left_point,right_point,max_point,max_dist
				);
			} else {
				printf("[CASE-2] Use origin image  \n");

				distance_result.image = image; // no rotated
				distance_result.rotated = false; // rotated or not
			}

			if (success){
				distance_result.success = true;
				distance_result.left_point = left_point;
				distance_result.right_point = right_point;
				distance_result.max_point = max_point;
				distance_result.max_dist = max_dist;

				// filter out some results
				bool is_filter_out = IsFilterOutDistanceResult(distance_result, distance_result.image.rows);
				if ( is_filter_out){
					distance_result.success = false;
				} else { // valid result 
					// for rail width
					GetRailWidth(
						width_mask, 
						distance_result.w_left_point,
						distance_result.w_right_point,
						distance_result.width
					);
				}
			}

			counter ++;
			
#ifdef DEBUG_TIME
			end = boost::posix_time::microsec_clock::local_time();
			cost = (end - start).total_milliseconds();
			std::cout<<"[GetDistanceResult] total = "<< cost << std::endl;
#endif 

			return distance_result; 
		}

		cv::Mat TopwireDistanceApi::DrawDistanceResult(
			const cv::Mat& image,
			const DistanceResult& distance_result
		)
		{
			bool success = distance_result.success;
			cv::Point2i left_point = distance_result.left_point;
			cv::Point2i right_point = distance_result.right_point;
			cv::Point2i max_point = distance_result.max_point;
			float max_dist = distance_result.max_dist;

			cv::Point2i w_left_point = distance_result.w_left_point;
			cv::Point2i w_right_point = distance_result.w_right_point;
			int width = distance_result.width;

			cv::Mat image_with_result;
			cv_cvtcolor_to_bgr(image, image_with_result);

			// (1) left right point with max dist
			cv::line(image_with_result, left_point, right_point, CV_BGR(0, 255, 0),3);
        	cv::circle(image_with_result, max_point, 2, CV_BGR(0, 0, 255),2);
        	cv::putText(image_with_result,std::to_string(max_dist),max_point,cv::FONT_HERSHEY_COMPLEX,2,CV_BGR(0,255,0),3);
			
			// (2) width
			cv::line(image_with_result, w_left_point, w_right_point, CV_BGR(0, 255, 255),3);
        	cv::putText(image_with_result,std::to_string(width),w_left_point,cv::FONT_HERSHEY_COMPLEX,2,CV_BGR(0, 255, 255),3);
			
			return image_with_result;
		}
*/

	}
}// end namespace