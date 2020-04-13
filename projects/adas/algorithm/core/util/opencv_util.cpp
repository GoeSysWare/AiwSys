#include "opencv_util.h"

#include "display_util.h"
#include "numpy_util.h"

// std
#include <iostream>
#include <map>
using namespace std;

// glog
#include <glog/logging.h>

#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imdecode imshow
using namespace cv;

// boost
#include <boost/date_time/posix_time/posix_time.hpp>  // boost::make_iterator_range
#include <boost/filesystem.hpp> // boost::filesystem


namespace watrix {
	namespace algorithm {

		void OpencvUtil::cv_resize(const cv::Mat& input, cv::Mat& output, const cv::Size& size)
		{
			cv::resize(input, output, size);
		}

		cv::Mat OpencvUtil::concat_mat(
			const cv::Mat& first_image,
			const cv::Mat& second_image
		)
		{
			cv::Mat result;
			cv::vconcat(first_image, second_image, result);
			return result;
		}

		void OpencvUtil::split_mat_horizon(
			const cv::Mat& image,
			cv::Mat& upper,
			cv::Mat& lower
		)
		{
			int height = image.rows;
			int width = image.cols;

			cv::Rect upper_rect(0, 0,          width, height / 2);
			cv::Rect lower_rect(0, height / 2, width, height / 2);
			upper = image(upper_rect);
			lower = image(lower_rect);
		}

		cv::Mat OpencvUtil::clip_mat(
			const cv::Mat& image,
			const float start_ratio,
			const float end_ratio
		)
		{
			/*
			1024,2048,  0.25,0.75 ===>1024, 1024
			*/
			int height = image.rows;
			int width = image.cols;
			int clip_start = int(width * start_ratio);
			int clip_end = int(width * end_ratio);

			cv::Rect rect(clip_start, 0, clip_end - clip_start, height);
			Mat clip_image = image(rect); // clip image 
			return clip_image;
		}

		cv::Mat OpencvUtil::rotate_mat(const cv::Mat& image)
		{
			Mat tmp;
			Mat result;
			cv::transpose(image, tmp);
			cv::flip(tmp, result, 0);
			return result;
		}

		cv::Mat OpencvUtil::rotate_mat2(const cv::Mat& image)
		{
			Mat tmp;
			Mat result;
			cv::transpose(image, tmp);
			cv::flip(tmp, result, 1);
			return result;
		}

		void OpencvUtil::dilate_mat(
			cv::Mat& diff,
			const int dilate_size
		)
		{
			if (dilate_size > 0) {
				cv::Mat dilate_element = getStructuringElement(MORPH_RECT, Size(dilate_size, dilate_size));
				cv::dilate(diff, diff, dilate_element);
			}
		}

		void OpencvUtil::erode_mat(
			cv::Mat& diff,
			const int erode_size
		)
		{
			if (erode_size > 0) {
				cv::Mat erode_element = getStructuringElement(MORPH_RECT, Size(erode_size, erode_size));
				cv::erode(diff, diff, erode_element);
			}
		}

		cv::Rect OpencvUtil::bounding_box(const boxs_t& boxs)
		{
			if (boxs.size() < 1) {
				return cv::Rect();
			}
			// get max bounding box 
			std::vector<int> vx1,vy1,vx2,vy2;
			for (size_t i = 0; i < boxs.size(); i++)
			{
				const cv::Rect& box = boxs[i];
				vx1.push_back(box.x);
				vy1.push_back(box.y);

				vx2.push_back(box.x+box.width);
				vy2.push_back(box.y+box.height);
			}
			
			auto xmin = *min_element(std::begin(vx1), std::end(vx1)); // c++11
			auto ymin = *min_element(std::begin(vy1), std::end(vy1)); // c++11
			auto xmax = *max_element(std::begin(vx2), std::end(vx2)); // c++11
			auto ymax = *max_element(std::begin(vy2), std::end(vy2)); // c++11

			cv::Rect max_bounding_box(xmin, ymin, xmax - xmin, ymax - ymin);
			return max_bounding_box;
		}

		cv::Rect OpencvUtil::boundary(
			const cv::Rect& rect_in,
			const cv::Size& max_size
		)
		{
			cv::Rect rect_out = rect_in;

			if (rect_out.x < 0) {
				rect_out.x = 0;
			}

			if (rect_out.y <0) {
				rect_out.y = 0;
			}

			if (rect_out.x + rect_out.width > max_size.width) {
				rect_out.width = max_size.width- rect_out.x;
			}

			if (rect_out.y + rect_out.height > max_size.height) {
				rect_out.height = max_size.height- rect_out.y;
			}
			return rect_out;
		}


		/*
		get pixel unique value count from mat
		*/
		size_t OpencvUtil::get_value_count(const cv::Mat& image)
		{
			CHECK(!image.empty()) << "invalid mat";
			std::map<int, int> value_count;
			for (int row = 0; row < image.rows; row++)
			{
				for (int col = 0; col < image.cols; col++)
				{
					int value = image.at<uchar>(row, col);
					value_count[value] ++;
				}
			}
			return value_count.size();
		}


		bool OpencvUtil::contour_compare(
			const std::vector<cv::Point>& contour1,
			const std::vector<cv::Point>& contour2
		)
		{
			return cv::contourArea(contour1) > cv::contourArea(contour2);
		}

		bool OpencvUtil::rect_compare(
			const cv::Rect& left_rect,
			const cv::Rect& right_rect
		)
		{
			return left_rect.area() > right_rect.area();
		}
	

		cv::Mat OpencvUtil::get_channel_mat(
			const cv::Mat& color_mat, 
			const unsigned int channel
		)
		{
			CHECK(!color_mat.empty())<<"invalid mat";

			// 0,1,2,3
			int channels = color_mat.channels();
			CHECK_GE(channels, 2) << "channels must >= 2";
			CHECK_LE(channel, (unsigned int)(channels-1)) << "channel must <"<< channels-1;
			int height = color_mat.rows;
			int width = color_mat.cols;

			cv::Mat result_mat(height, width, CV_8UC1);

			int c = channel;
			for (int h = 0; h < height; h++) // rows
			{
				for (int w = 0; w < width; w++) // cols
				{
					int value = color_mat.at<cv::Vec3b>(h, w)[c];
					result_mat.at<uchar>(h, w) = value;
				}
			}
			return result_mat;
		}

		// CV_32FC1 
		cv::Mat OpencvUtil::get_float_channel_mat(
			const cv::Mat& color_mat,
			const unsigned int channel
		)
		{
			CHECK(!color_mat.empty())<<"invalid mat";

			// 0,1,2,3
			int channels = color_mat.channels();
			CHECK_GE(channels, 2) << "channels must >= 2";
			CHECK_LE(channel, (unsigned int)(channels-1)) << "channel must <"<< channels-1;
			int height = color_mat.rows;
			int width = color_mat.cols;

			cv::Mat result_mat(height, width, CV_32FC1);

			int c = channel;
			for (int h = 0; h < height; h++) // rows
			{
				for (int w = 0; w < width; w++) // cols
				{
					float value = color_mat.at<cv::Vec3f>(h, w)[c];
					result_mat.at<float>(h, w) = value;
				}
			}
			return result_mat;
		}

		// hwc, bgr, CV_32FC3   mean === > chw, bgr
		void OpencvUtil::bgr_subtract_mean(
			const cv::Mat& bgr,
			const std::vector<float>& bgr_mean,
			float scale,
			channel_mat_t& channel_mat
		)
		{
			CHECK(!bgr.empty()) << "invalid mat";

			// 0,1,2,3
			int channels = bgr.channels();
			CHECK_EQ(channels, 3) << "channels must == 3";
			CHECK_EQ(bgr_mean.size(), 3) << "bgr_mean.size() must == 3";
			int height = bgr.rows;
			int width = bgr.cols;

			//channel_mat_t channel_mat;
			for (int c = 0; c < channels; c++)
			{
				cv::Mat c_mat(height, width, CV_32FC1);

				for (int h = 0; h < height; h++) // rows
				{
					for (int w = 0; w < width; w++) // cols
					{
						float value = bgr.at<cv::Vec3f>(h, w)[c];
						c_mat.at<float>(h, w) = (value - bgr_mean[c])*scale; // subtract mean and scale
					}
				}

				/*
				std::cout << "===================================\n";
				for (int row = 0; row <= 5; row++) {
					for (int col = 0; col <= 5; col++) {
						std::cout << c_mat.at<float>(row, col) << " ";
					}
					std::cout << std::endl;
				}
				std::cout << "===================================\n";
				*/

				channel_mat.push_back(c_mat); // b,g,r
			}
		}

		cv::Mat OpencvUtil::merge_mask(
			const cv::Mat& image,
			const cv::Mat& binary_mask,
			int b,int g,int r
		)
		{
			/*
			image            1024,1024,3   CV_8UC3
			binary_mask      1024,1024,1   CV_8UC1
			image_with_mask  1024,1024,3   CV_8UC3
			*/
			int height = image.rows;
			int width = image.cols;

			cv::Mat image_with_mask(height, width, CV_8UC3); // 0-255

			for (int h = 0; h < height; h++)
			{
				Vec3b *p = image_with_mask.ptr<cv::Vec3b>(h);

				const uchar *p_mask = binary_mask.ptr<uchar>(h);
				const Vec3b *p_origin = image.ptr<cv::Vec3b>(h);

				for (int w = 0; w < width; w++)
				{
					if ( 255 == *p_mask) {  // 255
						*p = Vec3b(b, g, r); // OK
					}
					else {  // 0
						*p = *p_origin; // OK
					}
					p++;

					p_mask++;
					p_origin++;
				}
			}
			return image_with_mask;
		}


		// 8UC3 + 8UC1
		cv::Mat OpencvUtil::merge_mask(
			const cv::Mat& image,
			const cv::Mat& binary_mask,
			cv::Scalar bgr
		)
		{
			return OpencvUtil::merge_mask(image, binary_mask, bgr[0], bgr[1], bgr[2] );
		}

		void OpencvUtil::print_mat(const cv::Mat& image)
		{
			std::cout << "image.channels()= " << image.channels() << std::endl; // 3
			std::cout << "image.size()=" << image.size() << std::endl; // [512 x 512]
			std::cout << "image.type()=" << image.type() << std::endl; // CV_8UC3=16
			//cv::imwrite("./image.jpg", image);

			std::cout << "================image===================\n";
			for (int row = 0; row <= 5; row++) {
				for (int col = 0; col <= 5; col++) {
					std::cout << image.at<cv::Vec3b>(row, col) << " ";
				}
				std::cout << std::endl; // 
			}
			std::cout << "=================image==================\n";
		}

		cv::Mat OpencvUtil::generate_mat(
			int height, int width, int b, int g, int r
		)
		{
			cv::Mat color_image(height, width, CV_8UC3); // 0-255

			for (int h = 0; h < height; h++)
			{
				Vec3b *p = color_image.ptr<cv::Vec3b>(h);

				for (int w = 0; w < width; w++)
				{
					/*
					p[0] = 0;  // ERROR
					p[1] = 0;
					p[2] = 255;

					(*p).val[0] = 0;  // OK
					(*p).val[1] = 0;
					(*p).val[2] = 255;

					*p = Vec3b(0, 0, 255); // OK
					*/

					*p = Vec3b(b, g, r); // OK

					p++;
				}
			}
			return color_image;
		}

		void OpencvUtil::mat_add_value(
			cv::Mat& mat,
			float value
		)
		{
			int height = mat.rows;
			int width = mat.cols;

			for (int h = 0; h < height; h++) // rows
			{
				for (int w = 0; w < width; w++) // cols
				{
					mat.at<float>(h, w) += value;
				}
			}
		}

		void OpencvUtil::mat_subtract_value(
			cv::Mat& mat,
			float value
		)
		{
			OpencvUtil::mat_add_value(mat, -value);
		}

		
		cv::Mat OpencvUtil::get_binary_image(
			const cv::Mat& image, 
			const float min_binary_threshold
		)
		{
			cv::Mat binary_image;
			int threshold = int(min_binary_threshold * 255);
			cv::threshold(image, binary_image, threshold, 255, CV_THRESH_BINARY);
			return binary_image;
		}

		/*
		expand box width and height, and make sure expand_box<= max_box 
		50*40, with delta =20,  90*80
		*/
		cv::Rect OpencvUtil::expand_box(
			const cv::Rect& box,
			const int expand_width,
			const int expand_height,
			const cv::Size& max_size
		)
		{
			Rect box_expanded = Rect(
				box.x - expand_width, 
				box.y - expand_height,
				box.width +  2 * expand_width, 
				box.height + 2 * expand_height
			);
			return boundary(box_expanded,max_size);
		}

		

#pragma region box to origin
		/*
		diff_size(128,512)中的box_in_diff 缩放到 origin_rect(224,1024)(1400,0) 获得box_in_origin
		然后平移origin_rect.x(1400)即可得到原图origin中的异常box
		*/
		cv::Rect OpencvUtil::diff_box_to_origin_box(
			const cv::Rect& box_in_diff,
			const cv::Size& diff_size,
			const cv::Size& origin_size,
			const int origin_x_offset
		)
		{
			float origin_x_scale = origin_size.width / (diff_size.width*1.0f);
			float origin_y_scale = origin_size.height / (diff_size.height*1.0f);

			// use x_scale,y_scale to scale rect to origin image
			// (128,512) ===>(224,1024)
			cv::Rect box_in_origin(
				(int)(box_in_diff.x * origin_x_scale),
				(int)(box_in_diff.y * origin_y_scale),
				(int)(box_in_diff.size().width * origin_x_scale),
				(int)(box_in_diff.size().height * origin_y_scale)
			);
			// ===>(2048,1024)
			box_in_origin.x += origin_x_offset; //平移1400
			return box_in_origin;
		}

		void OpencvUtil::diff_boxs_to_origin_boxs(
			const std::vector<cv::Rect>& v_box_in_diff,
			const cv::Size& diff_size,
			const cv::Size& origin_size,
			const int origin_x_offset,
			std::vector<cv::Rect>& v_box_in_origin
		)
		{
			float origin_x_scale = origin_size.width / (diff_size.width*1.0f);
			float origin_y_scale = origin_size.height / (diff_size.height*1.0f);

			for (size_t i = 0; i < v_box_in_diff.size(); i++)
			{
				const cv::Rect& box_in_diff = v_box_in_diff[i];
				// use x_scale,y_scale to scale rect to origin image
				// (128,512) ===>(224,1024)
				cv::Rect box_in_origin(
					(int)(box_in_diff.x * origin_x_scale),
					(int)(box_in_diff.y * origin_y_scale),
					(int)(box_in_diff.size().width * origin_x_scale),
					(int)(box_in_diff.size().height * origin_y_scale)
				);
				// ===>(2048,1024)
				box_in_origin.x += origin_x_offset; //平移1400
				v_box_in_origin.push_back(box_in_origin);
			}
		}
#pragma endregion

	
#pragma region contours

		bool OpencvUtil::get_contours(
			const cv::Mat& diff,
			const float box_min_binary_threshold,
			contours_t& contours
		)
		{
			CHECK(!diff.empty()) << "invalid mat";
			cv::Mat diff_8uc1;
			if (diff.type() == CV_32FC1) {
				diff.convertTo(diff_8uc1, CV_8UC1, 255, 0); //CV_8UC1 [0,255]
			}
			else {
				diff_8uc1 = diff;
			}

			cv::Mat binary_diff;
			int threshold = (int)(box_min_binary_threshold * 255);
			cv::threshold(diff_8uc1, binary_diff, threshold, 255, CV_THRESH_BINARY);

			cv::Mat element = getStructuringElement(MORPH_CROSS, Size(5, 5));
			morphologyEx(binary_diff, binary_diff, MORPH_CLOSE, element);

			vector<Vec4i> hierarchy;
			findContours(binary_diff, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			return contours.size()>0;
		}

		bool OpencvUtil::get_contours(
			const cv::Mat& diff,
			const float box_min_binary_threshold,
			const float contour_min_area,
			contours_t& contours
		)
		{
			contours_t all_contours;

			OpencvUtil::get_contours(
				diff,
				box_min_binary_threshold,
				all_contours
			);

			for (size_t i = 0; i < all_contours.size(); i++)
			{
				contour_t contour = all_contours[i];
				if (cv::contourArea(contour) > contour_min_area) {
					contours.push_back(contour);
				}
			}

			return contours.size()>0;
		}

#pragma endregion

	
#pragma region boxs
		/*
		对diff图进行处理获取异常boxs,返回有无boxs
		*/
		bool OpencvUtil::get_boxs(
			const cv::Mat& diff,
			const float box_min_binary_threshold,
			boxs_t& boxs
		)
		{
			/*
			diff的取值范围0,1,...255
			255代表白色，0代表黑色，越亮的区域代表异常的可能性越大。

			cv::threshold(diff, binary_diff, box_threshold,255, CV_THRESH_BINARY);
			binary_diff只有2个取值。0,255
			*/
			CHECK(!diff.empty()) << "invalid mat";

#ifdef shared_DEBUG
			{
				LOG(INFO)<<"[API] get_value_count(diff) = " << OpencvUtil::get_value_count(diff) << std::endl;
				// 180
				cv::imwrite("railway/compare_000_diff.jpg", diff);
			}
#endif
			
			cv::Mat diff_8uc1;
			if (diff.type() == CV_32FC1) {
				diff.convertTo(diff_8uc1, CV_8UC1, 255, 0); //CV_8UC1 [0,255]
			}
			else {
				diff_8uc1 = diff;
			}
			cv::Mat binary_diff;
			int box_threshold = int(box_min_binary_threshold*255);

			//std::cout << " box_threshold = " << box_threshold << std::endl;
			cv::threshold(diff_8uc1, binary_diff, box_threshold,255, CV_THRESH_BINARY);
			
#ifdef shared_DEBUG
			{
				LOG(INFO)<<"[API] get_value_count(binary_diff) = " << OpencvUtil::get_value_count(binary_diff) << std::endl; // only 0,255 two values
				// we'd better not use OTSU which may lead anomaly errors.
				cv::Mat binary_diff_error;
				cv::threshold(diff_8uc1, binary_diff_error, 0, 255, CV_THRESH_OTSU);
				cv::imwrite("railway/compare_111_ok.jpg", binary_diff);
				cv::imwrite("railway/compare_222_error.jpg", binary_diff_error);
			}
#endif
			cv::Mat element = getStructuringElement(MORPH_CROSS, Size(5, 5));
			morphologyEx(binary_diff, binary_diff, MORPH_CLOSE, element);

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(binary_diff, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			for (size_t i = 0; i < contours.size(); i++) {
				cv::Rect rect_in_diff;
				rect_in_diff = boundingRect(contours[i]);
				boxs.push_back(rect_in_diff);
			}

#ifdef shared_DEBUG
			cv::Mat diff_with_contours;
			DisplayUtil::draw_contours(diff, contours,2, diff_with_contours);
			cv::imwrite("railway/diff_with_contours.jpg", diff_with_contours);
#endif // shared_DEBUG

			return boxs.size() > 0;
		}

		bool OpencvUtil::get_boxs_and(
			const cv::Mat& diff,
			const float box_min_binary_threshold,
			const int box_min_height,
			const int box_min_width,
			boxs_t& boxs
		)
		{
			boxs_t full_boxs;
			OpencvUtil::get_boxs(diff, box_min_binary_threshold, full_boxs);
			for (size_t i = 0; i < full_boxs.size(); i++)
			{
				const cv::Rect& box = full_boxs[i];
				if (box.height >= box_min_height && box.width >= box_min_width)
				{
					boxs.push_back(box);
				}
			}
			return boxs.size() > 0;
		}

		bool OpencvUtil::get_boxs_or(
			const cv::Mat& diff,
			const float box_min_binary_threshold,
			const int box_min_height,
			const int box_min_width,
			boxs_t& boxs
		)
		{
			boxs_t full_boxs;
			OpencvUtil::get_boxs(diff, box_min_binary_threshold, full_boxs);
			for (size_t i = 0; i < full_boxs.size(); i++)
			{
				const cv::Rect& box = full_boxs[i];
				if (box.height >= box_min_height || box.width >= box_min_width)
				{
					boxs.push_back(box);
				}
			}
			return boxs.size() > 0;
		}


		/*
		(1)根据diff(256*128)获取boxs_in_diff
		(2)利用将diff(256*128)中的boxs_in_diff缩放到原图origin(2048*1024)
		(3)对于box过滤掉小于min_width,min_height的区域
		(4)传出原图origin(2048*1024)中的有效boxs_in_origin
		*/
		bool OpencvUtil::get_boxs_and(
			const cv::Mat& diff,
			const cv::Size& origin_size,
			const int origin_x_offset,
			const float box_min_binary_threshold,
			const int box_min_height,
			const int box_min_width,
			boxs_t& boxs
		)
		{
			// (1) get diff v_white_boxs
			std::vector<cv::Rect> diff_boxs,origin_boxs;
			bool has_diff_boxs = OpencvUtil::get_boxs(diff, box_min_binary_threshold, diff_boxs);
			if (!has_diff_boxs)
				return false;
		
			// (2) scale diff box to origin box
			diff_boxs_to_origin_boxs(diff_boxs, diff.size(), origin_size, origin_x_offset, origin_boxs);

			// (3) filter box by height and width
			for (size_t i = 0; i < origin_boxs.size(); i++)
			{
				const cv::Rect& origin_box = origin_boxs[i];
				if (origin_box.height>= box_min_height && origin_box.width >= box_min_width)
				{
					boxs.push_back(origin_box);
				}
			}
			return boxs.size() > 0;
		}

		bool OpencvUtil::get_boxs_or(
			const cv::Mat& diff,
			const cv::Size& origin_size,
			const int origin_x_offset,
			const float box_min_binary_threshold,
			const int box_min_height,
			const int box_min_width,
			boxs_t& boxs
		)
		{
			// (1) get diff v_white_boxs
			std::vector<cv::Rect> diff_boxs, origin_boxs;
			bool has_diff_boxs = OpencvUtil::get_boxs(diff, box_min_binary_threshold, diff_boxs);
			if (!has_diff_boxs)
				return false;

			// (2) scale diff box to origin box
			diff_boxs_to_origin_boxs(diff_boxs, diff.size(), origin_size, origin_x_offset, origin_boxs);

			// (3) filter box by height and width
			for (size_t i = 0; i < origin_boxs.size(); i++)
			{
				const cv::Rect& origin_box = origin_boxs[i];
				if (origin_box.height >= box_min_height || origin_box.width >= box_min_width)
				{
					boxs.push_back(origin_box);
				}
			}
			return boxs.size() > 0;
		}


		bool OpencvUtil::get_boxs_and(
			const boxs_t& boxs,
			const int box_min_height,
			const int box_min_width,
			boxs_t& results
		)
		{
			for (size_t i = 0; i < boxs.size(); i++)
			{
				const cv::Rect& box = boxs[i];
				if (box.height >= box_min_height && box.width >= box_min_width)
				{ // and
					results.push_back(box);
				}
			}
			return results.size() > 0;
		}

		bool OpencvUtil::get_boxs_or(
			const boxs_t& boxs,
			const int box_min_height,
			const int box_min_width,
			boxs_t& results
		)
		{
			for (size_t i = 0; i < boxs.size(); i++)
			{
				const cv::Rect& box = boxs[i];
				if (box.height >= box_min_height || box.width >= box_min_width)
				{ // and
					results.push_back(box);
				}
			}
			return results.size() > 0;
		}

		bool OpencvUtil::get_contours_and_boxs(
			const cv::Mat& diff,
			const float box_min_binary_threshold,
			contours_t& contours,
			boxs_t& boxs
		)
		{
			get_contours(diff, box_min_binary_threshold, contours);
			for (size_t i = 0; i < contours.size(); i++) {
				cv::Rect box;
				box = boundingRect(contours[i]);
				boxs.push_back(box);
			}
			return boxs.size() > 0;
		}
#pragma endregion

#pragma region filter topwire/railway boxs
		/*
		for railway过滤掉全部高亮的区域，保留水滴对应的异常矩形框
		对origin中的boxs按照像素的强度过滤
		如果区域box太亮，则非常有可能是水滴，不认为是异常。
		avg_pixel_intensity<threshold,则保留box
		*/
		bool OpencvUtil::filter_railway_boxs(
			const cv::Mat& origin,
			const boxs_t& boxs,
			const float box_pixel_threshold, // >threshold, filtered out
			boxs_t& result_boxs
		)
		{
			CHECK(origin.type()==CV_8UC1) << "invalid mat";

			for (size_t i = 0; i < boxs.size(); i++)
			{
				const cv::Rect& box = boxs[i];
				cv::Mat box_mat = origin(box);
				float avg_pixel_intensity = OpencvUtil::get_average_pixel(box_mat) / 255.0f;

				LOG(INFO) << "[API] avg_pixel_intensity = " << avg_pixel_intensity << std::endl;

				if (avg_pixel_intensity<= box_pixel_threshold)
				{
					result_boxs.push_back(box);
				}
				else {
#ifdef shared_DEBUG
					//LOG(INFO)<<"[API] [FILTER OUT] avg_pixel_intensity >= " << box_pixel_threshold << std::endl;
					//cv::imwrite("railway/box/" + std::to_string(i) + "_box.jpg", box_mat);
#endif // shared_DEBUG
				}
			}

			return result_boxs.size() > 0;
		}

		/*
		for railway按照box的标准差stdev过滤
		如果stdev比较小<=20,那么表明box区域和周围区域比较接近，则不是异常。
		如果stdev比较大，那么box区域和周围区域差异比较大，认为是异常。
		stdev>threshold，则保留box
		*/
		bool OpencvUtil::filter_railway_boxs2(
			const cv::Mat& origin,
			const boxs_t& boxs,
			const int box_expand_width,
			const int box_expand_height,
			const float box_stdev_threshold,
			boxs_t& result_boxs
		)
		{
			CHECK(origin.type() == CV_8UC1) << "invalid mat";

			for (size_t i = 0; i < boxs.size(); i++)
			{
				const cv::Rect box = boxs[i];
				cv::Rect expand_box = OpencvUtil::expand_box(
					box,
					box_expand_width, 
					box_expand_height, 
					origin.size()
				);

				cv::Mat expand_box_mat = origin(expand_box);
				float box_pixel_stdev = OpencvUtil::get_image_pixel_stdev(expand_box_mat);

				LOG(INFO) << "[API] box_pixel_stdev = " << box_pixel_stdev << std::endl;

				if (box_pixel_stdev >= box_stdev_threshold)
				{
					result_boxs.push_back(box);
				}
				else {
#ifdef shared_DEBUG
					//LOG(INFO)<<"[API] [FILTER OUT] box_pixel_stdev >= " << box_pixel_stdev << std::endl;
#endif // shared_DEBUG
				}
			}

			return result_boxs.size() > 0;
		}

		float OpencvUtil::get_average_pixel(const cv::Mat& image)
		{
			float sum = 0.f;
			for (size_t row = 0; row < image.rows; row++)
			{
				for (size_t col = 0; col < image.cols; col++)
				{
					sum += image.at<uchar>(row, col);
				}
			}
			return sum/(image.rows*image.cols*1.0f);
		}

		float OpencvUtil::get_average_pixel(const cv::Mat& image, const cv::Rect& box)
		{
			cv::Mat box_mat = image(box);
			return get_average_pixel(box_mat);
		}

		float OpencvUtil::get_image_pixel_stdev(const cv::Mat& image)
		{
			float avg_pixel = get_average_pixel(image);
			float variance = 0.f;
			for (size_t row = 0; row < image.rows; row++)
			{
				for (size_t col = 0; col < image.cols; col++)
				{
					float pixel = image.at<uchar>(row, col);
					variance += (pixel- avg_pixel)*(pixel - avg_pixel);
				}
			}
			return sqrt(variance / (image.rows*image.cols*1.0f));
		}

		/*
		for topwire:过滤掉avg白色区域>=avg黑色区域
		*/
		bool OpencvUtil::filter_topwire_boxs(
			const cv::Mat& origin_topwire_roi,
			const boxs_t& boxs_in_diff,
			const cv::Mat& diff,
			const float box_min_binary_threshold,
			boxs_t& result_boxs
		)
		{
			// roi : 224*1024
			// diff: 128*512
			CHECK(origin_topwire_roi.type() == CV_8UC1) << "invalid mat";

			cv::Mat diff_8uc1;
			if (diff.type() == CV_32FC1) {
				diff.convertTo(diff_8uc1, CV_8UC1, 255, 0); //CV_8UC1 [0,255]
			}
			else {
				diff_8uc1 = diff;
			}
			cv::Mat binary_diff;
			int box_threshold = int(box_min_binary_threshold * 255);
			cv::threshold(diff_8uc1, binary_diff, box_threshold, 255, CV_THRESH_BINARY);

			cv::Mat resized_origin;
			cv::resize(origin_topwire_roi, resized_origin, diff.size());

#ifdef shared_DEBUG
			LOG(INFO)<<"[API] ====================================================\n";
			LOG(INFO)<<"[API] filter_topwire_boxs\n";
			LOG(INFO)<<"[API] ====================================================\n";
			cv::imwrite("box/resized_origin.jpg", resized_origin);
			cv::imwrite("box/diff.jpg", diff);
#endif // shared_DEBUG

			for (size_t i = 0; i < boxs_in_diff.size(); i++)
			{
				const cv::Rect& box = boxs_in_diff[i];
				
				cv::Mat box_binary_diff = binary_diff(box);
				cv::Mat box_origin = resized_origin(box);

#ifdef shared_DEBUG
				cv::Mat box_diff = diff(box);
				cv::imwrite("box/" + to_string(i) + "_1_box_diff.jpg", box_diff);

				cv::imwrite("box/" + to_string(i) + "_2_box_binary_diff.jpg", box_binary_diff);
				cv::imwrite("box/" + to_string(i) + "_3_box_origin.jpg", box_origin);
#endif // shared_DEBUG

				float avg_white_pixel, avg_black_pixel;
				OpencvUtil::get_average_white_black_piexl(
					box_binary_diff, 
					box_origin, 
					avg_white_pixel,
					avg_black_pixel
				);

				if (avg_white_pixel<= avg_black_pixel)
				{ // 异常roi对应的均值 > 无异常区域均值  (异常区域所占的比重越大，越应该保留)
					result_boxs.push_back(box);
				}

#ifdef shared_DEBUG
				cv::Mat resized_origin_with_box;
				DisplayUtil::draw_box(resized_origin, box, 1, resized_origin_with_box);
				cv::imshow("gray", resized_origin_with_box);

				cv::Mat binary_diff_with_box;
				DisplayUtil::draw_box(binary_diff, box, 1, binary_diff_with_box);
				cv::imshow("binary", binary_diff_with_box);

				cv::waitKey(0);
#endif // shared_DEBUG

			}

#ifdef shared_DEBUG
			LOG(INFO)<<"[API] ====================================================\n";
			LOG(INFO)<<"[API] filter_topwire_boxs  result_boxs.size()="<< result_boxs.size()<<std::endl;
			LOG(INFO)<<"[API] ====================================================\n";
#endif // shared_DEBUG

			return result_boxs.size() > 0;
		}

		void OpencvUtil::get_average_white_black_piexl(
			const cv::Mat& binary_diff,
			const cv::Mat& gray,
			float& avg_white_pixel,
			float& avg_black_pixel
		)
		{
			/*
			binary_diff包含0(黑色),255(白色)
			gray包含0,1,2，...255
			计算255对应位置元素的avg_255  计算0对应位置元素的avg_0
			如果avg_255>=avg_0则drop该box (异常区域对应的原图像素过亮，则drop)
			利用了一个性质：如果原图中roi区域有异常，通常该区域比较黑暗。而比较亮的区域通常不是异常。
			*/
			CHECK(binary_diff.size() == gray.size()) << "mismatch mat size";

#ifdef shared_DEBUG
			LOG(INFO)<<"[API] *****************************multiply*****************************\n";
			LOG(INFO)<<"[API] binary_diff.size()=" << binary_diff.size() << std::endl;
			LOG(INFO)<<"[API] gray.size()=" << gray.size() << std::endl;

			LOG(INFO)<<"[API] get_value_count(binary_diff) =" << OpencvUtil::get_value_count(binary_diff) << std::endl;
			LOG(INFO)<<"[API] get_value_count(gray) =" << OpencvUtil::get_value_count(gray) << std::endl;

			//LOG(INFO)<<"[API] 11111111111111111111111111111111111111111111111\n";
			//std::cout << binary_diff << std::endl;
			//LOG(INFO)<<"[API] 11111111111111111111111111111111111111111111111\n";

			//LOG(INFO)<<"[API] 2222222222222222222222222222222222222222222\n";
			//std::cout << gray << std::endl;
			//LOG(INFO)<<"[API] 2222222222222222222222222222222222222222222\n";
#endif // shared_DEBUG

			float avg_255=0, avg_0 =0;
			int count_255=0, count_0=0;
			for (size_t row = 0; row < gray.rows; row++)
			{
				for (size_t col = 0; col < gray.cols; col++)
				{
					int b = binary_diff.at<uchar>(row, col);
					int g = gray.at<uchar>(row, col);
					if (b==255)
					{
						count_255++;
						avg_255 += g;
					}
					else {
						count_0++;
						avg_0 += g;
					}
				}
			}

			avg_255 /= (count_255*1.0);
			avg_0 /= (count_0*1.0);

			avg_white_pixel = avg_255;
			avg_black_pixel = avg_0;

#ifdef shared_DEBUG
			LOG(INFO)<<"[API] count_255=" << count_255 << std::endl;
			LOG(INFO)<<"[API] count_0=" << count_0 << std::endl;
			LOG(INFO)<<"[API] avg_255=" << avg_white_pixel << std::endl;
			LOG(INFO)<<"[API] avg_0=" << avg_black_pixel << std::endl;

			if (avg_255 >= avg_0) {
				LOG(INFO)<<"[API] avg_255>=avg_0 drop" << std::endl;
			}
			else {
				LOG(INFO)<<"[API] avg_255<avg_0 keep" << std::endl;
			}

#endif // shared_DEBUG
		}
#pragma endregion


		bool OpencvUtil::get_railway_box(
			const cv::Mat& diff,
			const cv::Size& origin_size,
			const int origin_x_offset,
			const float railway_box_ratio, // >=ratio则认为是有效的railway box区域
			cv::Rect& box
		)
		{
			/*
			updated at 20180720
			可能有2条或者多条railway box，目前为了接口的一致性，只取面积最大的一条。
			后续需要对多条railway box进行处理，目前所有的API接口需要重新设计。
			*/
			CHECK(!diff.empty()) << "invalid mat";

			int width_threshold_in_diff = 5; // diff图中railway的最小宽度  18,19

			// vertial project
			boxs_t possible_boxs;
			int min_width = width_threshold_in_diff;
			int min_height = diff.rows - 20;
			bool has_boxs = OpencvUtil::get_boxs_and(diff, 0.5f, min_height, min_width, possible_boxs);
			if (!has_boxs)
			{
#ifdef shared_DEBUG
				LOG(INFO)<<"[API] no railway/topwire boxs" << std::endl;
#endif 
				return false;
			}

			// 对N个boxs的面积进行sort,只取面积最大的box
			std::sort(possible_boxs.begin(), possible_boxs.end(), OpencvUtil::rect_compare);
			
#ifdef shared_DEBUG
			LOG(INFO)<<"[API] possible_boxs.size()=" << possible_boxs.size() << std::endl;
			for (size_t i = 0; i < possible_boxs.size(); i++)
			{
				std::cout << possible_boxs[i] << std::endl;
			}
#endif 
			cv::Rect max_box = possible_boxs[0]; // max box
			cv::Mat max_box_diff = diff(max_box);

#ifdef shared_DEBUG
			cv::imwrite("railway/max_box_diff.jpg", max_box_diff);
#endif 

			cv::Mat diff_8uc1;
			if (max_box_diff.type() == CV_32FC1) {
				max_box_diff.convertTo(diff_8uc1, CV_8UC1, 255, 0); //CV_8UC1 [0,255]
			}
			else {
				diff_8uc1 = max_box_diff;
			}
			cv::Mat binary_diff;
			float box_min_binary_threshold = 0.5f;
			int box_threshold = int(box_min_binary_threshold * 255);
			cv::threshold(diff_8uc1, binary_diff, box_threshold, 255, CV_THRESH_BINARY);

			// (2) 获取每列白色像素的数量count
			std::vector<int> v_col_white_pixel_count;
			OpencvUtil::get_col_white_pixel_count(binary_diff, v_col_white_pixel_count);

#ifdef shared_DEBUG
			{
				cv::Mat diff_horizontal = OpencvUtil::get_horizontal_project_mat(binary_diff);
				cv::Mat diff_vertical = OpencvUtil::get_vertical_project_mat(binary_diff);

				cv::imwrite("railway/diff_0.jpg", binary_diff);
				cv::imwrite("railway/diff_1_horizontal.jpg", diff_horizontal);
				cv::imwrite("railway/diff_2_vertical.jpg", diff_vertical);
			}
#endif

			int width = binary_diff.cols;
			int height = binary_diff.rows;

			//float railway_box_ratio = 0.95f; // 像素数量百分比>=ratio才认为是railway的box
			bool left_found, right_found;
			int left_col, right_col;

			// (3) 判断left+right 钢轨区域所在的col
			// (3.1)最左侧
			for (int col = 0; col < width; col++)
			{
				float ratio = v_col_white_pixel_count[col] / (height*1.0);

#ifdef shared_DEBUG
				//LOG(INFO)<<"[API] # " << col << ", ratio = " << ratio << std::endl;
#endif

				if (ratio >= railway_box_ratio)//进入钢轨最left侧区域 
				{
					left_found = true;
					left_col = col + max_box.x ; // plus box.x
					break;
				}
			}

#ifdef shared_DEBUG
			LOG(INFO)<<"[API] ======================================\n";
#endif

			// (3.2) 最右侧
			for (int col = width - 1; col >= 0; col--)
			{
				float ratio = v_col_white_pixel_count[col+1] / (height*1.0); // from last v[width]

#ifdef shared_DEBUG
				//LOG(INFO)<<"[API] # " << col << ", ratio = " << ratio << std::endl;
#endif
				if (ratio >= railway_box_ratio)//进入钢轨最right区域
				{
					right_found = true;
					right_col = col + max_box.x; // plus box.x
					break;
				}
			}

#ifdef shared_DEBUG
			{
				LOG(INFO)<<"[API] =================================================\n";
				LOG(INFO)<<"[API] left_col=" << left_col << std::endl;
				LOG(INFO)<<"[API] right_col=" << right_col << std::endl;
				LOG(INFO)<<"[API] width in diff =" << right_col - left_col << std::endl;
				LOG(INFO)<<"[API] =================================================\n";
			}
#endif

			if ( (!left_found) || (!right_found) || right_col - left_col <= width_threshold_in_diff)
			{ // no railway box
				LOG(INFO)<<"[API] =================================================\n";
				LOG(INFO)<<"[API]  no railway/topwire box cropped \n";
				LOG(INFO)<<"[API] left_col=" << left_col << std::endl;
				LOG(INFO)<<"[API] right_col=" << right_col << std::endl;
				LOG(INFO)<<"[API] width in diff =" << right_col - left_col <<"<="<< width_threshold_in_diff << std::endl;
				LOG(INFO)<<"[API] =================================================\n";

				cv::Mat diff_horizontal = OpencvUtil::get_horizontal_project_mat(binary_diff);
				cv::Mat diff_vertical = OpencvUtil::get_vertical_project_mat(binary_diff);

				cv::imwrite("railway/diff_0.jpg", binary_diff);
				cv::imwrite("railway/diff_1_horizontal.jpg", diff_horizontal);
				cv::imwrite("railway/diff_2_vertical.jpg", diff_vertical);

				cv::Rect box_in_diff(left_col, 0, right_col - left_col, height);
				cv::Mat diff_with_railway_box;
				DisplayUtil::draw_box(diff, box_in_diff, 2, diff_with_railway_box);
				cv::imwrite("railway/diff_3_railway_box.jpg", diff_with_railway_box);

				return false;
			}
			
			cv::Rect box_in_diff(left_col, 0, right_col - left_col, height);
#ifdef shared_DEBUG
			{
				cv::Mat diff_with_railway_box;
				DisplayUtil::draw_box(diff, box_in_diff, 2, diff_with_railway_box);

				cv::imwrite("railway/diff_3_railway_box.jpg", diff_with_railway_box);
			}
#endif

			cv::Rect box_in_origin = diff_box_to_origin_box(
				box_in_diff,
				diff.size(),
				origin_size,
				origin_x_offset
			);
			box_in_origin.y = 0;
			box_in_origin.height = origin_size.height; // keep 0 and 1024

			box = box_in_origin;
			return true;
		}

		bool OpencvUtil::get_topwire_box(
			const cv::Mat& diff,
			const cv::Size& origin_size,
			const int origin_x_offset,
			const float topwire_box_ratio,
			cv::Rect& box
		)
		{
			return OpencvUtil::get_railway_box(
				diff, 
				origin_size, 
				origin_x_offset, 
				topwire_box_ratio, 
				box
			);
		}

#pragma region gap and fill
		//水平投影,得到可能的唯一轨缝区域,返回has_gap
		bool OpencvUtil::get_horizontal_gap_box(
			const cv::Mat& diff, 
			const float gap_ratio, // v_row[row]/width > ratio则认为是有效的gap区域
			cv::Rect& fill_box,
			cv::Rect& gap_box_in_diff
		)
		{
			CHECK(!diff.empty()) << "invalid mat";
			int width = diff.cols;
			int height = diff.rows;

			// (1)对diff数据进行blur预处理
			cv::Mat blur_diff;
			cv::blur(diff, blur_diff, cv::Size(3, 3));
			cv::threshold(blur_diff, blur_diff, 0, 255, CV_THRESH_OTSU);

#ifdef shared_DEBUG
			{
				cv::imwrite("railway/gap/diff.jpg", diff);
				cv::imwrite("railway/gap/blur_diff.jpg", blur_diff);
			}
#endif

			// (2) 获取每行白色像素的数量count
			std::vector<int> v_row_white_pixel_count(height+1);  
			// 使用height+1行在最后面加一个tail确保一定能够退出白色区域 v[512] = 0
			// v[512]存储每行白色像素的数量：pxiel_value=255的数量(max_count = width = 128)

			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < width; col++)
				{
					int pxiel_value = blur_diff.at<uchar>(row, col);
					if (pxiel_value == 255)   //如果目标是白色  
					{
						v_row_white_pixel_count[row]++;
					}
				}
				//LOG(INFO)<<"[API] v_row_white_pixel_count[" << row << "]=" << v_row_white_pixel_count[row] << std::endl;
			}

			// (3) 寻找白色区域框v_white_boxs
			int gap_min_height = 10; // gap至少10个像素高
			vector<cv::Rect> v_white_boxs;//用于存储分割出来的每个白色区域框
			int start_row = 0;//记录进入白色区的row  
			int end_row = 0;//记录退出白色区的row  
			bool in_block = false;//是否在白色区内  
			for (int row = 0; row < v_row_white_pixel_count.size(); row++)
			{
				if (!in_block && v_row_white_pixel_count[row] != 0)//进入白色区  
				{
					in_block = true;
					start_row = row;
				}
				else if (in_block && v_row_white_pixel_count[row] == 0) // 已经进入白色区
				{
					/*
					退出白色区有2种方式：
					(1) row<height-1  v[row] = 0
					(2) row=height-1  最后一行即使v[row]>0，仍然退出白色区
					通过设置v[height] =0 ，确保一定能够退出白色区域。此时end_row = height
					*/
					end_row = row;
					in_block = false;
					//LOG(INFO)<<"[API] start_row,end_row=" << start_row << "," << end_row << std::endl;

					if (end_row - start_row >gap_min_height)
					{
						cv::Rect white_box(0, start_row, width, end_row - start_row);
						v_white_boxs.push_back(white_box);
					}
				}
			}
			//LOG(INFO)<<"[API] v_white_boxs.size()=" << v_white_boxs.size() << std::endl;

			// (4) 对v_white_boxs进行gap_ratio阈值过滤获取 v_refine_boxs
			std::vector<bool> v_refine_flag; //优化过的标记 flag 为 true
			std::vector<cv::Rect> v_refine_boxs; // 满足投影阈值的优化后的矩形框
			for (int row = 0; row < v_white_boxs.size(); row++)
			{
				int result_start_row = 0;
				int result_end_row = 0;
				bool flag_start = false;
				bool flag_end = false;

				cv::Rect& box = v_white_boxs[row];

				int min_height = box.y;
				int max_height = box.y + box.height;

				// 从上往下寻找start_row
				for (int j = min_height; j < max_height; j++)
				{
					if ((v_row_white_pixel_count[j] * 1.0 / width) > gap_ratio) 
					{
						result_start_row = j;
						flag_start = true;
						break;
					}
				}

				// 从下往上寻找end_row
				for (int k = max_height-1; k >= min_height; k--)
				{
					if ((v_row_white_pixel_count[k] * 1.0 / width) > gap_ratio) 
					{
						result_end_row = k;
						flag_end = true;
						break;
					}
				}

				cv::Rect refine_box;
				if ( flag_start && flag_end ) 
				{
					//满足投影阈值的优化后矩形框
					refine_box = cv::Rect(0, result_start_row, width, result_end_row - result_start_row + 1);
					v_refine_flag.push_back(true);
				}
				else {
					refine_box = cv::Rect(0, 0, 0, 0); // 确保refine boxs和white boxs的数量相等
					v_refine_flag.push_back(false);
				}
				v_refine_boxs.push_back(refine_box); 
			}
			//LOG(INFO)<<"[API] v_refine_boxs.size()=" << v_refine_boxs.size() << std::endl;

			// (5) 在满足投影阈值的所有优化矩形框v_refine_boxs中寻找面积最大的那个box作为最终结果
			double max_area = 0.0;
			bool has_gap = false;
			for (int i = 0; i < v_refine_flag.size(); i++)
			{
				if (v_refine_flag[i]) // 只判断优化过的矩形框
				{
					int cur_area = v_refine_boxs[i].area();
					if (cur_area >= max_area)
					{
						max_area = cur_area;
						gap_box_in_diff = v_refine_boxs[i];//面积最大的作为轨缝box
						fill_box = v_white_boxs[i];// 对应的填充box
						has_gap = true;
					}
				}
			}
			
#ifdef shared_DEBUG
			{
				cv::Mat mat_with_white_boxs;
				DisplayUtil::draw_boxs(blur_diff, v_white_boxs, 1, mat_with_white_boxs);
				cv::imwrite("railway/gap/mat_with_white_boxs.jpg", mat_with_white_boxs);

				cv::Mat mat_with_refine_boxs;
				DisplayUtil::draw_boxs(blur_diff, v_refine_boxs, 1, mat_with_refine_boxs);
				cv::imwrite("railway/gap/mat_with_refine_boxs.jpg", mat_with_refine_boxs);
				
				cv::Mat mat_with_2gap_box;
				DisplayUtil::draw_box(blur_diff, fill_box, 1, mat_with_2gap_box); // 红色填充box
				cv::rectangle(mat_with_2gap_box, gap_box_in_diff, cv::Scalar(0, 255, 0), 1, 1, 0); // 绿色轨缝box
				cv::imwrite("railway/gap/mat_with_2gap_box.jpg", mat_with_2gap_box);
			}
#endif

			return has_gap;
		}

		void OpencvUtil::fill_mat(
			cv::Mat& mat,
			const cv::Rect& fill_box,
			const int delta_height,
			const int fill_value
		)
		{
			CHECK(!mat.empty()) << "invalid mat";
			// fill value for mat
			cv::Rect expand_fill_box = fill_box;
			
			int min_height = fill_box.y - delta_height;
			int max_height = fill_box.y + fill_box.height + delta_height;
			if (min_height<0)
			{
				min_height = 0;
				expand_fill_box.y = 0;
			}
			if (max_height>mat.size().height)
			{
				max_height = mat.size().height;
				expand_fill_box.height = max_height - expand_fill_box.y;
			}

			cv::Mat mat_with_fill_box;
			DisplayUtil::draw_box(mat, fill_box, 1, mat_with_fill_box); // 红色填充box

			cv::Mat mat_with_expand_fill_box;
			DisplayUtil::draw_box(mat, expand_fill_box, 1, mat_with_expand_fill_box); // 红色填充box

#ifdef shared_DEBUG
			{
				cv::imwrite("railway/gap/fill_no.jpg", mat);
				cv::imwrite("railway/gap/fill_with_box.jpg", mat_with_fill_box);
				cv::imwrite("railway/gap/fill_with_expand_box.jpg", mat_with_expand_fill_box);
			}
#endif

			int min_width = 0;
			int max_width = expand_fill_box.width;

			for (int h = min_height; h < max_height; h++) {
				for (int w = min_width; w < max_width; w++) {
					mat.at<uchar>(h, w) = fill_value;
				}
			}

#ifdef shared_DEBUG
			{
				cv::imwrite("railway/gap/fill_yes.jpg", mat);
			}
#endif

		}
#pragma endregion

#pragma region horizontal and vertical mat 

#pragma region row-col pixel count
		// for vertical  垂直投影  按col划分 v_col_white_pixel_count
		// for horizontal 水平投影 按row划分 
		
		bool OpencvUtil::get_col_white_pixel_count(
			const cv::Mat& diff,
			std::vector<int>& v_col_white_pixel_count
		)
		{
			int width = diff.cols;
			int height = diff.rows;

			int black_pixel = 0;
			int white_pixel = 255;

			// (2) 获取每col白色像素的数量count
			v_col_white_pixel_count.resize(width + 1);//创建用于储存每col白色像素个数的数组  
			// 使用width+1行在最后一列加一个tail确保一定能够退出白色区域 v[640] = 0
			// v[640]存储每col白色像素的数量：pxiel_value=255的数量(max_count = height = 480)

			for (int col = 0; col < width; col++)
			{
				for (int row = 0; row < height; row++)
				{
					int pxiel_value = diff.at<uchar>(row, col);
					if (pxiel_value == white_pixel)//如果目标是白色  
					{
						v_col_white_pixel_count[col]++;
					}
				}
				//LOG(INFO)<<"[API] [" << col << "]=" << v_col_white_pixel_count[col] << std::endl;
			}

			return true;
		}

		bool OpencvUtil::get_row_white_pixel_count(
			const cv::Mat& diff,
			std::vector<int>& v_row_white_pixel_count
		)
		{
			int width = diff.cols;
			int height = diff.rows;

			int black_pixel = 0;
			int white_pixel = 255;

			// (2) 获取每row白色像素的数量count
			v_row_white_pixel_count.resize(height + 1);
			// 使用height+1行在最后面加一个tail确保一定能够退出白色区域 v[480] = 0
			// v[480]存储每row白色像素的数量：pxiel_value=255的数量(max_count = width = 640)

			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < width; col++)
				{
					int pxiel_value = diff.at<uchar>(row, col);
					if (pxiel_value == white_pixel)   //如果目标是白色  
					{
						v_row_white_pixel_count[row]++;
					}
				}
				//LOG(INFO)<<"[API] [" << row << "]=" << v_row_white_pixel_count[row] << std::endl;
			}

			return true;
		}

#pragma endregion

		cv::Mat OpencvUtil::get_vertical_project_mat(
			const cv::Mat& diff
		)
		{
			CHECK(!diff.empty()) << "invalid mat";
			int width = diff.cols;
			int height = diff.rows;

			int black_pixel = 0;
			int white_pixel = 255;

			// (1)对diff数据进行blur预处理
			cv::Mat blur_diff;
			//cv::blur(diff, blur_diff, cv::Size(3, 3));
			cv::threshold(diff, blur_diff, 128, 255, CV_THRESH_BINARY);
			
			// (2) 获取每列白色像素的数量count
			std::vector<int> v_col_white_pixel_count;
			OpencvUtil::get_col_white_pixel_count(blur_diff, v_col_white_pixel_count);
			
			// (3) 获取白色区域垂直投影的直方图
			Mat vertical_project_mat(height, width, CV_8UC1);//垂直投影的画布  

			// (3.1)初始化为黑色图像
			for (int col = 0; col < width; col++)
			{
				for (int row = 0; row < height; row++)
				{
					vertical_project_mat.at<uchar>(row, col) = black_pixel;//背景设置为黑色  
				}
			}

			// (3.2)根据白色像素的count填充垂直投影直方图  
			for (int col = 0; col < width; col++)
			{
				for (int row = 0; row < v_col_white_pixel_count[col]; row++)
				{
					vertical_project_mat.at<uchar>(row, col) = white_pixel; // 设置为白色
				}
			}
			return vertical_project_mat;
		}

		cv::Mat OpencvUtil::get_horizontal_project_mat(
			const cv::Mat& diff
			)
		{
			CHECK(!diff.empty()) << "invalid mat";
			int width = diff.cols;
			int height = diff.rows;

			int black_pixel = 0;
			int white_pixel = 255;

			// (1)对diff数据进行blur预处理
			cv::Mat blur_diff;
			//cv::blur(diff, blur_diff, cv::Size(3, 3));
			cv::threshold(diff, blur_diff, 128, 255, CV_THRESH_BINARY);
			
			// (2) 获取每行白色像素的数量count
			std::vector<int> v_row_white_pixel_count;
			OpencvUtil::get_row_white_pixel_count(blur_diff, v_row_white_pixel_count);
		
			// (3) 获取白色区域水平投影的直方图
			Mat horizontal_project_mat(height, width, CV_8UC1);

			// (3.1)初始化为黑色图像
			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < width; col++)
				{
					horizontal_project_mat.at<uchar>(row, col) = black_pixel; // 0 for black
				}
			}

			// (3.2)根据白色像素的count填充水平直方图  
			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < v_row_white_pixel_count[row]; col++)
				{
					horizontal_project_mat.at<uchar>(row, col) = white_pixel;//设置直方图为白色 
				}
			}
			return horizontal_project_mat;
		}

#pragma endregion

#pragma region for distance

		bool OpencvUtil::get_horizontal_distance_roi(
			const cv::Mat& origin_mat,
			cv::Mat& roi
		)
		{
			CHECK(!origin_mat.empty()) << "invalid mat";

			bool roi_found = false;
			int black_pixel = 0;
			int white_pixel = 255;
			// (1)对diff数据进行blur预处理
			Mat diff;
			//threshold(origin_mat, diff, 0, 255, CV_THRESH_OTSU);
			cv::threshold(origin_mat, diff, 128, 255, CV_THRESH_BINARY);

#ifdef shared_DEBUG
			{
				cv::imwrite("distance/1_diff.jpg", diff);
			}
#endif

			int perPixelValue = 0;//每个像素的值  
			int width = origin_mat.cols;
			int height = origin_mat.rows;

			// (2) 获取每行白色像素的数量count
			std::vector<int> v_row_white_pixel_count;
			OpencvUtil::get_row_white_pixel_count(diff, v_row_white_pixel_count);

			// (3) 寻找白色区域框
			//用于储存分割出来的距离线ROI
			int start_row = 0;//记录进入字符区的索引  
			int end_row = 0;//记录进入空白区域的索引  
			bool in_block = false;//是否遍历到了字符区内

			int max_white_pixel_total_count = 0; // start_row,end_row之间的白色像素的数量总和
			int in_out_white_threshold = 10; // 进入，退出白色区域的阈值（最好动态计算出来） 默认使用10
			for (int row = 0; row < v_row_white_pixel_count.size(); row++)
			{
				if (!in_block && v_row_white_pixel_count[row] > in_out_white_threshold)//进入字符区  
				{
					in_block = true;
					start_row = row;
				}
				else if (in_block && v_row_white_pixel_count[row] < in_out_white_threshold)//进入空白区  
				{
					end_row = row;
					in_block = false;

					int temp_count = 0; // 白色像素的数量总和
					for (size_t j = start_row; j < end_row; j++) {
						temp_count += v_row_white_pixel_count[j];
					}

					if (temp_count>max_white_pixel_total_count)
					{
						max_white_pixel_total_count = temp_count;

						int expand_height = 14; // 对start_row,end_row分别扩充
						start_row = max(0, start_row - expand_height);
						end_row = min(origin_mat.rows, end_row + expand_height);

						//LOG(INFO)<<"[API] start_row,end_row=" << start_row << "," << end_row << std::endl;
						cv::Rect roi_box(0, start_row, width, end_row - start_row);

						roi = origin_mat(roi_box);
						roi_found = true;
					}
				}
			}

			return roi_found;
		}

		bool OpencvUtil::get_vertical_project_distance(
			const cv::Mat& origin_mat,
			int& left_col,
			int& right_col
		)//垂直投影  
		{
			CHECK(!origin_mat.empty()) << "invalid mat";

			bool left_found = false;
			bool right_found = false;

			int width = origin_mat.cols;
			int height = origin_mat.rows;
			int black_pixel = 0;
			int white_pixel = 255;

			Mat diff;
			//cv::blur(origin_mat, diff, cv::Size(3, 3));
			cv::threshold(origin_mat, diff, 0, 255, CV_THRESH_OTSU);
			
#ifdef shared_DEBUG
			{
				cv::imwrite("distance/2_diff.jpg", diff);
			}
#endif
			

			// (2) 获取每列白色像素的数量count
			std::vector<int> v_col_white_pixel_count;
			OpencvUtil::get_col_white_pixel_count(diff, v_col_white_pixel_count);

			// (3) 判断进入和退出白色区域所在的col
			// (3.1)往右侧处理找到最左侧col ===>
			for (int col = 0; col < width; col++)
			{
				if (v_col_white_pixel_count[col] == 0 && v_col_white_pixel_count[col + 1] > 0)//进入白色区域 
				{
					left_found = true;
					left_col = col;
					break;
				}
			}

			// (3.2) 往左侧处理找到最右侧col <===
			for (int col = width - 1; col >= 0; col--)
			{
				if (v_col_white_pixel_count[col + 1] == 0 && v_col_white_pixel_count[col] > 0)//进入白色区域
				{
					right_found = true;
					right_col = col;
					break;
				}
			}

#ifdef shared_DEBUG
			{
				LOG(INFO)<<"[API] left_col=" << left_col << std::endl;
				LOG(INFO)<<"[API] right_col=" << right_col << std::endl;
			}
#endif

			return left_found && right_found;
		}

		

#pragma endregion


			

			

	}
}// end namespace



#pragma region comments 
 /*
 CV_<bit_depth>(S|U|F)C<number_of_channels>

 (1) bit_depth 比特数---代表8bit,16bits,32bits,64bits
 比如说,如如果你现在创建了一个存储--灰度图片的Mat对象,这个图像的大小为宽100,高100,那么,现在这张
 灰度图片中有10000个像素点，它每一个像素点在内存空间所占的空间大小是8bit,8位--所以它对应的就是CV_8

 (2) S|U|F
 S--代表---signed int---有符号整形
 U--代表--unsigned int--无符号整形
 F--代表--float---------单精度浮点型

 (3) C<number_of_channels>代表一张图片的通道数
 Gray灰度图片是1通道图像  mat.channels()==1, mat.type() == CV_8UC1=0, CV_32FC1=5
 RGB彩色图像是3通道图像   mat.channels()==3, mat.type() == CV_8UC3=16, CV_32FC3=21
 // 8U对应<Vec3b>  32F对应<Vec3f>
 RGBA图像是4通道图像      mat.channels()==4

 B1.convertTo(B1, CV_32FC1); // CV_8UC1 ---> CV_32FC1  [0,255] char(1个字节)--->[0,1] float(4个字节)


 LOG(INFO)<<"[API] CV_8UC1="<< CV_8UC1 << std::endl;
 LOG(INFO)<<"[API] CV_8UC3=" << CV_8UC3 << std::endl;
 LOG(INFO)<<"[API] CV_32FC1=" << CV_32FC1 << std::endl;
 LOG(INFO)<<"[API] CV_32FC3=" << CV_32FC3 << std::endl;

 CV_8UC1=0
 CV_8UC3=16

 CV_32FC1=5
 CV_32FC3=21

 CV_8UC(n)

 像素取值范围
 The conventional ranges for R, G, and B channel values are:
 -   0 to 255 for CV_8U images
 -   0 to 65535 for CV_16U images
 -   0 to 1 for CV_32F images


 int height = mat.rows;
 ing width = mat.cols;

 // 单通道Mat访问数据
 int ROWS = 100; // height
 int COLS = 200; // width
 Mat img1(ROWS , COLS , CV_32FC1);

 for (int i=0; i<ROWS ; i++)
 {
 for (int j=0; j<COLS ; j++)
 {
 img1.at<float>(i,j) = 3.2f;
 // float for CV_32FC1
 // uchar for CV_8UC1
 }
 }

 // 3通道Mat访问数据 默认BGR

 int ROWS = 100; // height
 int COLS = 200; // width
 Mat img1(ROWS , COLS , CV_8UC3);

 for (int i=0; i<ROWS ; i++)
 {
 for (int j=0; j<COLS ; j++)
 {

 // cv::Vec3f for CV_32FC3
 // cv::Vec3b for CV_8UC3
 img1.at<cv::Vec3b>(i,j)[0]= 3.2f;  // B 通道
 img1.at<cv::Vec3b>(i,j)[1]= 3.2f;  // G 通道
 img1.at<cv::Vec3b>(i,j)[2]= 3.2f;  // R 通道
 }
 }


 threshold_type可以使用CV_THRESH_OTSU类型，这样该函数就会使用大律法OTSU得到的全局自适应阈值
 来进行二值化图片，而参数中的threshold不再起作用。


#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
 */

#pragma endregion