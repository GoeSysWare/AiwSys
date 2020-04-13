#include "numpy_util.h"

#include "display_util.h"

// std
#include <iostream>
#include <map>
#include <set>
using namespace std;

#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imdecode imshow
using namespace cv;

#ifdef USE_EIGEN
#include <Eigen/Dense>
using namespace Eigen;
#endif // USE_EIGEN


namespace watrix {
	namespace algorithm {

		#define CV_BGR(b,g,r) CV_RGB(r,g,b)

#pragma region math		
		float NumpyUtil::get_rad_angle(int x1, int y1, int x2, int y2)
		{
			// in radian  
			float angle = 0;
			if (x1 != x2 ){
				angle = atan2((y2-y1)*1.0f,(x2-x1)*1.0f);
			} else {
				angle = CV_PI / 2.f;
			}
			return angle;
		}

		float NumpyUtil::rad_to_degree(float rad)
		{
			return rad * 180 /CV_PI; // rad ---> degree
		}

		float NumpyUtil::degree_to_rad(float degree)
		{
			return degree * CV_PI / 180.f; // degree ---> rad
		}

		float NumpyUtil::get_rad_angle(const cv::Point2i& left, const cv::Point2i& right)
		{
			float rad_angle = get_rad_angle(left.x, left.y, right.x, right.y);
			return rad_angle;
		}

		float NumpyUtil::get_degree_angle(const cv::Point2i& left, const cv::Point2i& right)
		{
			float rad_angle = get_rad_angle(left.x, left.y, right.x, right.y);
			return rad_to_degree(rad_angle);
		}


		void NumpyUtil::max_min_of_vector(const std::vector<int>& v, int& max_value, int& min_value)
		{
			if (v.size()>0){
				max_value = MAXV(v);
				min_value = MINV(v);
			}
		}

		int NumpyUtil::get_index(const std::vector<int>& v, int value)
		{
			for(int i=0;i<v.size();++i){
				if (value == v[i]){
					return i;
				}
			}
			return -1;
		}

		int NumpyUtil::max_index(const std::vector<int>& v)
		{
			//std::vector<float> v{f0,f1,f2};
			auto biggest = std::max_element(v.begin(), v.end());
			int index = std::distance(v.begin(), biggest);
			return index;
		}

		int NumpyUtil::max_index(const std::vector<float>& v)
		{
			//std::vector<float> v{f0,f1,f2};
			auto biggest = std::max_element(v.begin(), v.end());
			int index = std::distance(v.begin(), biggest);
			return index;
		}


		std::vector<int> NumpyUtil::add(const std::vector<int>& a, const std::vector<int>& b)
		{
			assert(a.size()==b.size());
			std::vector<int> c(a.size());
			for(int i=0;i<a.size();++i){
				c[i] = a[i] + b[i];
			}
			return c;
		}

		std::vector<int> NumpyUtil::sub(const std::vector<int>& a, const std::vector<int>& b)
		{
			assert(a.size()==b.size());
			std::vector<int> c(a.size());
			for(int i=0;i<a.size();++i){
				c[i] = a[i] - b[i];
			}
			return c;
		}
	
		std::vector<int> NumpyUtil::multipy_by(const std::vector<int>& v, int value)
		{
			std::vector<int> c(v.size());
			for(int i=0;i<v.size();++i){
				c[i] = v[i] * value;
			}
			return c;
		}

		std::vector<int> NumpyUtil::devide_by(const std::vector<int>& v, int value)
		{
			std::vector<int> c(v.size());
			for(int i=0;i<v.size();++i){
				c[i] = v[i] / value;
			}
			return c;
		}

		int NumpyUtil::dot(const std::vector<int>& a,const std::vector<int>& b)
		{
			int result = 0;
			assert(a.size()==b.size());
			for(int i=0;i<a.size();++i){
				result += a[i]*b[i];
			}
			return result;
		}

		float NumpyUtil::dot(const std::vector<float>& a,const std::vector<float>& b)
		{
			float result = 0;
			assert(a.size()==b.size());
			for(int i=0;i<a.size();++i){
				result += a[i]*b[i];
			}
			return result;
		}

		
		float NumpyUtil::module(const std::vector<float>& v)
		{
			// |v1| vector module = sqrt(x1*x1+y1*y1)
			float ret = 0.0;
			for (std::vector<float>::size_type i = 0; i != v.size(); ++i)
			{
				ret += v[i] * v[i];
			}
			return std::sqrt(ret);
		}

		
		float NumpyUtil::cosine(
			const std::vector<float>& v1,
			const std::vector<float>& v2
		)
		{
			// cos = v1.v2/(|v1|*|v2|)
			assert(v1.size() == v2.size());
			return dot(v1, v2) / (module(v1) * module(v2));
		}

#pragma endregion


#pragma region cv
		void NumpyUtil::cv_resize(const cv::Mat& input, cv::Mat& output, const cv::Size& size)
		{
			cv::resize(input, output, size);
		}

		void NumpyUtil::cv_cvtcolor_to_gray(const cv::Mat& image, cv::Mat& gray)
		{
			if (image.channels() == 1){
				gray = image;
			} else {
				cv::cvtColor(image, gray, CV_BGR2GRAY); // 88ms
			}
		}

		void NumpyUtil::cv_cvtcolor_to_bgr(const cv::Mat& image, cv::Mat& bgr)
		{
			if (image.channels() == 1){
				cv::cvtColor(image, bgr, CV_GRAY2BGR);
			}else {
				bgr = image.clone();
			}
		}

		void NumpyUtil::cv_add_weighted(const cv::Mat& overlay, cv::Mat& image, double alpha)
		{
			// https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
			cv::Mat bgr_overlay_ = overlay;
			if (overlay.channels()==1){
				cv::cvtColor(overlay, bgr_overlay_, CV_GRAY2BGR);
			}
			cv::addWeighted(bgr_overlay_, alpha,  image, 1-alpha, 0, image);
		}

		void NumpyUtil::cv_threshold(const cv::Mat& gray, cv::Mat& mask)
		{
			// mask: binary mask with 0 and 255
			cv::threshold(gray, mask, 128, 255, CV_THRESH_BINARY); // COST TIME = 74 ms
		}

		void NumpyUtil::cv_houghlinesp(const cv::Mat& mask, std::vector<cv::Vec4i>& lines)
		{
			lines.clear();
			cv::HoughLinesP(mask, lines, 1, CV_PI/180, 200, 50, 10);
		}

		void NumpyUtil::cv_canny(const cv::Mat& image, cv::Mat& edges, int min_val, int max_val)
		{
			cv::Canny(image, edges, min_val, max_val);
		}

		void NumpyUtil::cv_findcounters(const cv::Mat& mask, contours_t& contours)
		{
			std::vector<Vec4i> hierarchy;
			cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		}

		void NumpyUtil::cv_dilate_mat(cv::Mat& diff,int dilate_size)
		{
			if (dilate_size > 0) {
				cv::Mat dilate_element = cv::getStructuringElement(MORPH_RECT, Size(dilate_size, dilate_size));
				cv::dilate(diff, diff, dilate_element);
			}
		}

		void NumpyUtil::cv_erode_mat(cv::Mat& diff,int erode_size)
		{
			if (erode_size > 0) {
				cv::Mat erode_element = cv::getStructuringElement(MORPH_RECT, Size(erode_size, erode_size));
				cv::erode(diff, diff, erode_element);
			}
		}


		// 8UC1 (gray)
		cv::Mat NumpyUtil::cv_connected_component_mask(const cv::Mat& binary_mask,int min_area_threshold)
		{
			// binary mask = 0,1
			/*
			// 首先进行图像形态学运算
			int kernel_size=5;
			cv::Mat element = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
			cv::Mat binary;
			morphologyEx(binary_mask, binary, MORPH_CLOSE, element);
			*/
			cv::Mat binary = binary_mask.clone();

			// 进行连通域分析
			cv::Mat labels; // w*h  label = 0,1,2,3,...N-1 (0- background)       CV_32S = 4
			cv::Mat stats; // N*5  表示每个连通区域的外接矩形和面积 [x,y,w,h, area]   CV_32S = 4
			cv::Mat centroids; // N*2  (x,y)                                     CV_32S = 4
			int num_components = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

			//cv::imwrite("binary_mask.jpg",binary_mask);
			//cv::imwrite("binary.jpg",binary);

			//std::cout<<" num_components =" << num_components << std::endl; // 8
			//std::cout<<" stats hw =" << stats.cols <<","<<stats.rows << std::endl; // 8*5
			//std::cout<<" stats =" << stats << std::endl;

			// 排序连通域并删除过小的连通域 min_area_threshold = 200
			for(int index =0; index<num_components; index++)
			{
				if ( stats.at<int>(index, cv::CC_STAT_AREA) <= min_area_threshold ) 
				{
					// check label == index
					for(int row=0;row<labels.rows;++row) 
					{
						for(int col=0;col<labels.cols;++col)
						{
							int label = labels.at<int>(row,col);
							if (label == index )
							{
								binary.at<uchar>(row, col) = 0; // mark as black
							} 
						}
					}	
				}
			}

			//cv::imwrite("binary_result.jpg",binary);
			return binary; // binary mask 0,1
		}

		cv::Mat NumpyUtil::cv_rotate_image(const cv::Mat& image, float degree)
		{
			int height = image.rows;
			int width = image.cols; 
			float rad = degree_to_rad(degree);

			float sin_rad = abs ( sin(rad) );
			float cos_rad = abs ( cos(rad) );

			int heightNew = width * sin_rad + height * cos_rad;
			int widthNew = height * sin_rad + width * cos_rad;

			cv::Point2f center(width / 2, height / 2);
			
			//getRotationMatrix2D (2,3) value 6 = CV_64F  CV_64FC1 ===> double
			cv::Mat rotation_maxtix = cv::getRotationMatrix2D(center, degree, 1);
			//std::cout<<"rotation_maxtix = "<< rotation_maxtix.type() << std::endl;

			rotation_maxtix.at<double>(0,2) += (widthNew - width) / 2;
			rotation_maxtix.at<double>(1,2) += (heightNew - height) / 2;

			//std::cout<<"rotation_maxtix2 = "<< rotation_maxtix << std::endl;

			cv::Mat rotation_image;
			cv::warpAffine(image, rotation_image, rotation_maxtix, cv::Size(widthNew, heightNew),INTER_LINEAR );

			//cv::imwrite("1_origin.jpg",image);
			//cv::imwrite("2_rotate.jpg",rotation_image);
			return rotation_image;
		}

		cv::Mat NumpyUtil::cv_rotate_image_keep_size(const cv::Mat& image, float degree)
		{
			int height = image.rows;
			int width = image.cols; 
			cv::Size size(width, height);
			float rad = degree_to_rad(degree);

			cv::Point2f center(width / 2, height / 2);
			//getRotationMatrix2D (2,3) value 6 = CV_64F  CV_64FC1 ===> double
			cv::Mat rotation_maxtix = cv::getRotationMatrix2D(center, degree, 1);
			//std::cout<<"rotation_maxtix = "<< rotation_maxtix.type() << std::endl;

			cv::Mat rotation_image;
			cv::warpAffine(image, rotation_image, rotation_maxtix, size, INTER_LINEAR);

			//cv::imwrite("1_origin.jpg",image);
			//cv::imwrite("2_rotate.jpg",rotation_image);
			return rotation_image;
		}


		cv::Mat NumpyUtil::cv_vconcat(
			const cv::Mat& first_image,
			const cv::Mat& second_image
		)
		{
			cv::Mat result;
			cv::vconcat(first_image, second_image, result);
			return result;
		}

		void NumpyUtil::cv_split_mat_horizon(
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

		cv::Mat NumpyUtil::cv_clip_mat(
			const cv::Mat& image,
			float start_ratio,
			float end_ratio
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

		cv::Mat cv_rotate_mat(const cv::Mat& image)
		{
			Mat tmp;
			Mat result;
			cv::transpose(image, tmp);
			cv::flip(tmp, result, 0);
			return result;
		}

		cv::Mat cv_rotate_mat2(const cv::Mat& image)
		{
			Mat tmp;
			Mat result;
			cv::transpose(image, tmp);
			cv::flip(tmp, result, 1);
			return result;
		}

		
		cv::Rect NumpyUtil::cv_max_bounding_box(const std::vector<cv::Rect>& boxs)
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

		cv::Rect NumpyUtil::cv_boundary(const cv::Rect& rect_in,const cv::Size& max_size)
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
		size_t NumpyUtil::cv_get_value_count(const cv::Mat& image)
		{
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
#pragma endregion


#pragma region np 
		// y,x=np.where(image>0)
		void NumpyUtil::np_where_g(const cv::Mat& image, int value, std::vector<int>& v_y,std::vector<int>& v_x)
		{
			v_y.clear();
			v_x.clear();

			// gray image 
			int height = image.rows;
			int width = image.cols;

			//cv::Mat image_with_mask(height, width, CV_8UC3); // 0-255

			for (int h = 0; h < height; h++)
			{
				const uchar *p_mask = image.ptr<uchar>(h);
				
				for (int w = 0; w < width; w++)
				{
					if (*p_mask > value){ // update here
						v_y.push_back(h);
						v_x.push_back(w);
					}
					p_mask++;
				}
			}
		}

		// y,x = np.where(image == 1)
		void NumpyUtil::np_where_eq(const cv::Mat& image, int value, std::vector<int>& v_y,std::vector<int>& v_x)
		{
			v_y.clear();
			v_x.clear();

			// gray image 
			int height = image.rows;
			int width = image.cols;

			//cv::Mat image_with_mask(height, width, CV_8UC3); // 0-255

			for (int h = 0; h < height; h++)
			{
				const uchar *p_mask = image.ptr<uchar>(h);
				
				for (int w = 0; w < width; w++)
				{
					if (*p_mask == value){ // update here
						v_y.push_back(h);
						v_x.push_back(w);
					}
					p_mask++;
				}
			}
		}

		cv::Mat NumpyUtil::np_get_roi(
			const cv::Mat& image, 
			int y1, int y2, int x1, int x2
			)
		{
			// image[y1:y2,x1:x2];
			cv::Mat roi;
			cv::Rect rect(x1,y1, x2-x1, y2-y1); // left top, width, height
			roi = image(rect);
			return roi;
		}

		cv::Mat NumpyUtil::np_argmax_axis_channel2(const channel_mat_t& channel_mat)
		{
			/* chw 
			out = image.argmax(axis=0) 
			chw  axis=0 ===> c  shape=(h,w) value range[0,1,2] channel_index

			dl = 2,256,256; gz = 3,256,256
			output[0].argmax(axis=0)  # (2/3,256, 256) ===> (256, 256) value_range=[0,1,2]
			*/

			int channel = channel_mat.size();
			assert(2==channel);

			const cv::Mat& image0 = channel_mat[0];
			const cv::Mat& image1 = channel_mat[1];
			//const cv::Mat& image2 = channel_mat[2];

			int height = image0.rows;
			int width = image0.cols;

			cv::Mat result(height, width, CV_8UC1); // (256, 256) value_range=[0,1]
			
			for (int h = 0; h < height; h++)
			{
				uchar *b = result.ptr<uchar>(h);
			
				const float *p0 = image0.ptr<float>(h);
				const float *p1 = image1.ptr<float>(h);
				//const float *p2 = image2.ptr<float>(h);

				for (int w = 0; w < width; w++)
				{
					float f0_value = *p0++;
					float f1_value = *p1++;
					//float f2_value = *p2++;
					
					std::vector<float> v{f0_value,f1_value};
					
					*b = max_index(v); // set to value range [0,1]

					b++;
				}
			}
			return result;
		}

		cv::Mat NumpyUtil::np_argmax_axis_channel3(const channel_mat_t& channel_mat)
		{
			/* chw 
			out = image.argmax(axis=0) 
			chw  axis=0 ===> c  shape=(h,w) value range[0,1,2] channel_index

			dl = 2,256,256; gz = 3,256,256
			output[0].argmax(axis=0)  # (2/3,256, 256) ===> (256, 256) value_range=[0,1,2]
			*/

			int channel = channel_mat.size();
			assert(3==channel);

			const cv::Mat& image0 = channel_mat[0];
			const cv::Mat& image1 = channel_mat[1];
			const cv::Mat& image2 = channel_mat[2];

			int height = image0.rows;
			int width = image0.cols;

			cv::Mat result(height, width, CV_8UC1); // (256, 256) value_range=[0,1,2]
			
			for (int h = 0; h < height; h++)
			{
				uchar *b = result.ptr<uchar>(h);
			
				const float *p0 = image0.ptr<float>(h);
				const float *p1 = image1.ptr<float>(h);
				const float *p2 = image2.ptr<float>(h);

				for (int w = 0; w < width; w++)
				{
					float f0_value = *p0++;
					float f1_value = *p1++;
					float f2_value = *p2++;
					
					std::vector<float> v{f0_value,f1_value,f2_value};
					
					*b = max_index(v); // set to value range [0,1,2]

					b++;
				}
			}
			return result;
		}

		cv::Mat NumpyUtil::np_argmax(const channel_mat_t& channel_mat)
		{
			/*
			channel_mat:  2/3, 256, 256  CV_32FC1
			binary_mask:  256,256   CV_8FC1   bgr  value= 0,1
			final_binary_mask: 256,256   value = 0,255
			*/
			int channel = channel_mat.size();
			assert(2==channel || 3==channel);

			cv::Mat binary_mask; // [0,1], [0,1,2]
			if (2==channel){
				binary_mask = NumpyUtil::np_argmax_axis_channel2(channel_mat);
			} else {
				binary_mask = NumpyUtil::np_argmax_axis_channel3(channel_mat);
			}
			return binary_mask;
		}

		cv::Mat NumpyUtil::np_binary_mask_as255(const cv::Mat& binary_mask)
		{
			// mask 0,1 ===> mask 0,255
			
			int height = binary_mask.rows;
			int width = binary_mask.cols;
			//std::cout<<"width: "<<height<<" width: "<<height<<std::endl;
			cv::Mat binary_mask255(height, width, CV_8UC1); // 0/255
			
			for (int h = 0; h < height; h++)
			{
				uchar *b = binary_mask255.ptr<uchar>(h);

				const uchar *max_p = binary_mask.ptr<uchar>(h); // value [0,1,2]
			
				for (int w = 0; w < width; w++)
				{
					// argmax(axis=0)
					if (1 == *max_p ) { // set channel 1 to white
						*b = 255; // white
					}
					else { 
						*b = 0; // black
					}

					// to next 
					b ++;
					max_p ++;
				}
			}
			return binary_mask255;
		}


		std::vector<int> NumpyUtil::np_unique_return_index(const std::vector<int>& v)
		{
			std::vector<int> vindex;

			std::set<int> theSet(v.begin(), v.end());
			for(auto iter = theSet.begin() ; iter != theSet.end() ; ++iter)  
			{   
				for(auto i=0; i<v.size();++i)
				{
					if (*iter == v[i])
					{
						vindex.push_back(i);
						break; // omit the next same value
					}
				}
			}  
			//std::cout<<std::endl;

			/*
			std::cout<<"index ="<< std::endl;
			for(auto iter = vindex.begin() ; iter != vindex.end() ; ++iter)  
			{  
				std::cout<<*iter<<" ";  
			}
			std::cout<< std::endl;
			*/

			return vindex;
		}

		std::vector<int> NumpyUtil::np_argwhere_eq(const std::vector<int>& v, int value)
		{
			// [1,1,1,2,2,2,3,3,3,] value = 2;
			// return vindex = [3,4,5]
			std::vector<int> vindex;
			for(auto i=0; i<v.size();++i)
			{
				if (value == v[i])
				{
					vindex.push_back(i);
				}
			}
			return vindex;
		}


		/*
		y_mesh = (coord_trans[2,:]-y)*1000
        y_mesh = np.abs(y_mesh)
        y_mesh = y_mesh - np.min(y_mesh)
        y_id = np.where(y_mesh==0)
		*/
		std::vector<int> NumpyUtil::np_argwhere_eq(const std::vector<double>& v, double value)
		{
			std::vector<int> v_1000; 
			for(int i=0;i<v.size();i++){
				int  new_value = abs( (v[i] - value )*1000 );
				v_1000.push_back(new_value);
			}

			int min_value = 0;
			if (v_1000.size()>0){
				min_value = MINV(v_1000);
			}

			for(int i=0;i<v.size();i++){
				v_1000[i] -= min_value;
			}

			return NumpyUtil::np_argwhere_eq(v_1000, 0);
		}

		// coord = np.insert(coord, 2, values=1, axis=1) #x y 1   
		// n*2 (x,y) ===> n*3 (x,y,Value) 
		std::vector<dpoint_t> NumpyUtil::np_insert(const std::vector<dpoint_t>& points, float value)
		{
			 std::vector<dpoint_t> new_points;
			 for(auto& point: points)
			 {
				 dpoint_t new_point;
				 for(auto& p: point)
				 {
					 new_point.push_back(p);
				 }
				 new_point.push_back(value);
				 new_points.push_back(new_point);
			 }
			 return new_points;
		}

		std::vector<dpoint_t> NumpyUtil::np_delete(const std::vector<dpoint_t>& points, int column)
		{
			// n*3 (x,y,1) ===> n*2 (x,y)
			 std::vector<dpoint_t> new_points;
			 for(auto& point: points)
			 {
				 int m = point.size();
				 dpoint_t new_point;
				 for(int i=0; i< m; i++){
					 if (i != column){
						new_point.push_back(point[i]);
					 }
				 } 
				 new_points.push_back(new_point);
			 }
			 return new_points;

		}

		// coord_T = coord.T  # n*3 ===> 3*n
		std::vector<dpoint_t> NumpyUtil::np_transpose(const std::vector<dpoint_t>& points)
		{
			
			int n = points.size();
			assert(n>0);
			int dim = points[0].size();

			std::vector<dpoint_t> new_points;
			new_points.resize(dim);

			for(int d=0; d < dim; ++d)
			{
				new_points[d].resize(n);
				for(int i=0; i<n; ++i)
				{
					new_points[d][i] = points[i][d];
				}
			}
			return new_points;
		}

		std::vector<dpoint_t> NumpyUtil::np_inverse(const std::vector<dpoint_t>& points)
		{
			std::vector<dpoint_t> inverse_points;

			int n = points.size();
			assert(n>0);

			int m = points[0].size(); 
			assert(m>0);

			inverse_points.resize(n);

#ifdef USE_EIGEN
			Eigen::MatrixXd mat(n,m);
			for(int i=0; i< n; i++){
				for(int j=0; j< m; j++){
					mat(i,j) = points[i][j];
				}
			}
			Eigen::MatrixXd inverse_mat = mat.inverse();

			//std::cout <<"mat = "<< mat << endl;
			//std::cout <<"inverse_mat = "<< inverse_mat << endl;

			for(int i=0; i< n; i++){
				inverse_points[i].resize(m); 
				for(int j=0; j< m; j++){
					inverse_points[i][j] = inverse_mat(i,j);
				}
			}
#endif
			return inverse_points;

			/*
			('inverse_tr33 = ', matrix([[6.22219531e+03, 0.00000000e+00, 8.89046043e+02],
        [0.00000000e+00, 6.27332666e+03, 4.53648326e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))
			*/
		}

		//  coord_trans = np.matmul(Tr33, coord_T) # m*3, 3*n ===> m*n (x1,y1,z1) 
		std::vector<dpoint_t> NumpyUtil::np_matmul(
			const std::vector<dpoint_t>& a,
			const std::vector<dpoint_t>& b
			)
		{
			// c[i,j] = sum(aik* bkj) for k =0,1,...dim-1
			int m = a.size();
			assert(m>0);
			int dim = a[0].size();
			assert(dim>0);
			int dim2 = b.size();
			assert(dim == dim2); 
			int n = b[0].size();
			assert(n>0);
			
			// init m*n to 0
			std::vector<dpoint_t> c;
			c.resize(m);

			for(int i=0; i < m; ++i)
			{
				c[i].resize(n);
				for(int j=0; j<n; ++j)
				{
					c[i][j] = 0;
				}
			}

			// mul
			for(int i=0;i<m;i++)
			{
				for(int j=0; j<n; j++)
				{
					for(int k=0;k<dim;k++)
					{
						c[i][j] += a[i][k]*b[k][j];
					}
				}
			}
			return c;
		}
#pragma endregion


#pragma region algorithm

			bool NumpyUtil::sort_score_pair_descend(
				const std::pair<float, int>& pair1,
				const std::pair<float, int>& pair2
			)
			{
				return pair1.first > pair2.first;
			}

			float NumpyUtil::jaccard_overlap(
				const cv::Rect& bbox1,
				const cv::Rect& bbox2
			)
			{
				// overlap/(a1+a2-overlap)
				const float inter_xmin = std::max(bbox1.x, bbox2.x);
				const float inter_ymin = std::max(bbox1.y, bbox2.y);
				const float inter_xmax = std::min(bbox1.x + bbox1.width,  bbox2.x + bbox2.width);
				const float inter_ymax = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

				const float inter_width = inter_xmax - inter_xmin;
				const float inter_height = inter_ymax - inter_ymin;
				const float inter_size = inter_width * inter_height;

				const float bbox1_size = (bbox1.width) * (bbox1.height);
				const float bbox2_size = (bbox2.width) * (bbox2.height);

				return inter_size / (bbox1_size + bbox2_size - inter_size);
			}

			int NumpyUtil::nms_fast(
				const std::vector<cv::Rect>& boxs_,
				const float overlapThresh,
				std::vector<cv::Rect>& new_boxs
			)
			{
				std::vector<int> v_index;
				std::vector<cv::Rect> bbs = boxs_;

				if (bbs.size() < 1)
				{
					return 0;
				}

				// (1) get areas 
				//std::cout << bbs.size() << std::endl;
				std::vector<float> areas;
				for(int i=0;i<bbs.size();i++)
				{
					long w = bbs[i].width;
					long h = bbs[i].height;
					float s = w * h;
					areas.push_back(s);
				}

				// Generate index score pairs.
				std::vector<pair<float, int> > v_pair_score_index;
				for (int i = 0; i < areas.size(); ++i) {
					v_pair_score_index.push_back(std::make_pair(areas[i], i));
				}

				// Sort the score pair according to the scores in descending order
				std::stable_sort(
					v_pair_score_index.begin(),
					v_pair_score_index.end(),
					sort_score_pair_descend
				);

				// Do nms.
				while (v_pair_score_index.size() != 0)
				{
					const int idx = v_pair_score_index.front().second;
					bool keep = true;
					for (int k = 0; k < v_index.size(); ++k)
					{
						if (keep)
						{
							const int kept_idx = v_index[k];
							float overlap = jaccard_overlap(bbs[idx], bbs[kept_idx]);
							keep = (overlap <= overlapThresh);
						}
						else
						{
							break;
						}
					}

					if (keep) {
						v_index.push_back(idx);
					}

					v_pair_score_index.erase(v_pair_score_index.begin());
				}

				for (size_t i = 0; i < v_index.size(); i++)
				{
					int index = v_index[i];
					new_boxs.push_back(bbs[index]);
				}
				return v_index.size();
			}

#pragma endregion

	}
}// end namespace