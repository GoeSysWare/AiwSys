#include "polyfiter.h"

// std
#include <iostream>

// opencv
#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imwrite imdecode imshow


namespace watrix {
	namespace algorithm {

		Polyfiter::Polyfiter(
			const std::vector<cvpoint_t>& src_points_, 
			int n_,
			bool reverse_xy_,
			int x_range_min_,
			int x_range_max_,
			int y_range_min_,
			int y_range_max_
		): src_points(src_points_),
			n(n_), 
			reverse_xy(reverse_xy_),
			auto_x_range_min(0),
			auto_x_range_max(0),
			auto_y_range_min(0),
			auto_y_range_max(0),
			user_x_range_min(x_range_min_),
			user_x_range_max(x_range_max_),
			user_y_range_min(y_range_min_),
			user_y_range_max(y_range_max_),
			is_valid_input(false)
		{
			// n=1,2 points; n=2, 3 points
			is_valid_input = src_points.size()>n;
			//is_valid_input = true; 
			if (is_valid_input){
				// reverse1 (x,y) ===>(y,x)
				_reverse_points_xy();

				mat_k = _cal_mat(); 

				// reverse2 (y,x) ===>(x,y)
				_reverse_points_xy();

				// get auto range xy
				std::vector<int> vx,vy;
				for(auto& point: src_points){
					vx.push_back(point.x);
					vy.push_back(point.y);
				}
				auto_x_range_min = MINV(vx);
				auto_x_range_max = MAXV(vx);
				auto_y_range_min = MINV(vy);
				auto_y_range_max = MAXV(vy);
				
				//printf("[FIT] x_range = %d, %d \n", user_x_range_min, user_x_range_max); 
				//printf("[FIT] y_range = %d, %d \n", user_y_range_min, user_y_range_max); 

				assert(user_x_range_min <= user_x_range_max);
				assert(user_y_range_min <= user_y_range_max);
							
			} else {
				std::cout << "Not Enough Points to fit." << std::endl;
			}
		}

		void Polyfiter::_reverse_points_xy(){
			if(reverse_xy){
				for(auto& point: src_points){
					std::swap(point.x, point.y);
				}
			}
		}

		cv::Mat Polyfiter::_cal_mat()
		{
			int size = src_points.size();
			//unkonw parameters count
			int x_num = n + 1;

			//Mat U,Y
			cv::Mat mat_u(size, x_num, CV_64F);
			cv::Mat mat_y(size, 1, CV_64F);
		
			for (int i = 0; i < mat_u.rows; ++i)
				for (int j = 0; j < mat_u.cols; ++j)
				{
					mat_u.at<double>(i, j) = pow(src_points[i].x, j);
				}
		
			for (int i = 0; i < mat_y.rows; ++i)
			{
				mat_y.at<double>(i, 0) = src_points[i].y;
			}
		
			//Get coefficients mat K
			cv::Mat mat_k(x_num, 1, CV_64F);
			mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
			//std::cout << mat_k << std::endl;
			return mat_k;
		}

		double Polyfiter::_predict(double x)
		{
			double y = 0;
			for (int j = 0; j < n + 1; ++j)
			{
				y += mat_k.at<double>(j, 0)*pow(x,j);
			}
			return y;
		}

		std::vector<cvpoint_t>  Polyfiter::fit(bool use_auto_range)
		{
			if (!is_valid_input){
				return fitted_points;
			}

			fitted_points.clear();

			// get range xy
			int x_range_min,x_range_max;
			int y_range_min,y_range_max;
			if (use_auto_range){
				x_range_min = auto_x_range_min;
				x_range_max = auto_x_range_max;
				y_range_min = auto_y_range_min;
				y_range_max = auto_y_range_max;
			} else {
				x_range_min = user_x_range_min;
				x_range_max = user_x_range_max;
				y_range_min = user_y_range_min;
				y_range_max = user_y_range_max;
			}

			// reverse range xy
			if(reverse_xy){
				std::swap(x_range_min, y_range_min);
				std::swap(x_range_max, y_range_max);
			}

#ifdef DEBUG_INFO 
			printf("[FIT] x_range_min = %d, x_range_max = %d \n", x_range_min, x_range_max); 
			printf("[FIT] y_range_min = %d, y_range_max = %d \n", y_range_min, y_range_max); 
#endif

			for (int x = x_range_min; x < x_range_max; ++x)
			{
				int y = (int)_predict(x*1.0);
				if (y>=y_range_min && y<y_range_max){
					cvpoint_t ipt(x,y);
					if(reverse_xy){
						std::swap(ipt.x, ipt.y);
					}
					fitted_points.push_back(ipt);
				}
			}
			return fitted_points;
		}

		cv::Mat Polyfiter::draw_image(const cv::Size& image_size)
		{
			cv::Mat out(image_size.height, image_size.width, CV_8UC3, cv::Scalar::all(0));

			// source points
			for (auto& ipt: src_points){
				cv::circle(out, ipt, 1, cv::Scalar(0, 0, 255), CV_FILLED, CV_AA); // 1->3
			}

			//fitted points
			for (auto& ipt: fitted_points){
				cv::circle(out, ipt, 1, cv::Scalar(255, 255, 255), CV_FILLED, CV_AA); // 1->3
			}
			return out;
		}

	}
}// end namespace