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
		): 	src_points(src_points_),
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

				// mat_k = _cal_mat(); 
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
			// std::cout << "_cal_mat:" << mat_k << std::endl;
			return mat_k;
		}

		cv::Mat Polyfiter::cal_mat_add_points(dpoints_t d_src_points)
		{
			// d_src_points x,y -> y,x
			int size = d_src_points.size();
			//unkonw parameters count
			int x_num = n + 1;
			//Mat U,Y
			cv::Mat mat_u(size, x_num, CV_64F);
			cv::Mat mat_y(size, 1, CV_64F);

			for (int i = 0; i < mat_u.rows; ++i)
				for (int j = 0; j < mat_u.cols; ++j)
				{
					mat_u.at<double>(i, j) = pow(d_src_points[i][1], j);
				}
		
			for (int i = 0; i < mat_y.rows; ++i)
			{
				mat_y.at<double>(i, 0) = d_src_points[i][0];
			}
		
			//Get coefficients mat K
			cv::Mat mat_k(x_num, 1, CV_64F);
			mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
			// std::cout << "cal_mat:" << mat_k << std::endl;
			return mat_k;
		}

		void Polyfiter::set_mat_k(cv::Mat mat_t_new){
			mat_k = mat_t_new; 
		}

		double Polyfiter::_predict(double x)
		{
			double y = 0;
			for (int j = 0; j < n + 1; ++j)
			{
				y += mat_k.at<double>(j, 0)*pow(x,j);
				// std::cout << x << "," << mat_k.at<double>(j, 0) << "," << n-j << std::endl;
			}
			// std::cout << y << "\n" << std::endl;
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

		std::vector<dpoint_t>  Polyfiter::fit_dpoint(CAMERA_TYPE camera_type, dpoints_t d_src_points, bool use_auto_range)
		{	
			TABLE_TYPE table_type = TABLE_LONG_A;
			if (camera_type == CAMERA_SHORT){
				table_type = TABLE_SHORT_A;
			}

			std::vector<double> vx,vy;
			for(auto& point: d_src_points){
				vx.push_back(point[0]);
				vy.push_back(point[1]);
			}

			std::vector<dpoint_t> fitted_dpoints;
			if (!is_valid_input){
				return fitted_dpoints;
			}

			fitted_dpoints.clear();
		
			// get range xy
			double x_range_min,x_range_max;
			double y_range_min,y_range_max;
			if (use_auto_range){
				x_range_min = MINV(vx);
				x_range_max = MAXV(vx);
				y_range_min = MINV(vy);
				y_range_max = MAXV(vy);
				// // CAMERA_TYPE::CAMERA_LONG  1 
				// if (table_type == 1){
				// 	x_range_min = (MINV(vx) < -9) ? MINV(vx) : -9;
				// 	x_range_max = MAXV(vx);
				// 	y_range_min = (MINV(vy) < 31) ? MINV(vy) : 31;
				// 	y_range_max = MAXV(vy);
				// }
				// // CAMERA_TYPE::CAMERA_SHORT
				// else {
				// 	x_range_min = (MINV(vx) < -19) ? MINV(vx) : -19;
				// 	x_range_max = MAXV(vx);
				// 	y_range_min = (MINV(vy) < 1) ? MINV(vy) : 1;
				// 	y_range_max = MAXV(vy);
				// }
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
			std::cout << x_range_min << "," << x_range_max << std::endl;
			double x = x_range_min;
			while (x <= x_range_max){
				double y = _predict(x);
				// std::cout << x << "," << y << "," << y_range_min << "," << y_range_max << std::endl;
				// if (y>=y_range_min && y<=y_range_max){
				dpoint_t ipt;
				ipt.push_back(double(x));
				ipt.push_back(double(y));
				if(reverse_xy){
					std::swap(ipt[0], ipt[1]);
				}
				fitted_dpoints.push_back(ipt);
				// }
				x += 0.1;
				// x += 0.05;
			}
			// std::cout << "fitted_dpoints size:" << fitted_dpoints.size() << std::endl;
			// std::cout << "fitted_dpoints[0] size:" << fitted_dpoints[0].size() << std::endl;
			return fitted_dpoints;
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