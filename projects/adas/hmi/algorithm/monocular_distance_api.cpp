#include "monocular_distance_api.h"

#include <iostream>
#include <fstream>

// opencv
#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imwrite imdecode imshow

namespace watrix {
	namespace algorithm {

		float MonocularDistanceApi::INVALID_VALUE = 1000.0;
		float MonocularDistanceApi::map_x_short[COL][ROW];
		float MonocularDistanceApi::map_y_short[COL][ROW];

		float MonocularDistanceApi::map_x_long[COL][ROW];
		float MonocularDistanceApi::map_y_long[COL][ROW];

		bool MonocularDistanceApi::load_short_table(const char* table_filepath)
		{
			// table[col][row][2]
			//std::string table_filepath = "./table_map.bin";
			std::ifstream ifs(table_filepath, std::ifstream::binary);

			if(ifs.is_open()){
				for(int i=0; i<COL; ++i)
				{
					for(int j=0;j<ROW;++j)
					{
						float distance_x;
						ifs.read(reinterpret_cast<char*>(&distance_x), sizeof(float)); 
						
						float distance_y;
						ifs.read(reinterpret_cast<char*>(&distance_y), sizeof(float)); 

						map_x_short[i][j] = distance_x; 
						map_y_short[i][j] = distance_y; 

						/*
						if (distance_x != INVALID_VALUE && distance_y != INVALID_VALUE){
							std::cout<<"x,y = " <<distance_x <<","<< distance_y << std::endl;
						}
						*/
					}
				}
				ifs.close();
				return true;
			} else {
				return false;
			}
		}

		bool MonocularDistanceApi::load_long_table(const char* table_filepath)
		{
			// table[col][row][2]
			//std::string table_filepath = "./table_map.bin";
			std::ifstream ifs(table_filepath, std::ifstream::binary);

			if(ifs.is_open()){
				for(int i=0; i<COL; ++i)
				{
					for(int j=0;j<ROW;++j)
					{
						float distance_x;
						ifs.read(reinterpret_cast<char*>(&distance_x), sizeof(float)); 
						
						float distance_y;
						ifs.read(reinterpret_cast<char*>(&distance_y), sizeof(float)); 

						map_x_long[i][j] = distance_x; 
						map_y_long[i][j] = distance_y; 

						/*
						if (distance_x != INVALID_VALUE && distance_y != INVALID_VALUE){
							std::cout<<"x,y = " <<distance_x <<","<< distance_y << std::endl;
						}
						*/
					}
				}
				ifs.close();
				return true;
			} else {
				return false;
			}
		}

		bool MonocularDistanceApi::get_short_distance_coord(
			unsigned int row, 
			unsigned int col,
			float& x, 
			float& y
		)
		{
			if(0<=row<ROW && 0<=col<COL){
				x = map_x_short[col][row];
				y = map_y_short[col][row];

				bool valid = (x!= INVALID_VALUE) && (y!= INVALID_VALUE);
				return valid;
			} else {
				return false;
			}
		}

		bool MonocularDistanceApi::get_long_distance_coord(
			unsigned int row, 
			unsigned int col,
			float& x, 
			float& y
		)
		{
			if(0<=row<ROW && 0<=col<COL){
				x = map_x_long[col][row];
				y = map_y_long[col][row];

				bool valid = (x!= INVALID_VALUE) && (y!= INVALID_VALUE);
				return valid;
			} else {
				return false;
			}
		}

		distance_coord_t MonocularDistanceApi::image_coord_to_distance_coord(
			const image_coord_t& image_coord
			)
		{
			float x,y;
			bool valid = get_short_distance_coord(image_coord.y, image_coord.x, x, y); // row,col 
			distance_coord_t distance_coord{valid,x,y};
			return distance_coord;
		}

		overview_coord_t MonocularDistanceApi::distance_coord_to_overview_coord(
			const distance_coord_t& distance_coord, 
			const table_config_t& cfg,
			unsigned int overview_height, 
			unsigned int overview_width,
			float overview_height_factor, 
			float overview_width_factor
			)
		{
			unsigned int ox = 0, oy = 0;
			if (distance_coord.valid){
				oy = int(((distance_coord.y - cfg.distance_height_min) / cfg.grid_height) / overview_height_factor); //# 0 - 900
        		oy = overview_height - oy; //# 0-900   (left top ===> left bottom)
        		ox = int(((distance_coord.x - cfg.distance_width_min) / cfg.grid_width) / overview_width_factor);   //# 0 - 1000
			}
			overview_coord_t overview_coord{distance_coord.valid,ox,oy};
			return overview_coord;
		}

		overview_coord_t MonocularDistanceApi::image_coord_to_overview_coord(
			const image_coord_t& image_coord,
			const table_config_t& cfg,
			unsigned int overview_height, 
			unsigned int overview_width,
			float overview_height_factor, 
			float overview_width_factor
		)
		{
			distance_coord_t distance_coord = image_coord_to_distance_coord(image_coord);
			overview_coord_t overview_coord = distance_coord_to_overview_coord(
				distance_coord,
				cfg,
				overview_height,
				overview_width,
				overview_height_factor,
				overview_width_factor
			);
			return overview_coord;
		}

		bool MonocularDistanceApi::set_overview_pixel(
			cv::Mat& overview_image,
			unsigned int row, 
			unsigned int col,
			const table_config_t& cfg,
			unsigned int overview_height, 
			unsigned int overview_width,
			float overview_height_factor, 
			float overview_width_factor 
		)
		{
			unsigned int x = col; 
    		unsigned int y = row;

			image_coord_t image_coord{true,x,y};

			overview_coord_t overview_coord = image_coord_to_overview_coord(
				image_coord,
				cfg,
				overview_height,
				overview_width,
				overview_height_factor,
				overview_width_factor
			);

			if (overview_coord.valid){
				//overview
				int b = 0;
				int g = 255;
				int r = 0;

				int h = overview_coord.y; 
				int w = overview_coord.x;
				overview_image.at<cv::Vec3b>(h,w) = cv::Vec3b(b, g, r); // OK
			}

			return overview_coord.valid;
		}

		bool MonocularDistanceApi::draw_overview_circle(
			cv::Mat& overview_image,
			const cv::Point2i& center,
			const table_config_t& cfg,
			unsigned int overview_height, 
			unsigned int overview_width,
			float overview_height_factor, 
			float overview_width_factor 
		)
		{
			unsigned int x = center.x ; 
    		unsigned int y = center.y;

			image_coord_t image_coord{true,x,y};

			overview_coord_t overview_coord = image_coord_to_overview_coord(
				image_coord,
				cfg,
				overview_height,
				overview_width,
				overview_height_factor,
				overview_width_factor
			);

			if (overview_coord.valid){
				//overview
				cv::Point overview_center(overview_coord.x, overview_coord.y);
				cv::circle(overview_image, overview_center, 10, CV_RGB(255, 0, 0), 2);
			}

			return overview_coord.valid;
		}

		cv::Mat MonocularDistanceApi::generate_overview_image(
			const cv::Mat& binary_image,
			const std::vector<cv::Point2i>& v_centers,
			const table_config_t& cfg,
			unsigned int overview_height, 
			unsigned int overview_width
		)
		{
			float grid_height_total_pixel = cfg.distance_height /cfg.grid_height; //#  45/0.001 = 45000
   			float grid_width_total_pixel = cfg.distance_width /cfg.grid_width; //#  10/0.001 = 10000

    		float overview_height_factor = grid_height_total_pixel / overview_height; //# 50
    		float overview_width_factor = grid_width_total_pixel / overview_width; //# 10

			int height = binary_image.rows;
			int width = binary_image.cols;

    		//# generate overview image from binary mask
			cv::Mat overview_image(overview_height, overview_width, CV_8UC3); // 0-255

			for(int h=height-VALID_BINARY_HEIGHT; h < height; ++h)
			{
				for(int w=0;w < width; ++w)
				{
					if (binary_image.at<uchar>(h,w) == 255 ) // # only for black pixel
					{
						set_overview_pixel(
							overview_image, h, w, 
							cfg, overview_height, overview_width, overview_height_factor, overview_width_factor
						);
					}
				}
			}

			// # draw box center (circle point) to overview image
			for(int i=0; i< v_centers.size(); ++i)
			{
				const cv::Point2i&  box_center = v_centers[i];
				draw_overview_circle(
					overview_image, box_center, 
					cfg, overview_height, overview_width, overview_height_factor, overview_width_factor
				);
			}

			//cv::imwrite("overview.jpg", overview_image);
			return overview_image;
		}

	}
}// end namespace


