/*
* Software License Agreement
*
*  WATRIX.AI - www.watrix.ai
*  Copyright (c) 2016-2018, Watrix Technology, Inc.
*
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the copyright holder(s) nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: zunlin.ke@watrix.ai (Zunlin Ke)
*
*/
#pragma once
#include "algorithm_shared_export.h" 

// std
#include <vector>

// opencv
#include <opencv2/core.hpp> // Mat

namespace watrix {
	namespace algorithm {

		/*
		# image coord    (origin= left top(0,0), x=[0-1920],y=[0-1080])
		# distance coord (origin = camera(0,0), x=[-10,10],y=[10,200])
		# overview coord (origin= left bottom(0,0), x=[0-1000],y=[0-900])
		*/

		struct SHARED_EXPORT image_coord_t {
			bool valid;
			unsigned int x,y;
		};

		struct SHARED_EXPORT distance_coord_t {
			bool valid;
			float x,y;
		};

		typedef image_coord_t overview_coord_t;

		struct SHARED_EXPORT table_config_t{
			// # the grid size (meters)
			float grid_width = 0.001;
			float grid_height = 0.001;
			
			// # the scale of width of the scene
			int distance_width_min = -10;
			int distance_width_max = 10;

			// the scale of height of the scene
			int distance_height_min = 10;
			int distance_height_max = 200;

			//# the size of the whole scene
			int distance_width = distance_width_max - distance_width_min;    //# 10 
			int distance_height = distance_height_max - distance_height_min; //# 45
		};

		
		class SHARED_EXPORT MonocularDistanceApi
		{
			public:
				/*
				load distance table 1920*1080*[x,y]
				*/
				static bool load_short_table(const char* table_filepath);
				static bool load_long_table(const char* table_filepath);

				/*
				get image distance from 1920*1080*[x,y] map table
				*/
				static bool get_short_distance_coord(
					unsigned int row, 
					unsigned int col,
					float& x, 
					float& y
				);

				static bool get_long_distance_coord(
					unsigned int row, 
					unsigned int col,
					float& x, 
					float& y
				);

				/*
				use binary_image and box centers to generate overview image
				*/
				static cv::Mat generate_overview_image(
					const cv::Mat& binary_image,
					const std::vector<cv::Point2i>& v_centers,
					const table_config_t& cfg,
					unsigned int overview_height, 
					unsigned int overview_width
				);

			protected:

				static distance_coord_t image_coord_to_distance_coord(
					const image_coord_t& image_coord
				);

				static overview_coord_t distance_coord_to_overview_coord(
					const distance_coord_t& distance_coord, 
					const table_config_t& cfg,
					unsigned int overview_height, 
					unsigned int overview_width,
					float overview_height_factor, 
					float overview_width_factor
				);

				static	overview_coord_t image_coord_to_overview_coord(
					const image_coord_t& image_coord,
					const table_config_t& cfg,
					unsigned int overview_height, 
					unsigned int overview_width,
					float overview_height_factor, 
					float overview_width_factor
				);

				static bool set_overview_pixel(
					cv::Mat& overview,
					unsigned int row, 
					unsigned int col,
					const table_config_t& cfg,
					unsigned int overview_height, 
					unsigned int overview_width,
					float overview_height_factor, 
					float overview_width_factor 
				);

				static bool draw_overview_circle(
					cv::Mat& overview_image,
					const cv::Point2i& center,
					const table_config_t& cfg,
					unsigned int overview_height, 
					unsigned int overview_width,
					float overview_height_factor, 
					float overview_width_factor 
				);

			private:
				static float INVALID_VALUE; 

				static const int VALID_BINARY_HEIGHT = 512; // binary mask valid height

				// demo: 5-4 
				static const int COL = 1920; // xy
				static const int ROW = 1080;
				static float map_x_short[COL][ROW];
				static float map_y_short[COL][ROW]; 

				static float map_x_long[COL][ROW];
				static float map_y_long[COL][ROW]; 
		};

	}
}// end namespace