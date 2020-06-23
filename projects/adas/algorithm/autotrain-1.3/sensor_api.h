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
#include "projects/adas/algorithm/algorithm_shared_export.h" 
#include "projects/adas/algorithm/algorithm_type.h"

// std
#include <vector>

// opencv
#include <opencv2/core.hpp> // Mat

namespace watrix {
	namespace algorithm {

		class SHARED_EXPORT SensorApi
		{
			public:
				static lidar_camera_param_t params;
				static const int IMAGE_HEIGHT = 1080;
				static const int IMAGE_WIDTH = 1920;

				// camera 
				static cv::Mat camera_matrix_short_;
				static cv::Mat camera_distCoeffs_short_;

				static cv::Mat camera_matrix_long_;
				static cv::Mat camera_distCoeffs_long_;

				// lidar
				static cv::Mat a_matrix_;
				static cv::Mat r_matrix_;
				static cv::Mat t_matrix_;

			public:
				static void init(lidar_camera_param_t& params);

				static cvpoint_t image_cvpoint_a2b(
					CAMERA_TYPE camera_type,
					cvpoint_t inputPoint
				);

				static cv::Mat image_a2b(
					CAMERA_TYPE camera_type, 
					const cv::Mat& image_a
				);

				// NOT USE
				//static cv::Mat image_b2a(
				//	CAMERA_TYPE camera_type, 
				//	const cv::Mat& image_b
				//);

			protected:

				static bool lidar_3d_to_2d(
					cv::Point3d point3, 
					std::vector<double> extra_params, 
					cv::Point2i& point_a
				);

				static void load_params();
		};

	}
}// end namespace