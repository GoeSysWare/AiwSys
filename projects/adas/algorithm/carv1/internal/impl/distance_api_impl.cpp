#include "distance_api_impl.h"

#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"

#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imdecode imshow

// glog
#include <glog/logging.h>

// std
#include <iostream>


namespace watrix {
	namespace algorithm {
		namespace internal{

			bool DistanceApiImpl::get_distance(
				const cv::Mat& left,
				const cv::Mat& right,
				const int image_distance,
				int& distance
			)
			{
				cv::Mat left_roi, right_roi;
				bool left_roi_found  = OpencvUtil::get_horizontal_distance_roi(left, left_roi);
				bool right_roi_found = OpencvUtil::get_horizontal_distance_roi(right, right_roi);
				if (!left_roi_found || !right_roi_found)
				{
					LOG(INFO)<<"[API]  left or right roi not found\n";
					return false;
				}

#ifdef shared_DEBUG
				{
					cv::Mat left_horizontal = OpencvUtil::get_horizontal_project_mat(left);
					cv::Mat left_vertical = OpencvUtil::get_vertical_project_mat(left);

					cv::Mat right_horizontal = OpencvUtil::get_horizontal_project_mat(right);
					cv::Mat right_vertical = OpencvUtil::get_vertical_project_mat(right);

					cv::imwrite("distance/left.jpg", left);
					cv::imwrite("distance/right.jpg", right);

					cv::imwrite("distance/left_horizontal.jpg", left_horizontal);
					cv::imwrite("distance/left_vertical.jpg", left_vertical);

					cv::imwrite("distance/right_horizontal.jpg", right_horizontal);
					cv::imwrite("distance/right_vertical.jpg", right_vertical);

					cv::imwrite("distance/left_roi.jpg", left_roi);
					cv::imwrite("distance/right_roi.jpg", right_roi);
				}
#endif

				int left_col, right_col;
				bool success = OpencvUtil::get_vertical_project_distance(left_roi, left_col, right_col);
				int left_distance = (left_roi.cols - right_col);

				if (success)
				{
					LOG(INFO)<<"[API] left_distance = " << left_distance << std::endl;
				}
				else {
					LOG(INFO)<<"[API] left failed. \n";
				}

				int left_col2, right_col2;
				bool success2 = OpencvUtil::get_vertical_project_distance(right_roi, left_col2, right_col2);
				int right_distance = left_col2;

				distance = left_distance + image_distance + right_distance;

				if (success2)
				{
					LOG(INFO)<<"[API] right_distance = " << right_distance << std::endl;
				}
				else {
					LOG(INFO)<<"[API] right failed. \n";
				}

				return (success && success2);
			}


		}
	}
}// end namespace

