#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <stack>
#include <string>
#include <list>
#include <time.h>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
//#include <pcl/surface/convex_hull.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef DEBUG_INFO_FWC
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#endif


#include "projects/adas/algorithm/algorithm_type.h" 

#define DIVIDE_DIS 10.0f
#define DETECT_DISTANCE 53.0f
#define GROUND_JUDGE 23
#define PIC_WIDTH 1920
#define PIC_HEIGHT 1080
#define SUSPENDED_OBJECT true
#define HOLE_JUDGE_HEIGHT -2.2f
#define SLOPE_LIMIT 3.5f
#define POINT_EXPAND 0.5f
#define INVASION_EXPAND 0.05f   //0.7825f
#define GROUND_HEIGHT -1.17f


namespace watrix {
	namespace algorithm {
		class LOD {
			public:
				static std::unordered_map<int, std::pair<int, int>> getInvasionMap(std::vector<cv::Point2i> input_l, std::vector<cv::Point2i> input_r, int& top_y);
				static pcl::PointCloud<pcl::PointXYZ>::Ptr getPointFrom2DAnd3D(const std::vector<cv::Point3f>& cloud_cv, std::unordered_map<int, std::pair<int, int>> invasionP,
					int top_y, InvasionData invasion, Eigen::Matrix4d& rotation, float distance_limit);
				static float calHeightVar(std::vector<Point3> object);
				static cv::Point2f calHeightVar(std::vector<float> object);
				static void getInvasionData(InvasionData& invasion_data, std::string csv_file);
				static void getInvasionData(std::vector<cv::Point2i>& input_l, std::vector<cv::Point2i>& input_r,
					std::string csv_file_l, std::string csv_file_r);
				static FitLineData fitNextPoint(std::vector<cv::Point2d> lines, float row_radio);
				static void calLowestPoint(TrackData& trackdata, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointXYZ min_p,
					pcl::PointXYZ max_p, float row_radio, InvasionData invasion, std::string image_file);
				static void create_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Grid_data_init& grid, pcl::PointXYZ min_p, pcl::PointXYZ max_p,
					float row_radio, float col_radio, TrackData trackdata);
				static std::vector<OB_index_data> cluster(Grid_data_init& grid, TrackData trackdata);
				static void search_grid(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
					OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, TrackData trackdata);
				static void search_fork(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
					OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, int row, int col, TrackData trackdata);
				static float boxOverlap(OB_Size train_box, OB_Size ob_box);
				static void drawImage(std::vector<LidarBox> obstacle_box, std::string image_file);
				static std::vector<LidarBox> object_detection(pcl::PointCloud<pcl::PointXYZ>::Ptr& points, Eigen::Matrix4d rotation, InvasionData invasion, std::string image_file);

				static std::vector<lidar_invasion_cvbox> lidarboxTocvbox(std::vector<LidarBox> obstacle_box);
		};
	} 
}
