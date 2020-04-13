#pragma once

#include "watrix/proto/point_cloud.pb.h"
#include "lidar_point_struct.h"

#include <iostream>
#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
//#include <pcl/surface/convex_hull.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>


class LOD {

public:
	static void create_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Grid_data_init& grid, pcl::PointXYZ min_p, pcl::PointXYZ max_p,
		float row_radio, float col_radio);
	static std::vector<OB_index_data> cluster(Grid_data_init& grid);
	static void search_grid(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
		OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size);
	static void search_fork(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
		OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, int row, int col);
	static std::vector<watrix::proto::PointCloud> object_detection(watrix::proto::PointCloud check_pointclouds);
};
