#pragma once

#include "modules/drivers/proto/pointcloud.pb.h"
#include "lidar_point_struct.h"
#include <iostream>
#include <vector>
#include <stack>
#include <string>
#include <list>
#include <time.h>
#include <opencv2/opencv.hpp>
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
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <torch/script.h> 
#include <torch/torch.h> 

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::FT                                                FT;
typedef K::Point_2                                           Point2;
typedef K::Segment_2                                         Segment;
typedef CGAL::Alpha_shape_vertex_base_2<K>                   Vb;
typedef CGAL::Alpha_shape_face_base_2<K>                     Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>          Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds>                Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2>                 Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator            Alpha_shape_edges_iterator;

class FindContours_v2 {

public:
	std::vector<cv::Point2d> start_contours(std::vector<apollo::drivers::PointXYZIT> points);
	void load_params();
	void OnPointCloud(const apollo::drivers::PointCloud& data, apollo::drivers::PointCloud& lidar2image_paint_,
		apollo::drivers::PointCloud& lidar_safe_area_, std::vector<cv::Point3f>& lidar_cloud_buf, int& effect_point);
private:
	std::vector<std::vector<std::pair<int, int>>> distortTable;
	std::vector<float> distCoeff;
	cv::Mat cameraMatrix;
	cv::Mat rvec, tvec;

	void alphaShape(std::list<Point2> points, pcl::PointCloud<pcl::PointXYZ>::Ptr& tmp, float min_h);
	void search_fork(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
		OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, int row, int col);
	void create_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Grid_data_init& grid, pcl::PointXYZ min_p, pcl::PointXYZ max_p,
		std::vector<float> grid_row_index, float col_radio);
	std::vector<OB_index_data> cluster(Grid_data_init& grid, int& screen_num);
	void search_grid(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
		OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size);
	void downSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_train, std::list<Point2>& points);

};
