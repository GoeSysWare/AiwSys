#pragma once


#include <opencv2/opencv.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
//#include <pcl/surface/convex_hull.h>
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "modules/drivers/proto/pointcloud.pb.h"
#include "lidar_point_struct.h"

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

namespace watrix
{
namespace projects
{
namespace adas
{



class FindContours_v2 {

public:
	static std::vector<cv::Point2d> start_contours(std::vector<apollo::drivers::PointXYZIT> points);
	static void load_params();
	static void OnPointCloud(const apollo::drivers::PointCloud& data, apollo::drivers::PointCloud& lidar2image_paint_,
		apollo::drivers::PointCloud& lidar_safe_area_, std::vector<cv::Point3f>& lidar_cloud_buf, int& effect_point);

	static std::vector<std::vector<std::pair<int, int>>> distortTable;
	static std::vector<float> distCoeff;
	static cv::Mat cameraMatrix;
	static cv::Mat rvec, tvec;
private:
	static void alphaShape(std::list<Point2> points, pcl::PointCloud<pcl::PointXYZ>::Ptr& tmp, float min_h);
	static void search_fork(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
		OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, int row, int col);
	static void create_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Grid_data_init& grid, pcl::PointXYZ min_p, pcl::PointXYZ max_p,
		std::vector<float> grid_row_index, float col_radio);
	static std::vector<OB_index_data> cluster(Grid_data_init& grid, int& screen_num);
	static void search_grid(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
		OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size);
	static void downSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_train, std::list<Point2>& points);

};

}
}
}
