#pragma once

#include "watrix/proto/point_cloud.pb.h"
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
/*
struct Bounding_Box {

	double x;
	double y;
	double height;
	double width;
};

struct OB_Size {

	float max_x = 0.0f;
	float max_y = 0.0f;
	float max_z = 0.0f;
	float min_x = 0.0f;
	float min_y = 0.0f;
	float min_z = 0.0f;
};

struct Point3
{
	float x, y, z;
	int intensity;
};

struct Grid_data {
	union Point3F {
		struct { float x, y, z; };
		float vData[3];
		Point3F() {};
		template<typename T>
		Point3F(const T& v) :x(v.x), y(v.y), z(v.z) {}
		Point3F(float _x, float _y, float _z) :x(_x), y(_y), z(_z) { }
		float& operator[](int n) {
			return vData[n];
		}
	};
	std::vector<Point3F> grid_points;
	float max_x = 0.0f;
	float min_x = 0.0f;
	float total_x = 0.0f;
	float total_z = 0.0f;
	float max_y = 0.0f;
	float min_y = 0.0f;
	float max_z = 0.0f;
	float min_z = 0.0f;
};

struct OB_index_data {

	int pointsNum = 0;
	float maxSize = 0.0f;
	OB_Size ob_size;
	std::vector<std::pair<int, int>> gridIndex;
};

struct Grid_data_init
{
	Grid_data* pData;
	size_t grid_cols, grid_rows;
	Grid_data_init(size_t _rows, size_t _cols) :grid_rows(_rows), grid_cols(_cols) { pData = new Grid_data[_rows * _cols]; }
	Grid_data_init(const Grid_data_init&) = delete;
	Grid_data_init(Grid_data_init&&) = delete;
	const Grid_data_init& operator=(const Grid_data_init&) = delete;
	const Grid_data_init& operator=(Grid_data_init&&) = delete;
	~Grid_data_init() { delete[] pData; }
	Grid_data* operator[](size_t _nrow) { return pData + grid_cols * _nrow; }
};
*/
class FindContours {

public:
	static watrix::proto::PointCloud start_contours(std::vector<watrix::proto::LidarPoint> points);

protected:
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
