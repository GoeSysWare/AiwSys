#include "projects/adas/component/perception/FindContours_v2.h"
#include "projects/adas/component/common/util.h"
#include "projects/adas/configs/config_gflags.h"
#include <torch/script.h> // One-stop header.
#include <torch/torch.h> // One-stop header.


namespace watrix
{
namespace projects
{
namespace adas
{

std::vector<std::vector<std::pair<int, int>>> FindContours_v2::distortTable;
 std::vector<float> FindContours_v2::distCoeff;
 cv::Mat FindContours_v2::cameraMatrix;
cv::Mat FindContours_v2::rvec;
cv::Mat FindContours_v2::tvec;

 void FindContours_v2::load_params() {

	const int size = 1920 * 1080 * 2;
	int* memblock = new int[size];

	std::string filepath = apollo::cyber::common::GetAbsolutePath(
		watrix::projects::adas:: GetAdasWorkRoot(),
		FLAGS_calibrator_cfg_distortTable);
	std::ifstream file(filepath, std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open())
	{
		file.seekg(0, std::ios::beg);
		file.read((char*)memblock, sizeof(float) * size);
		file.close();

		std::vector<std::pair<int, int>> temp;
		for (int i = 0; i < size; i += 2) {
			cv::Point2f pt;
			pt.x = memblock[i];
			pt.y = memblock[i + 1];
			temp.push_back(std::make_pair(memblock[i], memblock[i + 1]));
			if (temp.size() == 1920) {
				distortTable.push_back(temp);
				temp.clear();
			}

		}

		delete[] memblock;
	}
	file.close();

	distCoeff.push_back(-0.2283);
	distCoeff.push_back(0.1710);
	distCoeff.push_back(-0.0013);
	distCoeff.push_back(-8.2250e-06);
	distCoeff.push_back(0);

	float tempMatrix[3][3] = { { 2.1334e+03, 0, 931.1503 }, { 0, 2.1322e+03, 580.8112 }, { 0, 0, 1.0 } };

	cameraMatrix = cv::Mat(3, 3, CV_32F);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			cameraMatrix.at<float>(i, j) = tempMatrix[i][j];
		}
	}

	double tempRvec[3][3] = { {1,0,0},
							 {0,1,0},
							 {0,0,1} };

	rvec = cv::Mat(3, 3, CV_64F);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			rvec.at<double>(i, j) = tempRvec[i][j];
	cv::Rodrigues(rvec, rvec);

	double tempTvec[3] = { 0.0,0.80,0.177 };

	tvec = cv::Mat(3, 1, CV_64F);
	for (int i = 0; i < 3; ++i)
		tvec.at<double>(i, 0) = tempTvec[i];
}

 void FindContours_v2::OnPointCloud(const apollo::drivers::PointCloud& data, apollo::drivers::PointCloud& lidar2image_paint_,
	apollo::drivers::PointCloud& lidar_safe_area_, std::vector<cv::Point3f>& lidar_cloud_buf, int& effect_point)
{
	int image_height = 1080;
	int image_width = 1920;
	int * imagexy_check= (int *)malloc(1080*1920*sizeof(int));
	// int imagexy_check[1080][1920] = { 0,0 };
	effect_point = 0;

	apollo::drivers::PointCloud lidar2image_check_;
	lidar2image_check_.clear_point();
	lidar2image_check_.set_measurement_time(data.measurement_time());

	lidar2image_paint_.clear_point();
	lidar2image_paint_.set_measurement_time(data.measurement_time());

	lidar_safe_area_.clear_point();
	lidar_safe_area_.set_measurement_time(data.measurement_time());
	lidar_cloud_buf.clear();
	//lidar_cloud_buf[lidar_buf_index_];// = new pcl::PointCloud<pcl::PointXYZ>();
	int p1_ep = 0;
	int p2_ep = 0;
	cv::Mat current_point = cv::Mat::ones(3, 1, CV_64FC1);
	long lidar_mem_length = data.point_size() * 3;
	double* lidar_buffer = (double*)malloc(lidar_mem_length * sizeof(double));
	//double lidar[lidar_mem_length];
	//WATRIX_ERROR <<lidar2image_check_[lidar_buf_index_].timestamp_msec();
	int p_count = 0;
	for (int i = 0; i < data.point_size(); i++) {
		lidar_buffer[p_count++] = data.point(i).x();
		lidar_buffer[p_count++] = data.point(i).y();
		lidar_buffer[p_count++] = data.point(i).z();
		lidar_cloud_buf.push_back(cv::Point3f(data.point(i).x(), data.point(i).y(), data.point(i).z()));  //return
	}
	//cv::Mat image_with_lidar = Mat::zeros(image_height, image_width, CV_8UC3); // 0-255
	// double lidar[300000];
	// memset(lidar,0,sizeof(double)*300000);
	// lidar[p] = y;
	// lidar[p+1] = -x;
	// lidar[p+2] = z; 

	cv::Mat lidar_points = cv::Mat(data.point_size(), 3, CV_64FC1, lidar_buffer);
	cv::Mat lidar_points_t;
	transpose(lidar_points, lidar_points_t);

	torch::DeviceType device_type = torch::kCUDA;  //torch::kCUDA  and torch::kCPU
	torch::Device device(device_type, 0);

	torch::Tensor tensor_rotation = torch::tensor({ 0.999832,0.0,-0.015292,0.00086496258196527018,0.99999655211205762,-0.0026255536555242422,
													0.015294316876925231,0.0026259908600024411,0.99982955268435725 }, at::kDouble).view({ 3,3 }).to(device);   //rotation
	torch::Tensor tensor_rotation_cv = torch::tensor({ 0.9999960856928221, 0.0003353891285046393, -0.002777789258835963,-0.0003158570188030314,
													  0.9999752464831646, 0.007028986788848552,0.002780077944536163, -0.007028081891000952, 0.9999714382078898 }, at::kDouble).view({ 3,3 }).to(device);   //rotation_cv
	torch::Tensor tensor_image = torch::from_blob(lidar_points_t.data, { lidar_points_t.rows, lidar_points_t.cols }, torch::kDouble).to(device);   //cloud_points

	torch::Tensor tensor_change_points = tensor_rotation.mm(tensor_image);
	std::vector<torch::Tensor> sp_mat_points = at::split(tensor_change_points, 1, 0);
	torch::Tensor tensor_x = sp_mat_points[1];
	torch::Tensor tensor_y = sp_mat_points[0].mul(-1.0);
	torch::Tensor tensor_z = sp_mat_points[2].add(0.2);
	torch::Tensor tensor_yxz = at::cat({ tensor_x,tensor_y,tensor_z }, 0);
	tensor_change_points = tensor_rotation_cv.mm(tensor_yxz);

	tensor_change_points = tensor_change_points.cpu();

	double* point_result = (double*)tensor_change_points.data_ptr();
	
	int WIDTH = data.point_size();
	std::vector<cv::Point3d> all_cloud;
	std::vector<cv::Point2d> projectedPoints;
	for (int w = 0; w < WIDTH; ++w) {

		double pt_x = *(point_result + w);
		double pt_y = *(point_result + WIDTH + w);
		double pt_z = *(point_result + 2 * WIDTH + w);
		
		all_cloud.push_back(cv::Point3d(pt_x, pt_y, pt_z));
	}

	cv::projectPoints(all_cloud, rvec, tvec, cameraMatrix, distCoeff, projectedPoints);
	
	for (int w = 0; w < WIDTH; w++) {	

		int cv_ori_point_j = projectedPoints[w].x + 7;
		int cv_ori_point_i = projectedPoints[w].y + 5;

		if (cv_ori_point_i < 0 || cv_ori_point_i + 1 >= image_height || cv_ori_point_j < 0 || cv_ori_point_j + 1 >= image_width) {
			continue;
		}

		// if (imagexy_check[(int)cv_ori_point_i][(int)cv_ori_point_j] == 1) {
		if (imagexy_check[cv_ori_point_i* image_width +cv_ori_point_j] == 1) {
			continue;
		}
		imagexy_check[cv_ori_point_i* image_width +cv_ori_point_j] = 1;
		
		// imagexy_check[(int)cv_ori_point_i][(int)cv_ori_point_j] = 1;
		apollo::drivers::PointXYZIT* lidar_point1 = lidar2image_check_.add_point();
		lidar_point1->set_x(cv_ori_point_j);
		lidar_point1->set_y(cv_ori_point_i);
		//////////////////////////////////////////////
		apollo::drivers::PointXYZIT * lpt1 = lidar2image_paint_.add_point();
		if (cv_ori_point_i < 1 || cv_ori_point_i + 1 >= image_height || cv_ori_point_j < 1 || cv_ori_point_j + 1 >= image_width) {
			lpt1->set_x(1);
			lpt1->set_y(1);
		}
		else {
			lpt1->set_x(cv_ori_point_j);
			lpt1->set_y(cv_ori_point_i);
		}

		//safe area data 
		apollo::drivers::PointXYZIT * lpt2 = lidar_safe_area_.add_point();
		lpt2->set_x(data.point(w).x());
		lpt2->set_y(data.point(w).y());
		lpt2->set_z(data.point(w).z());

		effect_point++;   //return
	}


	free(lidar_buffer);
	free( imagexy_check);
	if (effect_point < 1) {

		return;
	}


}


template <class OutputIterator>
void alpha_edges(const Alpha_shape_2& A, OutputIterator out)
{
	Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
		end = A.alpha_shape_edges_end();
	for (; it != end; ++it)
		* out++ = A.segment(*it);
}

 std::vector<cv::Point2d> FindContours_v2::start_contours(std::vector<apollo::drivers::PointXYZIT> points) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_train(new pcl::PointCloud<pcl::PointXYZ>);

	Eigen::Matrix4f rotation, rotation1, rotation2;
	rotation1 << 0.999832, 0.000000, -0.015292, 0.000000,
		0.000905, 1.000000, -0.000008, 0.000000,
		0.015292, 0.000008, 0.999833, 0.000000,
		0.000000, 0.000000, 0.000000, 1.000000;
	rotation2 << 1, 0, 0, 0,
		0, cos(0.15 * CV_PI / 180), -sin(0.15 * CV_PI / 180), 0.000000,
		0, sin(0.15 * CV_PI / 180), cos(0.15 * CV_PI / 180), 0.000000,
		0, 0, 0, 1;

	rotation = rotation2 * rotation1;

	std::vector<cv::Point2d> result;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ori(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_box(new pcl::PointCloud<pcl::PointXYZ>);

	for (int i = 0;i < points.size();++i) {
		pcl::PointXYZ pt;
		pt.x = (points[i].x());
		pt.y = (points[i].y());
		pt.z = (points[i].z());
		cloud_ori->points.push_back(pt);
	}

	if (cloud_ori->points.size() == 0)
		return result;

	pcl::transformPointCloud(*cloud_ori, *cloud_ori, rotation);
	for (int i = 0; i < cloud_ori->points.size(); ++i) {

		if (cloud_ori->points[i].x >= -0.5f && cloud_ori->points[i].x <= 3.3f)  //-0.38  3.0
			cloud_box->points.push_back(cloud_ori->points[i]);

	}

	if (cloud_box->points.size() == 0)
		return result;

	std::list<Point2> train;

	pcl::PointXYZ min_p, max_p;
	pcl::getMinMax3D(*cloud_box, min_p, max_p);

	std::vector<float> grid_row_index;
	float pt_distance = 0.0f;
	float cur_grid_height;
	float limit_distance = max_p.z;
	if (max_p.z > 300)
		limit_distance = 300.0f;
	do {
		cur_grid_height = 268.7608f - 269.0594f * sqrtf(1 - pt_distance * pt_distance / 90000.0f) + 0.35f;
		pt_distance += cur_grid_height;
		grid_row_index.push_back(pt_distance);
	} while (pt_distance < limit_distance);

	float col_radio = 0.10f;
	int grid_row = grid_row_index.size();
	int grid_col = (max_p.y - min_p.y) / col_radio + 1;
	Grid_data_init* temp = new Grid_data_init(grid_row, grid_col);

	auto& grid = *temp;
	for (int i = 0; i < grid_row; i++) {
		for (int j = 0; j < grid_col; j++)
		{
			grid[i][j].grid_points.clear();
			grid[i][j].min_x = 0.0f;
			grid[i][j].max_x = 0.0f;
			grid[i][j].total_x = 0.0f;
			grid[i][j].total_z = 0.0f;
			grid[i][j].max_y = 0.0f;
			grid[i][j].min_y = 0.0f;
			grid[i][j].max_z = 0.0f;
			grid[i][j].min_z = 0.0f;
		}
	}
	int screen_num = 0;
	create_grid(cloud_box, grid, min_p, max_p, grid_row_index, col_radio);
	auto obstacle_grid = cluster(grid, screen_num);

	if(obstacle_grid.size()==0)
		return result;

	screen_num += 1;
	if (obstacle_grid.size() < screen_num)
		screen_num = obstacle_grid.size();


	std::vector<int> ob_index;
	std::vector<float> index_tmp;
	for (int i = 0; i < obstacle_grid.size(); ++i)
		index_tmp.push_back(obstacle_grid[i].maxSize);

	int max_points = 0;
	int max_index = -1;
	for (int i = 0; i < screen_num; ++i) {
		for (int j = 0; j < index_tmp.size(); ++j) {
			if (max_points < index_tmp[j]) {
				max_index = j;
				max_points = index_tmp[j];
			}
		}
		ob_index.push_back(max_index);
		max_points = 0;
		index_tmp[max_index] = 0;
		max_index = -1;
	}
	for (int i = 0; i < ob_index.size(); ++i) {
		for (int j = 0; j < obstacle_grid[ob_index[i]].gridIndex.size(); ++j) {
			for (int k = 0; k < grid[obstacle_grid[ob_index[i]].gridIndex[j].first][obstacle_grid[ob_index[i]].gridIndex[j].second].grid_points.size(); ++k) {
				pcl::PointXYZ pt;
				pt.x = grid[obstacle_grid[ob_index[i]].gridIndex[j].first][obstacle_grid[ob_index[i]].gridIndex[j].second].grid_points[k][0];
				pt.y = grid[obstacle_grid[ob_index[i]].gridIndex[j].first][obstacle_grid[ob_index[i]].gridIndex[j].second].grid_points[k][1];
				pt.z = grid[obstacle_grid[ob_index[i]].gridIndex[j].first][obstacle_grid[ob_index[i]].gridIndex[j].second].grid_points[k][2];
				//std::cout<<"i = "<<i<<"  j="<<j<<"  k="<<k<<std::endl;
				cloud_train->points.push_back(pt);
			}
		}
	}
	delete& grid;

	downSampling(cloud_train, train);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);

	alphaShape(train, cloud_temp, min_p.x);

	for (int i = 0;i < cloud_temp->points.size();++i) {

		pcl::PointXYZ pt = cloud_temp->points[i];
		float temp = pt.x;
		pt.x = pt.y;
		pt.y = -temp;
		pt.z = pt.z + 0.2f;
		cloud_temp->points[i].x = pt.x;
		cloud_temp->points[i].y = pt.y;
		cloud_temp->points[i].z = pt.z;

	}

	Eigen::Matrix4d rotation_cv;
	rotation_cv << 0.9999960856928221, 0.0003353891285046393, -0.002777789258835963, 0,
		-0.0003158570188030314, 0.9999752464831646, 0.007028986788848552, 0,
		0.002780077944536163, -0.007028081891000952, 0.9999714382078898, 0,
		0.000000, 0.000000, 0.000000, 1.000000;

	pcl::transformPointCloud(*cloud_temp, *cloud_temp, rotation_cv);

	std::vector<cv::Point3d> all_cloud;
	for (int i = 0; i < cloud_temp->points.size(); ++i) {

		pcl::PointXYZ pt = cloud_temp->points[i];
		all_cloud.push_back(cv::Point3d(pt.x, pt.y, pt.z));
	}
	cv::projectPoints(all_cloud, rvec, tvec, cameraMatrix, distCoeff, result);

	std::vector<cv::Point2d> result_modify;
	for (int i = 0;i < result.size();++i) {

		result[i].x += 7;
		result[i].y += 5;

		if (result[i].x >= 0 && result[i].x < 1920 && result[i].y >= 0 && result[i].y < 1080) 
			result_modify.push_back(cv::Point2d(result[i].x, result[i].y));	
		
	}

	return result_modify;
}

void FindContours_v2::alphaShape(std::list<Point2> points, pcl::PointCloud<pcl::PointXYZ>::Ptr& tmp, float min_h) {

	Alpha_shape_2 A(points.begin(), points.end(),
		FT(10000),
		Alpha_shape_2::GENERAL);

	std::vector<Segment> segments;
	alpha_edges(A, std::back_inserter(segments));

	for (int i = 0; i < segments.size(); ++i) {
		pcl::PointXYZ pt;
		pt.x = min_h;
		pt.y = segments[i].source().hx();
		pt.z = segments[i].source().hy();
		tmp->points.push_back(pt);
		pt.x = min_h;
		pt.y = segments[i].target().hx();
		pt.z = segments[i].target().hy();
		tmp->points.push_back(pt);
	}

}

void FindContours_v2::search_fork(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
	OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, int row, int col) {

	if (row >= 0 && row < grid.grid_rows && col >= 0 && col < grid.grid_cols && visited[row][col] == 0) {

		if (grid[row][col].grid_points.size() >= 1) {

			extend.push(std::make_pair(row, col));
			visited[row][col] = 1;

			obstacle_grid_group.gridIndex.push_back(std::make_pair(row, col));
			grids_total_x += grid[row][col].total_x;
			obstacle_grid_group.pointsNum += grid[row][col].grid_points.size();
			if (ob_size.max_x < grid[row][col].max_x)
				ob_size.max_x = grid[row][col].max_x;
			if (ob_size.min_x > grid[row][col].min_x)
				ob_size.min_x = grid[row][col].min_x;
			if (ob_size.max_y < grid[row][col].max_y)
				ob_size.max_y = grid[row][col].max_y;
			if (ob_size.min_y > grid[row][col].min_y)
				ob_size.min_y = grid[row][col].min_y;
			if (ob_size.max_z < grid[row][col].max_z)
				ob_size.max_z = grid[row][col].max_z;
			if (ob_size.min_z > grid[row][col].min_z)
				ob_size.min_z = grid[row][col].min_z;
		}
	}
}

void FindContours_v2::create_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Grid_data_init& grid, pcl::PointXYZ min_p, pcl::PointXYZ max_p,
	std::vector<float> grid_row_index, float col_radio) {

	for (int i = 0; i < cloud->points.size(); ++i) {

		pcl::PointXYZ pt = cloud->points[i];
		if (cloud->points[i].z > 300)
			continue;

		int grid_cur_row = 0;
		int grid_cur_col = (pt.y - min_p.y) / col_radio;

		for (int j = 0; j < grid_row_index.size(); ++j) {
			if (j == 0) {
				if (pt.z >= 0 && pt.z <= grid_row_index[j]) {
					grid_cur_row = j;
					break;
				}
			}
			else
			{
				if (pt.z > grid_row_index[j - 1] && pt.z <= grid_row_index[j]) {
					grid_cur_row = j;
					break;
				}
			}
		}

		grid[grid_cur_row][grid_cur_col].grid_points.push_back({ pt.x,pt.y,pt.z });
		if (grid[grid_cur_row][grid_cur_col].grid_points.size() == 1) {
			grid[grid_cur_row][grid_cur_col].min_x = pt.x;
			grid[grid_cur_row][grid_cur_col].max_x = pt.x;
			grid[grid_cur_row][grid_cur_col].min_y = pt.y;
			grid[grid_cur_row][grid_cur_col].max_y = pt.y;
			grid[grid_cur_row][grid_cur_col].min_z = pt.z;
			grid[grid_cur_row][grid_cur_col].max_z = pt.z;
		}
		else {
			if (grid[grid_cur_row][grid_cur_col].max_x < pt.x)
				grid[grid_cur_row][grid_cur_col].max_x = pt.x;
			if (grid[grid_cur_row][grid_cur_col].min_x > pt.x)
				grid[grid_cur_row][grid_cur_col].min_x = pt.x;
			if (grid[grid_cur_row][grid_cur_col].max_y < pt.y)
				grid[grid_cur_row][grid_cur_col].max_y = pt.y;
			if (grid[grid_cur_row][grid_cur_col].min_y > pt.y)
				grid[grid_cur_row][grid_cur_col].min_y = pt.y;
			if (grid[grid_cur_row][grid_cur_col].max_z < pt.z)
				grid[grid_cur_row][grid_cur_col].max_z = pt.z;
			if (grid[grid_cur_row][grid_cur_col].min_z > pt.z)
				grid[grid_cur_row][grid_cur_col].min_z = pt.z;
		}
		grid[grid_cur_row][grid_cur_col].total_x += pt.x;
		grid[grid_cur_row][grid_cur_col].total_z += pt.z;
	}
}

std::vector<OB_index_data> FindContours_v2::cluster(Grid_data_init& grid, int& screen_num) {

	std::stack<std::pair<int, int>> extend;

	float grids_total_x = 0.0f;
	float dis_thres = 0.0f;
	OB_Size ob_size;

	std::vector<std::vector<bool>> visited(grid.grid_rows, std::vector<bool>(grid.grid_cols, 0));
	std::vector<OB_index_data> obstacle_grid;
	OB_index_data obstacle_grid_group;

	for (int i = 0; i < grid.grid_rows; ++i) {
		for (int j = 0; j < grid.grid_cols; ++j) {

			if (grid[i][j].grid_points.size() < 1 || visited[i][j]) {

				visited[i][j] = true;
				continue;
			}

			/*if (grid[i][j].max_x >= 3.05f) {
				visited[i][j] = true;
				continue;
			}*/

			grids_total_x = 0.0f;
			obstacle_grid_group.pointsNum = 0;
			ob_size.max_x = grid[i][j].max_x;
			ob_size.max_y = grid[i][j].max_y;
			ob_size.max_z = grid[i][j].max_z;
			ob_size.min_x = grid[i][j].min_x;
			ob_size.min_y = grid[i][j].min_y;
			ob_size.min_z = grid[i][j].min_z;
			extend.push(std::make_pair(i, j));
			obstacle_grid_group.gridIndex.push_back(std::make_pair(i, j));
			grids_total_x = grid[i][j].total_x;

			obstacle_grid_group.pointsNum = grid[i][j].grid_points.size();
			visited[i][j] = 1;

			search_grid(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size);

			if (ob_size.max_x >= 1.0f && obstacle_grid_group.pointsNum >= 20) {

				obstacle_grid_group.ob_size = ob_size;
				if (ob_size.max_y - ob_size.min_y > ob_size.max_z - ob_size.min_z)
					obstacle_grid_group.maxSize = ob_size.max_y - ob_size.min_y;
				else
					obstacle_grid_group.maxSize = ob_size.max_z - ob_size.min_z;
				obstacle_grid.push_back(obstacle_grid_group);
			}
			obstacle_grid_group.gridIndex.clear();
			obstacle_grid_group.pointsNum = 0;
			obstacle_grid_group.maxSize = 0.0f;
		}
	}

	OB_index_data maxScale_ob;
	for (int i = 0; i < obstacle_grid.size(); ++i) {
		if (maxScale_ob.maxSize < obstacle_grid[i].maxSize) {
			maxScale_ob = obstacle_grid[i];
		}
	}

	if (maxScale_ob.ob_size.max_y - maxScale_ob.ob_size.min_y == maxScale_ob.maxSize) {
		if (maxScale_ob.ob_size.max_z > 0) {
			for (int i = 0; i < obstacle_grid.size();) {
				OB_index_data ob_tmp = obstacle_grid[i];
				if (ob_tmp.ob_size.max_y - ob_tmp.ob_size.min_y <= 1.5f && ob_tmp.ob_size.max_z - ob_tmp.ob_size.min_z <= 2.0f &&
					ob_tmp.ob_size.max_x - ob_tmp.ob_size.min_x >= 1.6f && ob_tmp.ob_size.min_z < maxScale_ob.ob_size.max_z) {
					screen_num += 1;
					obstacle_grid.erase(obstacle_grid.begin() + i);
				}
				else
					++i;
			}
		}
		else if (maxScale_ob.ob_size.min_z < 0) {
			for (int i = 0; i < obstacle_grid.size();) {
				OB_index_data ob_tmp = obstacle_grid[i];
				if (ob_tmp.ob_size.max_y - ob_tmp.ob_size.min_y <= 1.5f && ob_tmp.ob_size.max_z - ob_tmp.ob_size.min_z <= 2.0f &&
					ob_tmp.ob_size.max_x - ob_tmp.ob_size.min_x >= 1.6f && ob_tmp.ob_size.max_z > maxScale_ob.ob_size.min_z) {
					screen_num += 1;
					obstacle_grid.erase(obstacle_grid.begin() + i);
				}
				else
					++i;
			}
		}
	}
	else {
		if (maxScale_ob.ob_size.max_y > 0) {
			for (int i = 0; i < obstacle_grid.size();) {
				OB_index_data ob_tmp = obstacle_grid[i];
				if (ob_tmp.ob_size.max_y - ob_tmp.ob_size.min_y <= 1.5f && ob_tmp.ob_size.max_z - ob_tmp.ob_size.min_z <= 2.0f &&
					ob_tmp.ob_size.max_x - ob_tmp.ob_size.min_x >= 1.6f && ob_tmp.ob_size.min_y < maxScale_ob.ob_size.max_y) {
					screen_num += 1;
					obstacle_grid.erase(obstacle_grid.begin() + i);
				}
				else
					++i;
			}
		}
		else if (maxScale_ob.ob_size.min_y < 0) {
			for (int i = 0; i < obstacle_grid.size();) {
				OB_index_data ob_tmp = obstacle_grid[i];
				if (ob_tmp.ob_size.max_y - ob_tmp.ob_size.min_y <= 1.5f && ob_tmp.ob_size.max_z - ob_tmp.ob_size.min_z <= 2.0f &&
					ob_tmp.ob_size.max_x - ob_tmp.ob_size.min_x >= 1.6f && ob_tmp.ob_size.max_y > maxScale_ob.ob_size.min_y) {
					screen_num += 1;
					obstacle_grid.erase(obstacle_grid.begin() + i);
				}
				else
					++i;
			}
		}
	}

	return obstacle_grid;
}

void FindContours_v2::search_grid(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
	OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size) {

	while (!extend.empty())
	{
		std::pair<int, int> cur = extend.top();
		float grid_avr_z = grid[cur.first][cur.second].total_z / grid[cur.first][cur.second].grid_points.size();
		int col_extend = grid_avr_z / 17 + 1;
		if (grid_avr_z > 150)
			col_extend += 3;
		if (grid_avr_z > 200)
			col_extend += 3;
		extend.pop();
		//top
		search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1, cur.second);
		//bottom
		search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1, cur.second);
		//left
		for (int i = 0; i < col_extend; ++i)
			search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first, cur.second - 1 - i);
		//right
		for (int i = 0; i < col_extend; ++i)
			search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first, cur.second + 1 + i);

		if (grid[cur.first][cur.second].total_z / grid[cur.first][cur.second].grid_points.size() >= 20) {

			//top left
			for (int i = 0; i < col_extend; ++i)
				search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1, cur.second - 1 - i);
			//right left
			for (int i = 0; i < col_extend; ++i)
				search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1, cur.second + 1 + i);
			//bottom left
			for (int i = 0; i < col_extend; ++i)
				search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1, cur.second - 1 - i);
			//bottom right
			for (int i = 0; i < col_extend; ++i)
				search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1, cur.second + 1 + i);
		}

	}
}

void FindContours_v2::downSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_train, std::list<Point2>& points) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud_train);
	sor.setLeafSize(0.1f, 0.1f, 0.1f);
	sor.filter(*cloud_filter);

	for (int i = 0;i < cloud_filter->points.size();++i) {
		pcl::PointXYZ pt = cloud_filter->points[i];
		Point2 pt2(pt.y, pt.z);
		points.push_back(pt2);
	}

}

}
}
}