#include"LittleObjectDetection.h"

void LOD::create_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Grid_data_init& grid, pcl::PointXYZ min_p, pcl::PointXYZ max_p,
	float row_radio, float col_radio) {

	for (int i = 0; i < cloud->points.size(); ++i) {

		pcl::PointXYZ pt = cloud->points[i];

		int grid_cur_row = (pt.z - min_p.z) / row_radio;
		int grid_cur_col = (pt.y - min_p.y) / col_radio;

		if (pt.x < -1.14f) {
			grid[grid_cur_row][grid_cur_col].loss_points.push_back({ pt.x,pt.y,pt.z });
			if (grid[grid_cur_row][grid_cur_col].loss_points.size() == 1)
				grid[grid_cur_row][grid_cur_col].loss_min_x = pt.x;
			else
				if (grid[grid_cur_row][grid_cur_col].loss_min_x > pt.x)
					grid[grid_cur_row][grid_cur_col].loss_min_x = pt.x;
			continue;
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

std::vector<OB_index_data> LOD::cluster(Grid_data_init& grid) {

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

			grids_total_x = 0.0f;
			obstacle_grid_group.pointsNum = 0;
			obstacle_grid_group.lossNum = 0;
			ob_size.max_x = grid[i][j].max_x;
			ob_size.max_y = grid[i][j].max_y;
			ob_size.max_z = grid[i][j].max_z;
			ob_size.min_x = grid[i][j].min_x;
			ob_size.min_y = grid[i][j].min_y;
			ob_size.min_z = grid[i][j].min_z;
			ob_size.loss_min_x = grid[i][j].loss_min_x;
			extend.push(std::make_pair(i, j));
			obstacle_grid_group.gridIndex.push_back(std::make_pair(i, j));
			grids_total_x = grid[i][j].total_x;
			obstacle_grid_group.lossNum = grid[i][j].loss_points.size();
			obstacle_grid_group.pointsNum = grid[i][j].grid_points.size();
			visited[i][j] = 1;

			search_grid(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size);

			int needNum = 0;
			needNum = 24 - 4 * int(ob_size.min_z / 10);
			if (ob_size.min_z > 50)
				needNum = 6;
			if (ob_size.min_z > 60)
				needNum = 3;

			if (obstacle_grid_group.pointsNum > 1 && obstacle_grid_group.pointsNum + obstacle_grid_group.lossNum >= needNum) {

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
			obstacle_grid_group.lossNum = 0;
		}
	}

	return obstacle_grid;
}

void LOD::search_grid(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
	OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size) {

	while (!extend.empty())
	{
		std::pair<int, int> cur = extend.top();
		float grid_avr_z = grid[cur.first][cur.second].total_z / grid[cur.first][cur.second].grid_points.size();
		int col_extend = grid_avr_z / 10 + 1;
		if (grid_avr_z > 40)
			col_extend = 5;

		extend.pop();
		//��
		search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1, cur.second);
		//��
		search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1, cur.second);
		//��
		for (int i = 0; i < col_extend; ++i)
			search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first, cur.second - 1 - i);
		//��
		for (int i = 0; i < col_extend; ++i)
			search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first, cur.second + 1 + i);

		if (grid[cur.first][cur.second].total_z / grid[cur.first][cur.second].grid_points.size() >= 20) {

			//����
			for (int i = 0; i < col_extend; ++i)
				search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1, cur.second - 1 - i);
			//����
			for (int i = 0; i < col_extend; ++i)
				search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1, cur.second + 1 + i);
			//����
			for (int i = 0; i < col_extend; ++i)
				search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1, cur.second - 1 - i);
			//����
			for (int i = 0; i < col_extend; ++i)
				search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1, cur.second + 1 + i);
		}

	}
}

void LOD::search_fork(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
	OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, int row, int col) {

	if (row >= 0 && row < grid.grid_rows && col >= 0 && col < grid.grid_cols && visited[row][col] == 0) {

		if (grid[row][col].grid_points.size() >= 1) {

			extend.push(std::make_pair(row, col));
			visited[row][col] = 1;

			obstacle_grid_group.gridIndex.push_back(std::make_pair(row, col));
			grids_total_x += grid[row][col].total_x;
			obstacle_grid_group.pointsNum += grid[row][col].grid_points.size();
			obstacle_grid_group.lossNum += grid[row][col].loss_points.size();
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
			if (ob_size.loss_min_x > grid[row][col].loss_min_x)
				ob_size.loss_min_x = grid[row][col].loss_min_x;
		}
	}
}

std::vector<watrix::proto::PointCloud> LOD::object_detection(watrix::proto::PointCloud check_pointclouds) {

	Eigen::Matrix4f rotation;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_init(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_limit(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_result(new pcl::PointCloud<pcl::PointXYZ>);
	//7��10���״���ת����
	rotation << 0.998985, 0.000000, -0.045041, 0.000000,
		0.000000, 1.000000, 0.000000, 0.000000,
		0.045041, 0.000000, 0.998963, 0.000000,
		0.000000, 0.000000, 0.000000, 1.000000;

	for (int i = 0;i < check_pointclouds.points_size();++i) {
		pcl::PointXYZ pp;
		pp.x = check_pointclouds.points(i).x();
		pp.y = check_pointclouds.points(i).y();
		pp.z = check_pointclouds.points(i).z();
		cloud_init->points.push_back(pp);
		if(i <3){
			std::cout<<pp.x<<" "<<pp.y<<" "<<pp.z<<" "<<std::endl;
		}
	}

	pcl::transformPointCloud(*cloud_init, *cloud_init, rotation);

	for (int i = 0;i < cloud_init->points.size();++i) {
		pcl::PointXYZ pt = cloud_init->points[i];
		if (pt.x <= 1.9f && pt.z <= 55.0f) {
			cloud_limit->points.push_back(pt);
		}
	}

	pcl::PointXYZ min_p, max_p;
	pcl::getMinMax3D(*cloud_limit, min_p, max_p);

	float row_radio = 0.10f;
	float col_radio = 0.05f;
	int grid_row = (max_p.z - min_p.z) / row_radio + 1;
	int grid_col = (max_p.y - min_p.y) / col_radio + 1;

	Grid_data_init* temp = new Grid_data_init(grid_row, grid_col);

	auto& grid = *temp;
	for (int i = 0; i < grid_row; i++) {
		for (int j = 0; j < grid_col; j++)
		{
			grid[i][j].grid_points.clear();
			grid[i][j].loss_points.clear();
			grid[i][j].min_x = 0.0f;
			grid[i][j].max_x = 0.0f;
			grid[i][j].total_x = 0.0f;
			grid[i][j].total_z = 0.0f;
			grid[i][j].max_y = 0.0f;
			grid[i][j].min_y = 0.0f;
			grid[i][j].max_z = 0.0f;
			grid[i][j].min_z = 0.0f;
			grid[i][j].loss_min_x = 0.0f;
		}
	}

	create_grid(cloud_limit, grid, min_p, max_p, row_radio, col_radio);
	auto obstacle_grid = cluster(grid);

	Eigen::Matrix4f r_t;
	r_t = rotation.inverse();
	std::vector<watrix::proto::PointCloud> obstacle_box;
	
	for (int i = 0;i < obstacle_grid.size();++i) {
		watrix::proto::PointCloud obstacle_temp;
		cloud_result->points.clear();
		OB_Size ob_s = obstacle_grid[i].ob_size;
		if (ob_s.loss_min_x > ob_s.min_x)
			ob_s.loss_min_x = ob_s.min_x;
		cloud_result->points.push_back({ ob_s.loss_min_x, ob_s.min_y, ob_s.min_z });
		cloud_result->points.push_back({ ob_s.max_x, ob_s.min_y, ob_s.min_z });
		cloud_result->points.push_back({ ob_s.max_x, ob_s.max_y, ob_s.min_z });
		cloud_result->points.push_back({ ob_s.loss_min_x, ob_s.max_y, ob_s.min_z });
		pcl::transformPointCloud(*cloud_result, *cloud_result, r_t);

		watrix::proto::LidarPoint* pt_tmp = obstacle_temp.add_points();
		pt_tmp->set_x(cloud_result->points[0].x);
		pt_tmp->set_y(cloud_result->points[0].y);
		pt_tmp->set_z(cloud_result->points[0].z);
		watrix::proto::LidarPoint* pt_tmp1 = obstacle_temp.add_points();
		pt_tmp1->set_x(cloud_result->points[1].x);
		pt_tmp1->set_y(cloud_result->points[1].y);
		pt_tmp1->set_z(cloud_result->points[1].z);
		watrix::proto::LidarPoint* pt_tmp2 = obstacle_temp.add_points();
		pt_tmp2->set_x(cloud_result->points[2].x);
		pt_tmp2->set_y(cloud_result->points[2].y);
		pt_tmp2->set_z(cloud_result->points[2].z);
		watrix::proto::LidarPoint* pt_tmp3 = obstacle_temp.add_points();
		pt_tmp3->set_x(cloud_result->points[3].x);
		pt_tmp3->set_y(cloud_result->points[3].y);
		pt_tmp3->set_z(cloud_result->points[3].z);

		obstacle_box.push_back(obstacle_temp);
	}
	delete& grid;
	return obstacle_box;
}


// int main() {

// 	std::vector<watrix::proto::LidarPoint> points;
// 	std::vector<std::vector<watrix::proto::LidarPoint>> obstacle_box;

// 	obstacle_box = LOD::object_detection(points);

// 	return 0;
// }