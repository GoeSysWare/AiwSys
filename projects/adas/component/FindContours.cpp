#include "FindContours.h"

template <class OutputIterator>
void alpha_edges(const Alpha_shape_2& A, OutputIterator out)
{
	Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
		end = A.alpha_shape_edges_end();
	for (; it != end; ++it)
		* out++ = A.segment(*it);
}

watrix::proto::PointCloud FindContours::start_contours(std::vector<watrix::proto::LidarPoint> points) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_train(new pcl::PointCloud<pcl::PointXYZ>);

	FILE* fp = NULL;
	Eigen::Matrix4f rotation;
	rotation << 0.998985, 0.000149, -0.045041, 0.000000,
		0.00149, 0.999978, 0.0066, 0.000000,
		0.0045041, -0.0066, 0.998963, 0.000000,
		0.000000, 0.000000, 0.000000, 1.000000;

	watrix::proto::PointCloud result;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ori(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_box(new pcl::PointCloud<pcl::PointXYZ>);

	for (int i = 0;i < points.size();++i) {
		pcl::PointXYZ pt;
		pt.x = (points[i].x());
		pt.y =(points[i].y());
		pt.z =(points[i].z());
		cloud_ori->points.push_back(pt);
	}

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

	screen_num += 1;
	if (obstacle_grid.size() < screen_num)
		screen_num = obstacle_grid.size();

	//�ڵ�������Ϊn
	//�����������Ծ���Ŀ��������򣬵õ�n+1�ε���
	std::vector<int> ob_index;
	std::vector<float> index_tmp;
	for (int i = 0; i < obstacle_grid.size(); ++i){
		index_tmp.push_back(obstacle_grid[i].maxSize);

	}

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
	Eigen::Matrix4f r_t;
	r_t = rotation.inverse();
	pcl::transformPointCloud(*cloud_temp, *cloud_temp, r_t);
	for (int i = 0; i < cloud_temp->points.size(); ++i) {
		watrix::proto::LidarPoint* pt = result.add_points();
		pt->set_x ( cloud_temp->points[i].x);
		pt->set_y ( cloud_temp->points[i].y);
		pt->set_z ( cloud_temp->points[i].z);	
	}

	return result;
}

void FindContours::alphaShape(std::list<Point2> points, pcl::PointCloud<pcl::PointXYZ>::Ptr& tmp, float min_h) {

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

void FindContours::search_fork(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
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

void FindContours::create_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Grid_data_init& grid, pcl::PointXYZ min_p, pcl::PointXYZ max_p,
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

std::vector<OB_index_data> FindContours::cluster(Grid_data_init& grid, int& screen_num) {

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

	//�ڵ�����
	//����y�ᡢz��ĳ��Ⱥ͸߶��ж�Ŀ���Ƿ����ڵ���
	//������n���ڵ�����г�����Ϊn+1��
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

void FindContours::search_grid(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
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

void FindContours::downSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_train, std::list<Point2>& points) {

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

// int main() {

// 	std::vector<watrix::proto::LidarPoint> points;
// 	FindContours::start_contours(points);

// 	return 0;
// }

