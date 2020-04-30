
#pragma once
#include <vector>
struct Point3
{
	float x, y, z;

};

struct OB_Size {

	float max_x = 0.0f;
	float max_y = 0.0f;
	float max_z = 0.0f;
	float min_x = 0.0f;
	float min_y = 0.0f;
	float min_z = 0.0f;
	float loss_min_x = 0.0f;
};

struct OB_index_data {

	int pointsNum = 0;
	float maxSize = 0.0f;
	OB_Size ob_size;
	int lossNum = 0;
	std::vector<std::pair<int, int>> gridIndex;
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
	std::vector<Point3F> loss_points;
	float max_x = 0.0f;
	float min_x = 0.0f;
	float total_x = 0.0f;
	float total_z = 0.0f;
	float max_y = 0.0f;
	float min_y = 0.0f;
	float max_z = 0.0f;
	float min_z = 0.0f;
	float loss_min_x = 0.0f;
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