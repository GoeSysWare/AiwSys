#include"ObjectAndTrainDetection.h"
#include <chrono>
using namespace std;
namespace watrix {
	namespace algorithm {

		std::unordered_map<int, std::pair<int, int>> LOD::getInvasionMap(std::vector<cv::Point2i> input_l, std::vector<cv::Point2i> input_r, int& top_y) {

			for (int i = 0;i < input_l.size();) {

				if (input_l[i].x < 0 || input_l[i].x >= PIC_WIDTH ||
					input_l[i].y < 0 || input_l[i].y >= PIC_HEIGHT)
					input_l.erase(input_l.begin() + i);
				else
					++i;
			}
			for (int i = 0;i < input_r.size();) {

				if (input_r[i].x < 0 || input_r[i].x >= PIC_WIDTH ||
					input_r[i].y < 0 || input_r[i].y >= PIC_HEIGHT)
					input_r.erase(input_r.begin() + i);
				else
					++i;
			}
			
			std::unordered_map<int, std::pair<int, int>> invasionP;
			std::vector<InvasionData2D> input;
			for (int i = 0; i < PIC_HEIGHT; ++i) {

				InvasionData2D ivd = { -1,-1,i };
				input.push_back(ivd);
			}
			int input_l_index = input_l.size() - 1;
			int input_r_index = input_r.size() - 1;
			top_y = input_l[input_l_index].y > input_r[input_r_index].y ? input_l[input_l_index].y : input_r[input_r_index].y;

			int loss_l = input_l[0].y;
			int loss_r = input_r[0].y;

			for (int i = PIC_HEIGHT - 1; i > loss_l; --i)
				input[i].left_x = input_l[0].x;
			for (int i = PIC_HEIGHT - 1; i > loss_r; --i)
				input[i].right_x = input_r[0].x;

			for (int i = 0; i < input_l.size(); ++i) {

				if (i == 0)
					input[input_l[i].y].left_x = input_l[i].x;
				else {
					int start_y = input_l[i].y;
					int start_x = input_l[i].x;
					int end_y = input_l[i - 1].y;
					int end_x = input_l[i - 1].x;
					int dis_y = end_y - start_y;
					int dis_x = end_x - start_x;
					if (dis_y > 1) {

						for (int j = end_y - 1; j > start_y; --j) {

							input[j].left_x = (j - start_y) * dis_x / dis_y + start_x;
						}
					}

					input[input_l[i].y].left_x = input_l[i].x;
				}
			}

			for (int i = 0; i < input_r.size(); ++i) {

				if (i == 0)
					input[input_r[i].y].right_x = input_r[i].x;
				else {
					int start_y = input_r[i].y;
					int start_x = input_r[i].x;
					int end_y = input_r[i - 1].y;
					int end_x = input_r[i - 1].x;
					int dis_y = end_y - start_y;
					int dis_x = end_x - start_x;
					if (dis_y > 1) {

						for (int j = end_y - 1; j > start_y; --j) {

							input[j].right_x = (j - start_y) * dis_x / dis_y + start_x;
						}
					}

					input[input_r[i].y].right_x = input_r[i].x;
				}
			}

			for (int i = 0; i < input.size(); ++i) {

				if (input[i].left_x != -1 && input[i].right_x != -1)
					invasionP.insert(std::make_pair(input[i].y, std::make_pair(input[i].left_x, input[i].right_x)));
			}

			return invasionP;
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr LOD::getPointFrom2DAnd3D(const std::vector<cv::Point3f>& cloud_cv, std::unordered_map<int, std::pair<int, int>> invasionP,
			int top_y, InvasionData invasion, Eigen::Matrix4d& rotation, float distance_limit) {

			pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_change_standard(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_change(new pcl::PointCloud<pcl::PointXYZ>);

			int min_y = top_y;

			cv::Mat cameraMatrix(3, 3, CV_32F);//������ڲξ���?
			std::vector<float> distCoeff;//����������
			cv::Mat rvec(3, 3, CV_64F), tvec(3, 1, CV_64F);//�任����

			distCoeff.push_back(-0.2759f);
			distCoeff.push_back(0.4355f);
			distCoeff.push_back(0.0010f);
			distCoeff.push_back(-1.2753e-04f);
			distCoeff.push_back(0.0f);

			float tempMatrix[3][3] = { { 2.1159e+03f, 0.0f, 974.2096f }, { 0.0f, 2.1175e+03f, 619.7170f }, { 0.0f, 0.0f, 1.0f } };

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					cameraMatrix.at<float>(i, j) = tempMatrix[i][j];
				}
			}

			double tempRvec[3][3] = { {1,0,0},
									{0,1,0},
									{0,0,1} };

			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					rvec.at<double>(i, j) = tempRvec[i][j];
			cv::Rodrigues(rvec, rvec);

			double tempTvec[3] = { 0.0,0.92,0.137 };

			for (int i = 0; i < 3; ++i)
				tvec.at<double>(i, 0) = tempTvec[i];

			Eigen::Matrix4d rotation1, rotation2, rotation3;

			rotation1 << 0.999829, 0.000000, -0.012079, 0.000000,
				0.010205, 0.999944, -0.000053, 0.000000,
				0.012079, -0.000053, 0.999935, 0.000000,
				0.000000, 0.000000, 0.000000, 1.000000;

			rotation2 << 1, 0, 0, 0,
				0, cos(-0.21 * CV_PI / 180), sin(-0.21 * CV_PI / 180), 0.000000,
				0, -sin(-0.21 * CV_PI / 180), cos(-0.21 * CV_PI / 180), 0.000000,
				0, 0, 0, 1;

			rotation3 << 1, 0, 0, 0,
				0, cos(-0.15 * CV_PI / 180), -sin(-0.15 * CV_PI / 180), 0.000000,
				0, sin(-0.15 * CV_PI / 180), cos(-0.15 * CV_PI / 180), 0.000000,
				0, 0, 0, 1;

			rotation = rotation3 * rotation1;

			float* lidar_buffer = (float*)malloc(cloud_cv.size() * 3 * sizeof(float));
			float* cloud = (float*) malloc(cloud_cv.size() * 3 * sizeof(float));
			float* cloud_standard = (float*) malloc(cloud_cv.size() * 3 * sizeof(float));
			float rotation1_3[9] = { 0.999829f, 0.000000f, -0.012079f, 0.010205f, 0.999944f, -0.000053f, 0.012079f, -0.000053f, 0.999935f };
			float rotation_3[9] = {(float)rotation(0,0),(float)rotation(0,1),(float)rotation(0,2),(float)rotation(1,0),(float)rotation(1,1),(float)rotation(1,2),
				(float)rotation(2,0),(float)rotation(2,1),(float)rotation(2,2)};
			
			for (int i = 0; i < cloud_cv.size(); i++) {
				lidar_buffer[i] = cloud_cv[i].x;
				lidar_buffer[i+cloud_cv.size()] = cloud_cv[i].y;
				lidar_buffer[i+2*cloud_cv.size()] = cloud_cv[i].z;
			}

			float *g_rotation1_3, *g_rotation_3,*g_lidar_buffer, *g_cloud,*g_cloud_standard;
			cudaMalloc((void **)&g_rotation1_3, sizeof(float) * 9);
			cudaMalloc((void **)&g_rotation_3, sizeof(float) * 9);
			cudaMalloc((void **)&g_lidar_buffer, sizeof(float) * cloud_cv.size()*3);
			cudaMalloc((void **)&g_cloud, sizeof(float) * cloud_cv.size()*3);
			cudaMalloc((void **)&g_cloud_standard, sizeof(float) * cloud_cv.size()*3);

			// initialize CUBLAS context
			cublasHandle_t handle;
			cublasCreate(&handle);

			cudaMemcpy(g_rotation1_3,&rotation1_3,9*sizeof(float),cudaMemcpyHostToDevice);
			cudaMemcpy(g_rotation_3,&rotation_3,9*sizeof(float),cudaMemcpyHostToDevice);
       		cudaMemcpy(g_lidar_buffer,lidar_buffer,3*cloud_cv.size()*sizeof(float),cudaMemcpyHostToDevice);
        	cudaMemset(g_cloud,0,3*cloud_cv.size()*sizeof(float));
			cudaMemset(g_cloud_standard,0,3*cloud_cv.size()*sizeof(float));

			float al = 1.0f;
			float bet = 0.0f;
			
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
				cloud_cv.size(),3 , 3, &al, g_lidar_buffer, 
				cloud_cv.size(), g_rotation_3, 3, &bet, g_cloud, cloud_cv.size());

			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
				cloud_cv.size(),3 , 3, &al, g_lidar_buffer, 
				cloud_cv.size(), g_rotation1_3, 3, &bet, g_cloud_standard, cloud_cv.size());      //g_rotation_3

			cudaMemcpy(cloud,g_cloud,3*cloud_cv.size()*sizeof(float),cudaMemcpyDeviceToHost);
			cudaMemcpy(cloud_standard,g_cloud_standard,3*cloud_cv.size()*sizeof(float),cudaMemcpyDeviceToHost);

			int left_num = invasion.coeff_left.size();
			int right_num = invasion.coeff_right.size();

			std::vector<float> points_z;
			std::vector<pcl::PointXYZ> mat_need_points;
			bool isShort = false;   //ͼ���й���ָ����Զ�����Ƿ�������������?
			float z_limit = DIVIDE_DIS + 20.0f;
			if (z_limit > distance_limit) {

				z_limit = distance_limit;
				isShort = true;
			}

			cloud_change->points.reserve(cloud_cv.size());
			cloud_change_standard->points.reserve(cloud_cv.size());
			points_z.reserve(cloud_cv.size());
			mat_need_points.reserve(cloud_cv.size());
			for (int i = 0;i < cloud_cv.size();++i) {

				pcl::PointXYZ pt;
				pt.x = cloud_standard[i];
				pt.y = cloud_standard[i + cloud_cv.size()];
				pt.z = cloud_standard[i + 2*cloud_cv.size()];

				if (pt.z >= DIVIDE_DIS + 18.0f && pt.z <= DETECT_DISTANCE) {    //50

					pcl::PointXYZ pt_change;
					pt_change.x = cloud[i];
					pt_change.y = cloud[i + cloud_cv.size()];
					pt_change.y = cloud[i + 2*cloud_cv.size()];
					cloud_change->points.push_back({ pt_change.y, -pt_change.x, pt_change.z + 0.1f });
					cloud_change_standard->points.push_back(pt);
				}

				if (pt.z >= z_limit) 
					continue;
				//float y_limit = 1.0f/15.0f * pt.z + 2.0f; //0,2  ----->30,4
				if( pt.y <= -4.0f || pt.y >= 4.0f)    //最小转�?���?150m
					continue;
				points_z.push_back(pt.z);
				mat_need_points.push_back(pt);
			}

			if (points_z.size() == 0){
				free(lidar_buffer);
				free(cloud);
				free(cloud_standard);
				cudaFree(g_rotation1_3);
				cudaFree(g_rotation_3);
				cudaFree(g_lidar_buffer);
				cudaFree(g_cloud);
				cudaFree(g_cloud_standard);
				return output;
			}

			cv::Mat m_combine;
			cv::Mat m_x0 = cv::Mat::ones(1, points_z.size(), CV_32F);
			cv::Mat m_x1(1, points_z.size(), CV_32F, points_z.data());
			cv::Mat m_x2 = m_x1.mul(m_x1);
			cv::Mat m_x3 = m_x1.mul(m_x2);
			cv::Mat m_x4 = m_x1.mul(m_x3);
			cv::Mat m_x5 = m_x1.mul(m_x4);

			m_x5.push_back(m_x4);
			m_x5.push_back(m_x3);
			m_x5.push_back(m_x2);
			m_x5.push_back(m_x1);
			m_x5.push_back(m_x0);
			m_combine = m_x5.t();

			cv::Mat left_coeff = (cv::Mat_<float>(left_num, 1) << invasion.coeff_left[0], invasion.coeff_left[1],
				invasion.coeff_left[2], invasion.coeff_left[3], invasion.coeff_left[4], invasion.coeff_left[5]);
			cv::Mat right_coeff = (cv::Mat_<float>(right_num, 1) << invasion.coeff_right[0], invasion.coeff_right[1],
				invasion.coeff_right[2], invasion.coeff_right[3], invasion.coeff_right[4], invasion.coeff_right[5]);

			cv::Mat left_result, right_result;
			left_result = m_combine * left_coeff;
			right_result = m_combine * right_coeff;

			for (int i = 0;i < mat_need_points.size();++i) {

				pcl::PointXYZ pt = mat_need_points[i];
				float l_limit = left_result.at<float>(i, 0);
				if (pt.y < l_limit - POINT_EXPAND)
					continue;
				float r_limit = right_result.at<float>(i, 0);

				double virtual_mid;

				if (r_limit - l_limit > 1.435f) {
					virtual_mid = (l_limit + r_limit) / 2.0f;
					if (pt.y >= virtual_mid - 0.7175f - POINT_EXPAND && pt.y <= virtual_mid + 0.7175f + POINT_EXPAND) {
						output->points.push_back(pt);
					}

				}
				else {
					if (pt.y >= l_limit - POINT_EXPAND && pt.y <= r_limit + POINT_EXPAND) {  //0.7825
						output->points.push_back(pt);
					}
				}
			}

			if (isShort)
			{
				free(lidar_buffer);
				free(cloud);
				free(cloud_standard);
				cudaFree(g_rotation1_3);
				cudaFree(g_rotation_3);
				cudaFree(g_lidar_buffer);
				cudaFree(g_cloud);
				cudaFree(g_cloud_standard);
				return output;

			}
				
			std::vector<cv::Point2d> projectedPoints;
			std::vector<cv::Point3d> point_need;

			Eigen::Matrix4d rotation_cv;

			rotation_cv << 0.9999994138848853, -0.0007077764296917184, -0.0008193182600682035, 0,
				0.000688561989143834, 0.9997301624711764, -0.02321913279479644, 0,
				0.0008355311321636263, 0.02321855503430087, 0.9997300638621639, 0,
				0.000000, 0.000000, 0.000000, 1.000000;

			pcl::transformPointCloud(*cloud_change, *cloud_change, rotation_cv);

			//将点云压缩到同一高度(地面高度)，再映射到图像上
			for (int i = 0; i < cloud_change->points.size(); ++i) {
				pcl::PointXYZ pt = cloud_change->points[i];
				point_need.push_back(cv::Point3d(pt.x, -GROUND_HEIGHT, pt.z));
			}

			if (point_need.size() == 0){

				free(lidar_buffer);
				free(cloud);
				free(cloud_standard);
				cudaFree(g_rotation1_3);
				cudaFree(g_rotation_3);
				cudaFree(g_lidar_buffer);
				cudaFree(g_cloud);
				cudaFree(g_cloud_standard);
				return output;
			}
				

			cv::projectPoints(point_need, rvec, tvec, cameraMatrix, distCoeff, projectedPoints);

			for (int i = 0; i < projectedPoints.size(); i++)
			{
				cv::Point2i p = projectedPoints[i];
				p.x += 0;
				p.y += 0;

				if (p.y < min_y)
					continue;

				if (p.x >= invasionP[p.y].first && p.x <= invasionP[p.y].second) {

					if (invasionP[p.y].first != 0 && invasionP[p.y].second != 0) {

						output->points.push_back(cloud_change_standard->points[i]);
					}

				}
			}

			free(lidar_buffer);
			free(cloud);
			free(cloud_standard);
			cudaFree(g_rotation1_3);
			cudaFree(g_rotation_3);
			cudaFree(g_lidar_buffer);
			cudaFree(g_cloud);
			cudaFree(g_cloud_standard);
		
			return output;
		}

		float LOD::calHeightVar(std::vector<Point3> object) {  //����һ��Ŀ��ĸ߶ȷ���??����ж����Ƿ���һ��ƽ��?������˳�?

			float total_x = 0.0f;
			for (int j = 0; j < object.size(); ++j) {

				Point3 pt = object[j];
				total_x += pt.x;
			}
			float ave_x = total_x / object.size();

			float total_var_x = 0.0f, total_var_y = 0.0f, total_var_z = 0.0f;

			for (int j = 0; j < object.size(); ++j) {

				total_var_x += (object[j].x - ave_x) * (object[j].x - ave_x);

			}

			float var_x = total_var_x / object.size();

			return sqrtf(var_x);
		}

		//return ave and var
		cv::Point2f LOD::calHeightVar(std::vector<float> object) {

			float total = 0.0f;
			for (int j = 0; j < object.size(); ++j)
				total += object[j];

			float ave = total / object.size();

			float total_var = 0.0f;

			for (int j = 0; j < object.size(); ++j) {

				total_var += (object[j] - ave) * (object[j] - ave);

			}

			float var_x = total_var / object.size();

			return cv::Point2f(ave, sqrtf(var_x));
		}

		void LOD::getInvasionData(InvasionData& invasion_data, std::string csv_file) {

			ifstream inFile(csv_file.c_str(), ios::in);
			if (!inFile)
			{
				cout << "��csvʧ�ܣ�" << endl;
				exit(1);
			}

			std::string line;
			std::string field;
			getline(inFile, line);

			std::istringstream sin(line);
			std::vector<double> coeff;

			getline(sin, field, ',');

			while (getline(sin, field, ',')) {

				coeff.push_back(atof(field.c_str()));
			}

			int num = coeff.size() / 2;

			for (int i = 0; i < num; ++i) {
				invasion_data.coeff_left.push_back(coeff[i]);
			}
			for (int i = num; i < coeff.size(); ++i) {
				invasion_data.coeff_right.push_back(coeff[i]);
			}

			inFile.close();
		}

		void LOD::getInvasionData(std::vector<cv::Point2i>& input_l, std::vector<cv::Point2i>& input_r,
			std::string csv_file_l, std::string csv_file_r) {

			ifstream inFile_l(csv_file_l.c_str(), ios::in);
			ifstream inFile_r(csv_file_r.c_str(), ios::in);

			if (!inFile_l || !inFile_r)
			{
				cout << "��csvʧ�ܣ�" << endl;
				exit(1);
			}

			std::string line;
			std::string field;
			getline(inFile_l, line);
			while (getline(inFile_l, line))
			{
				std::string field;
				std::istringstream sin(line);
				cv::Point2i pt;

				getline(sin, field, ',');
				pt.x = atof(field.c_str());
				getline(sin, field, ',');
				pt.y = atof(field.c_str());

				input_l.push_back(pt);

			}
			inFile_l.close();

			while (getline(inFile_r, line))
			{
				std::string field;
				std::istringstream sin(line);
				cv::Point2i pt;

				getline(sin, field, ',');
				pt.x = atof(field.c_str());
				getline(sin, field, ',');
				pt.y = atof(field.c_str());

				input_r.push_back(pt);

			}
			inFile_r.close();

		}

		FitLineData LOD::fitNextPoint(std::vector<cv::Point2d> lines, float row_radio) {

			float A = 0.0f;
			float B = 0.0f;
			float C = 0.0f;
			float D = 0.0f;

			for (int i = 0; i < lines.size(); ++i) {

				A += lines[i].x * lines[i].x;
				B += lines[i].x;
				C += lines[i].x * lines[i].y;
				D += lines[i].y;
			}

			float k, b, temp = 0;

			if (temp = (lines.size() * A - B * B)) {// �жϷ�ĸ��Ϊ0

				k = (lines.size() * C - B * D) / temp;
				b = (A * D - B * C) / temp;

			}
			else {

				k = 1;
				b = 0;

			}

			FitLineData line_data;
			line_data.k = k;
			line_data.b = b;
			int lines_index = lines.size() - 1;
			line_data.nextPoint = k * (lines[lines_index].x + row_radio) + b;

			return line_data;
		}

		void LOD::calLowestPoint(TrackData& trackdata, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointXYZ min_p,
			pcl::PointXYZ max_p, float row_radio, InvasionData invasion, std::string image_file) {

			int grid_row = (max_p.z - min_p.z) / row_radio + 1;

			for (int i = 0; i < grid_row; ++i)
				trackdata.trackBottomFit.push_back(100.0f);

			for (int i = 0; i < GROUND_JUDGE; ++i)
				trackdata.trackTop.push_back(-100.0f);

			std::vector<float> mid_y;
			std::vector<cv::Point2d> track_coor;
			//��ȡ���޽����ݵĵ��Ʋ���
			int left_num = invasion.coeff_left.size();
			int right_num = invasion.coeff_right.size();
			for (int i = 0; i < trackdata.trackBottomFit.size(); ++i) {

				double dis = i * row_radio + row_radio / 2 + min_p.z;

				double l_limit = 0;
				double r_limit = 0;

				for (int j = 0; j < left_num; ++j)
					l_limit += invasion.coeff_left[j] * std::pow(dis, left_num - 1 - j);
				for (int j = 0; j < right_num; ++j)
					r_limit += invasion.coeff_right[j] * std::pow(dis, right_num - 1 - j);

				mid_y.push_back((l_limit + r_limit) / 2.0f);
				track_coor.push_back({ l_limit,r_limit });
			}

			for (int i = 0; i < cloud->points.size(); ++i) {

				pcl::PointXYZ pt = cloud->points[i];
				int index = (pt.z - min_p.z) / row_radio;

				if (pt.y >= mid_y[index] - 0.9f && pt.y <= mid_y[index] + 0.9f) {

					if (trackdata.trackBottomFit[index] > pt.x) {
						trackdata.trackBottomFit[index] = pt.x;

					}
				}

				if (pt.x > 1.0f)
					continue;
				if (index < GROUND_JUDGE) {

					float height_limit = pt.z * 4 / 100;

					if (pt.y >= track_coor[index].x - 0.3f && pt.y <= track_coor[index].x + 0.3f ||
						pt.y >= track_coor[index].y - 0.3f && pt.y <= track_coor[index].y + 0.3f) {

						if (fabs(pt.x + 1.18f) <= height_limit && trackdata.trackTop[index] < pt.x) {
							trackdata.trackTop[index] = pt.x;
						}
					}
				}
			}

			bool isHole = false;
			int hole_judge_index = DETECT_DISTANCE / row_radio;
			if (hole_judge_index > trackdata.trackBottomFit.size())
				hole_judge_index = trackdata.trackBottomFit.size();
			for (int i = GROUND_JUDGE;i < hole_judge_index;) {

				if (trackdata.trackBottomFit[i] == 100.0f) {

					i += 1;
					continue;
				}

				float hole_standard = trackdata.trackBottomFit[i];
				std::vector<float> hole_judge_points;
				hole_judge_points.push_back(hole_standard);

				int last_index = i + 15;
				if (last_index - hole_judge_index >= 8)
					break;
				if (last_index > hole_judge_index)
					last_index = hole_judge_index;
				for (int j = i + 1;j < last_index;++j) {
					if (fabs(trackdata.trackBottomFit[j] - hole_standard) > 0.2f)
						continue;
					hole_judge_points.push_back(trackdata.trackBottomFit[j]);
				}
				if (hole_judge_points.size() < 3) {
					i += 1;
					continue;
				}
				cv::Point2f hole_judge = calHeightVar(hole_judge_points);
				if (hole_judge.x < HOLE_JUDGE_HEIGHT && hole_judge.y < 0.04f) {
					isHole = true;
					break;
				}
				i += 1;
			}

			std::vector<cv::Point2d> track_lines;
			float height_diff;
			bool isFind = false;
			for (int i = 3;i < trackdata.trackTop.size();++i) {

				cv::Point2d pt;
				pt.x = (float)i * row_radio + row_radio / 2;
				pt.y = trackdata.trackTop[i];
				if (!isFind) {
					if (fabs(trackdata.trackTop[i] + 1.18f) > 0.1f)
						continue;
					else {
						isFind = true;
						height_diff = pt.y;
					}
				}
				else {
					if (fabs(trackdata.trackTop[i] - height_diff) > 0.05f)
						continue;
					else
						height_diff = pt.y;

				}

				track_lines.push_back(pt);
			}
			//���ܳ��ֵ������������������?
			if (track_lines.size() < 3)
				return;
			FitLineData track_line_data = fitNextPoint(track_lines, row_radio);
			trackdata.trackBottom.assign(trackdata.trackBottomFit.begin(), trackdata.trackBottomFit.end());

			int nearby_fit_num = GROUND_JUDGE;
			if (GROUND_JUDGE > trackdata.trackBottomFit.size())
				nearby_fit_num = trackdata.trackBottomFit.size();
			for (int i = 0; i < nearby_fit_num; ++i) {

				float predict_track = track_line_data.k * ((float)i * row_radio + row_radio / 2) + track_line_data.b;
				if (fabs(trackdata.trackTop[i] - predict_track) > 0.06f)
					trackdata.trackTop[i] = predict_track;

				trackdata.trackBottomFit[i] = trackdata.trackTop[i] - 0.25f;     //25cm�������ڵĹ����ײ�ֵ

			}

			//���ж��ڳ����ڡ�����֮���пӵ�����£�ִ�����¹���?
			if (isHole) {
				for (int i = GROUND_JUDGE;i < trackdata.trackBottomFit.size();++i)
					trackdata.trackBottomFit[i] = trackdata.trackBottomFit[GROUND_JUDGE - 1];

				trackdata.trackBottom.clear();
				trackdata.trackBottom.assign(trackdata.trackBottomFit.begin(), trackdata.trackBottomFit.end());

				return;
			}

			//���ݹ���½�����ѡȡ�?�����ϵ�
			std::vector<cv::Point2d> whole_line;
			float whole_diff = trackdata.trackBottomFit[5];
			float slope_limit_k = -SLOPE_LIMIT / 100.0f;
			float slope_limit_b = whole_diff - slope_limit_k * (5.0f * row_radio + row_radio / 2);
			for (int i = 5; i < trackdata.trackBottomFit.size(); ++i) {

				cv::Point2d pt;
				pt.x = (float)i * row_radio + row_radio / 2;
				pt.y = trackdata.trackBottomFit[i];
				if (pt.y == 100.0f || fabs(pt.y - whole_diff) >= 0.5f)
					continue;
				float slope_judge = slope_limit_k * (float)pt.x + slope_limit_b;
				float slope_diff = slope_judge - pt.y;
				if (slope_diff > 0.03f)
					continue;
				whole_line.push_back(pt);
				whole_diff = pt.y;
			}

			FitLineData whole_line_data = fitNextPoint(whole_line, row_radio);

			//�������GROUND_JUDGE����ĵ����߶�?
			for (int i = GROUND_JUDGE; i < trackdata.trackBottomFit.size(); ++i) {

				std::vector<cv::Point2d> lines_data;
				for (int j = i - 10;j < i;++j) {
					cv::Point2d pt;
					pt.x = (float)j * row_radio + row_radio / 2;
					pt.y = trackdata.trackBottomFit[j];
					lines_data.push_back(pt);
				}
				FitLineData last_line_data = fitNextPoint(lines_data, row_radio);
				float predict = last_line_data.nextPoint;

				if (trackdata.trackBottomFit[i] == 100.0f || fabs(trackdata.trackBottomFit[i] - predict) >= 0.05f ||
					fabs(trackdata.trackBottomFit[i] - trackdata.trackBottomFit[i - 1]) >= 0.05f) {

					float whole_predict = whole_line_data.k * ((float)i * row_radio + row_radio / 2) + whole_line_data.b;
					if (fabs(trackdata.trackBottomFit[i] - whole_predict) >= 0.05f) {

						float last_point = last_line_data.k * ((float)(i - 1) * row_radio + row_radio / 2) + last_line_data.b;
						float predict_b = last_point - whole_line_data.k * ((float)(i - 1) * row_radio + row_radio / 2);
						float last_predict = whole_line_data.k * ((float)i * row_radio + row_radio / 2) + predict_b;
						if (fabs(trackdata.trackBottomFit[i] - last_predict) >= 0.05f)
							trackdata.trackBottomFit[i] = last_predict;
					}

					else
						continue;
				}
			}

			trackdata.trackBottom[0] = trackdata.trackBottomFit[0];
			for (int i = 1; i < trackdata.trackBottomFit.size(); ++i) {

				if (trackdata.trackBottomFit[i] > trackdata.trackBottom[i] || trackdata.trackBottom[i] == 100.0f)
					trackdata.trackBottom[i] = trackdata.trackBottomFit[i];
				else {
					float bottom_modify = whole_line_data.k * ((float)i * row_radio + row_radio / 2) + whole_line_data.b;
					if (fabs(trackdata.trackBottom[i] - bottom_modify) > 0.25f)
						trackdata.trackBottom[i] = trackdata.trackBottom[i - 1];
				}
			}

		}

		void LOD::create_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Grid_data_init& grid, pcl::PointXYZ min_p, pcl::PointXYZ max_p,
			float row_radio, float col_radio, TrackData trackdata) {

			for (int i = 0; i < cloud->points.size(); ++i) {

				pcl::PointXYZ pt = cloud->points[i];

				int grid_cur_row = (pt.z - min_p.z) / row_radio;
				int grid_cur_col = (pt.y - min_p.y) / col_radio;

				if (pt.x > trackdata.trackBottomFit[grid_cur_row] + 3.5f)
					continue;

				if (pt.z <= DIVIDE_DIS && pt.x - trackdata.trackTop[grid_cur_row] <= 0.03f) {   //0.0
					grid[grid_cur_row][grid_cur_col].loss_points.push_back({ pt.x,pt.y,pt.z });
					if (grid[grid_cur_row][grid_cur_col].loss_points.size() == 1)
						grid[grid_cur_row][grid_cur_col].loss_min_x = pt.x;
					else
						if (grid[grid_cur_row][grid_cur_col].loss_min_x > pt.x)
							grid[grid_cur_row][grid_cur_col].loss_min_x = pt.x;
					continue;
				}

				if (pt.z > DIVIDE_DIS && pt.x - trackdata.trackBottom[grid_cur_row] <= 0.05f) {
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

		std::vector<OB_index_data> LOD::cluster(Grid_data_init& grid, TrackData trackdata) {

			std::stack<std::pair<int, int>> extend;

			float grids_total_x = 0.0f;
			float dis_thres = 0.0f;
			OB_Size ob_size;

			std::vector<std::vector<bool>> visited(grid.grid_rows, std::vector<bool>(grid.grid_cols, 0));
			std::vector<OB_index_data> obstacle_grid;
			OB_index_data obstacle_grid_group;

			for (int i = 0; i < grid.grid_rows; ++i) {
				for (int j = 0; j < grid.grid_cols; ++j) {

					if (grid[i][j].grid_points.size() < 1 || visited[i][j] ||
						(grid[i][j].min_z <= 5.0f && grid[i][j].grid_points.size() <= 2)) {

						visited[i][j] = true;
						continue;
					}

					if (grid[i][j].min_z <= DIVIDE_DIS && grid[i][j].max_x - trackdata.trackTop[i] < 0.05f) {

						visited[i][j] = true;
						continue;
					}
					//����С��30m���������?30cm�ߵĵ���Ϊ�ϰ����?
					if (grid[i][j].min_z > DIVIDE_DIS && grid[i][j].max_z < 30.0f &&
						grid[i][j].max_x - trackdata.trackBottom[i] < 0.15f) {    //0.3

						visited[i][j] = true;
						continue;
					}
					//�������?30m���������?10cm�ߵĵ���Ϊ�������ϰ����?
					if (grid[i][j].max_z >= 30.0f && grid[i][j].max_x - trackdata.trackBottom[i] < 0.1f) {

						visited[i][j] = true;
						continue;
					}
					obstacle_grid_group.gridIndex.clear();
					obstacle_grid_group.pointsNum = 0;
					obstacle_grid_group.maxSize = 0.0f;
					obstacle_grid_group.lossNum = 0;
					grids_total_x = 0.0f;
					ob_size.max_x = grid[i][j].max_x;
					ob_size.max_y = grid[i][j].max_y;
					ob_size.max_z = grid[i][j].max_z;
					ob_size.min_x = grid[i][j].min_x;
					ob_size.min_y = grid[i][j].min_y;
					ob_size.min_z = grid[i][j].min_z;
					ob_size.total_z = grid[i][j].total_z;
					ob_size.loss_min_x = grid[i][j].loss_min_x;
					extend.push(std::make_pair(i, j));
					obstacle_grid_group.gridIndex.push_back(std::make_pair(i, j));
					grids_total_x = grid[i][j].total_x;
					obstacle_grid_group.lossNum = grid[i][j].loss_points.size();
					obstacle_grid_group.pointsNum = grid[i][j].grid_points.size();
					visited[i][j] = 1;

					search_grid(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, trackdata);

					/*if (ob_size.max_x - ob_size.min_x < 0.05)
						continue;*/

					int needNum = 0;
					needNum = 24 - 4 * int(ob_size.min_z / 10);
					if (ob_size.min_z > 50)
						needNum = 6;
					if (ob_size.min_z > 60)
						needNum = 3;
					bool isTrain = false;
					if (ob_size.max_y - ob_size.min_y > 1.5f && ob_size.max_x - ob_size.min_x > 1.0f && ob_size.max_x >= 0.7f)
						isTrain = true;
					if (isTrain) {
						if (obstacle_grid_group.pointsNum > 6) {
							obstacle_grid_group.ob_size = ob_size;
							obstacle_grid_group.ob_size.loss_min_x = ob_size.min_x;
							obstacle_grid_group.maxSize = std::max(ob_size.max_y - ob_size.min_y, ob_size.max_z - ob_size.min_z);

							obstacle_grid_group.distance_z = ob_size.total_z / obstacle_grid_group.pointsNum;
							obstacle_grid.push_back(obstacle_grid_group);
						}
					}
					else {
						if (ob_size.min_z < DETECT_DISTANCE && obstacle_grid_group.pointsNum > 2 &&       //52
							obstacle_grid_group.pointsNum + obstacle_grid_group.lossNum >= needNum) {
							if (ob_size.min_z > DIVIDE_DIS || obstacle_grid_group.pointsNum >= needNum) {

								if (ob_size.min_z <= DIVIDE_DIS && ob_size.max_x - ob_size.min_x < 0.1f)   //0.15
									continue;
								if (ob_size.min_z > DIVIDE_DIS && ob_size.max_x - ob_size.min_x < 0.25f)
									continue;

								//�����⵽��������5m�ڣ�����y���ϵĿ���С��20cm�������˳�
								if (ob_size.max_z < 10.0f && ob_size.max_y - ob_size.min_y <= 0.25f)
									continue;
								if (ob_size.max_z < 25.0f && ob_size.max_y - ob_size.min_y <= 0.12f)
									continue;

								//��⵽������?10m���ڣ���߶ȵı�׼�����С��3cm�����䰴��촦��?
								if (ob_size.min_z < DIVIDE_DIS) {
									std::vector<Point3> object;
									for (int grid_index = 0; grid_index < obstacle_grid_group.gridIndex.size(); ++grid_index) {

										int grid_i = obstacle_grid_group.gridIndex[grid_index].first;
										int grid_j = obstacle_grid_group.gridIndex[grid_index].second;
										for (int p_i = 0; p_i < grid[grid_i][grid_j].grid_points.size(); ++p_i) {

											object.push_back(grid[grid_i][grid_j].grid_points[p_i]);
										}
									}
									if (calHeightVar(object) < 0.03f)
										continue;
								}

								obstacle_grid_group.ob_size = ob_size;
								obstacle_grid_group.maxSize = std::max(ob_size.max_y - ob_size.min_y, ob_size.max_z - ob_size.min_z);

								obstacle_grid_group.distance_z = ob_size.total_z / obstacle_grid_group.pointsNum;
								obstacle_grid.push_back(obstacle_grid_group);
							}
						}
					}
				}
			}

			return obstacle_grid;
		}

		void LOD::search_grid(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
			OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, TrackData trackdata) {

			while (!extend.empty())
			{
				std::pair<int, int> cur = extend.top();
				float grid_avr_z = grid[cur.first][cur.second].total_z / grid[cur.first][cur.second].grid_points.size();
				int col_extend = grid_avr_z / 40 + 1;
				int row_extend = 1;
				/*int col_extend = grid_avr_z / 10 + 1;
				if (grid_avr_z > 40) {
					col_extend = 5;
					row_extend = 2;
				}*/
				extend.pop();
				//��
				for (int i = 0; i < row_extend; ++i)
					search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1 - i, cur.second, trackdata);
				//��
				for (int i = 0; i < row_extend; ++i)
					search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1 + i, cur.second, trackdata);
				//��
				for (int i = 0; i < col_extend; ++i)
					search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first, cur.second - 1 - i, trackdata);
				//��
				for (int i = 0; i < col_extend; ++i)
					search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first, cur.second + 1 + i, trackdata);

				if (grid[cur.first][cur.second].total_z / grid[cur.first][cur.second].grid_points.size() >= 20) {

					//����
					for (int i = 0; i < row_extend; ++i)
						for (int j = 0; j < col_extend; ++j)
							search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1 - i, cur.second - 1 - j, trackdata);
					//����
					for (int i = 0; i < row_extend; ++i)
						for (int j = 0; j < col_extend; ++j)
							search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first - 1 - i, cur.second + 1 + j, trackdata);
					//����
					for (int i = 0; i < row_extend; ++i)
						for (int j = 0; j < col_extend; ++j)
							search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1 + i, cur.second - 1 - j, trackdata);
					//����
					for (int i = 0; i < row_extend; ++i)
						for (int j = 0; j < col_extend; ++j)
							search_fork(grid, visited, extend, obstacle_grid_group, grids_total_x, ob_size, cur.first + 1 + i, cur.second + 1 + j, trackdata);
				}

			}
		}

		void LOD::search_fork(Grid_data_init& grid, std::vector<std::vector<bool>>& visited, std::stack<std::pair<int, int>>& extend,
			OB_index_data& obstacle_grid_group, float& grids_total_x, OB_Size& ob_size, int row, int col, TrackData trackdata) {

			if (row >= 0 && row < grid.grid_rows && col >= 0 && col < grid.grid_cols && visited[row][col] == 0) {

				visited[row][col] = 1;

				if (grid[row][col].min_z <= 5.0f && grid[row][col].grid_points.size() <= 2)
					return;

				if (grid[row][col].grid_points.size() == 0)
					return;

				if (grid[row][col].min_z <= DIVIDE_DIS && grid[row][col].max_x - trackdata.trackTop[row] < 0.05f)
					return;

				if (grid[row][col].min_z > DIVIDE_DIS && grid[row][col].max_z < 30.0f &&
					grid[row][col].max_x - trackdata.trackBottom[row] < 0.3f)
					return;

				if (grid[row][col].max_z >= 30.0f && grid[row][col].max_x - trackdata.trackBottom[row] < 0.1f)
					return;

				extend.push(std::make_pair(row, col));

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
				ob_size.total_z += grid[row][col].total_z;

			}
		}

		float LOD::boxOverlap(OB_Size train_box, OB_Size ob_box) {

			if (ob_box.min_y > train_box.max_y)
				return 0.0f;
			if (ob_box.max_y < train_box.min_y)
				return 0.0f;
			if (ob_box.min_x > train_box.max_x)
				return 0.0f;
			if (ob_box.max_x < train_box.min_x)
				return 0.0f;
			float colInt = std::min(train_box.max_y, ob_box.max_y) - std::max(train_box.min_y, ob_box.min_y);
			float rowInt = std::min(train_box.max_x, ob_box.max_x) - std::max(train_box.min_x, ob_box.min_x);
			float intersection = colInt * rowInt;
			float area = (ob_box.max_x - ob_box.min_x) * (ob_box.max_y - ob_box.min_y);

			return intersection / area;
		}

		void LOD::drawImage(std::vector<LidarBox> obstacle_box, std::string image_file) {

			std::vector<std::vector<cv::Point3d>> all_points;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_box(new pcl::PointCloud<pcl::PointXYZ>);
			std::vector<cv::Point3d> all_temp;
			cv::Mat cameraMatrix(3, 3, CV_32F);
			std::vector<float> distCoeff;
			cv::Mat rvec(3, 3, CV_64F), tvec(3, 1, CV_64F);

			//11�¶̽��������?
			distCoeff.push_back(-0.2283);
			distCoeff.push_back(0.1710);
			distCoeff.push_back(-0.0013);
			distCoeff.push_back(-8.2250e-06);
			distCoeff.push_back(0);

			//11�¶̽��ڲβ���
			float tempMatrix[3][3] = { { 2.1334e+03, 0, 931.1503 }, { 0, 2.1322e+03, 580.8112 }, { 0, 0, 1.0 } };

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					cameraMatrix.at<float>(i, j) = tempMatrix[i][j];
				}
			}

			double tempRvec[3][3] = { {1,0,0},
									{0,1,0},
									{0,0,1} };

			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					rvec.at<double>(i, j) = tempRvec[i][j];
			cv::Rodrigues(rvec, rvec);

			double tempTvec[3] = { 0.0,0.80,0.177 };

			for (int i = 0; i < 3; ++i)
				tvec.at<double>(i, 0) = tempTvec[i];

			for (int i = 0; i < obstacle_box.size(); ++i) {

				cloud_box->points.push_back({ obstacle_box[i].left_bottom.x, obstacle_box[i].left_bottom.y, obstacle_box[i].left_bottom.z });
				cloud_box->points.push_back({ obstacle_box[i].left_top.x, obstacle_box[i].left_top.y, obstacle_box[i].left_top.z });
				cloud_box->points.push_back({ obstacle_box[i].right_top.x, obstacle_box[i].right_top.y, obstacle_box[i].right_top.z });
				cloud_box->points.push_back({ obstacle_box[i].right_bottom.x, obstacle_box[i].right_bottom.y, obstacle_box[i].right_bottom.z });

			}

			Eigen::Matrix4d rotation, rotation_cv;
			rotation << 1, 0, 0, 0,
				0, cos(0.15 * CV_PI / 180), -sin(0.15 * CV_PI / 180), 0.000000,
				0, sin(0.15 * CV_PI / 180), cos(0.15 * CV_PI / 180), 0.000000,
				0, 0, 0, 1;

			pcl::transformPointCloud(*cloud_box, *cloud_box, rotation);

			for (int i = 0;i < cloud_box->points.size();++i) {

				pcl::PointXYZ pt = cloud_box->points[i];
				float temp = pt.x;
				pt.x = pt.y;
				pt.y = -temp;
				pt.z = pt.z + 0.2f;
				cloud_box->points[i].x = pt.x;
				cloud_box->points[i].y = pt.y;
				cloud_box->points[i].z = pt.z;

			}

			rotation_cv << 0.9999960856928221, 0.0003353891285046393, -0.002777789258835963, 0,
				-0.0003158570188030314, 0.9999752464831646, 0.007028986788848552, 0,
				0.002780077944536163, -0.007028081891000952, 0.9999714382078898, 0,
				0.000000, 0.000000, 0.000000, 1.000000;

			pcl::transformPointCloud(*cloud_box, *cloud_box, rotation_cv);

			for (int i = 0;i < cloud_box->points.size();i += 4) {

				all_temp.clear();
				all_temp.push_back(cv::Point3d(cloud_box->points[i].x, cloud_box->points[i].y, cloud_box->points[i].z));
				all_temp.push_back(cv::Point3d(cloud_box->points[i + 1].x, cloud_box->points[i + 1].y, cloud_box->points[i + 1].z));
				all_temp.push_back(cv::Point3d(cloud_box->points[i + 2].x, cloud_box->points[i + 2].y, cloud_box->points[i + 2].z));
				all_temp.push_back(cv::Point3d(cloud_box->points[i + 3].x, cloud_box->points[i + 3].y, cloud_box->points[i + 3].z));

				all_points.push_back(all_temp);

			}

			cv::Mat image = cv::imread("E:\\Cal_data\\results\\cam1\\"
				+ image_file + "_imag.png");

			for (int i = 0; i < all_points.size(); ++i) {

				std::vector<cv::Point2d> projectedPoints;
				cv::projectPoints(all_points[i], rvec, tvec, cameraMatrix, distCoeff, projectedPoints);
				projectedPoints[0].x += 7;
				projectedPoints[1].x += 7;
				projectedPoints[2].x += 7;
				projectedPoints[3].x += 7;
				projectedPoints[0].y += 5;
				projectedPoints[1].y += 5;
				projectedPoints[2].y += 5;
				projectedPoints[3].y += 5;

				cv::line(image, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 2);
				cv::line(image, projectedPoints[1], projectedPoints[2], cv::Scalar(0, 0, 255), 2);
				cv::line(image, projectedPoints[2], projectedPoints[3], cv::Scalar(0, 0, 255), 2);
				cv::line(image, projectedPoints[3], projectedPoints[0], cv::Scalar(0, 0, 255), 2);
				std::string str = "Dis: " + std::to_string(obstacle_box[i].distance_z) + "m";
				cv::putText(image, str, cv::Point(projectedPoints[0].x, projectedPoints[0].y + 15), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
				cv::putText(image, std::to_string(obstacle_box.size()), cv::Point(100, 100), cv::FONT_HERSHEY_TRIPLEX, 2.0, cv::Scalar(0, 0, 255), 5);
			}

			cv::imwrite("E:\\Cal_data\\results\\pic_result\\"
				+ image_file + "_imag.png", image);
		}

		std::vector<LidarBox> LOD::object_detection(pcl::PointCloud<pcl::PointXYZ>::Ptr& points, Eigen::Matrix4d rotation, InvasionData invasion, std::string image_file) {

			std::vector<LidarBox> obstacle_box;
			LidarBox obstacle_temp;

			if (points->points.size() == 0)
				return obstacle_box;

			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_limit(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_result(new pcl::PointCloud<pcl::PointXYZ>);

#ifdef SHOW_PCD
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr show_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
#endif

			pcl::PointXYZ min_p(1000, 1000, 1000), max_p(-1000, -1000, -1000);
			for (int i = 0; i < points->points.size(); ++i) {
				pcl::PointXYZ pt = points->points[i];
				cloud_limit->points.push_back(pt);

				if (pt.x > max_p.x)
					max_p.x = pt.x;
				if (pt.x < min_p.x)
					min_p.x = pt.x;
				if (pt.y > max_p.y)
					max_p.y = pt.y;
				if (pt.y < min_p.y)
					min_p.y = pt.y;
				if (pt.z > max_p.z)
					max_p.z = pt.z;
				if (pt.z < min_p.z)
					min_p.z = pt.z;

#ifdef SHOW_PCD
				pcl::PointXYZRGB pp;
				pp.x = pt.x;
				pp.y = pt.y;
				pp.z = pt.z;

				pp.r = 255;
				pp.g = 255;
				pp.b = 255;

				show_cloud->points.push_back(pp);
#endif


			}

			float row_radio = 0.4f;   //0.1
			float col_radio = 0.2f;   //0.05   ��ֵ����
			int grid_row = (max_p.z - min_p.z) / row_radio + 1;
			int grid_col = (max_p.y - min_p.y) / col_radio + 1;

			TrackData trackdata;
			calLowestPoint(trackdata, cloud_limit, min_p, max_p, row_radio, invasion, image_file);
			if (trackdata.trackBottom.size() == 0)
				return obstacle_box;

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
					grid[i][j].loss_min_x = 100.0f;
				}
			}

			create_grid(cloud_limit, grid, min_p, max_p, row_radio, col_radio, trackdata);
			auto obstacle_grid = cluster(grid, trackdata);

			//��������ͼ���н��޽���Ŀ�������޽��ڵ�����
			int left_num = invasion.coeff_left.size();
			int right_num = invasion.coeff_right.size();
			for (int i = 0; i < obstacle_grid.size();) {

				double dis = obstacle_grid[i].distance_z;

				double l_limit = 0;
				double r_limit = 0;

				for (int j = 0; j < left_num; ++j)
					l_limit += invasion.coeff_left[j] * std::pow(dis, left_num - 1 - j);
				for (int j = 0; j < right_num; ++j)
					r_limit += invasion.coeff_right[j] * std::pow(dis, right_num - 1 - j);

				if (obstacle_grid[i].ob_size.min_y > r_limit + INVASION_EXPAND || obstacle_grid[i].ob_size.max_y < l_limit - INVASION_EXPAND)
					obstacle_grid.erase(obstacle_grid.begin() + i);
				else
					++i;
			}

			//�����������������Ŀ��?(�������µ�)
			for (int i = 0;i < obstacle_grid.size();) {

				float lowest_height = obstacle_grid[i].ob_size.loss_min_x;
				float highest_height = obstacle_grid[i].ob_size.max_x;

				//�˳����ڴ��ڵ��µ���ɵ����������
				if (lowest_height != 100.0f) {
					double trackBottomAvr = 0.0f;
					for (int j = 0;j < obstacle_grid[i].gridIndex.size();++j) {

						trackBottomAvr += trackdata.trackBottomFit[obstacle_grid[i].gridIndex[j].first];
					}
					trackBottomAvr /= obstacle_grid[i].gridIndex.size();
					if (trackBottomAvr - lowest_height >= 0.05f) {     //�߶���ֵ
						if (obstacle_grid[i].ob_size.min_x - trackBottomAvr > 0.05f)
							lowest_height = obstacle_grid[i].ob_size.min_x;
						else
							lowest_height = trackBottomAvr;
						if (highest_height - lowest_height < 0.33f) {   //Ŀ��߶���ֵ�����ڸ���ֵ�˳�?

							obstacle_grid.erase(obstacle_grid.begin() + i);
							continue;
						}
					}
					else if (trackBottomAvr - lowest_height >= 0.0f) {

						lowest_height = (trackBottomAvr + lowest_height) / 2.0f;
					}
				}

				//�ж�Ŀ��߶��Ƿ�ﵽ30cm
				if (lowest_height == 100.0f) {
					lowest_height = obstacle_grid[i].ob_size.min_x;
					if (highest_height - obstacle_grid[i].ob_size.min_x <= 0.33f) {  //Ŀ��߶���ֵ�����ڸ���ֵ�˳�?

						obstacle_grid.erase(obstacle_grid.begin() + i);
						continue;
					}

				}
				else {
					if (highest_height - lowest_height <= 0.33f) {   //Ŀ��߶���ֵ�����ڸ���ֵ�˳�?

						obstacle_grid.erase(obstacle_grid.begin() + i);
						continue;
					}

				}

				//�˳����յ�
				float limit_height = 7.0f * lowest_height / 12.0f + 5.0f * highest_height / 12.0f;   // 2/3

				if (SUSPENDED_OBJECT && obstacle_grid[i].ob_size.min_x > obstacle_grid[i].ob_size.loss_min_x + 1.0f) {   //0.5
					obstacle_grid.erase(obstacle_grid.begin() + i);
					continue;
				}

				if (limit_height <= lowest_height + 0.35f) {
					++i;
					continue;
				}

				int high_num = 0;
				float high_lowest = highest_height;
				float low_highest = lowest_height;
				for (int j = 0; j < obstacle_grid[i].gridIndex.size(); ++j) {
					for (int k = 0; k < grid[obstacle_grid[i].gridIndex[j].first][obstacle_grid[i].gridIndex[j].second].grid_points.size(); ++k) {
						float now_height = grid[obstacle_grid[i].gridIndex[j].first][obstacle_grid[i].gridIndex[j].second].grid_points[k][0];

						if (now_height > limit_height) {
							high_num++;
							if (now_height < high_lowest)
								high_lowest = now_height;
						}
						else {

							if (now_height > low_highest)
								low_highest = now_height;
						}

					}
				}
				if (low_highest < lowest_height + 0.3f && high_num > 0 && high_num <= 5 && high_lowest - low_highest > 0.15f)
					obstacle_grid.erase(obstacle_grid.begin() + i);
				else if (SUSPENDED_OBJECT && low_highest < lowest_height + 0.3f && high_num > 5 && high_lowest - low_highest > 0.15f) {

					obstacle_grid.erase(obstacle_grid.begin() + i);
				}

				else
					++i;
			}

			int idx_train = 0;
			float max_sizey = 0;
			bool isTrain = false;

			for (int i = 0; i < obstacle_grid.size(); ++i) {
				if (obstacle_grid[i].maxSize > max_sizey) {
					idx_train = i;
					max_sizey = obstacle_grid[i].maxSize;
				}
				if (obstacle_grid[i].ob_size.max_y - obstacle_grid[i].ob_size.min_y > 1.5f &&
					obstacle_grid[i].ob_size.max_x - obstacle_grid[i].ob_size.min_x > 1.0f &&
					obstacle_grid[i].ob_size.max_x > 0.7f)
					isTrain = true;
			}

			//���г�ǰ�����г���Ϊͬһ������ϰ���ľ����滻�г�����
			float modifyDis = 0.0f;
			if (isTrain)
				modifyDis = obstacle_grid[idx_train].ob_size.min_z;
			for (int i = 0; i < obstacle_grid.size(); ++i) {

				if (isTrain && i != idx_train) {

					float glass_dis_front = obstacle_grid[i].ob_size.max_z - obstacle_grid[idx_train].ob_size.min_z;
					float glass_dis_behind = obstacle_grid[i].ob_size.min_z - obstacle_grid[idx_train].ob_size.min_z;
					if (glass_dis_front > -0.6f && glass_dis_front < 0.0f) {
						if (boxOverlap(obstacle_grid[idx_train].ob_size, obstacle_grid[i].ob_size) >= 0.3f) {
							if (modifyDis > obstacle_grid[i].ob_size.min_z)
								modifyDis = obstacle_grid[i].ob_size.min_z;

							continue;
						}

					}
					if (glass_dis_behind > 0.0f) {

						continue;
					}
					//if (glass_dis_behind > 0.0f && glass_dis_behind <= 5.0f)

				}
				cloud_result->points.clear();
				OB_Size ob_s = obstacle_grid[i].ob_size;
				if (ob_s.loss_min_x > ob_s.min_x)
					ob_s.loss_min_x = ob_s.min_x;
				cloud_result->points.push_back({ ob_s.loss_min_x, ob_s.min_y, ob_s.min_z });
				cloud_result->points.push_back({ ob_s.max_x, ob_s.min_y, ob_s.min_z });
				cloud_result->points.push_back({ ob_s.max_x, ob_s.max_y, ob_s.min_z });
				cloud_result->points.push_back({ ob_s.loss_min_x, ob_s.max_y, ob_s.min_z });

				obstacle_temp.left_bottom = Point3(cloud_result->points[0]);
				obstacle_temp.left_top = Point3(cloud_result->points[1]);
				obstacle_temp.right_top = Point3(cloud_result->points[2]);
				obstacle_temp.right_bottom = Point3(cloud_result->points[3]);

				obstacle_temp.distance_z = obstacle_grid[i].ob_size.min_z;

				obstacle_box.push_back(obstacle_temp);
			}
			//�޽����⵼�»��޷����?
			/*if (isTrain)
				obstacle_box[idx_train].distance_z = modifyDis;*/

#ifdef SHOW_PCD			
			srand((unsigned)time(NULL));
			pcl::PointXYZRGB point_view;
			for (int i = 0; i < obstacle_grid.size(); ++i) {
				int r = rand() % 255;
				int g = rand() % 255;
				int b = rand() % 255;
				r = 255;
				g = 0;
				b = 0;
				for (int j = 0; j < obstacle_grid[i].gridIndex.size(); ++j) {
					for (int k = 0; k < grid[obstacle_grid[i].gridIndex[j].first][obstacle_grid[i].gridIndex[j].second].grid_points.size(); ++k) {
						point_view.x = grid[obstacle_grid[i].gridIndex[j].first][obstacle_grid[i].gridIndex[j].second].grid_points[k][0];
						point_view.y = grid[obstacle_grid[i].gridIndex[j].first][obstacle_grid[i].gridIndex[j].second].grid_points[k][1];
						point_view.z = grid[obstacle_grid[i].gridIndex[j].first][obstacle_grid[i].gridIndex[j].second].grid_points[k][2];
						point_view.r = r;
						point_view.g = g;
						point_view.b = b;

						show_cloud->points.push_back(point_view);
					}
				}
			}
			pcl::visualization::CloudViewer viewer("Show");
			viewer.showCloud(show_cloud);
			// system("pause");
			while (!viewer.wasStopped()){ };
#endif

			delete& grid;
			return obstacle_box;
		}



		
		// fwc
		std::vector<lidar_invasion_cvbox> LOD::lidarboxTocvbox(std::vector<LidarBox> obstacle_box){
			std::vector<lidar_invasion_cvbox> v_cvbox;

			std::vector<std::vector<cv::Point3d>> all_points;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_box(new pcl::PointCloud<pcl::PointXYZ>);
			std::vector<cv::Point3d> all_temp;
			cv::Mat cameraMatrix(3, 3, CV_32F);
			std::vector<float> distCoeff;
			cv::Mat rvec(3, 3, CV_64F), tvec(3, 1, CV_64F);

			//11�¶̽��������?
			distCoeff.push_back(-0.2759f);
			distCoeff.push_back(0.4355f);
			distCoeff.push_back(0.0010f);
			distCoeff.push_back(-1.2753e-04f);
			distCoeff.push_back(0.0f);

			//11�¶̽��ڲβ���
			float tempMatrix[3][3] = { { 2.1159e+03f, 0.0f, 974.2096f }, { 0.0f, 2.1175e+03f, 619.7170f }, { 0.0f, 0.0f, 1.0f } };

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					cameraMatrix.at<float>(i, j) = tempMatrix[i][j];
				}
			}

			double tempRvec[3][3] = { {1,0,0},
									{0,1,0},
									{0,0,1} };

			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					rvec.at<double>(i, j) = tempRvec[i][j];
			cv::Rodrigues(rvec, rvec);

			double tempTvec[3] = { 0.0,0.92,0.137 };

			for (int i = 0; i < 3; ++i)
				tvec.at<double>(i, 0) = tempTvec[i];

			for (int i = 0; i < obstacle_box.size(); ++i) {
#ifdef SHOW_PCD_BOX_NUM
				std::cout << "obstacle_box " << i << ":" << std::endl;
				std::cout << obstacle_box[i].left_bottom.x << " " << obstacle_box[i].left_bottom.y << " " << obstacle_box[i].left_bottom.z << std::endl;
				std::cout << obstacle_box[i].left_top.x << " " << obstacle_box[i].left_top.y << " " << obstacle_box[i].left_top.z << std::endl;
				std::cout << obstacle_box[i].right_top.x << " " << obstacle_box[i].right_top.y << " " << obstacle_box[i].right_top.z << std::endl;
				std::cout << obstacle_box[i].right_bottom.x << " " << obstacle_box[i].right_bottom.y << " " << obstacle_box[i].right_bottom.z << std::endl;
#endif
				cloud_box->points.push_back({ obstacle_box[i].left_bottom.x, obstacle_box[i].left_bottom.y, obstacle_box[i].left_bottom.z });
				cloud_box->points.push_back({ obstacle_box[i].left_top.x, obstacle_box[i].left_top.y, obstacle_box[i].left_top.z });
				cloud_box->points.push_back({ obstacle_box[i].right_top.x, obstacle_box[i].right_top.y, obstacle_box[i].right_top.z });
				cloud_box->points.push_back({ obstacle_box[i].right_bottom.x, obstacle_box[i].right_bottom.y, obstacle_box[i].right_bottom.z });

			}

			Eigen::Matrix4d rotation, rotation_cv;
			rotation << 1, 0, 0, 0,
				0, cos(-0.15 * CV_PI / 180), -sin(-0.15 * CV_PI / 180), 0.000000,
				0, sin(-0.15 * CV_PI / 180), cos(-0.15 * CV_PI / 180), 0.000000,
				0, 0, 0, 1;

			pcl::transformPointCloud(*cloud_box, *cloud_box, rotation);

			for (int i = 0;i < cloud_box->points.size();++i) {

				pcl::PointXYZ pt = cloud_box->points[i];
				float temp = pt.x;
				pt.x = pt.y;
				pt.y = -temp;
				pt.z = pt.z + 0.1f;
				cloud_box->points[i].x = pt.x;
				cloud_box->points[i].y = pt.y;
				cloud_box->points[i].z = pt.z;

			}

			rotation_cv << 0.9999994138848853, -0.0007077764296917184, -0.0008193182600682035, 0,
				0.000688561989143834, 0.9997301624711764, -0.02321913279479644, 0,
				0.0008355311321636263, 0.02321855503430087, 0.9997300638621639, 0,
				0.000000, 0.000000, 0.000000, 1.000000;

			pcl::transformPointCloud(*cloud_box, *cloud_box, rotation_cv);

			for (int i = 0;i < cloud_box->points.size();i += 4) {

				all_temp.clear();
				all_temp.push_back(cv::Point3d(cloud_box->points[i].x, cloud_box->points[i].y, cloud_box->points[i].z));
				all_temp.push_back(cv::Point3d(cloud_box->points[i + 1].x, cloud_box->points[i + 1].y, cloud_box->points[i + 1].z));
				all_temp.push_back(cv::Point3d(cloud_box->points[i + 2].x, cloud_box->points[i + 2].y, cloud_box->points[i + 2].z));
				all_temp.push_back(cv::Point3d(cloud_box->points[i + 3].x, cloud_box->points[i + 3].y, cloud_box->points[i + 3].z));

				all_points.push_back(all_temp);

			}


			for (int i = 0; i < all_points.size(); ++i) {

				std::vector<cv::Point2d> projectedPoints;
				cv::projectPoints(all_points[i], rvec, tvec, cameraMatrix, distCoeff, projectedPoints);

				int x_min = 10000, y_min = 10000;
				int x_max = -1, y_max = -1;

#ifdef SHOW_PCD_BOX_NUM
				std::cout << "projectedPoints " << i << ":" << std::endl;
#endif

				for (int idx=0; idx < projectedPoints.size(); idx++){
#ifdef SHOW_PCD_BOX_NUM
					std::cout << projectedPoints[idx].x << " " << projectedPoints[idx].y << std::endl;
#endif
					if (projectedPoints[idx].x <=  x_min) x_min = projectedPoints[idx].x;
					if (projectedPoints[idx].x >=  x_max) x_max = projectedPoints[idx].x;
					if (projectedPoints[idx].y <=  y_min) y_min = projectedPoints[idx].y;
					if (projectedPoints[idx].y >=  y_max) y_max = projectedPoints[idx].y;
				}
				x_min += 0;
				x_max += 0;
				y_min += 0;
				y_max += 0;

				lidar_invasion_cvbox cvbox;
				cvbox.xmin = x_min;
				cvbox.xmax = x_max;
				cvbox.ymin = y_min;
				cvbox.ymax = y_max;
				cvbox.dist = obstacle_box[i].distance_z;

				v_cvbox.push_back(cvbox);
			}

			return v_cvbox;
		}


	}
} //namespace
