#include "lane_util.h"

#include "algorithm/core/util/filesystem_util.h"
#include "algorithm/core/util/opencv_util.h"
#include "algorithm/core/util/numpy_util.h"
#include "algorithm/third/cluster_util.h"

// std
#include <iostream>

// opencv
#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imwrite imdecode imshow

using namespace std;
using namespace cv;


namespace watrix {
	namespace algorithm {
		
			// 8UC1 (gray)
			cv::Mat LaneUtil::connected_component_binary_mask(
				const cv::Mat& binary_mask,
				int min_area_threshold 
			)
			{
				// binary mask = 0,1
				// 首先进行图像形态学运算
				int kernel_size=5;
				cv::Mat element = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
				cv::Mat binary;
				morphologyEx(binary_mask, binary, MORPH_CLOSE, element);

				// 进行连通域分析
				cv::Mat labels; // w*h  label = 0,1,2,3,...N-1 (0- background)       CV_32S = 4
				cv::Mat stats; // N*5  表示每个连通区域的外接矩形和面积 [x,y,w,h, area]   CV_32S = 4
				cv::Mat centroids; // N*2  (x,y)                                     CV_32S = 4
				int num_components = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

				//cv::imwrite("binary_mask.jpg",binary_mask);
				//cv::imwrite("binary.jpg",binary);

				if (false){
					std::cout<<" num_components =" << num_components << std::endl; // 8
					std::cout<<" stats hw =" << stats.cols <<","<<stats.rows << std::endl; // 8*5
					std::cout<<" stats =" << stats << std::endl;
				}

				// 排序连通域并删除过小的连通域 min_area_threshold = 200
				for(int index =1; index<num_components; index++)
				{
					if ( stats.at<int>(index, cv::CC_STAT_AREA) < min_area_threshold ) 
					{
						// check label == index
						for(int row=0;row<labels.rows;++row) 
						{
							for(int col=0;col<labels.cols;++col)
							{
								int label = labels.at<int>(row,col);
								if (label == index )
								{
									binary.at<uchar>(row, col) = 0; // mark as black
								} 
							}
						}	
					}
				}

				//cv::imwrite("binary_result.jpg",binary);

				return binary; // binary mask 0,1
			}


			void LaneUtil::get_largest_connected_component(
				const cv::Mat& binary_mask, // 0-255
				cv::Mat& out,
				component_stat_t& largest_stat
			)
			{
				// binary mask = 0,1
				// 首先进行图像形态学运算
				int kernel_size=5;
				cv::Mat element = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
				morphologyEx(binary_mask, out, MORPH_CLOSE, element);

				// 进行连通域分析
				cv::Mat labels; // w*h  label = 0,1,2,3,...N-1 (0- background)       CV_32S = 4
				cv::Mat stats; // N*5  表示每个连通区域的外接矩形和面积 [x,y,w,h, area]   CV_32S = 4
				cv::Mat centroids; // N*2  (x,y)                                     CV_32S = 4
				int num_components = connectedComponentsWithStats(out, labels, stats, centroids, 8, CV_32S);


#ifdef DEBUG_IMAGE 
				{
					cv::imwrite("xxx_0_binary_mask.jpg",binary_mask);
					cv::imwrite("xxx_1_binary.jpg",out);
				}
#endif 

#ifdef DEBUG_INFO 
				{
					std::cout<<" num_components =" << num_components << std::endl; // 8
					std::cout<<" stats hw =" << stats.cols <<","<<stats.rows << std::endl; // 8*5
					std::cout<<" stats =" << stats << std::endl;

					/*
					// stats不是按照面积排序的
					stats =[0, 0, 512, 384, 195126;
						438, 272, 28, 4, 86;
						125, 275, 295, 20, 1305;
						56, 344, 5, 25, 91]
					 */
				}
#endif

				int largest_index = 0;
				int largest_area = 0;
				for(int index =1; index<num_components; index++)
				{
					int area = stats.at<int>(index, cv::CC_STAT_AREA);
					if (largest_area < area){
						largest_area = area;
						largest_index = index;
					}
				}
				if (largest_index == 0){
					largest_area = 1; // NOTICE HERE !!! 
				} else {
					largest_area = stats.at<int>(largest_index, cv::CC_STAT_AREA);
				}
				largest_stat.x = stats.at<int>(largest_index, cv::CC_STAT_LEFT);
				largest_stat.y = stats.at<int>(largest_index, cv::CC_STAT_TOP);
				largest_stat.w = stats.at<int>(largest_index, cv::CC_STAT_WIDTH);
				largest_stat.h = stats.at<int>(largest_index, cv::CC_STAT_HEIGHT);
				largest_stat.area = largest_area;

#ifdef DEBUG_INFO 
				{
					std::cout<<" largest_index = "<< largest_index << std::endl;
					std::cout<<" largest_area = "<< largest_stat.area << std::endl;
				}
#endif				
          
				// 面积小于largest的都去掉, mark as  0
				for(int index =1; index<num_components; index++)
				{
					if ( stats.at<int>(index, cv::CC_STAT_AREA) < largest_stat.area ) 
					{
						// check label == index
						for(int row=0;row<labels.rows;++row) 
						{
							for(int col=0;col<labels.cols;++col)
							{
								int label = labels.at<int>(row,col);
								if (label == index )
								{
									out.at<uchar>(row, col) = 0; // mark as black
								} 
							}
						}	
					}
				}

#ifdef DEBUG_IMAGE
			{
				cv::imwrite("xxx_3_binary.jpg",out);
			}
#endif

			}


			// 8UC1 (gray)
			dpoints_t LaneUtil::filter_out_noise(
				const cv::Mat& binary_mask, 
				const dpoints_t& one_lane_points,
				int min_area_threshold 
			)
			{
				// binary mask = 0,1
				// 首先进行图像形态学运算
				int kernel_size=5; // default 5
				cv::Mat element = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
				cv::Mat binary;
				morphologyEx(binary_mask, binary, MORPH_CLOSE, element);

				// 进行连通域分析
				cv::Mat labels; // w*h  label = 0,1,2,3,...N-1 (0- background)       CV_32S = 4
				cv::Mat stats; // N*5  表示每个连通区域的外接矩形和面积 [x,y,w,h, area]   CV_32S = 4
				cv::Mat centroids; // N*2  (x,y)                                     CV_32S = 4
				int num_components = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

				//cv::imwrite("binary_mask.jpg",binary_mask);
				//cv::imwrite("binary.jpg",binary);

				if (false){
					std::cout<<" num_components =" << num_components << std::endl; // 8
					std::cout<<" stats hw =" << stats.cols <<","<<stats.rows << std::endl; // 8*5
					std::cout<<" stats =" << stats << std::endl;
				}
			
				dpoints_t filtered_out_lane_points;
				for(int index =0; index<num_components; index++)
				{
					if ( index>0 && stats.at<int>(index, cv::CC_STAT_AREA) > min_area_threshold ) 
					{
						// check label == index
						for(int row=0;row<labels.rows;++row) 
						{
							for(int col=0;col<labels.cols;++col)
							{
								int label = labels.at<int>(row,col);
								if (label == index )
								{
									dpoint_t point{(double)col,(double)row}; // xy
									filtered_out_lane_points.push_back(point);
								} 
							}
						}	
					}
				}
				return filtered_out_lane_points; // [x,y], [x,y],...
			}



#pragma region get grid features

			ftpoint_t LaneUtil::get_feature(const channel_mat_t& instance_mask, int y, int x)
			{
				// instance_seg = (8, 256, 1024)
				int feature_dim = instance_mask.size(); // 8
				//std::cout<<"feature_dim = "<<feature_dim<<std::endl;
				ftpoint_t feature; // 8-tuple feature
				feature.resize(feature_dim);

				for(int f=0; f< feature_dim; ++f)
				{
					feature[f] = instance_mask[f].at<float>(y,x);
				}
				return feature;
			}
			
			/*
			grid_id 0-310
				feature = grid_features[grid_id] 　===> cluster_points(4), cluster_ids(310)
				points_in_grid = grid_points[grid_id]
				cluster_id = cluster_ids[grid_id]
			*/
			void LaneUtil::get_grid_features_and_points(
				const cv::Mat& binary_mask, 
				const channel_mat_t& instance_mask,
				const int grid_size, // default = 8
				std::vector<ftpoint_t>& grid_features,
				std::vector<dpoints_t>& grid_points
			)
			{
				// binary_mask + instance_mask
				// # binary_seg = (256, 1024) v=[0,1];   instance_seg = (8, 256, 1024)

				//int height = binary_mask.rows;
				//int width = binary_mask.cols;

				std::vector<int> vy,vx;
				NumpyUtil::np_where_eq(binary_mask, 1, vy, vx); // find white positions
				
				// y/8, x/8 ===>    y/8 * 1000 + x/8   max range 1024-1024 ===>  1023-1023
				const int factor = 10000;
				std::vector<int> vy_8 = NumpyUtil::devide_by(vy, grid_size); // 0-300
				std::vector<int> vy_10000 = NumpyUtil::multipy_by(vy_8, factor); 
				std::vector<int> vx_8 = NumpyUtil::devide_by(vx, grid_size); // 0-300

				std::vector<int> yxgrid = NumpyUtil::add(vy_10000,vx_8); // 001-050 300-200, 
				std::vector<int> vindex = NumpyUtil::np_unique_return_index(yxgrid);

#ifdef DEBUG_INFO
				std::cout<<"vy.size()="<< vy.size()<<std::endl; // 9274
				std::cout<<"vindex.size()="<< vindex.size()<<std::endl; // 310
#endif 

				grid_features.clear(); 
				grid_points.clear();
				grid_features.resize(vindex.size()); // 310 of feature-8
				grid_points.resize(vindex.size()); // 310 of grid points
				
				for(int i=0; i< vindex.size();++i)
				{
					// image[y,x] = 1
					int index = vindex[i];
					int y = vy[index];
					int x = vx[index];
					//printf(" y = %d, x = %d \n", y, x);
					ftpoint_t feature = get_feature(instance_mask, y, x);
					grid_features[i] = feature;

					// # get all y,x within same grid
					dpoints_t grid_pts; 

					//m_idx = np.argwhere(yxgrid==yxgrid[0][i]) # yxgrid[y,x] = 1+50*j
					std::vector<int> midx = NumpyUtil::np_argwhere_eq(yxgrid, yxgrid[index]);
					//std::cout<<"midx.size()="<< midx.size() << std::endl;
					for(auto& k: midx)
					{
						int y = vy[k]; 
						int x = vx[k]; 
						dpoint_t point;
						point.push_back(x);
						point.push_back(y);
						grid_pts.push_back(point);
					}
					grid_points[i] = grid_pts;
				}
			}
#pragma endregion


#pragma region get clustered lane points

			void LaneUtil::get_clustered_lane_points_from_features(
				const LaneInvasionConfig& config,
				const cv::Mat& binary_mask, 
				const channel_mat_t& instance_mask,
				std::vector<dpoints_t>& v_src_lane_points
			)
			{
				// (1) get grid features and grid points
				std::vector<ftpoint_t> grid_features;
				std::vector<dpoints_t> grid_points;
				get_grid_features_and_points(
					binary_mask, instance_mask, config.grid_size, grid_features, grid_points
				);
				
				// (2) get clustered points and cluster ids by clustering [grid_features]
				mean_shift_result_t result;
				result = ClusterUtil::cluster(grid_features, config);
				
				// (3) get clustered lane points
				// (3.1) keep some clustered ids
				int num_of_clusters = result.cluster_centers.size();
				std::vector<int>& cluster_ids = result.cluster_ids;

#ifdef DEBUG_INFO
				std::cout<<"cluster_centers.size() = "<< result.cluster_centers.size() << std::endl;
				std::cout<<"cluster_ids.size() = "<< result.cluster_ids.size() << std::endl;
#endif 

				std::vector<bool> v_keep_cluster_id;// false by default
				v_keep_cluster_id.resize(num_of_clusters);

				std::vector<int> v_cluster_grid_count;
				for(int i=0; i< num_of_clusters; ++i)
				{
					std::vector<int> grid_idx = NumpyUtil::np_argwhere_eq(cluster_ids, i);
					int grid_count_in_cluster = grid_idx.size();

#ifdef DEBUG_INFO
					printf("cluster %d, grid count = %d \n", i, grid_count_in_cluster);
#endif
					if (grid_count_in_cluster >= config.min_grid_count_in_cluster){
						v_keep_cluster_id[i] = true;
					} else {
						v_keep_cluster_id[i] = false;
					}
					v_cluster_grid_count.push_back(grid_count_in_cluster);
				}
				/*
				cluster 0, grid count = 74 
				cluster 1, grid count = 74 
				cluster 2, grid count = 94 
				cluster 3, grid count = 68 
				*/
			
				// (3.2) fill lane with all grid points
				int lane_id = 0;
				for(int i=0; i< num_of_clusters; ++i){
					if (v_keep_cluster_id[i]){
						std::vector<int> grid_idx = NumpyUtil::np_argwhere_eq(cluster_ids, i); // 0-310
						dpoints_t one_lane_points;

						for(auto& grid_id: grid_idx){
							dpoints_t&  points_in_grid = grid_points[grid_id];

							for(auto& pointxy: points_in_grid){
								one_lane_points.push_back(pointxy);
							}
						}

						// filter out lane noise
						if (config.filter_out_lane_noise){
							dpoints_t filtered_out_lane_points = filter_out_lane_nosie(
								binary_mask, 
								config.min_area_threshold,
								one_lane_points, 
								lane_id
							);
							lane_id ++;

							if (filtered_out_lane_points.size()>config.min_lane_pts){
								v_src_lane_points.push_back(filtered_out_lane_points);
							}
						} else {
							v_src_lane_points.push_back(one_lane_points);
						}
						
					}
				}

#ifdef DEBUG_INFO
				std::cout<<"v_src_lane_points.size()="<< v_src_lane_points.size()<<std::endl; // 4
#endif
			}


			void LaneUtil::get_clustered_lane_points_from_left_right(
				const LaneInvasionConfig& config,
				const cv::Mat& binary_mask,   // [128,480] v=[0,1]  surface 
				const channel_mat_t& instance_mask, // [2, 128,480]  v=[0,1] left/right lane points
				std::vector<dpoints_t>& v_src_lane_points
			)
			{
				//std::cout<<" __pt_simple_get_clustered_lane_points \n";
				int size = instance_mask.size();
				const int PT_SIMPLE_LANE_COUNT = 2;
				assert(size == PT_SIMPLE_LANE_COUNT);// left and right

				for(int i=0; i< size; i++){
					dpoints_t one_lane_points;
					const cv::Mat& lane_image = instance_mask[i];
					for(int r=0; r<lane_image.rows;r++){
						for(int c=0; c<lane_image.cols;c++){
							if (lane_image.at<uchar>(r,c) == 1) {
								dpoint_t pointxy;
								pointxy.push_back(c); // x
								pointxy.push_back(r); // y
								one_lane_points.push_back(pointxy);
							}
						}
					}
					v_src_lane_points.push_back(one_lane_points);
				}
			}

#pragma endregion


#pragma region filter lane noise
			cv::Mat LaneUtil::get_lane_binary_image(
				const cv::Size& size, 
				const dpoints_t& one_lane_points
			)
			{
				//cv::Mat id_mask_image(size.height, size.width, CV_8UC3, cv::Scalar(0,0,0));
				cv::Mat id_mask_image(size.height, size.width, CV_8UC1, cv::Scalar(0));
				for(auto& pointxy: one_lane_points){
					int x = (int)pointxy[0];
					int y = (int)pointxy[1];
					// (row,col) <===> (y,x)
					//printf(" x = %d, y = %d \n", x, y);
					//id_mask_image.at<cv::Vec3b>(y, x) = cv::Vec3b(0,255,255);
					id_mask_image.at<uchar>(y, x) = 255; 
				}
				return id_mask_image;
			}

			dpoints_t LaneUtil::filter_out_lane_nosie(
				const cv::Mat& binary_mask, 
				int min_area_threshold,
				const dpoints_t& one_lane_points,
				unsigned int lane_id
			)
			{
				/*
				# one_lane_points in binary_mask (256,1024)
        		# [  [165, 23], [166, 23], [167, 23], [172, 22], [173, 22] ]
				*/
				cv::Mat id_mask_image = get_lane_binary_image(
					binary_mask.size(), one_lane_points
				);

				dpoints_t filtered_out_lane_points = filter_out_noise(
					id_mask_image, one_lane_points, min_area_threshold
				);

				cv::Mat filtered_id_mask_image = get_lane_binary_image(
					binary_mask.size(), filtered_out_lane_points
				);

				if(1){
					std::string filepath = "./" + std::to_string(lane_id)+"_a.jpg";
					cv::imwrite(filepath, id_mask_image);

					std::string filepath2 = "./" + std::to_string(lane_id)+"_b.jpg";
					cv::imwrite(filepath2, filtered_id_mask_image);
				}
				
#ifdef DEBUG_INFO
				printf("[FILTER LANE] lane_id %d, %d  pts ---> %d  pts\n", 
					lane_id, (int)one_lane_points.size(), (int)filtered_out_lane_points.size()
				);
#endif
				return filtered_out_lane_points;
			}
#pragma endregion


			cv::Mat LaneUtil::get_largest_lane_mask(
				const channel_mat_t& instance_mask
			)
			{
				std::vector<cv::Mat> v_largest_lane;
				std::vector<int> v_maxarea;
				for(auto& lane: instance_mask){ // 3 lanes
					cv::Mat lane_largest;
					component_stat_t largest_stat;
					LaneUtil::get_largest_connected_component(lane, lane_largest, largest_stat);
					v_largest_lane.push_back(lane_largest);
					v_maxarea.push_back(largest_stat.area);
				}

				int max_index = NumpyUtil::max_index(v_maxarea);
				cv::Mat& largest_lane = v_largest_lane[max_index];

#ifdef DEBUG_IMAGE
				{
					cv::imwrite("5_largest.png", largest_lane);
				}
#endif

				return largest_lane;
			}


			cvpoints_t LaneUtil::get_lane_cvpoints(
				const cv::Mat& lane_binary
			)
			{
				// only 0  and 255, return all 255 points(x,y)
				cvpoints_t lane_cvpoints;

				for(int row=0;row<lane_binary.rows;++row)  // y 
				{
					for(int col=0;col<lane_binary.cols;++col) // x 
					{
						unsigned char label = lane_binary.at<uchar>(row,col);
						if (label == 255 )
						{
							cvpoint_t point;
							point.x = col;
							point.y = row; // xy
							lane_cvpoints.push_back(point);
						} 
					}
				}	
				return lane_cvpoints;
			}


			int LaneUtil::get_average_x(
				const cv::Mat& lane_binary
			)
			{
				// only 0  and 255, return all 255 points(x,y)
				cvpoints_t lane_cvpoints;

				float avg_x = 0;
				int count = 0;

				for(int row=0;row<lane_binary.rows;++row)  // y 
				{
					for(int col=0;col<lane_binary.cols;++col) // x 
					{
						unsigned char label = lane_binary.at<uchar>(row,col);
						if (label == 255 )
						{
							cvpoint_t point;
							point.x = col;
							point.y = row; // xy

							avg_x += point.x;
							count ++;
						} 
					}
				}
        if (count==0){avg_x = INT_MAX;}	
				else{avg_x /= (count*1.0);}

				return (int)avg_x; 
			}
			


	}
}// end namespace
