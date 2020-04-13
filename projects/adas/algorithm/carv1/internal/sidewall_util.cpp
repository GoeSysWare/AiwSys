#include "sidewall_util.h"

#include "algorithm/core/util/display_util.h"
#include "algorithm/core/util/opencv_util.h"
#include "algorithm/core/util/gpu_util.h"

#include "distortion_fixer.h"

// for opencv orb extracttor
#include <opencv2/core.hpp> // Mat
#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imdecode imshow
#include <opencv2/features2d.hpp> // KeyPoint orb
#include <opencv2/calib3d.hpp> //findHomography

#include <glog/logging.h>

#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

#define PI 3.1415926

namespace watrix {
	namespace algorithm {
		namespace internal {

			SidewallType::sidewall_param_t SidewallUtil::sidewall_param;

			std::stringstream SidewallUtil::ss;

			void SidewallUtil::init_sidewall_param(
				const SidewallType::sidewall_param_t& param
			)
			{
				sidewall_param = param;
			}

			std::vector<Point2f> SidewallUtil::keypoints_to_points2f(keypoints_t keypoints)
			{
				std::vector<Point2f> res;
				for (unsigned i = 0; i < keypoints.size(); i++) {
					res.push_back(keypoints[i].pt);
				}
				return res;
			}

			int SidewallUtil::detect_and_compute(
				const cv::Mat& image,
				keypoints_t& keypoints,
				cv::Mat& descriptor
			)
			{
				/*
				FEATURE_ORB,
				FEATURE_SIFT,
				FEATURE_SURF, //surf same as sift
				*/
				/*
				orb:  descriptor.type() == CV_8UC1   0
				sift/surf: descriptor.type() == CV_32F    5
				*/
				//cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
				cv::Ptr<cv::ORB> orb = cv::ORB::create();
				orb->detect(image, keypoints);
				orb->compute(image, keypoints, descriptor);

				return 0;
			}
			
			void SidewallUtil::concat_images(
				const std::vector<cv::Mat>& images_history,
				const std::vector<keypoints_t> &keypoints_history,
				const std::vector<cv::Mat> & descriptors_history,
				cv::Mat& image_scene,
				keypoints_t& keypoint_scene,
				cv::Mat& descriptor_scene
			)
			{
				// concat image, KeyPoint and Descriptor from N history image
				int y_shift = 0;

				image_scene = images_history[0];
				descriptor_scene = descriptors_history[0];
				keypoint_scene = keypoints_history[0];

#ifdef RELEASE_DEBUG
				//LOG(INFO)<<"[API] " << descriptors_history[0].size().width << "," << descriptors_history[0].size().height << std::endl;
				//LOG(INFO)<<"[API] descriptor type = " << descriptors_history[0].type() << std::endl;
#endif // shared_DEBUG

				for (size_t i = 1; i < images_history.size(); i++) {

#ifdef RELEASE_DEBUG
					//LOG(INFO)<<"[API] " << descriptors_history[i] .size().width <<"," << descriptors_history[i].size().height <<std::endl;
					//LOG(INFO)<<"[API] descriptor type = "<< descriptors_history[i].type() << std::endl;
#endif // shared_DEBUG

					vconcat(image_scene, images_history[i], image_scene);
					vconcat(descriptor_scene, descriptors_history[i], descriptor_scene);
					y_shift += images_history[i].rows; // 1000,2000,...

					keypoints_t keypoints_i = keypoints_history[i];

					for (size_t k = 0; k < keypoints_i.size(); k++) {
						KeyPoint tmp = keypoints_i[k];
						tmp.pt.y += y_shift; // shift KeyPoint y  by 1000*i
						keypoint_scene.push_back(tmp);
					}
				}

#ifdef RELEASE_DEBUG
				LOG(INFO)<<"[API] FINAL descriptor_scene=" << descriptor_scene.size().width << "," << descriptor_scene.size().height << std::endl;
				ss << "[API] FINAL descriptor_scene=" << descriptor_scene.size().width << "," << descriptor_scene.size().height << std::endl;
#endif // shared_DEBUG
			}

			void SidewallUtil::match_and_filter(
				const cv::Mat& descriptor_object, 
				const cv::Mat& descriptor_scene,
				const keypoints_t& keypoint_object, 
				const keypoints_t& keypoint_scene, 
				const float nn_match_ratio,
				keypoints_t& matched1, 
				keypoints_t& matched2
			)
			{
				// roi vs history
				//Matching object and scene descriptor vectors using knn matcher and filter out good matches
				std::vector<std::vector<DMatch> > matches;
				/*
				FlannBasedMatcher( const Ptr<flann::IndexParams>& indexParams=new flann::KDTreeIndexParams(),
				const Ptr<flann::SearchParams>& searchParams=new flann::SearchParams() );

				By default FlannBasedMatcher use `KDTreeIndexParams`.

				https://blog.csdn.net/u012812963/article/details/52998845
				1.对于params的取值为 AutotunedIndexParams、LinearIndexParams、KDTreeIndexParams时需要使用float型的descriptor Mat

				2.当param为LshIndexParams时，使用的是FLANN_DIST_HAMMING,其定义为
				typedef ::cvflann::Hamming<uchar> HammingDistance;
				features是uchar的descriptor Mat.

				https://stackoverflow.com/questions/29694490/flann-error-in-opencv-3?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
				in order to use FlannBasedMatcher you need to convert your CV_8UC1 descriptors to CV_32F:

				if(descriptors_1.type()!=CV_32F) {
				descriptors_1.convertTo(descriptors_1, CV_32F);
				}
				*/

				Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
				matcher->knnMatch(descriptor_object, descriptor_scene, matches, 2); // uchar descriptor Mat

				/*
				if (FEATURE_TYPE::FEATURE_ORB == FT) {
					Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
					matcher->knnMatch(descriptor_object, descriptor_scene, matches, 2); // uchar descriptor Mat
				}
				 else if (FEATURE_TYPE::FEATURE_SIFT == FT || FEATURE_TYPE::FEATURE_SURF == FT) {
					 FlannBasedMatcher matcher;
					 matcher.knnMatch(descriptor_object, descriptor_scene, matches, 2); // float descriptor Mat
				 }
				 else {
					LOG(INFO)<<"[API] unknown FEATURE_TYPE" << std::endl;
					 return;
				 }*/

				 // filter out good matches by distance
				for (unsigned i = 0; i < matches.size(); i++)
				{
					std::vector<DMatch>& one_match = matches[i]; // 2
					DMatch& m = one_match[0];
					DMatch& n = one_match[1];

					// filter good matches by distance
					if (m.distance < nn_match_ratio * n.distance)
					{
						matched1.push_back(keypoint_object[m.queryIdx]);
						matched2.push_back(keypoint_scene[m.trainIdx]);
					}
				}

#ifdef RELEASE_DEBUG
				LOG(INFO)<<"[API] match size " << matches.size() << std::endl; // 500
				LOG(INFO)<<"[API] [good] match1 size " << matched1.size() << std::endl; // 135
				LOG(INFO)<<"[API] [good] match2 size " << matched1.size() << std::endl;// 135

				ss << "[API] match size " << matches.size() << std::endl; // 500
				ss << "[API] [good] match1 size " << matched1.size() << std::endl; // 135
				ss << "[API] [good] match2 size " << matched1.size() << std::endl;// 135
#endif // shared_DEBUG
			}

			bool SidewallUtil::find_homography(
				const keypoints_t& matched1, 
				const keypoints_t& matched2,
				cv::Mat& homography, 
				cv::Mat& inlier_mask
			)
			{
				// Find homography, we need at least 4 keypoints
				if (matched1.size() < sidewall_param.min_good_match_size 
					|| matched2.size() < sidewall_param.min_good_match_size)
				{
					LOG(INFO) << "[SIDEWALL-API FAILED]  find_homography good matched points = "<<matched1.size()<<" <="<< sidewall_param.min_good_match_size<< "\n";
				    ss << "[SIDEWALL-API FAILED]  find_homography good matched points = " << matched1.size() << " <=" << sidewall_param.min_good_match_size << "\n";
					return false;
				}

				std::vector<Point2f> src_pts = keypoints_to_points2f(matched1);
				std::vector<Point2f> dst_pts = keypoints_to_points2f(matched2);

				const double ransac_thresh = 5.0f; // RANSAC inlier threshold

				homography = cv::findHomography(src_pts, dst_pts, RANSAC, ransac_thresh, inlier_mask);
				if (homography.empty())
				{
					LOG(INFO) << "[SIDEWALL-API FAILED] cannot findHomography" << std::endl;
					ss << "[SIDEWALL-API FAILED] cannot findHomography" << std::endl;
					return false;
				}
				else
				{
					return true;
				}
			}

			void SidewallUtil::get_average_move(
				const keypoints_t& matched1, 
				const keypoints_t& matched2,
				const cv::Mat& inlier_mask,
				int& count, 
				float& avg_x_move, 
				float& avg_y_move
			)
			{
				// Calculate x,y average move pixels
				count = 0;
				for (int i = 0; i < matched1.size(); i++)
				{ // 135
					if (inlier_mask.at<uchar>(i))
					{  // 119
						count +=1;
						avg_x_move = avg_x_move + (matched2[i].pt.x - matched1[i].pt.x);
						avg_y_move = avg_y_move + (matched2[i].pt.y - matched1[i].pt.y);
					}
				}
				avg_x_move = avg_x_move / (count*1.0f);
				avg_y_move = avg_y_move / (count*1.0f); // count = 119

#ifdef RELEASE_DEBUG
				LOG(INFO)<<"[API] count: " << count << " avg_x_move: " << avg_x_move << " avg_y_move " << avg_y_move << std::endl;
				ss << "[API] count: " << count << " avg_x_move: " << avg_x_move << " avg_y_move " << avg_y_move << std::endl;
				// count: 119 avg_x_move: -0.830108 avg_y_move 5647.89
#endif // shared_DEBUG
			}

			void SidewallUtil::select_best_roi(
				const cv::Mat& image_scene, const cv::Mat& image_object, uint32_t y,
				uint32_t& best_y, cv::Mat& best_roi
			)
			{
#ifdef RELEASE_DEBUG
				//LOG(INFO)<<"[API] [notice] we skip **get_mssim_diff** to get best_roi" << std::endl;
				//LOG(INFO)<<"[API] image_scene size = " << image_scene.size() << std::endl;
				//LOG(INFO)<<"[API] image_object size = " << image_object.size() << std::endl;
#endif // shared_DEBUG

				best_y = y;
				cv::Rect best_rect(0, best_y, image_object.cols, image_object.rows);
				//LOG(INFO)<<"[API] best_rect size = " << best_rect.x <<","<< best_rect.y << std::endl;

				best_roi = image_scene(best_rect);
				//cv::imwrite("image/best_roi.jpg", best_roi);
				//cv::imshow("best_roi", best_roi);
				//cv::waitKey(0);
			}

			void  SidewallUtil::get_image_index_and_offset(
				const std::vector<cv::Mat>& images_history, uint32_t best_y,
				uint32_t& image_index, uint32_t& y_offset
			)
			{
				// Get history image index and pixel offset for best y
				/*
				0-1024-2048-3072-**3599**-4096, best_y= 3599 ===> return image_index = 3, y_offset = 527

				best_y
				0  ===>0,0
				10 ===>0,10
				1024===>1,0
				1025===>1,1
				3599===>3,527
				*/

#ifdef RELEASE_DEBUG
				LOG(INFO)<<"[API] get_image_index_and_offset for " << best_y << std::endl;
				LOG(INFO)<<"[API] images_history.size() " << images_history.size() << std::endl;

				ss << "[API] get_image_index_and_offset for " << best_y << std::endl;
				ss << "[API] images_history.size() " << images_history.size() << std::endl;
#endif // shared_DEBUG

				image_index = 0;
				y_offset = 0;
				for (size_t i = 0; i < images_history.size(); i++)
				{
					// must use signed int value
					if ((int)best_y - images_history[i].rows >= 0)
					{
						best_y -= images_history[i].rows;
					}
					else
					{
						image_index = (uint32_t)i;
						y_offset = best_y;
						return;
					}
				}
			}

			void SidewallUtil::match_and_filter(
				const bool enable_gpu,
				const cv::Mat& descriptor_object,
				const cv::Mat& descriptor_scene,
				const keypoints_t& keypoint_object,
				const keypoints_t& keypoint_scene,
				const float nn_match_ratio,
				keypoints_t& matched1,
				keypoints_t& matched2
			)
			{
				if (enable_gpu) {
#ifdef ENABLE_OPENCV_CUDA
					GpuUtil::gpu_match_and_filter(
						descriptor_object, descriptor_scene,
						keypoint_object, keypoint_scene, nn_match_ratio,
						matched1, matched2
					);
#endif // ENABLE_OPENCV_CUDA
				}
				else {
					match_and_filter(
						descriptor_object, descriptor_scene,
						keypoint_object, keypoint_scene, nn_match_ratio,
						matched1, matched2
					);
				}
			}

			int SidewallUtil::detect_and_compute(
				const bool enable_gpu,
				const cv::Mat& image,
				keypoints_t& keypoints,
				cv::Mat& descriptor
			)
			{
				if (enable_gpu) {
#ifdef ENABLE_OPENCV_CUDA
					return GpuUtil::gpu_detect_and_compute(image, keypoints, descriptor);
#endif // ENABLE_OPENCV_CUDA
					return 0;
				}
				else {
					return detect_and_compute(image, keypoints, descriptor);
				}
			}

			bool SidewallUtil::sidewall_match(
				const bool enable_gpu,
				const cv::Mat& image_object,
				const keypoints_t keypoint_object,
				const cv::Mat descriptor_object,
				const std::vector<cv::Mat>& images_history,
				const std::vector<keypoints_t>& keypoints_history,
				const std::vector<cv::Mat>& descriptors_history,
				cv::Mat& best_roi,
				uint32_t& image_index,
				uint32_t& y_offset
			)
			{
				// >=3 history images 
				CHECK_GE(images_history.size(), sidewall_param.min_history_size) << "images_history must >="<< sidewall_param.min_history_size;
				CHECK_GE(keypoints_history.size(), sidewall_param.min_history_size) << "keypoints_history must >=" << sidewall_param.min_history_size;
				CHECK_GE(descriptors_history.size(), sidewall_param.min_history_size) << "descriptors_history must >=" << sidewall_param.min_history_size;

				ss.clear();//

				//Step 1:  concat image, KeyPoint and Descriptor from N history image
				cv::Mat image_scene;
				keypoints_t keypoint_scene;
				cv::Mat descriptor_scene;

				concat_images(
					images_history, 
					keypoints_history, 
					descriptors_history,
					image_scene,
					keypoint_scene, 
					descriptor_scene
				);

#ifdef shared_DEBUG
				cv::imwrite("sidewall/image_scene.jpg", image_scene);
				cv::imwrite("sidewall/image_object.jpg", image_object);
#endif // shared_DEBUG

				//Step 2: Detect the keypoints and extract descriptors of current image by using ORB
				// image_object,keypoint_object,descriptor_object

				// Step 3: Matching object and scene descriptor vectors using knn matcher and filter out good matches
				keypoints_t matched1, matched2;
				match_and_filter(
					enable_gpu,
					descriptor_object, 
					descriptor_scene,
					keypoint_object, 
					keypoint_scene, 
					sidewall_param.min_nn_match_ratio, // Nearest-neighbour matching ratio
					matched1, 
					matched2
				);

				// Step 4: Find Homography with src and dst pts.
				cv::Mat homography, inlier_mask;
				bool success = find_homography(matched1, matched2, homography, inlier_mask);
				if (!success) {
					return false;
				}

				// Step 5: Transform four corners to get new box and filter out by abs(x) value.
				std::vector<Point2f> bb;
				float max_x = (image_object.cols - 1)*1.0f;
				float max_y = (image_object.rows - 1)*1.0f;
				bb.push_back(cv::Point2f(0, 0));
				bb.push_back(cv::Point2f(max_x, 0));
				bb.push_back(cv::Point2f(max_x, max_y));
				bb.push_back(cv::Point2f(0, max_y));

				std::vector<Point2f> new_bb;
				cv::perspectiveTransform(bb, new_bb, homography);

				//LOG(INFO)<<"[API] -------------[src]perspectiveTransform-------------------" << std::endl;
				for (size_t i = 0; i < bb.size(); i++) {
					//std::cout << bb[i].x << " " << bb[i].y << std::endl;
				}
				//LOG(INFO)<<"[API] -------------homography-------------------" << std::endl;
				//std::cout << homography << std::endl;
				//LOG(INFO)<<"[API] -------------[dst]perspectiveTransform-------------------" << std::endl;
				for (size_t i = 0; i < new_bb.size(); i++) {
					//std::cout << new_bb[i].x << " " << new_bb[i].y << std::endl;
				}
				/*
				0 0
				2047 0
				2047 1023
				0 1023

				=================>

				-2.59238 5647.38
				2047.31 5648.96
				2045.15 6670.13
				0.044066 6669.05
				*/
				float y_min = std::min({ new_bb[0].y, new_bb[1].y, new_bb[2].y, new_bb[3].y }); // 5647
				float y_max = std::max({ new_bb[0].y, new_bb[1].y, new_bb[2].y, new_bb[3].y }); // 5647

				// filter1
				if (y_max > image_scene.size().height) {
					LOG(INFO) << "[SIDEWALL-API FAILED] match failed y_max=" << y_max << " > height=" << image_scene.size().height << std::endl;

					ss << "[SIDEWALL-API FAILED] match failed y_max=" << y_max << " > height=" << image_scene.size().height << std::endl;
					return false;
				}
				
				//float td = (float)tan(angle * PI / 180.0);
				double t1 = abs(new_bb[1].y - new_bb[0].y) / abs(new_bb[1].x - new_bb[0].x);
				double t2 = abs(new_bb[2].x - new_bb[1].x) / abs(new_bb[2].y - new_bb[1].y);
				double t3 = abs(new_bb[3].y - new_bb[2].y) / abs(new_bb[3].x - new_bb[2].x);
				double t4 = abs(new_bb[0].x - new_bb[3].x) / abs(new_bb[0].y - new_bb[3].y);

				double a1 = atan(t1) * 180.0 / PI;
				double a2 = atan(t2) * 180.0 / PI;
				double a3 = atan(t3) * 180.0 / PI;
				double a4 = atan(t4) * 180.0 / PI;

#ifdef RELEASE_DEBUG
				LOG(INFO)<<"[API] ***[ANGLE]*** = " << a1 << "," << a2 << "," << a3 << "," << a4 << std::endl;

				ss << "[API] ***[ANGLE]*** = " << a1 << "," << a2 << "," << a3 << "," << a4 << std::endl;
#endif // shared_DEBUG

				// filter2
				if (abs(a1) > sidewall_param.max_corner_angle ||
					abs(a2) > sidewall_param.max_corner_angle ||
					abs(a3) > sidewall_param.max_corner_angle ||
					abs(a4) > sidewall_param.max_corner_angle
					)
				{
					LOG(INFO) << "[SIDEWALL-API FAILED]  match failed angle > " << sidewall_param.max_corner_angle << " degree" << std::endl;
					
					ss << "[SIDEWALL-API FAILED]  match failed angle > " << sidewall_param.max_corner_angle << " degree" << std::endl;
					return false;
				}

				// Step 6: Calculate x,y average move pixels
				int count = 0;
				float avg_x_move = 0, avg_y_move = 0;
				get_average_move(matched1, matched2, inlier_mask, count, avg_x_move, avg_y_move);

				// fix:
				// avg_y_move = -0.00337414<0 ===>Match Success

				// filter3
				if (avg_y_move < 0) {
					if ( abs(avg_y_move)< sidewall_param.y_move_negative_max_offset ){
						avg_y_move = 0; // force to match success
					}
					else {
						LOG(INFO) << "[SIDEWALL-API FAILED]  asb(avg_y_move) = " << avg_y_move << std::endl;
						ss << "[SIDEWALL-API FAILED]  asb(avg_y_move) = " << avg_y_move << std::endl;
						return false;
					}
				}

				// filter4
				if (count < sidewall_param.y_move_min_valid_pixel_count) {
					//console_write_lock.lock();
					LOG(INFO) << "[SIDEWALL-API FAILED]  filter by count,find homography,but count=" << count << std::endl;
					ss << "[SIDEWALL-API FAILED]  filter by count,find homography,but count=" << count << std::endl;
					//console_write_lock.unlock();
					return false;
				}

				// Step 7: Select best roi by moveing one pixel up and down at y direction.
				uint32_t y = (uint32_t)round(avg_y_move); // 使用平均移动距离avg_y_move，而不是y_min（可能和最佳差距10个像素以上） 
				uint32_t best_y;
				select_best_roi(image_scene, image_object, y, best_y, best_roi);

#ifdef RELEASE_DEBUG
				LOG(INFO)<<"[API] y_min=" << y_min << " ,best_y=" << best_y << " ,best_y-y_min=" << best_y - y_min << std::endl;
#endif // shared_DEBUG

				// Step 8: Get history image index and pixel offset for best y.
				get_image_index_and_offset(images_history, best_y, image_index, y_offset);

#ifdef RELEASE_DEBUG
				LOG(INFO)<<"[API] image_index #" << image_index << " y_offset= " << y_offset << std::endl;

				ss << "[API] image_index #" << image_index << " y_offset= " << y_offset << std::endl;
#endif // shared_DEBUG

				return true;
			}

			void SidewallUtil::sidewall_fix(
				const cv::Mat& roi,
				const cv::Mat& image,
				cv::Mat& result_image
			)
			{
				DistortionFixer fixer = DistortionFixer(roi, image);
				fixer.result.copyTo(result_image);

#ifdef shared_DEBUG
				cv::imwrite("sidewall/fixed_1_image.jpg", image);
				cv::imwrite("sidewall/fixed_2_roi.jpg", roi);
				cv::imwrite("sidewall/fixed_3_fixed.jpg", result_image);
#endif // shared_DEBUG
			}

			void SidewallUtil::sidewall_fix(
				const std::vector<cv::Mat>& v_roi,
				const std::vector<cv::Mat>& v_image,
				std::vector<cv::Mat>& v_result_image
			)
			{
				for (size_t i = 0; i < v_roi.size(); i++)
				{
					cv::Mat result_image;
					SidewallUtil::sidewall_fix(v_roi[i], v_image[i], result_image);
					v_result_image.push_back(result_image);
				}
			}

		}
	}
}// end namespace