#include "gpu_util.h"

#include "display_util.h"

#include <iostream>
#include <map>

// glog
#include <glog/logging.h>

#include <opencv2/imgproc.hpp> // cvtColor
#include <opencv2/highgui.hpp> // imdecode imshow

#ifdef ENABLE_OPENCV_CUDA

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>  // boost::make_iterator_range
#include <boost/filesystem.hpp> // boost::filesystem

using namespace std;
using namespace cv;

namespace watrix {
	namespace algorithm {

		/*
		(1) OpenCV modules
		// cudev core cudaarithm flann imgproc ml video cudabgsegm cudafilters cudaimgproc cudawarping imgcodecs photo shape videoio cudacodec highgui objdetect ts features2d calib3d cudafeatures2d cudalegacy cudaobjdetect cudaoptflow cudastereo stitching superres videostab python2


		(2) opencv使用cuda版本的lib,dll；
		caffe任然使用无cuda版本的opencv编译的lib,dll
		exe中用opencv dll,lib替换为cuda版本即可。无需重新编译caffe。

		(3) cpu vs gpu for orb
		https://blog.csdn.net/m0_37857300/article/details/79039214
		对于分辨率不特别大的图片间的ORB特征匹配，CPU运算得比GPU版的快（由于图像上传到GPU消耗了时间）；但对于分辨率较大的图片，或者GPU比CPU好的机器（比如Nvidia Jetson系列），GPU版的ORB算法比CPU版的程序更高效。

		(4) 发现一个问题
		Problems:
		(1) 使用cuda版本的opencv caffe网络的第一次创建非常耗时，后面的网络创建则非常快。(dropped)
		(2) opencv的gpu代码比cpu代码慢，初次启动多耗费20s左右。
		https://stackoverflow.com/questions/12074281/why-opencv-gpu-code-is-slower-than-cpu/16038287#16038287
		https://www.zhihu.com/question/28449000/answer/55287980
		
		Reasons:
		`Your problem is that CUDA needs to initialize! And it will generally takes between 1-10 seconds`

		Why first function call is slow?
		That is because of initialization overheads. On first GPU function call `Cuda Runtime API` is initialized implicitly.

		The first gpu function call is always takes more time, because CUDA initialize context for device. The following calls will be faster.
		http://answers.opencv.org/question/1670/huge-time-to-upload-data-to-gpu/#1676

		Not Reasons:
		(1) CPU clockspeed is 10x faster than GPU clockspeed.
		(2) memory transfer times between host (CPU) and device (GPU)  (upload,downloa data)

		首先对于任何一个CUDA程序，在调用它的第一个CUDA API时后都要花费秒级的时间去初始化运行环境，后续还要分配显存，传输数据，启动内核，每一样都有延迟。这样如果你一个任务CPU运算都仅要几十毫秒，相比而言必须带上这些延迟的GPU程序就会显得非常慢。其次，一个运算量很小的程序，你的CUDA内核不可能启动太多的线程，没有足够的线程来屏蔽算法执行时从显存加载数据到GPU SM中的时延，这就没有发挥GPU的真正功能。

		总结原因：
		对于任何一个CUDA程序，在调用它的第一个CUDA API后都要花费较长的时间(1-10s左右)去初始化运行环境
		这个过程非常耗时。
		This may also come from the fact that creating a CUDA context takes time.
		之后在GPU上调用api速度就非常快了。


		(5) gtx 1060 编译的opencv caffe在gtx 970m上运行出现错误

		`Check failed: error == cudaSuccess (8 vs. 0) invalid device function`

		gtx 1060   sm_61
		gtx 970m   sm_52
		*/

		int GpuUtil::test_gpu(bool enable_gpu)
		{
			try
			{
				cv::Mat src_host = cv::imread("1.jpg", CV_LOAD_IMAGE_GRAYSCALE);

				cv::Mat result_host;

				if (enable_gpu)
				{
					cv::cuda::GpuMat dst, src;
					src.upload(src_host);

					cv::cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
					dst.download(result_host);
				}
				else {
					cv::threshold(src_host, result_host, 128.0, 255.0, CV_THRESH_BINARY);
				}
				// gpu 3.11 s
				// cpu 1.53 s
				
				cv::imshow("Result", result_host);
				cv::waitKey();
			}
			catch (const cv::Exception& ex)
			{
				LOG(INFO)<<"[API] Error: " << ex.what() << std::endl;
			}
			return 0;
		}

		int GpuUtil::gpu_detect_and_compute(
			const cv::Mat& image,
			keypoints_t& keypoints,
			cv::Mat& descriptor
		)
		{
			/*
			error:
			Invalid pitch argument  cv::cuda::GpuMat::copyTo

			https://devtalk.nvidia.com/default/topic/408428/-quot-invalid-pitch-argument-error-quot-from-cublassetmatrix-limitation-on-size-of-matrix/

			The maximum memory pitch is 262144 bytes ( it is reported by deviceQuery).
			*/
			/*
			FEATURE_ORB,
			FEATURE_SIFT,
			FEATURE_SURF, //surf same as sift
			*/
			/*
			orb:  descriptor.type() == CV_8UC1   0
			sift/surf: descriptor.type() == CV_32F    5
			*/
			cuda::GpuMat g_image;

			g_image.upload(image);
			//g_image.convertTo(g_image_32F, CV_32F); // TEST ERROR, why ???

			//Ptr<cuda::ORB> g_orb = cuda::ORB::create(500, 1.2f, 6, 31, 0, 2, 0, 31, 20, true);
			Ptr<cuda::ORB> g_orb = cuda::ORB::create();

			cuda::GpuMat g_keypoints;
			cuda::GpuMat g_descriptors;

			g_orb->detectAndComputeAsync(g_image, cuda::GpuMat(), g_keypoints, g_descriptors);
			// must use 8UC1 to detect and compute,otherwise error occur.

			g_orb->convert(g_keypoints, keypoints);
			g_descriptors.download(descriptor);

#ifdef shared_DEBUG
			LOG(INFO)<<"[API] [gpu_detect_and_compute] descriptor.type()=" << descriptor.type() << std::endl; // 8UC1
#endif // shared_DEBUG

			return 0;
		}

		int GpuUtil::gpu_match_and_filter(
			const cv::Mat& descriptor_object, 
			const cv::Mat& descriptor_scene,
			const keypoints_t& keypoint_object, 
			const keypoints_t& keypoint_scene,
			const float nn_match_ratio,
			keypoints_t& matched1, 
			keypoints_t& matched2
		)
		{
			cuda::GpuMat g_descriptorsL, g_descriptorsR;
			cuda::GpuMat g_descriptorsL_32F, g_descriptorsR_32F;

			g_descriptorsL.upload(descriptor_object);
			g_descriptorsR.upload(descriptor_scene);

			g_descriptorsL.convertTo(g_descriptorsL_32F, CV_32F); // use float on GPU
			g_descriptorsR.convertTo(g_descriptorsR_32F, CV_32F); // use float on GPU

			Ptr<cv::cuda::DescriptorMatcher> g_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

			std::vector<DMatch> matches;
			g_matcher->match(g_descriptorsL_32F, g_descriptorsR_32F, matches);

			// filter out good matches by distance
			int sz = matches.size();
			float max_dist = 0;
			float min_dist = UINT64_MAX*1.0f;

			for (int i = 0; i < sz; i++)
			{
				float dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
#ifdef shareg_DEBUG
			LOG(INFO)<<"[API] Max dist : " << max_dist << std::endl;
			LOG(INFO)<<"[API] Min dist : " << min_dist << std::endl;
#endif // shareg_DEBUG

			for (int i = 0; i < sz; i++)
			{
				const cv::DMatch m = matches[i];
				if (m.distance < nn_match_ratio*max_dist)
				{
					// good matches
					matched1.push_back(keypoint_object[m.queryIdx]);
					matched2.push_back(keypoint_scene[m.trainIdx]);
				}
			}

			return 0;
		}

		int GpuUtil::gpu_orb_and_match(
			const cv::Mat& image_object,
			const cv::Mat& image_scene,
			const float nn_match_ratio,
			keypoints_t& matched1,
			keypoints_t& matched2
		)
		{
			// object vs scene  (object vs history images)
			cuda::GpuMat g_srcL, g_srcR;

			g_srcL.upload(image_object);
			g_srcR.upload(image_scene);

			Ptr<cuda::ORB> g_orb = cuda::ORB::create(500, 1.2f, 6, 31, 0, 2, 0, 31, 20, true);

			cuda::GpuMat g_keypointsL, g_keypointsR;
			cuda::GpuMat g_descriptorsL, g_descriptorsR, g_descriptorsL_32F, g_descriptorsR_32F;

			keypoints_t keyPoints_1, keyPoints_2;

			Ptr<cv::cuda::DescriptorMatcher> g_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

			std::vector<DMatch> matches;

			g_orb->detectAndComputeAsync(g_srcL, cuda::GpuMat(), g_keypointsL, g_descriptorsL);
			g_orb->convert(g_keypointsL, keyPoints_1);
			g_descriptorsL.convertTo(g_descriptorsL_32F, CV_32F); // use float on GPU

			g_orb->detectAndComputeAsync(g_srcR, cuda::GpuMat(), g_keypointsR, g_descriptorsR);
			g_orb->convert(g_keypointsR, keyPoints_2);
			g_descriptorsR.convertTo(g_descriptorsR_32F, CV_32F); // use float on GPU

			g_matcher->match(g_descriptorsL_32F, g_descriptorsR_32F, matches);

			// filter out good matches by distance
			int sz = matches.size();
			float max_dist = 0;
			float min_dist = UINT64_MAX*1.0f;

			for (int i = 0; i < sz; i++)
			{
				float dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
#ifdef shareg_DEBUG
			LOG(INFO)<<"[API] Max dist : " << max_dist << std::endl;
			LOG(INFO)<<"[API] Min dist : " << min_dist << std::endl;
#endif // shareg_DEBUG

			for (int i = 0; i < sz; i++)
			{
				const cv::DMatch m = matches[i];
				if (m.distance < nn_match_ratio*max_dist)
				{
					// good matches
					matched1.push_back(keyPoints_1[m.queryIdx]);
					matched2.push_back(keyPoints_2[m.trainIdx]);
				}
			}

			return 0;
		}


#pragma region gpu vs cpu demo

		int GpuUtil::gpu_orb_demo()
		{
			Mat img_1 = imread("./orb/001_1.jpg");
			Mat img_2 = imread("./orb/001_2.jpg");

			if (!img_1.data || !img_2.data)
			{
				LOG(INFO)<<"[API] error reading images " << endl;
				return -1;
			}

			LOG(INFO)<<"[API] ---------------------------------------------\n";
			int times = 0;
			double startime = cv::getTickCount();
			
			int64 start, end;
			double time;

			vector<Point2f> recognized;
			vector<Point2f> scene;

			int max_times = 10;
			for (times = 0; times< max_times; times++)
			{
				start = getTickCount();

				recognized.resize(500);
				scene.resize(500);

				cuda::GpuMat g_img1, g_img2;
				cuda::GpuMat g_srcL, g_srcR;

				g_img1.upload(img_1); g_img2.upload(img_2);

				Mat img_matches, des_L, des_R;

				cuda::cvtColor(g_img1, g_srcL, COLOR_BGR2GRAY);
				cuda::cvtColor(g_img2, g_srcR, COLOR_BGR2GRAY);

				Ptr<cuda::ORB> g_orb = cuda::ORB::create(500, 1.2f, 6, 31, 0, 2, 0, 31, 20, true);

				cuda::GpuMat g_keypointsL, g_keypointsR;
				cuda::GpuMat g_descriptorsL, g_descriptorsR, g_descriptorsL_32F, g_descriptorsR_32F;

				vector<KeyPoint> keyPoints_1, keyPoints_2;

				Ptr<cv::cuda::DescriptorMatcher> g_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

				std::vector<DMatch> matches;
				std::vector<DMatch> goog_matches;

				g_orb->detectAndComputeAsync(g_srcL, cuda::GpuMat(), g_keypointsL, g_descriptorsL);
				g_orb->convert(g_keypointsL, keyPoints_1);
				g_descriptorsL.convertTo(g_descriptorsL_32F, CV_32F); // use float on GPU

				g_orb->detectAndComputeAsync(g_srcR, cuda::GpuMat(), g_keypointsR, g_descriptorsR);
				g_orb->convert(g_keypointsR, keyPoints_2);
				g_descriptorsR.convertTo(g_descriptorsR_32F, CV_32F);

				g_matcher->match(g_descriptorsL_32F, g_descriptorsR_32F, matches);

				int sz = matches.size();
				double max_dist = 0; 
				double min_dist = 100;

				for (int i = 0; i < sz; i++)
				{
					double dist = matches[i].distance;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				//LOG(INFO)<<"[API] \n-- Max dist : " << max_dist << endl;
				//LOG(INFO)<<"[API] \n-- Min dist : " << min_dist << endl;

				for (int i = 0; i < sz; i++)
				{
					if (matches[i].distance < 0.6*max_dist)
					{
						goog_matches.push_back(matches[i]);
					}
				}

				for (size_t i = 0; i < goog_matches.size(); ++i)
				{
					scene.push_back(keyPoints_2[goog_matches[i].trainIdx].pt);
				}

				//for (unsigned int j = 0; j < scene.size(); j++)
				//	cv::circle(img_2, scene[j], 2, cv::Scalar(0, 255, 0), 2);

				//cv::imshow("img_2", img_2);
				//cv::waitKey(1);

				end = getTickCount();
				time = (double)(end - start) * 1000 / getTickFrequency();
				LOG(INFO)<<"[API] ---------------------------------------------\n";
				LOG(INFO)<<"[API] Total time : " << time << " ms" << endl;

				if (times == 9)
				{
					double maxvalue = (cv::getTickCount() - startime) / cv::getTickFrequency();
					LOG(INFO)<<"[API] #frames " << times / maxvalue << endl;
				}
				LOG(INFO)<<"[API] The number of frame is :  " << times << endl;
			}

			return 0;
		}

		int GpuUtil::cpu_orb_demo()
		{
			Mat img_1 = imread("./orb/001_1.jpg");
			Mat img_2 = imread("./orb/001_2.jpg");

			if (!img_1.data || !img_2.data)
			{
				LOG(INFO)<<"[API] error reading images " << endl;
				return -1;
			}

			int times = 0;
			double startime = cv::getTickCount();

			int64 start, end;
			double time;

			vector<Point2f> recognized;
			vector<Point2f> scene;

			int max_times = 10;
			for (times = 0; times< max_times; times++)
			{
				start = getTickCount();

				recognized.resize(500);
				scene.resize(500);

				Mat g_srcL, g_srcR;

				Mat img_matches, des_L, des_R;

				cvtColor(img_1, g_srcL, COLOR_BGR2GRAY);
				cvtColor(img_2, g_srcR, COLOR_BGR2GRAY);

				Ptr<ORB> g_orb = ORB::create(500, 1.2f, 6, 31, 0, 2);

				Mat g_descriptorsL, g_descriptorsR, g_descriptorsL_32F, g_descriptorsR_32F;

				vector<KeyPoint> keyPoints_1, keyPoints_2;

				Ptr<DescriptorMatcher> g_matcher = cv::BFMatcher::create("BruteForce");

				std::vector<DMatch> matches;
				std::vector<DMatch> goog_matches;

				g_orb->detectAndCompute(g_srcL, Mat(), keyPoints_1, g_descriptorsL);

				g_orb->detectAndCompute(g_srcR, Mat(), keyPoints_2, g_descriptorsR);

				g_matcher->match(g_descriptorsL, g_descriptorsR, matches);

				int sz = matches.size();
				double max_dist = 0; double min_dist = 100;

				for (int i = 0; i < sz; i++)
				{
					double dist = matches[i].distance;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				//LOG(INFO)<<"[API] \n-- Max dist : " << max_dist << endl;
				//LOG(INFO)<<"[API] \n-- Min dist : " << min_dist << endl;

				for (int i = 0; i < sz; i++)
				{
					if (matches[i].distance < 0.6*max_dist)
					{
						goog_matches.push_back(matches[i]);
					}
				}

				for (size_t i = 0; i < goog_matches.size(); ++i)
				{
					scene.push_back(keyPoints_2[goog_matches[i].trainIdx].pt);
				}

				//for (unsigned int j = 0; j < scene.size(); j++)
				//	cv::circle(img_2, scene[j], 2, cv::Scalar(0, 255, 0), 2);

				//imshow("img_2", img_2);
				//waitKey(1);

				end = getTickCount();
				time = (double)(end - start) * 1000 / getTickFrequency();
				LOG(INFO)<<"[API] Total time : " << time << " ms" << endl;

				if (times == 1000)
				{
					double maxvalue = (cv::getTickCount() - startime) / cv::getTickFrequency();
					LOG(INFO)<<"[API] zhenshu " << times / maxvalue << "  zhen" << endl;
				}
				LOG(INFO)<<"[API] The number of frame is :  " << times << endl;
			}

			return 0;
		}
#pragma endregion

	}
}// end namespace


#endif // ENABLE_OPENCV_CUDA

 
/*

#include <vector>

using namespace cv;
using namespace cuda;
using namespace std;


bool stop = false;
void sigIntHandler(int signal)
{
stop = true;
LOG(INFO)<<"[API] Honestly, you are out!" << endl;
}



*/