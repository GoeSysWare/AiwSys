#include "caffe_api.h"

//add macro GLOG_NO_ABBREVIATED_SEVERITIES 
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cassert>

// caffe
#include <caffe/caffe.hpp>

using namespace std;
using namespace caffe;

namespace watrix {
	namespace algorithm {

#pragma region set caffe mode

		// set mode is thread-local.
		/*
		Caffe set_mode GPU 在多线程下失效
		在main thread中设置GPU模式，在worker thread中调用网络进行检测，GPU模式不起效，默认仍然使用CPU模式，所以速度很慢，和GPU相比慢了10倍左右。
		Caffe fails to use GPU in a new thread ？？？
		https://github.com/BVLC/caffe/issues/4178

		解决方案：在子线程中set_mode,然后调用网络进行检测。
		the `Caffe::mode_` variable that controls this is thread-local, so ensure you’re calling `caffe.set_mode_gpu()` in each thread before running any Caffe functions. That should solve your issue.
		(1)创建网络在main thread。static 网络存储在全局静态数据区。worker thread可以直接使用。
		(2) 在worker thread中检测，需要在子线程中set_mode,然后调用网络进行检测。
		*/
		void CaffeApi::set_mode(bool gpu_mode,int device_id, unsigned int seed)
		{
			if (gpu_mode) {
				Caffe::set_mode(Caffe::GPU); // thread-local
			}
			else {
				Caffe::set_mode(Caffe::CPU);
			}
			Caffe::SetDevice(device_id);
			Caffe::set_random_seed(seed);
		}

#pragma endregion

#pragma region glog

		void CaffeApi::init_glog(
			const std::string& log_name,
			const std::string& log_fatal_dir,
			const std::string& log_error_dir,
			const std::string& log_warning_dir,
			const std::string& log_info_dir
			)
		{
			//FLAGS_logtostderr = true;
			//FLAGS_alsologtostderr = true;
			//FLAGS_colorlogtostderr = true;
			FLAGS_log_prefix = true;
			FLAGS_logbufsecs = 0;  //0 means realtime
			FLAGS_max_log_size = 100;  // MB

			google::InitGoogleLogging(log_name.c_str()); // init google logging
			google::SetStderrLogging(google::GLOG_WARNING);
			google::SetLogDestination(google::GLOG_FATAL,log_fatal_dir.c_str());
			google::SetLogDestination(google::GLOG_ERROR,log_error_dir.c_str());
			google::SetLogDestination(google::GLOG_WARNING,log_warning_dir.c_str());
			google::SetLogDestination(google::GLOG_INFO,log_info_dir.c_str());
		}

		void CaffeApi::shutdown_glog()
		{
			google::ShutdownGoogleLogging();
		}

#pragma endregion

	}
}// end namespace

