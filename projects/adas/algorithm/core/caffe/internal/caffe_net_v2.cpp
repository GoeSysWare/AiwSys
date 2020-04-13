#include "caffe_net_v2.h"

// glog
#include <glog/logging.h>

// caffe
#include <caffe/caffe.hpp>
using namespace caffe;
using namespace std;

namespace watrix {
	namespace algorithm {
		namespace internal {

#pragma region init and free nets

			void CaffeNetV2::Init(
				int net_count,
				std::string proto_filepath,
				std::string weight_filepath,
				std::vector<shared_caffe_net_t>& v_net
			)
			{
				enum caffe::Phase phase = caffe::Phase::TEST;
				v_net.resize(net_count);
				for (int i = 0; i < v_net.size(); i++)
				{
					v_net[i].reset(
						new caffe_net_t(proto_filepath, phase)
					);
					v_net[i]->CopyTrainedLayersFrom(weight_filepath);
				};
			}

			void CaffeNetV2::Free(std::vector<shared_caffe_net_t>& v_net)
			{
				//LOG(INFO)<<"[API]  CaffeNet::free \n";
				for (int i = 0; i < v_net.size(); i++)
				{
					v_net[i] = nullptr;
				}
			}
#pragma endregion

		}
	}
}