#include "mosun_api.h"
#include "internal/mosun_api_impl.h"


namespace watrix {
	namespace algorithm {


		void MosunApi::init(
			const MosunParam& params,
			int net_count
		)
		{
			internal::MosunApiImpl::init(params, net_count);
		}

		void MosunApi::free()
		{
			internal::MosunApiImpl::free();
		}

		MosunResult MosunApi::detect(
			int net_id,
			const cv::Mat& image
		)
		{
			return internal::MosunApiImpl::detect(net_id, image);
		}

	}
}// end namespace

