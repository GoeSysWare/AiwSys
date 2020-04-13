#include "ocr_util.h"

#include <cassert>
#include <cmath>

#ifdef USE_DLIB
using namespace dlib;
#endif

using namespace std;

namespace watrix {
	namespace algorithm {
		namespace internal {

			// v1.v2
			float OcrUtil::dot_product(
				const std::vector<float>& v1,
				const std::vector<float>& v2
			)
			{
				assert(v1.size() == v2.size());
				float ret = 0.0;
				for (std::vector<float>::size_type i = 0; i != v1.size(); ++i)
				{
					ret += v1[i] * v2[i];
				}
				return ret;
			}

			// |v1| vector module = sqrt(x1*x1+y1*y1)
			float OcrUtil::module(const std::vector<float>& v)
			{
				float ret = 0.0;
				for (std::vector<float>::size_type i = 0; i != v.size(); ++i)
				{
					ret += v[i] * v[i];
				}
				return std::sqrt(ret);
			}

			// cos = v1.v2/(|v1|*|v2|)
			float OcrUtil::cosine(
				const std::vector<float>& v1,
				const std::vector<float>& v2
			)
			{
				assert(v1.size() == v2.size());
				return dot_product(v1, v2) / (module(v1) * module(v2));
			}

			bool OcrUtil::sort_score_pair_descend(
				const std::pair<float, int>& pair1,
				const std::pair<float, int>& pair2
			)
			{
				return pair1.first > pair2.first;
			}

			float OcrUtil::jaccard_overlap(
				const cv::Rect& bbox1,
				const cv::Rect& bbox2
			)
			{
				// overlap/(a1+a2-overlap)
				const float inter_xmin = std::max(bbox1.x, bbox2.x);
				const float inter_ymin = std::max(bbox1.y, bbox2.y);
				const float inter_xmax = std::min(bbox1.x + bbox1.width,  bbox2.x + bbox2.width);
				const float inter_ymax = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

				const float inter_width = inter_xmax - inter_xmin;
				const float inter_height = inter_ymax - inter_ymin;
				const float inter_size = inter_width * inter_height;

				const float bbox1_size = (bbox1.width) * (bbox1.height);
				const float bbox2_size = (bbox2.width) * (bbox2.height);

				return inter_size / (bbox1_size + bbox2_size - inter_size);
			}

			void OcrUtil::get_max_area_index(
				const std::vector<float>& scores,
				const float threshold,
				const int top_k,
				std::vector<std::pair<float, int>>& score_index_vec
			)
			{
				// Generate index score pairs.
				for (int i = 0; i < scores.size(); ++i) {
					if (scores[i] > threshold) {
						score_index_vec.push_back(std::make_pair(scores[i], i));
					}
				}

				// Sort the score pair according to the scores in descending order
				std::stable_sort(
					score_index_vec.begin(),
					score_index_vec.end(),
					sort_score_pair_descend
				);

				// Keep top_k scores if needed.
				if (top_k > -1 && top_k < score_index_vec.size())
				{
					score_index_vec.resize(top_k);
				}
			}

			int OcrUtil::nms_fast(
				const std::vector<cv::Rect>& boxs_,
				const float overlapThresh,
				std::vector<cv::Rect>& new_boxs
			)
			{
				std::vector<int> v_index;
				std::vector<cv::Rect> bbs = boxs_;

				if (bbs.size() < 1)
				{
					return 0;
				}

				// (1) get areas 
				//std::cout << bbs.size() << std::endl;
				std::vector<float> areas;
				for(int i=0;i<bbs.size();i++)
				{
					long w = bbs[i].width;
					long h = bbs[i].height;
					float s = w * h;
					areas.push_back(s);
				}

				// Generate index score pairs.
				std::vector<pair<float, int> > v_pair_score_index;
				for (int i = 0; i < areas.size(); ++i) {
					v_pair_score_index.push_back(std::make_pair(areas[i], i));
				}

				// Sort the score pair according to the scores in descending order
				std::stable_sort(
					v_pair_score_index.begin(),
					v_pair_score_index.end(),
					sort_score_pair_descend
				);

				// Do nms.
				while (v_pair_score_index.size() != 0)
				{
					const int idx = v_pair_score_index.front().second;
					bool keep = true;
					for (int k = 0; k < v_index.size(); ++k)
					{
						if (keep)
						{
							const int kept_idx = v_index[k];
							float overlap = jaccard_overlap(bbs[idx], bbs[kept_idx]);
							keep = (overlap <= overlapThresh);
						}
						else
						{
							break;
						}
					}

					if (keep) {
						v_index.push_back(idx);
					}

					v_pair_score_index.erase(v_pair_score_index.begin());
				}

				for (size_t i = 0; i < v_index.size(); i++)
				{
					int index = v_index[i];
					new_boxs.push_back(bbs[index]);
				}
				return v_index.size();
			}

#ifdef USE_DLIB
			float OcrUtil::jaccard_overlap(
				const dlib::rectangle& bbox1,
				const dlib::rectangle& bbox2
			)
			{
				// overlap/(a1+a2-overlap)
				if (bbox2.left() > bbox1.right() || bbox2.right() < bbox1.left()
					|| bbox2.top() > bbox1.bottom() || bbox2.bottom() < bbox1.top())
				{
					return 0;
				}
				else
				{
					const float inter_xmin = std::max(bbox1.left(), bbox2.left());
					const float inter_ymin = std::max(bbox1.top(), bbox2.top());
					const float inter_xmax = std::min(bbox1.right(), bbox2.right());
					const float inter_ymax = std::min(bbox1.bottom(), bbox2.bottom());

					const float inter_width = inter_xmax - inter_xmin;
					const float inter_height = inter_ymax - inter_ymin;
					const float inter_size = inter_width * inter_height;

					const float bbox1_size = (bbox1.right() - bbox1.left())
						* (bbox1.bottom() - bbox1.top());
					const float bbox2_size = (bbox2.right() - bbox2.left())
						* (bbox2.bottom() - bbox2.top());

					return inter_size / (bbox1_size + bbox2_size - inter_size);
				}
			}

			int OcrUtil::nms_fast(
				const std::vector<dlib::rectangle>& bbs_,
				const float overlapThresh,
				const float min_area,
				std::vector<int>& v_index
			)
			{
				std::vector<dlib::rectangle> bbs = bbs_;

				if (bbs.size() < 1)
				{
					return 0;
				}

				//std::cout << bbs.size() << std::endl;
				std::vector<float> areas;
				int cur = 0;
				while (bbs.size() > 0 && cur < bbs.size())
				{
					long w = bbs[cur].right() - bbs[cur].left();
					long h = bbs[cur].bottom() - bbs[cur].top();
					float s = w * h;
					if (s < min_area)
					{
						bbs.erase(bbs.begin() + cur);
						continue;
					}
					areas.push_back(s);
					cur++;
				}

				std::vector<pair<float, int> > score_index_vec;
				get_max_area_index(areas, 0, 5, score_index_vec);
				// Do nms.

				while (score_index_vec.size() != 0)
				{
					const int idx = score_index_vec.front().second;
					bool keep = true;
					for (int k = 0; k < v_index.size(); ++k)
					{
						if (keep)
						{
							const int kept_idx = v_index[k];
							float overlap = jaccard_overlap(bbs[idx], bbs[kept_idx]);
							keep = overlap <= overlapThresh;
						}
						else
						{
							break;
						}
					}
					if (keep) {
						v_index.push_back(idx);
					}

					/*
					else {
					bbs.erase(idx);
					}*/
					score_index_vec.erase(score_index_vec.begin());
					/*
					if (keep && eta < 1 && adaptive_threshold > 0.5) {
					adaptive_threshold *= eta;
					}*/
				}
				return v_index.size();
			}
#endif

			

		}
	}
}