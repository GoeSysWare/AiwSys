/*
* Software License Agreement
*
*  WATRIX.AI - www.watrix.ai
*  Copyright (c) 2016-2018, Watrix Technology, Inc.
*
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the copyright holder(s) nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: zunlin.ke@watrix.ai (Zunlin Ke)
*
*/
#pragma once
 

#include <vector>
#include <opencv2/core.hpp>

#ifdef USE_DLIB
#include <dlib/geometry.h>
#endif

namespace watrix {
	namespace algorithm {
		namespace internal {

			class OcrUtil
			{
			public:
				// v1.v2
				static float dot_product(
					const std::vector<float>& v1,
					const std::vector<float>& v2
				);

				// |v1| vector module = sqrt(x1*x1+y1*y1)
				static float module(const std::vector<float>& v);

				// cos = v1.v2/(|v1|*|v2|)
				static float cosine(
					const std::vector<float>& v1,
					const std::vector<float>& v2
				);

				static bool sort_score_pair_descend(
					const std::pair<float, int>& pair1,
					const std::pair<float, int>& pair2
				);

				static float jaccard_overlap(
					const cv::Rect& bbox1,
					const cv::Rect& bbox2
				);

				static void get_max_area_index(
					const std::vector<float>& scores,
					const float threshold,
					const int top_k,
					std::vector<std::pair<float, int>>& score_index_vec
				);

				static int nms_fast(
					const std::vector<cv::Rect>& boxs,
					const float overlapThresh, // 0.1
					//const float min_area, // 400
					std::vector<cv::Rect>& new_boxs
				);

#ifdef USE_DLIB
				static float jaccard_overlap(
					const dlib::rectangle& bbox1,
					const dlib::rectangle& bbox2
				);

				static int nms_fast(
					const std::vector<dlib::rectangle> &bbs,
					const float overlapThresh, // 0.1
					const float min_area, // 400
					std::vector<int>& v_index
				);
#endif

			};

		}
	}
}