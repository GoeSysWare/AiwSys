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
#include "projects/adas/algorithm/algorithm_shared_export.h"

#include "projects/adas/algorithm/algorithm_type.h"

#include "projects/adas/algorithm/core/timer.h"
#include "projects/adas/algorithm/core/profiler.h"

#include "projects/adas/algorithm/core/util/opencv_util.h"
#include "projects/adas/algorithm/core/util/display_util.h"
#include "projects/adas/algorithm/core/util/filesystem_util.h"
#include "projects/adas/algorithm/core/util/polyfiter.h"

#include "projects/adas/algorithm/core/caffe/caffe_api.h"

// // carv1
// #include "carv1/topwire_api.h"
// #include "carv1/railway_api.h"
// #include "carv1/lockcatch_type.h"
// #include "carv1/lockcatch_api.h"
// #include "carv1/sidewall_type.h"
// #include "carv1/sidewall_api.h"
// #include "carv1/ocr_type.h"
// #include "carv1/ocr_api.h"
// #include "carv1/distance_api.h"

// // refinedet
// #include "refinedet/refinedet_api.h"

// yolo
//#include "yolov3/yolov3_api.h"

// autotrain
// caffe api
#include "projects/adas/algorithm/autotrain/yolo_darknet_api.h"  // caffe
#include "projects/adas/algorithm/autotrain/yolo_api.h"  // caffe
#include "projects/adas/algorithm/autotrain/trainseg_api.h" // caffe
#include "projects/adas/algorithm/autotrain/laneseg_api.h" // caffe/pytorch backend
#include "projects/adas/algorithm/autotrain/monocular_distance_api.h"
#include "projects/adas/algorithm/autotrain/sensor_api.h"

// // carv2 
// #include "carv2/lahu_api.h"
// #include "carv2/topwire_distance_api.h"
// #include "carv2/mosun_api.h"

namespace watrix {
	namespace algorithm {

		SHARED_EXPORT void InitModuleAlgorithm();
		SHARED_EXPORT void FreeModuleAlgorithm();

	}
} // end namespace