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
*  Author: jilong.ma@watrix.ai (JiLong Ma) zunlin.ke@watri.ai(ZunLin ke)
*
*/
#pragma once

// opencv
#include <opencv2/core.hpp> // Mat

namespace watrix {
	namespace algorithm {
		namespace internal{

			class DistortionFixer {
			public:
				cv::Mat img1; // history image 
				cv::Mat img2; //current 
				cv::Mat result; //output
				void fix();
				DistortionFixer(char* filename1, char* filename2);
				DistortionFixer(cv::Mat A, cv::Mat B);

			private:
				const int HEIGHT = 1024;
				const int WIDTH = 2048;
				const int RADIUS = 30;
				const int BLOCK_WIDTH = 90;
				const int v_radius = 10;
				cv::Mat target = cv::Mat(WIDTH, 2 * RADIUS + 1, CV_32F);

				void getVector(int i, cv::Mat const line_pic, cv::Mat dst);
				int getPosition(int index, cv::Mat &source_vector);
			};

		}
	}
}