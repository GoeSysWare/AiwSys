#include "distortion_fixer.h"

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

namespace watrix {
	namespace algorithm {
		namespace internal{

			DistortionFixer::DistortionFixer(char* filename1, char* filename2) {
				img1 = cv::imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
				img2 = cv::imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
				result = cv::Mat::zeros(cv::Size(img2.cols, img2.rows), img2.type());
				fix();
			}

			DistortionFixer::DistortionFixer(cv::Mat A, cv::Mat B) {
				img1 = A;   // assign image reference
				img2 = B;
				result = cv::Mat::zeros(cv::Size(img2.cols, img2.rows), img2.type());
				fix();
			}

			void DistortionFixer::getVector(int index, cv::Mat const line_pic, cv::Mat dst) {
				cv::Mat vector = cv::Mat::zeros(1, 2 * RADIUS + 1, CV_32F);
				int start = index - RADIUS < 0 ? 0 : index - RADIUS;
				int end = index + RADIUS > WIDTH ? WIDTH : index + RADIUS;

				line_pic
					.colRange(start, end)
					.copyTo(vector.colRange(start - (index - RADIUS), end - index + RADIUS));
				vector.convertTo(vector, CV_32F);
				// get the feature vector within the range of 2*Radius and normalize it.
				double minVal = 0, maxVal = 0;
				cv::Point minPt, maxPt;
				minMaxLoc(vector, &minVal, &maxVal, &minPt, &maxPt);
				vector = (vector - mean(vector)[0]) / (maxVal - minVal);
				vector.copyTo(dst);
			}

			int DistortionFixer::getPosition(int index, cv::Mat &source_vector) {
				if (index < 0)
					index = 0;
				int start = (index - v_radius) < 0 ? 0 : (index - v_radius);
				int end = (index + v_radius) > WIDTH ? WIDTH : (index + v_radius);

				for (int i = start; i < end; i++)
				{
					getVector(i, img1.row(HEIGHT / 2), target.row(i));
				}

				// find the most similar vector and its index.
				std::vector<std::vector<double>> dist(end - start, std::vector<double>(2));
				// double dist[end-start][2];
				for (int j = start; j < end; j++)
				{
					cv::Mat sub = source_vector - target.row(j);
					double d = sub.dot(sub);
					dist[j - start][0] = j;
					dist[j - start][1] = d;
				}
				double min = 10000000;
				int temp_pos;
				for (int i = 0; i < end - start; i++)
				{
					if (dist[i][1] < min)
					{
						min = dist[i][1];
						temp_pos = dist[i][0];
					}
				}
				return temp_pos;
			}

			void DistortionFixer::fix() {
				cv::Mat r_vector = cv::Mat(1, 2 * RADIUS + 1, CV_32F);
				cv::Mat l_vector = cv::Mat(1, 2 * RADIUS + 1, CV_32F);

				// process left area(from 0 to 750)
				int l_last_col = 0;
				int left_block = 0;
				for (int i = 0; i < 1050; i += BLOCK_WIDTH)
				{
					getVector(i, img2.row(HEIGHT / 2), l_vector);
					getVector(i + BLOCK_WIDTH, img2.row(HEIGHT / 2), r_vector);
					int r = getPosition(i + BLOCK_WIDTH, r_vector);
					int l = getPosition(i, l_vector);
					cv::Mat test2 = img2.colRange(i, i + BLOCK_WIDTH).clone();
					cv::resize(test2, test2, cv::Size(r - l, HEIGHT), 0, 0, cv::INTER_CUBIC);
					test2.copyTo(result.colRange(l_last_col, l_last_col + (r - l)));
					l_last_col = l_last_col + (r - l);
					left_block = i + BLOCK_WIDTH;
				}

				// process right area(from 1248 to 2048)
				int r_last_col = 2048;
				int right_block = 0;
				for (int i = 2048; i > 1150; i -= BLOCK_WIDTH)
				{
					getVector(i, img2.row(HEIGHT / 2), r_vector);
					getVector(i - BLOCK_WIDTH, img2.row(HEIGHT / 2), l_vector);
					int r = getPosition(i, r_vector);
					int l = getPosition(i - BLOCK_WIDTH, l_vector);
					cv::Mat test2 = img2.colRange(i - BLOCK_WIDTH, i).clone();
					cv::resize(test2, test2, cv::Size(r - l, HEIGHT), 0, 0, cv::INTER_CUBIC);
					test2.copyTo(result.colRange(r_last_col - (r - l), r_last_col));
					r_last_col = r_last_col - (r - l);
					right_block = i - BLOCK_WIDTH;
				}
				cv::Mat test2 = img2.colRange(left_block, right_block).clone();
				cv::resize(test2, test2, cv::Size(r_last_col - l_last_col, HEIGHT), 0, 0, cv::INTER_CUBIC);
				test2.copyTo(result.colRange(l_last_col, r_last_col));

			}

		}
	}
}


/*
//Summary:
//  use img1 as a benchmark to correct radial distortion of img2
//Parameters:
//  img1(Type: openCV Mat): a benchmark picture.
//  img2((Type: openCV Mat): picture to correct
//  OutputImage(Type: openCV Mat): Output Matrix
//Notice:
//  This function can only work on 1024*2048(height*width) size picture.

void RadialDistortionFixer(Mat img1, Mat img2, Mat OutputImage) {

DistortionFixer a = DistortionFixer(img1, img2);
a.result.copyTo(OutputImage);
}
*/