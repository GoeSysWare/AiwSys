#include "yolo_darknet_api.h"
#include <opencv2/opencv.hpp>

namespace watrix {
	namespace algorithm {

        DarknetYoloConfig YoloDarknetApi::config_;
        network* YoloDarknetApi::net_dk; // darknet 模型文件

        int YoloDarknetApi::class_count_; // 
        std::vector<std::string> YoloDarknetApi::class_labels_; // class labels: person,car,...
        int YoloDarknetApi::batchsize;
        int YoloDarknetApi::CHANNELS; // 3
        int YoloDarknetApi::INPUT_H; // 416
        int YoloDarknetApi::INPUT_W; // 416

		void YoloDarknetApi::Init(const DarknetYoloConfig& config)
		{
            config_ = config;
            SetClassLabels(config_.label_filepath);
            batchsize = 1;
            net_dk = load_network_custom((char*)config_.cfg_filepath.c_str(), (char*)config_.weight_filepath.c_str(), 0, batchsize);
            CHANNELS = 3;
            INPUT_H = net_dk->h;
            INPUT_W = net_dk->w;
            // INPUT_H = config_.input_size.height;
            // INPUT_W = config_.input_size.width;
		}

		void YoloDarknetApi::Free()
		{
            free_network(*net_dk);
		}
        
        void YoloDarknetApi::SetClassLabels(const std::string& filepath)
        {
            std::ifstream infile (filepath);
            if (! infile.is_open())
            {
                std::cout << "Error opening file "<< filepath; 
                return;
            }
            
            char buffer[256];
            while (! infile.eof() ) {
                infile.getline (buffer,256);
    
                std::string label(buffer); // last label is empty 81 ===>80
                if (label!=""){
                    //std::cout << label << std::endl;
                    class_labels_.push_back(label);
                }
            }
            infile.close();

            class_count_ = (int)class_labels_.size();
            // CHECK_GE(class_count_, 0) << "class_labels count must >= 0";

            //printf("[YOLO] class_labels_.size() = %d \n",class_count_);
        }

        void YoloDarknetApi::img2buffer(cv::Mat img, float* data){
            float *input_buffer = data;
            cv::Mat dst;
            cv::cvtColor(img, dst, cv::COLOR_RGB2BGR);
            cv::resize(dst, dst, cv::Size(INPUT_W, INPUT_H));
            dst.convertTo(dst, CV_32FC3, 1/255.0);

            for(int c = 0; c < CHANNELS; ++c)
            {
                for(int h = 0; h < INPUT_H; ++h)
                {
                    for(int w = 0; w < INPUT_W; ++w)
                    {
                        *input_buffer++ = float(dst.at<cv::Vec3f>(h,w)[c]);
                    }
                }
            }

        }


        image YoloDarknetApi::Mat2Image(cv::Mat mat)
        {
            cv::Mat dst;
            cv::cvtColor(mat, dst, cv::COLOR_RGB2BGR);

            int w = dst.cols;
            int h = dst.rows;
            int c = dst.channels();
            image im = make_image(w, h, c);
            unsigned char *data = (unsigned char *)dst.data;
            int step = dst.step;
            for (int y = 0; y < h; ++y) {
                for (int k = 0; k < c; ++k) {
                    for (int x = 0; x < w; ++x) {
                        im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
                    }
                }
            }
            return im;
        }


        void YoloDarknetApi::DealWithDetection(detection *dets, int nboxes, cv::Mat img, detection_boxs_t& output){
    
            for (int i = 0; i < nboxes; i++){
                bool flag = 0;
                int class_index;
                float best_conf = config_.confidence_threshold;
                for(int j = 0;j < class_count_; j++){
                    if(dets[i].prob[j] >= best_conf){
                        flag = 1;
                        class_index = j;
                        best_conf = dets[i].prob[j];
                    }
                }
                if(flag){
                    int left = (dets[i].bbox.x - dets[i].bbox.w / 2.)*img.cols;
                    int right = (dets[i].bbox.x + dets[i].bbox.w / 2.)*img.cols;
                    int top = (dets[i].bbox.y - dets[i].bbox.h / 2.)*img.rows;
                    int bot = (dets[i].bbox.y + dets[i].bbox.h / 2.)*img.rows;
    
                    if (left < 0)
                        left = 0;
                    if (right > img.cols - 1)
                        right = img.cols - 1;
                    if (top < 0)
                        top = 0;
                    if (bot > img.rows - 1)
                        bot = img.rows - 1;

                    detection_box_t detection_box{ 
								left,
								top,
								right,
								bot,
								dets[i].prob[class_index],
								class_index,
								class_labels_[class_index],
								false,
								0,
								0
							};
					output.push_back(detection_box);
                }
            }// end for

        }

		bool YoloDarknetApi::Detect(
			const std::vector<cv::Mat>& v_image,
			std::vector<detection_boxs_t>& v_output
		)
		{
			bool success = false;
            if (v_image.size() != 0){
                for(int i=0; i < v_image.size(); i++){
                    int ori_img_w = v_image[i].cols;
                    int ori_img_h = v_image[i].rows;

                    // image im = Mat2Image(v_image[i]);
                    // image sized = resize_image(im, INPUT_W, INPUT_H);
                    // float *input = sized.data;

                    float *input = (float*) malloc (batchsize * CHANNELS * INPUT_H * INPUT_W * sizeof(float));
                    img2buffer(v_image[i], input);
                    
                    network_predict_ptr(net_dk, input);// net infer
                    int nboxes = 0;
                    int letter_box = 0; // 0为直接resize; 1为保持比例resize
                    detection *dets = get_network_boxes(net_dk, ori_img_w, ori_img_h, config_.confidence_threshold, config_.hier_thresh, 0, 1, &nboxes, letter_box);

                    do_nms_sort(dets, nboxes, class_count_, config_.iou_thresh);

                    detection_boxs_t output;
                    DealWithDetection(dets, nboxes, v_image[i], output);

                    v_output.push_back(output);

                    free_detections(dets, nboxes);
                }
                
                success = true;
            }

			return success;
		}
	}
}// end namespace