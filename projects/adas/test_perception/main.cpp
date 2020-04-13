

 #include "test_perception.h"
 #include "FindContours_v2.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> //
#include <opencv2/highgui.hpp> // imwrite
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


int main(int argc, char *argv[]) {

	// watrix::algorithm::YoloNetConfig cfg;
	// watrix::algorithm::YoloApi::Init(cfg);
   std::cout<< "ceshi"<<std::endl;
    watrix::Test_Perception app;
    app.Init(argv[0]);
    app.Start();
    apollo::cyber::WaitForShutdown();

  return 0;
}
