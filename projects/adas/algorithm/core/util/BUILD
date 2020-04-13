
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "algorithm_util_api",
    srcs = [
        "display_util.cpp" ,      
        "filesystem_util.cpp" ,    
       "gpu_util.cpp" ,    
       "lane_util.cpp" ,    
        "numpy_util.cpp" ,    
       "opencv_util.cpp" ,    
        "polyfiter.cpp" ,    
    ],
    hdrs = [
        "display_util.h" ,      
        "filesystem_util.h" ,    
       "gpu_util.h" ,    
       "lane_util.h" ,    
        "numpy_util.h" ,    
       "opencv_util.h" ,    
        "polyfiter.h" ,    
    ],
    deps = [
            "//projects/adas/algorithm:algorithm_header",   
            "//projects/adas/algorithm/third:algorithm_third_api",
            "@pcl",
            "@eigen",
            "@cuda",
           "@pytorch",
             "@vtk",
            "@caffe",  
            "@glog",
            "//external:gflags",   
    ],
)
