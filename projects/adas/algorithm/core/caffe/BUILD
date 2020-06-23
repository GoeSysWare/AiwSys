
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "algorithm_caffe_api",
    srcs = [
        "internal/caffe_def.cpp",
        "internal/caffe_net_v2.cpp" ,    
        "internal/caffe_net.cpp" ,    
        "caffe_api.cpp" ,    
    ],
    hdrs = [
        "internal/caffe_def.h",
        "internal/caffe_net_v2.h" ,    
        "internal/caffe_net.h" ,    
        "caffe_api.h" ,   
    ],
    deps = [
            "//projects/adas/algorithm:algorithm_header",   
            "@caffe",  
            "@pcl",      
            "@vtk",             
            "@eigen",   
            "@cuda",             
    ],
)
