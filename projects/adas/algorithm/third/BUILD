
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "algorithm_third_api",
    srcs = [
        "cluster_util.cpp" ,
        "MeanShift/UserMeanShift.cpp" ,       
    ],
    hdrs = [
        "cluster_util.h",
        "MeanShift/UserMeanShift.h" ,       
    ],
    deps = [
            "//projects/adas/algorithm:algorithm_header",   
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
