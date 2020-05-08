

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "adas_sdk",
    srcs = [
        "GetLaneStatus.cpp",
         "laneseg_api.cpp",       
        "monocular_distance_api.cpp",
         "ObjectAndTrainDetection.cpp",  
        "sensor_api.cpp",
         "trainseg_api.cpp",  
        "yolo_api.cpp",
        "internal/caffe/caffe_laneseg_api_impl.cpp",
        "internal/caffe/laneseg_net.cpp",
        "internal/caffe/trainseg_api_impl.cpp",
        "internal/caffe/trainseg_net.cpp",
        "internal/caffe/yolo_net.cpp",
        "internal/pytorch/pt_simple_laneseg_api_impl.cpp",
        "internal/laneseg_util.cpp",
    ],
    hdrs = [
        "GetLaneStatus.h",
         "laneseg_api.h",       
        "monocular_distance_api.h",
         "ObjectAndTrainDetection.h",  
        "sensor_api.h",
         "trainseg_api.h",  
        "yolo_api.h",
        "internal/caffe/caffe_laneseg_api_impl.h",
        "internal/caffe/laneseg_net.h",
        "internal/caffe/trainseg_api_impl.h",
        "internal/caffe/trainseg_net.h",
        "internal/caffe/yolo_net.h",
        "internal/pytorch/pt_simple_laneseg_api_impl.h",
        "internal/laneseg_util.h",
    ],
    deps = [
             "//projects/adas/algorithm:algorithm_header",
            "//projects/adas/algorithm/core:algorithm_core_api",
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
