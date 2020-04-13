load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "test_algorithm",
    srcs = [
        "main.cpp",
    ],
    deps = [
                ":test_perception_lib",  
         "//cyber",
        "//cyber/parameter",
            "//projects/adas/algorithm:algorithm_sdk",
            "@mkldnn",
        "@vtk",
        "@glog",  
            "@opencv",
               "@caffe",         
        ],
    linkopts = [
         "-L/usr/lib/x86_64-linux-gnu", 
       "-lboost_filesystem",
        "-lboost_atomic",
        "-lboost_system",
        "-lboost_date_time",
        # "-lgmpxx",
        "-lgmp",
         "-L /usr/local/lib", 
         "-ldl", 
    ],
    
)



cc_binary(
    name = "test_perception",
    srcs = [
        "main.cpp",
    ],
    deps = [
        ":test_perception_lib",  

        ],
    linkopts = [
         "-L/usr/lib/x86_64-linux-gnu", 
       "-lboost_filesystem",
        "-lboost_atomic",
        "-lboost_system",
        "-lboost_date_time",
        # "-lgmpxx",
        "-lgmp",
         "-L /usr/local/lib", 
         "-ldl", 
    ],
)

cc_library(
    name = "test_perception_lib",
    srcs = [
       "test_perception.cpp",
         "datetime_util.cpp",  

    ],
    hdrs = [
        "test_perception.h",
        "test_uitls.h",
        "datetime_util.h",
    ],

    deps = [
        ":FindContours",  
        "//cyber",
        "//cyber/parameter",
        "//modules/common/status",
          "//modules/common/time",
       "//modules/drivers/proto:sensor_proto",
         "//projects/adas/algorithm:algorithm_sdk",
       "//projects/adas/proto:point_cloud_proto",
       "//projects/adas/proto:camera_image_proto",
       "//projects/adas/proto:test_config_proto",
       "//projects/adas/proto:node_config_proto",
       "@opencv",
        "@glog",
            "@caffe",  
            "@pytorch",          
    ],
)

cc_library(
    name = "FindContours",
    srcs = [
        "FindContours_v2.cpp",        
           "gflags/test_perception_flags.cpp",
    ],
    hdrs = [
        "FindContours_v2.h",
        "lidar_point_struct.h",
               "gflags/test_perception_flags.h"
    ],

    deps = [
        "//projects/adas/proto:point_cloud_proto",
        "//projects/adas/algorithm:algorithm_sdk",
        "@cgal",  
        "@mkldnn",
        "@vtk",

    ],
)

cpplint()
