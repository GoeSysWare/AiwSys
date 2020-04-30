load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libadas_perception_component.so",
    linkshared = True,
    linkstatic = False,
    deps = [":adas_perception_lib"],
)

cc_library(
    name = "adas_perception_lib",
    srcs = [
         "adas_perception_component.cpp",
        "FindContours_v2.cpp",        
         "perception_task.cpp",      
    ],
    hdrs = [
        "adas_perception_component.h",
        "perception_task.h",
        "FindContours_v2.h",
        "lidar_point_struct.h",
    ],
    copts = [
        '-DMODULE_NAME=\\"ADAS_PERCEPTION\\"',
    ],
    deps = [
        "//cyber",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:geometry_proto",
        "//modules/common/proto:header_proto",
        "//modules/common/time",
        "//modules/drivers/proto:sensor_proto",
        "//projects/adas/proto:adas_config_proto",
        "//projects/adas/configs:config_gflags",     
         "//projects/adas/component/common",
        "//projects/adas/algorithm:algorithm_sdk",
        "@yaml_cpp//:yaml",
        "@cgal",  
      ],
)



# cc_library(
#     name = "FindContours",
#     srcs = [
#         "FindContours_v2.cpp",        
#     ],
#     hdrs = [
#         "FindContours_v2.h",
#         "lidar_point_struct.h",
#     ],
#     deps = [
#         "//modules/common/proto:error_code_proto",
#         "//modules/common/proto:geometry_proto",
#         "//modules/common/proto:header_proto",
#         "//modules/drivers/proto:sensor_proto",
#         "//projects/adas/configs:config_gflags",  
#         "//projects/adas/component/common",
#         "@cgal",  
#         "@mkldnn",
#         "@vtk",
#         "@pytorch",
#          "@pcl",
#          "@eigen",
#     ],
# )



cpplint()
