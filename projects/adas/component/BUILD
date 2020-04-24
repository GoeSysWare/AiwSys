load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libadas_camera_component.so",
    linkshared = True,
    linkstatic = False,
    deps = [":adas_cemara_component"],
)

cc_library(
    name = "adas_cemara_component",
    srcs = [
        "adas_camera_component.cpp",
    ],
    hdrs = [
        "adas_camera_component.h",
        "lidar_point_struct.h",
    ],
    copts = [
        '-DMODULE_NAME=\\"ADAS_CAMERA\\"',
    ],
    deps = [
        ":FindContours",  
        "//cyber",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:geometry_proto",
        "//modules/common/proto:header_proto",
        "//modules/common/time",
        "//modules/drivers/proto:sensor_proto",
        "//modules/perception/base",
        "//modules/perception/camera/common",
        "//projects/common/inner_component_messages",
        "//projects/common/registerer",
        "//projects/adas/proto:adas_camera_proto",
        "//projects/adas/proto:adas_detection_proto",
        "//projects/adas/proto:node_config_proto",
        "//projects/adas/configs:config_gflags",     
        "//projects/adas/detector/interface:base_adas_detector",
        "//projects/adas/algorithm:algorithm_sdk",
        "@yaml_cpp//:yaml",
      ],
)



cc_library(
    name = "FindContours",
    srcs = [
        "FindContours_v2.cpp",        
    ],
    hdrs = [
        "FindContours_v2.h",
        "lidar_point_struct.h",
    ],

    deps = [
        "//cyber",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:geometry_proto",
        "//modules/common/proto:header_proto",
        "//modules/common/time",
        "//modules/drivers/proto:sensor_proto",
        "//projects/adas/proto:adas_camera_proto",
        "//projects/adas/proto:adas_detection_proto",
        "//projects/adas/proto:node_config_proto",
        "//projects/adas/configs:config_gflags",     
        "//projects/adas/algorithm:algorithm_sdk",
    ],
)

cpplint()
