load("//tools:cpplint.bzl", "cpplint")
load("//tools:qt.bzl", "qt_cc_library")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "perception_hmi",
    includes = [
        ".",
    ],
    linkopts = [
        "-lpthread",
        "-L/usr/lib/x86_64-linux-gnu", 
       "-lboost_filesystem",
        "-lboost_atomic",
        "-lboost_system",
        "-lboost_date_time",
        "-lasound",
        
    ],
    deps = [
        ":perception_hmi_lib",
        "@glog",
        "@opencv",
    ],
)

# name, src, hdr, uis = [], res = [], normal_hdrs = [], deps = None
qt_cc_library(
    name = "perception_hmi_lib",
    
    src = glob(
        [
            "*.cpp",
            "**/*.cpp",
        ],
    ),
    copts = [
        "-Imodules/drivers/proto",
    ],
    hdr = glob([
        "*.h",
           "**/*.h",
    ]),
    includes = [
        ".",
    ],
    linkstatic = 1,
    res = glob([
        "*.qrc",
    ]),
    uis = glob([
        "*.ui",
    ]),
    deps = [
        "//cyber",
        "//modules/drivers/proto:sensor_proto",
        "@qt//:qt_gui",
        "@qt//:qt_media",
        "//projects/adas/proto:point_cloud_proto",
       "//projects/adas/proto:camera_image_proto",
       

    ],
)

cpplint()
