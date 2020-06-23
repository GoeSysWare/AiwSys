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
        "-ldl",
        "-L/usr/lib/x86_64-linux-gnu", 
       "-lboost_filesystem",
        "-lboost_atomic",
        "-lboost_system",
        "-lboost_date_time",
    ],
    copts = [
        '-DMODULE_NAME=\\"ADAS_HMI\\"',
    ],
    deps = [
        ":perception_hmi_lib",
        "@glog",
  
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
        "//cyber/common:log",
        "//modules/drivers/proto:sensor_proto",
        "@qt//:qt_gui",
        # "@qt//:qt_media",
        "//projects/adas/proto:adas_config_proto",
        "//projects/adas/configs:config_gflags",
        "//projects/adas/component/common",
        "//projects/adas/algorithm:algorithm_sdk",
    ],
)

cpplint()
