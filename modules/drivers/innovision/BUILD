load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libinnovision_driver_component.so",
    linkopts = ["-shared"],
    linkstatic = False,
    deps = [":innovision_driver_component_lib"],
)

cc_library(
    name = "innovision_driver_component_lib",
    srcs = [
        "innovision_component.cc",
        "inno_lidar.cc", 
        ],
    hdrs = [
        "innovision_component.h",
        "inno_lidar.h", 

        ],
    copts = ['-DMODULE_NAME=\\"innovision\\"'],
    deps = [
        "//cyber",
        "//modules/common/util:message_util",
        "//modules/drivers/proto:sensor_proto",
        "//modules/drivers/innovision/proto:innolidar_proto",
        "@innovision",
    ],
)



cpplint()
