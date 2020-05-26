load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "innolidar_proto",
    deps = [
        ":sensor_innolidar_proto",
    ],
)

cc_proto_library(
    name = "sensor_innolidar_proto",
    deps = [
        ":innovision_proto_lib",
    ],
)

proto_library(
    name = "innovision_proto_lib",
    srcs = [
        "config.proto",
        "inno_lidar.proto",
    ],
    deps = [
    ],
)

cpplint()
