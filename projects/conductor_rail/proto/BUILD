load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "conductor_rail_proto",
    deps = [
        ":conductor_rail_proto_lib",
    ],
)

proto_library(
    name = "conductor_rail_proto_lib",
    srcs = [
        "rail_detection.proto",
        "rail_camera.proto",
    ],
    deps = [
        "//modules/common/proto:error_code_proto_lib",
        "//modules/common/proto:geometry_proto_lib",
        "//modules/common/proto:header_proto_lib",
    ],
)




cpplint()
