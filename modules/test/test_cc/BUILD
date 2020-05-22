load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "test_usb_cam",
    srcs = ["test_cam.cpp"],
    deps = [
        ":usb_camera_lib",
    ],
    linkopts = [
        "-lswscale",
        "-lavformat",
        "-lavcodec",
        "-latomic",
       "-lavutil",
    ],
)



cc_library(
    name = "usb_camera_lib",
    srcs = ["usb_cam.cpp"],
    hdrs = ["usb_cam.h"],
    deps = [

    ],
)


cc_binary(
    name = "test_async",
    srcs = [
        "test_async.cpp",
    ],
    linkopts = [
        "-lpthread",
    ],
    deps = [
    ],
)


cc_binary(
    name = "test_thread",
    srcs = [
        "test_thread.cpp",
    ],
    deps = [
    ],
)

cc_binary(
    name = "test_thread_detach",
    srcs = [
        "test_thread_detach.cpp",
    ],
    deps = [

    ],
    linkopts = [
        "-lpthread",
    ],
)
cc_binary(
    name = "test_move",
    srcs = [
        "test_move.cpp",
    ],
    deps = [

    ],
    linkopts = [
        "-lpthread",
    ],
)


cc_binary(
    name = "test_string",
    srcs = [
        "test_string.cpp",
    ],
    deps = [

    ],
    linkopts = [
    ],
)



cpplint()
