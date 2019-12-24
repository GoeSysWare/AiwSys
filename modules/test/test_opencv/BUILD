load("//tools:cpplint.bzl", "cpplint")
load("//tools:qt.bzl", "qt_cc_library")

package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "test_opencv",
    deps = [":opencv_lib"],
)

cc_library(
    name = "opencv_lib",
    srcs = [
        "test_opencv.cpp",
    ],
    hdrs = [
    ],
    copts = [
    ],
    deps = [
        "@opencv",
    ],
)



cpplint()