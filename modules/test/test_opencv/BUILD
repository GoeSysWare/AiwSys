load("//tools:cpplint.bzl", "cpplint")


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

cc_binary(
    name = "test_video",
    srcs = [
        "test_video.cpp",
    ],
    deps = [
        "@opencv",
    ],
)


cpplint()