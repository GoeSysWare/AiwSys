package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "test_opencv",
    deps = [":main_lib"],
)

cc_library(
    name = "main_lib",
    srcs = [
        "main.cpp",
    ],
    hdrs = [
    ],
    copts = [
    ],
    linkopts = [
        "-lopencv_core",
        "-lopencv_highgui",
        "-lopencv_imgproc",
        "-lopencv_calib3d",
        "-lopencv_imgcodecs",
    ],
    deps = [
    ],
)
