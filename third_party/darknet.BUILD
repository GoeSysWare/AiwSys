package(default_visibility = ["//visibility:public"])


cc_library(
    name = "darknet",
    srcs = glob([
        "build_release/*.so.*",
        "build_release/*.so",
    ]),
    includes = [
        "include/darknet"
    ],
)