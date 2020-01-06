load("//tools:cpplint.bzl", "cpplint")


package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "test_pcl",
    deps = [":pcl_lib"],
)

cc_library(
    name = "pcl_lib",
    srcs = [
        "test_pcl.cpp",
    ],
    hdrs = [
    ],
    copts = [
    ],
    deps = [
        "@eigen",
        "@pcl",
    ],
)


cpplint()