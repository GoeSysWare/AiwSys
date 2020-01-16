load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "registerer",
    srcs = ["registerer.cc"],
    hdrs = ["registerer.h"],
    deps = [
        "//cyber",
    ],
)

cc_test(
    name = "test_pattern",
    size = "small",
    srcs = [
        "test_pattern.cpp",
    ],
    deps = [
        ":registerer",
        "@gtest//:gtest_main",
    ],
)

cpplint()
