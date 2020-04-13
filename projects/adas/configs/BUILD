load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "config_gflags",
    srcs = [
        "config_gflags.cc",
    ],
    hdrs = [
        "config_gflags.h",
    ],
    deps = [
        "//external:gflags",
    ],
)

cpplint()
