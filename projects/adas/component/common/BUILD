
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])



cc_library(
    name = "common",
    srcs = [
        "util.cpp",
    ],
    hdrs = [
        "util.h",
        "timer.h",
        "threadpool.h"
    ],
    deps = [
        "//cyber/common",
        "//cyber/base",       
    ],
)


cpplint()
