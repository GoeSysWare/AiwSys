load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "resource_monitor",
    srcs = ["resource_monitor.cc"],
    hdrs = ["resource_monitor.h"],
    # linkopts = [
    #     "-lboost_filesystem",
    # ],
    deps = [
        "//modules/common/util",
        "//modules/view/proto:hmi_mode_proto",
        "//modules/monitor/common:monitor_manager",
        "//modules/monitor/common:recurrent_runner",
        "//modules/monitor/software:summary_monitor",
        "@boost"
    ],
)
cpplint()
