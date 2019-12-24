load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "recurrent_runner",
    srcs = ["recurrent_runner.cc"],
    hdrs = ["recurrent_runner.h"],
    deps = [
        "//cyber",
    ],
)

cc_test(
    name = "recurrent_runner_test",
    size = "small",
    srcs = ["recurrent_runner_test.cc"],
    deps = [
        ":recurrent_runner",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "monitor_manager",
    srcs = ["monitor_manager.cc"],
    hdrs = ["monitor_manager.h"],
    deps = [
        "//modules/common/adapters:adapter_gflags",
        "//modules/common/monitor_log",
        "//modules/common/util",
        "//modules/common/util:map_util",
        "//modules/view/proto:hmi_config_proto",
        "//modules/view/proto:hmi_mode_proto",
        "//modules/view/proto:hmi_status_proto",
        "//modules/monitor/proto:system_status_proto",
        ":monitor_conf",
    ],
)
cc_library(
    name = "monitor_conf",
    srcs = ["monitor_conf.cc"],
    hdrs = ["monitor_conf.h"],
    deps = [
        "//modules/common/util",
        "//modules/common/util:map_util",
        "//modules/common/util:message_util",
        "//modules/common/adapters:adapter_gflags",
        "//modules/view/proto:hmi_config_proto",
        "//modules/view/proto:hmi_mode_proto",
        "//modules/view/proto:hmi_status_proto",
        "//cyber",
        "@com_google_protobuf//:protobuf",
    ],
)

cpplint()
