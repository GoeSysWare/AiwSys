load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libmonitor.so",
    srcs = [
        "monitor.cc",
        "monitor.h",
    ],
    copts = ['-DMODULE_NAME=\\"monitor\\"'],
    linkshared = True,
    linkstatic = False,
    deps = [
        "//cyber",
        "//modules/common/util:message_util",
        "//modules/monitor/common:recurrent_runner",
        "//modules/monitor/hardware:resource_monitor",
        # "//modules/monitor/software:channel_monitor",
        "//modules/monitor/software:process_monitor",
        # "//modules/monitor/software:recorder_monitor",
        "//modules/monitor/software:summary_monitor",
    ],
)

cpplint()
