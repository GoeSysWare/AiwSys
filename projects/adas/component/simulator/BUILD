load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libadas_simulator_component.so",
    linkshared = True,
    linkstatic = False,
    deps = [
        ":adas_simulator_lib",
        ],
    linkopts = [
        "-pthread",
         "-L/usr/lib/x86_64-linux-gnu",
        # "-lboost_filesystem",
        # "-lboost_atomic",
         "-lboost_system",
        # "-lboost_date_time",
    ],
)

cc_library(
    name = "adas_simulator_lib",
    srcs = [
        "adas_simulator_component.cpp",
    ],
    hdrs = [
        "adas_simulator_component.h",
    ],
    copts = [
        '-DMODULE_NAME=\\"ADAS_SIM\\"',
    ],
    deps = [
        "//cyber",
        "//projects/adas/proto:adas_config_proto",
        "//projects/adas/component/common",
        "//projects/adas/configs:config_gflags",
        "@opencv",
    ],
)




cpplint()
