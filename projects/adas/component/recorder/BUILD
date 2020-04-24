load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libadas_recorder_component.so",
    linkshared = True,
    linkstatic = False,
    deps = [
        ":rec_recorder_component",
        ":rec_player_component",
        ],
    linkopts = [
        "-pthread",
        "-L/usr/lib/x86_64-linux-gnu",
        "-lboost_filesystem",
        # "-lboost_atomic",
        # "-lboost_system",
        # "-lboost_date_time",
    ],
)

cc_library(
    name = "rec_recorder_component",
    srcs = [
        "adas_rec_recorder_component.cpp",
    ],
    hdrs = [
        "adas_rec_recorder_component.h",
    ],
    copts = [
        '-DMODULE_NAME=\\"ADAS_RECORDER\\"',
    ],
    deps = [
        "//cyber",
        "//cyber/parameter",
        "//cyber/proto:record_cc_proto",
        "//projects/adas/proto:adas_config_proto",
    ],
)



cc_library(
    name = "rec_player_component",
    srcs = [
        "adas_rec_player_component.cpp",
    ],
    hdrs = [
        "adas_rec_player_component.h",
    ],
    copts = [
        '-DMODULE_NAME=\\"ADAS_PLAYER\\"',
    ],
    deps = [
        "//cyber",
        "//cyber/parameter",
         "//cyber/timer",
        "//cyber/proto:record_cc_proto",
        "//projects/adas/proto:adas_config_proto",
        "//projects/adas/proto:adas_record_proto",
          ":player",
    ],
)


cc_library(
    name = "player",
    srcs = [
        "player/play_task.cc",
        "player/play_task_buffer.cc",
        "player/play_task_consumer.cc",
        "player/play_task_producer.cc",
        "player/player.cc",
    ],
    hdrs = [
        "player/play_param.h",
        "player/play_task.h",
        "player/play_task_buffer.h",
        "player/play_task_consumer.h",
        "player/play_task_producer.h",
        "player/player.h",
    ],
    deps = [
        "//cyber",
        "//cyber/common:log",
        "//cyber/proto:record_cc_proto",
        "//cyber/record:record_reader",
    ],
)

cpplint()
