package(default_visibility = ["//visibility:public"])


cc_proto_library(
    name = "hmi_config_proto",
    deps = [
        ":hmi_config_proto_lib",
    ],
)

proto_library(
    name = "hmi_config_proto_lib",
    srcs = ["hmi_config.proto"],
)

cc_proto_library(
    name = "hmi_mode_proto",
    deps = [
        ":hmi_mode_proto_lib",
    ],
)

proto_library(
    name = "hmi_mode_proto_lib",
    srcs = ["hmi_mode.proto"],
    deps = [
        "//cyber/proto:scheduler_conf_proto",
    ],
)

cc_proto_library(
    name = "hmi_status_proto",
    deps = [
        ":hmi_status_proto_lib",
    ],
)

proto_library(
    name = "hmi_status_proto_lib",
    srcs = ["hmi_status.proto"],
    deps = [
        "//modules/common/proto:header_proto_lib",
        "//modules/monitor/proto:system_status_proto_lib",
    ],
)