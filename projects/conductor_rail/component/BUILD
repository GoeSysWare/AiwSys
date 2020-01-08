load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])



cc_binary(
    name = "librail_wearcomponent.so",
    linkshared = True,
    linkstatic = False,
    deps = [":rail_wear_component"],
)

cc_library(
    name = "rail_wear_component",
    srcs = [
        "wear_component.cc",
    ],
    hdrs = [
        "wear_component.h",
    ],
    copts = [
        '-DMODULE_NAME=\\"RAIL_WEAR\\"',
    ],
    deps = [
        "//cyber",
        "//modules/common/math",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:geometry_proto",
        "//modules/common/proto:header_proto",
        "//modules/common/time",
        "//modules/drivers/proto:sensor_proto",
        "//modules/perception/base",
        "//modules/perception/camera/common",
        "//projects/conductor_rail/proto:conductor_rail_proto",
        "//projects/common/inner_component_messages:inner_component_messages",
        "//projects/common/registerer:registerer",
        "@yaml_cpp//:yaml",
    ],
)


cpplint()
