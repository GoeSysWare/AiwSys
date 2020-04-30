load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library","py_proto_library")

cc_binary(
    name = "py_talker",
    srcs = ["py_talker.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:unit_test_cc_proto",
        "//cyber/py_wrapper:py_node",
    ],
)

cc_binary(
    name = "py_listener",
    srcs = ["py_listener.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:unit_test_cc_proto",
        "//cyber/py_wrapper:py_node",
    ],
)

cc_binary(
    name = "py_service",
    srcs = ["py_service.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:unit_test_cc_proto",
        "//cyber/py_wrapper:py_node",
    ],
)

cc_binary(
    name = "py_record",
    srcs = ["py_record.cc"],
    deps = [
        "//cyber",
        "//cyber/proto:unit_test_cc_proto",
        "//cyber/py_wrapper:py_record",
    ],
)

py_proto_library(
    name = "unit_test_py_proto",
    deps = [
        "//cyber/proto:unit_test_proto",
    ],
)


cpplint()
