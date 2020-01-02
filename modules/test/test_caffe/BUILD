load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "caffe_lib",
    srcs = [
        "test_caffe.cpp",
    ],
    hdrs = [
    ],
    deps = [
        "//external:gflags",
        "@caffe",
        "@glog",
        "@opencv",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "test_caffe",
    srcs = [
    ],
    deps = [
        ":caffe_lib",
    ],
)

cpplint()
