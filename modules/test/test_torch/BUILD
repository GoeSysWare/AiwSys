load("//tools:cpplint.bzl", "cpplint")


package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "test_torch",
    deps = [":test_torch_lib"],
)

cc_library(
    name = "test_torch_lib",
    srcs = [
        "test_torch.cpp",
    ],
    hdrs = [
    ],
    copts = [
    ],
    deps = [
        "@pytorch",
    ],
)

cpplint()
