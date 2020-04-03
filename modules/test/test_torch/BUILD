load("//tools:cpplint.bzl", "cpplint")


package(default_visibility = ["//visibility:public"])


load("@rules_python//python:defs.bzl", "py_library", "py_test","py_binary")


py_binary(
    name = "test_torch_py",
    srcs = [
        "test_torch.py",
    ],
     main="test_torch.py"
)



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
