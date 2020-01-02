package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@rules_python//python:defs.bzl", "py_library", "py_test","py_binary")

py_library(
    name = "helloworld_lib",
    srcs = ["helloworld.py"],
)

py_binary(
    name = "helloworld",
    srcs = [
        "helloworld.py",
        "helloworld_test.py",
    ],
    main="helloworld.py"
)


py_test(
    name = "helloworld_test",
    srcs = ["helloworld_test.py"],
    deps = [":helloworld"],
    python_version = "PY3",
)
