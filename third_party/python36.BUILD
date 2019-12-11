package(default_visibility = ["//visibility:public"])

licenses(["notice"])
#需要安装python3-dev。每个版本的python路径不尽相同，需要根据python版本调整
cc_library(
    name = "python36",
    srcs = glob([
        "lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6.so",
    ]),
    hdrs = glob([
        "include/python3.6/*.h",
    ]),
    includes = ["include/python3.6"],
)
