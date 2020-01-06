licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "boost",
    includes = [
        ".",
    ],
    # 链接全部的库
    # srcs = glob([
    #     "stage/lib/*.so.*",
    #     "stage/lib/*.so",
    # ]),
    deps = [
        "@python36",
    ],

    # 利用bazel规则链接指定的库
    srcs = glob([
        "stage/lib/libboost_filesystem.so*",
        "stage/lib/libboost_atomic.so*",
        "stage/lib/libboost_system.so*",
    ]),

    # 利用gcc规则链接指定的库
    # linkopts = [
    #     "-Wl,-rpath,/usr/lib/x86_64-linux-gnu/",
    #     "-lboost_filesystem",
    #     "-lboost_atomic",
    # ],
)
