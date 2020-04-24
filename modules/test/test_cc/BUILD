load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])



cc_binary(
    name = "test_thread",
    srcs = [
        "test_thread.cpp",
    ],
    deps = [
    ],
)

cc_binary(
    name = "test_thread_detach",
    srcs = [
        "test_thread_detach.cpp",
    ],
    deps = [

    ],
    linkopts = [
        "-lpthread",
    ],
)
cc_binary(
    name = "test_move",
    srcs = [
        "test_move.cpp",
    ],
    deps = [

    ],
    linkopts = [
        "-lpthread",
    ],
)




cpplint()
