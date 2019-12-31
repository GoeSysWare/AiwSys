licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "caffe",
    includes = [
        ".",
        "include",
    ],
    srcs = glob([
        "lib/*.so.*",
        "lib/*.so",
    ]),
    deps = [
        "@opencv",
        "@boost",
        "@cuda",
        "@com_google_protobuf//:protobuf",
    ]
)
