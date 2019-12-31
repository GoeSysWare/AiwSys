licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "boost",
    includes = [
        ".",
    ],
    srcs = glob([
        "stage/lib/*.so.*",
        "stage/lib/*.so",
    ]),
    deps = [
        "@python36",
    ]
)
