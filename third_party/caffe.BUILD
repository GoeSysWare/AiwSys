licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "caffe",
    includes = [
        ".",
        "include",
    ],
    linkopts = [
        "-lpthread",
        "-lblas",
        "-lcblas",
        "-lhdf5_hl",
        "-lhdf5",
        "-lz",
        "-ldl",
        "-lm",
        "-Llib",
        "-lcaffe",
    ],
    deps = [
        "@opencv",
        "@boost"
    ]
)
