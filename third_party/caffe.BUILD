licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "caffe",
    includes = [
        "include",
    ],
    hdrs = glob([
        "caffe/*",
        "caffe/**/*.h",
    ]),
    # 利用bazel规则链接全部vtk的库
    # srcs = glob([
    #     "lib/*.so.*",
    #     "lib/*.so",
    # ]),
    # 利用bazel规则链接指定的库
    srcs = glob([
          "lib/*.so",
        "lib/*.so*",
        ]),
    # #链接指定的库
    # linkopts = [
    #     "-Llib",
    #     "-lcaffe",
    # ],
    deps = [
        # "@opencv",
        # "@boost",
        # "@cuda",
        # "@com_google_protobuf//:protobuf_headers",
    ]
)
