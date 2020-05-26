licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "innovision",
    includes = [
        "include",
    ],
    # 利用bazel规则链接指定的库
    srcs = glob([
          "lib/libinnolidar.so*",
        ]),
)
