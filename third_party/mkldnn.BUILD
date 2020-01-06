package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "mkldnn",
    # srcs = [
    #     "lib/libmkldnn.so",
    # ],
    srcs = glob([
        "lib/*.so.*",
        "lib/*.so",
    ]),
    hdrs = glob([
        "include/*.h",
    ]),
)
