package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "mklml",
    # srcs = [
    #     "lib/libiomp5.so",
    #     "lib/libmklml_intel.so",
    # ],
    srcs = glob([
        "lib/*.so.*",
        "lib/*.so",
    ]),
    hdrs = glob([
        "include/*.h",
    ]),
)
