package(default_visibility = ["//visibility:public"])

# 根据实际RSTP CDR的版本编译后的情况修改目录
cc_library(
    name = "fastrtps",
    srcs = glob([
        "build/src/cpp/*.so.*",
        "build/src/cpp/*.so",
        "build/external/install/lib/*.so.*",
        "build/external/install/lib/*.so",
    ]),
    includes = [
        "include",
        "build/include/fastrtps/",
        "build/include/",
        "build/external/install/include",
    ],
)