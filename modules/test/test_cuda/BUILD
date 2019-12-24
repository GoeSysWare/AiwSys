load("//tools:cpplint.bzl", "cpplint")
load("//tools:cuda_library.bzl", "cuda_library","cuda_binary")
load("//tools:qt.bzl", "qt_cc_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "cuda_lib",
    srcs = [
        "test_CUDA.cu",
    ],
    hdrs = [
    ],
    deps = [
        "@cuda",
    ],
)
# cuda_binary(
#     name = "test_cuda_exe",
#     srcs = [
#         "test_CUDA.cu",
#     ],
#     hdrs = [
#     ],
#     deps = [
#         "@cuda",
#     ],
# )
cc_binary(
    name = "test_cuda_exe",
    srcs = [
         ":cuda_lib"
    ],
)


cpplint()