load("//tools:cpplint.bzl", "cpplint")
load("//tools:cuda_library.bzl", "cuda_library","cuda_binary")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "test_cuda_lib",
    srcs = [
        "test_CUDA.cu",
    ],
    hdrs = [
    ],
    deps = [
        "@cuda",
    ],
)
cuda_binary(
    name = "test_cuda",
    deps = [
        ":test_cuda_lib"
    ],
)