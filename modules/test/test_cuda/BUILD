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

# cuda的cc_binary依赖项比较特殊，必须在srcs里，而不是deps里
cc_binary(
    name = "test_cuda",
    srcs = [
         ":cuda_lib"
    ],
)


cpplint()