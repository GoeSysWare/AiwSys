package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "osqp",
    include_prefix = "osqp",
    includes = [
        "include",
    ],
    # linkopts = [
    #     "-Lbuild/out",
    #     "-Wl,-rpath,/usr/lib/x86_64-linux-gnu/",
    #     "-losqp",
    # ],
    linkopts = [
        "-Lbuild/out",
        "-Wl,-rpath,",
        "-losqp,-lqdldla",
    ],
)