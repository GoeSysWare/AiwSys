licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "boost",
    includes = [
        ".",
    ],
    
    linkopts = [
        "-Lstage/lib",
        "-lboost_system",
        "-lboost_thread",
        "-lboost_filesystem",
    ],
)
