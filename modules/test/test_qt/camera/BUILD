

load("//tools:cpplint.bzl", "cpplint")
load("//tools:qt.bzl", "qt_cc_library")

package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "test_qt_camera",
    includes = [
        ".",
    ],
    linkopts = [
    ],
    deps = [
        ":camera",
    ],
)
qt_cc_library(
    name = "camera",
    src = glob(
        # ["*.cc"],
        ["*.cpp"],
    ),
    copts = [
    ],
    hdr = glob([
        "*.h",
    ]),
    includes = [
        ".",
    ],
    res = glob([
        "*.qrc",
    ]),
    uis = glob([
        "*.ui",
    ]),
    deps = [
        "@qt//:qt_gui",
        "@qt//:qt_quick",
        "@qt//:qt_media",
    ],
)