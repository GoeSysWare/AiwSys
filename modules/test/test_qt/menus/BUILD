

load("//tools:cpplint.bzl", "cpplint")
load("//tools:qt.bzl", "qt_cc_library")

package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "test_qt_menus",
    includes = [
        ".",
    ],
    linkopts = [
    ],
    deps = [
        ":menus",
    ],
)
qt_cc_library(
    name = "menus",
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
    ],
)