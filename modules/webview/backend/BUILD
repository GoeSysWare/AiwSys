load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "backend",
    srcs = [
        "dreamview.cc",
        "main.cc",
    ],
    hdrs = [
        "dreamview.h",
    ],
    copts = ['-DMODULE_NAME=\\"dreamview\\"'],
    deps = [
        "//modules/common/status",
        "//modules/webview/backend/common:dreamview_gflags",
        "//modules/webview/backend/handlers:image_handler",
        "//modules/webview/backend/handlers:websocket_handler",
        "@civetweb//:civetweb++",
    ],
)

cpplint()
