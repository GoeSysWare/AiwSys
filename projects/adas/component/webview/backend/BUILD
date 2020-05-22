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
    copts = ['-DMODULE_NAME=\\"ADAS_WEB\\"'],
    deps = [
        "//modules/common/status",
        "//projects/adas/component/webview/backend/common:dreamview_gflags",
        "//projects/adas/component/webview/backend/handlers:image_handler",
        "//projects/adas/component/webview/backend/handlers:websocket_handler",
        "@civetweb//:civetweb++",
    ],
)

cpplint()
