load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "frontend",
    srcs = glob(["frontend/dist/**/*"]),
)

cc_binary(
    name = "webview",
    data = [
        ":frontend",
    ],
    deps = [
        "//projects/adas/component/webview/backend",
        "@opencv",
    ],
)

cpplint()
