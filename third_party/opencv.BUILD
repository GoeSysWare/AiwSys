licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opencv",
    includes = [
        "include",
    ],
    # hdrs = glob([
    #     "include/*.h",
    # ]),
    linkopts = [
        "-Llib",
        "-lopencv_core",
        "-lopencv_highgui",
        "-lopencv_imgproc",
        "-lopencv_calib3d",
        "-lopencv_imgcodecs",
    ],
)
