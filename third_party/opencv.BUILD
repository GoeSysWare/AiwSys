licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opencv",
    # 利用bazel规则链接全部的库
    # srcs = glob([
    #     "lib/*.so.*",
    #     "lib/*.so",
    # ]),
    # hdrs = glob([
    #     "include/*.h",
    # ]),

    
    # 利用bazel规则链接指定的库
    srcs =glob( [
        "lib/libopencv_core.so*",
        "lib/libopencv_highgui.so*",
        "lib/libopencv_imgcodecs.so*",
        "lib/libopencv_imgproc.so*",
        "lib/libopencv_video.so*",
        "lib/libopencv_videoio.so*",
    ]),
    includes = [
        "include",
    ],

    # #链接指定的库
    linkopts = [
        "-Wl,-rpath=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/3rd/opencv-3.4.0/build/install/lib",
        # "-L/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/3rd/opencv-3.4.0/build/install/lib",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_imgcodecs",
        "-lopencv_video",
        "-lopencv_videoio",
    ],

)
