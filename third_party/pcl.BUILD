licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pcl",
    defines = ["PCL_NO_PRECOMPILE"],
    includes = ["include"],
    srcs = glob([
        "build/src/cpp/*.so.*",
        "build/src/cpp/*.so",
        "build/external/install/lib/*.so.*",
        "build/external/install/lib/*.so",
    ]),
    hdrs = glob([
        "Eigen/*",
        "Eigen/**/*.h",
        "unsupported/Eigen/*",
        "unsupported/Eigen/**/*.h",
    ]),

    linkopts = [
        "-Linstall/lib",
        "-lboost_system",
        "-lpcl_common",
        "-lpcl_features",
        "-lpcl_filters",
        "-lpcl_io_ply",
        "-lpcl_io",
        "-lpcl_kdtree",
        "-lpcl_keypoints",
        "-lpcl_octree",
        "-lpcl_outofcore",
        "-lpcl_people",
        "-lpcl_recognition",
        "-lpcl_registration",
        "-lpcl_sample_consensus",
        "-lpcl_search",
        "-lpcl_segmentation",
        "-lpcl_surface",
        "-lpcl_tracking",
        "-lpcl_visualization",
    ],
)
