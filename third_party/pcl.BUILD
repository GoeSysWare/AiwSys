licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pcl",
    defines = ["PCL_NO_PRECOMPILE"],
    includes = ["include"],
    srcs = glob([
        "lib/libpcl_common.so",
        "lib/libpcl_features.so",
        "lib/libpcl_filters.so",
        "lib/libpcl_io_ply.so",
        "lib/libpcl_io.so",
        "lib/libpcl_kdtree.so",
        "lib/libpcl_keypoints.so",
        "lib/libpcl_octree.so",
        "lib/libpcl_people.so",
        "lib/libpcl_recognition.so",
        "lib/libpcl_registration.so",
        "lib/libpcl_sample_consensus.so",
        "lib/libpcl_search.so",
        "lib/libpcl_segmentation.so",
        "lib/libpcl_surface.so",                         
        "lib/libpcl_tracking.so",    
        "lib/libpcl_visualization.so",                                                                                                           
    ]),
    hdrs = glob([
    ]),


#     linkopts = [
#         "-Wl,-rpath,lib",
#         "-lpcl_common",
#         "-lpcl_features",
#         "-lpcl_filters",
#         "-lpcl_io_ply",
#         "-lpcl_io",
#         "-lpcl_kdtree",
#         "-lpcl_keypoints",
#         "-lpcl_octree",
#         "-lpcl_outofcore",
#         "-lpcl_people",
#         "-lpcl_recognition",
#         "-lpcl_registration",
#         "-lpcl_sample_consensus",
#         "-lpcl_search",
#         "-lpcl_segmentation",
#         "-lpcl_surface",
#         "-lpcl_tracking",
#         "-lpcl_visualization",
#     ],
)
