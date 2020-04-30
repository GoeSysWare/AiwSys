load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_proto_library(
    name = "adas_config_proto",
    deps = [
        ":adas_config_proto_lib",
    ],
)
proto_library(
    name = "adas_config_proto_lib",
    srcs = [
        "adas_config.proto",
    ],
    deps = [
        ":adas_camera_proto_lib",
        ":adas_detection_proto_lib",
        ":adas_perception_proto_lib",
        ":adas_record_proto_lib",
        ":adas_simulator_proto_lib",       
    ],
)

cc_proto_library(
    name = "adas_camera_proto",
    deps = [
        ":adas_camera_proto_lib",
    ],
)

proto_library(
    name = "adas_camera_proto_lib",
    srcs = [
        "adas_camera.proto",
    ],
    deps = [

    ],
)

cc_proto_library(
    name = "adas_record_proto",
    deps = [
        ":adas_record_proto_lib",
    ],
)
proto_library(
    name = "adas_record_proto_lib",
    srcs = [
        "adas_record.proto",
    ],
    deps = [

    ],
)

cc_proto_library(
    name = "adas_perception_proto",
    deps = [
        ":adas_perception_proto_lib",
    ],
)

proto_library(
    name = "adas_perception_proto_lib",
    srcs = [
        "adas_perception.proto",
    ],
    deps = [
    ],
)


cc_proto_library(
    name = "adas_detection_proto",
    deps = [
        ":adas_detection_proto_lib",
    ],
)
proto_library(
    name = "adas_detection_proto_lib",
    srcs = [
        "adas_detection.proto",
    ],
    deps = [
        "//modules/common/proto:error_code_proto_lib",
        "//modules/common/proto:geometry_proto_lib",
        "//modules/common/proto:header_proto_lib",
        "//modules/drivers/proto:sensor_proto_lib",

    ],
)
cc_proto_library(
    name = "adas_simulator_proto",
    deps = [
        ":adas_simulator_proto_lib",
    ],
)
proto_library(
    name = "adas_simulator_proto_lib",
    srcs = [
        "adas_simulator.proto",
    ],
    deps = [
        "//modules/common/proto:error_code_proto_lib",
        "//modules/common/proto:geometry_proto_lib",
        "//modules/common/proto:header_proto_lib",
        "//modules/drivers/proto:sensor_proto_lib",
    ],
)



# cc_proto_library(
#     name = "watrix_adas_proto",
#     deps = [
#         ":watrix_adas_proto_lib",
   
#     ],

# proto_library(
#     name = "watrix_adas_proto_lib",
#     deps = [
#         ":node_config_proto_lib",
#         ":camera_image_proto_lib",
#         ":error_code_proto_lib",
#         ":header_proto_lib",
#         ":point_cloud_proto_lib",
#         ":test_config_proto_lib",        
#     ],
# )


# cc_proto_library(
#     name = "node_config_proto",
#     deps = [
#         ":node_config_proto_lib",
#     ],
# )

# proto_library(
#     name = "node_config_proto_lib",
#     srcs = [
#         "node_config.proto",
#     ],
#     deps = [

#     ],
# )


# cc_proto_library(
#     name = "camera_image_proto",
#     deps = [
#         ":camera_image_proto_lib",
#     ],
# )

# proto_library(
#     name = "camera_image_proto_lib",
#     srcs = [
#         "camera_image.proto",
#     ],
#     deps = [
#     ],
# )
# cc_proto_library(
#     name = "error_code_proto",
#     deps = [
#         ":error_code_proto_lib",
#     ],
# )

# proto_library(
#     name = "error_code_proto_lib",
#     srcs = [
#         "error_code.proto",
#     ],
#     deps = [
#     ],
# )
# cc_proto_library(
#     name = "header_proto",
#     deps = [
#         ":header_proto_lib",
#     ],
# )

# proto_library(
#     name = "header_proto_lib",
#     srcs = [
#         "header.proto",
#     ],
#     deps = [
#     ],
# )

# cc_proto_library(
#     name = "point_cloud_proto",
#     deps = [
#         ":point_cloud_proto_lib",
#     ],
# )

# proto_library(
#     name = "point_cloud_proto_lib",
#     srcs = [
#         "point_cloud.proto",
#     ],
#     deps = [
#     ],
# )
# cc_proto_library(
#     name = "test_config_proto",
#     deps = [
#         ":test_config_proto_lib",
#     ],
# )

# proto_library(
#     name = "test_config_proto_lib",
#     srcs = [
#         "test_config.proto",
#     ],
#     deps = [
#     ],
# )

cpplint()
