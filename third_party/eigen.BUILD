package(default_visibility = ["//visibility:public"])

licenses([
    # Note: Eigen is an MPL2 library that includes GPL v3 and LGPL v2.1+ code.
    #       We've taken special care to not reference any restricted code.
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["COPYING.MPL2"])

cc_library(
    name = "eigen",
    hdrs = glob([
        "Eigen/*",
        "Eigen/**/*.h",
        "unsupported/Eigen/*",
        "unsupported/Eigen/**/*.h",
    ]),
    defines = ["EIGEN_MPL2_ONLY"],
    includes = ["."],
)