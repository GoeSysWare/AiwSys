workspace(name = "AiwSys")
#########################################################################################
#bazel 编译的基础工具链
#########################################################################################
#加载bazel自带的工具链,这样可以运行http_archive，git_repository等函数
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive","http_file","http_jar")
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository',"new_git_repository")

#对自带可以bazel构建的开源包，压缩文件就可以，使用时直接调用http_archive/local_repository规则即可
# strip_prefix 关键字可以解决解压时会发生两层文件名情况
# 本地非bazel构建的库，需要自己定义BUILD文件，然后利用bazel规则，像make文件一样去编写
#  注意new_local_repository 是当地文件夹，不是tar.gz压缩文件
#########################################################################################
#bazel 编译的语言支持
#########################################################################################
local_repository(
    name = "bazel_skylib",
    path=  "../3rd/bazel-skylib-1.0.2",
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()


#rules_cc
local_repository(
    name = "rules_cc",
    path=  "../3rd/rules_cc-master",
)
load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")
rules_cc_dependencies()
load("@rules_cc//cc:defs.bzl", "cc_library")


#rules_python
local_repository(
    name = "rules_python",
    path=  "../3rd/rules_python-master",
)

# This call should always be present.
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()
# This one is only needed if you're using the packaging rules.
load("@rules_python//python:pip.bzl", "pip_repositories")
pip_repositories()

load("@rules_python//python:defs.bzl", "py_library", "py_test")

# python
new_local_repository(
    name = "python36",
    build_file = "third_party/python36.BUILD",
    path = "/usr",
)



#Go
local_repository(
    name = "io_bazel_rules_go",
    path=  "../3rd/rules_go-master",
)
load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies", "go_register_toolchains")
go_rules_dependencies()
go_register_toolchains()
load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_binary","go_test","go_source")




#rules_java
local_repository(
    name = "rules_java",
    path=  "../3rd/rules_java-master",
)
load("@rules_java//java:repositories.bzl", "rules_java_dependencies", "rules_java_toolchains")
rules_java_dependencies()
rules_java_toolchains()
load("@rules_java//java:defs.bzl", "java_library")


# qt语言 bazel原生没有qt的生成规则，需要自定义
new_local_repository(
    name = "qt",
    build_file = "third_party/qt.BUILD",
    path = "/opt/Qt5.5.1/5.5/gcc_64",
)


#########################################################################################
# 开发应用的通用基础库如glog gflag  gtest protobuf
#########################################################################################
new_local_repository(
    name = "zlib",
    build_file = "third_party/zlib.BUILD",
    path=  "../3rd/zlib-1.2.11",
)

# googletest (GTest and GMock)

local_repository(
    name = "gtest",
    path=  "../3rd/googletest-release-1.8.1",
)

# gflags
local_repository(
    name = "com_github_gflags_gflags",
    path=  "../3rd/gflags-2.2.2",
)
#bind是用来重命名的，使用时必须使用"//external:"规则使用
bind(
    name = "gflags",
    actual= "@com_github_gflags_gflags//:gflags",
)

# glog

local_repository(
    name = "glog",
    path=  "../3rd/glog-0.4.0",
)

# 本地非bazel构建的库，需要自己定义BUILD文件，然后利用bazel规则，像make文件一样去编写
#  注意new_local_repository 是当地文件夹，不是tar.gz压缩文件
# cpplint from google style guide
new_local_repository(
    name = "google_styleguide",
    build_file = "third_party/google_styleguide.BUILD",
    path = "../3rd/styleguide-gh-pages"
)


# protobuf

# This com_google_protobuf repository is required for proto_library rule.
# It provides the protocol compiler binary (i.e., protoc).
local_repository(
    name = "com_google_protobuf",
    path=  "../3rd/protobuf-3.9.1",
)
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps",)
protobuf_deps()

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library","py_proto_library")

# This com_google_protobuf_cc repository is required for cc_proto_library
# rule. It provides protobuf C++ runtime. Note that it actually is the same
# repo as com_google_protobuf but has to be given a different name as
# required by bazel.
# local_repository(
#     name = "com_google_protobuf_cc",
#     path=  "../3rd/protobuf-3.9.1",
# )

# # Similar to com_google_protobuf_cc but for Java (i.e., java_proto_library).
# local_repository(
#     name = "com_google_protobuf_java",
#     path=  "../3rd/protobuf-3.9.1",
# )


# Similar to com_google_protobuf_cc but for Java lite. If you are building
# for Android, the lite version should be prefered because it has a much
# smaller code size.
# "https://github.com/protocolbuffers/protobuf/archive/javalite.zip"
# local_repository(
#     name = "com_google_protobuf_javalite",
#     path=  "../3rd/protobuf-javalite",
# )


#########################################################################################
# 开发应用的功能基础库、框架、中间件
#########################################################################################

#BOOST
new_local_repository(
    name = "boost",
    build_file = "third_party/boost.BUILD",
    path = "../3rd/boost_1_70_0",
)
#FastRTPS 1.5
new_local_repository(
    name = "fastrtps",
    build_file = "third_party/fastrtps.BUILD",
    path = "../3rd/Fast-RTPS",
)

#  pocoproject / poco , 自己添加的
new_local_repository(
    name = "poco",
    build_file = "third_party/poco.BUILD",
    path = "../3rd/poco-1.9.4/build/install"
)
# Curl-CPP
new_local_repository(
    name = "curlpp",
    build_file = "third_party/curlpp.BUILD",
    path = "../3rd/curlpp-0.8.1"
)

# YAML-CPP
#  https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.5.3.zip
new_local_repository(
    name = "yaml_cpp",
    build_file = "third_party/yaml_cpp.BUILD",
    path = "../3rd/yaml-cpp-0.6.3/",
)

# CivetWeb (web server)
new_local_repository(
    name = "civetweb",
    build_file = "third_party/civetweb.BUILD",
    path = "../3rd/civetweb-1.11",
)

#########################################################################################
# 图像处理、点云功能库
#########################################################################################
# PCL 1.9
# =======
# This requires libpcl-dev to be installed in your Ubuntu/Debian.
new_local_repository(
    name = "pcl",
    build_file = "third_party/pcl.BUILD",
    path = "../3rd/pcl-1.9.0/build/install",
)
# VTK 8.2.0 
new_local_repository(
    name = "vtk",
    build_file = "third_party/vtk.BUILD",
    path = "../3rd/VTK-8.2.0/build/install",
)
#CGAL 4.14.1
new_local_repository(
    name = "cgal",
    build_file = "third_party/cgal.BUILD",
    path = "../3rd/CGAL-4.14.1/build/install",
)

# opencv  3.4
# =======
# This requires libpcl-dev to be installed in your Ubuntu/Debian.
new_local_repository(
    name = "opencv",
    build_file = "third_party/opencv.BUILD",
    path = "../3rd/opencv-3.4.0/build/install",
)

#########################################################################################
# 数学计算库
#########################################################################################
# eigen https://github.com/eigenteam/eigen-git-mirror.git
new_local_repository(
    name = "eigen",
    build_file = "third_party/eigen.BUILD",
    path = "../3rd/eigen-git-mirror-3.2.10",
)
# OSQP 算子分解QP解算  https://github.com/oxfordcontrol/osqp.git
new_local_repository(
    name = "osqp",
    build_file = "third_party/osqp.BUILD",
    path = "../3rd/osqp-0.6.0/",
)
# qpOASES来源于  https://www.coin-or.org/download/source/qpOASES/qpOASES-3.2.1.zip ，一种控制策略算法
new_local_repository(
    name = "qpOASES",
    build_file = "third_party/qpOASES.BUILD",
    # strip_prefix = "qpOASES-3.2.1",
    path = "../3rd/qpOASES-3.2.1",
)
# Proj.4
new_local_repository(
    name = "proj4",
    build_file = "third_party/proj4.BUILD",
    path = "../3rd/PROJ-4.9.3",
)

#########################################################################################
# 人工智能框架和工具
#########################################################################################
# Cuda
new_local_repository(
    name = "cuda",
    build_file = "third_party/cuda.BUILD",
    path = "/usr/local/cuda",
)
# Caffe
new_local_repository(
    name = "caffe",
    build_file = "third_party/caffe.BUILD",
    path = "../3rd/caffe-watrix/build/install",
)
# PyTorch
new_local_repository(
    name = "pytorch",
    build_file = "third_party/pytorch.BUILD",
    path = "../3rd/pytorch/build/install",
)

# # PyTorch GPU
# new_local_repository(
#     name = "pytorch_gpu",
#     build_file = "third_party/pytorch_gpu.BUILD",
#     path = "/usr/local/lib",
# )

# # paddlepaddle
# new_local_repository(
#     name = "paddlepaddle",
#     build_file = "third_party/paddlepaddle.BUILD",
#     path = "/usr/local/apollo/paddlepaddle",
# )

# mkldnn
new_local_repository(
    name = "mkldnn",
    build_file = "third_party/mkldnn.BUILD",
    path = "../3rd/mkl-dnn-0.14/build/install",
)
# darknet
new_local_repository(
    name = "darknet",
    build_file = "third_party/darknet.BUILD",
    path = "../3rd/darknet",
)
# # mklml
# new_local_repository(
#     name = "mklml",
#     build_file = "third_party/mklml.BUILD",
#     path = "/usr/local/apollo/local_third_party/mklml",
# )
# tensorrt
# new_local_repository(
#     name = "tensorrt",
#     build_file = "third_party/tensorrt.BUILD",
#     path = "../3rd/TensorRT-6.0.1/build/install",
# )


# #snappystream
# new_local_repository(
#     name = "snappystream",
#     build_file = "third_party/snappystream.BUILD",
#     path = "/usr/local/apollo/paddlepaddle_dep/snappystream",
# )

#########################################################################################
# 硬件驱动
#########################################################################################
# innovision公司的硬件驱动
new_local_repository(
    name = "innovision",
    build_file = "third_party/innovision.BUILD",
    path = "../3rd/innovision-1.4.1",
)


#########################################################################################
# 其他工具和框架
#########################################################################################

