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
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

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
local_repository(
    name = "com_google_protobuf",
    path=  "../3rd/protobuf-3.9.1",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps",)
protobuf_deps()

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library","py_proto_library")



#########################################################################################
# 开发应用的功能基础库、框架、中间件
#########################################################################################
#FastRTPS
new_local_repository(
    name = "fastrtps",
    build_file = "third_party/fastrtps.BUILD",
    path = "/home/shuimujie/01.works/3rd/Fast-RTPS",
)

#  pocoproject / poco , 自己添加的
new_local_repository(
    name = "poco",
    build_file = "third_party/poco.BUILD",
    path = "/home/shuimujie/01.works/3rd/poco-poco-1.9.4-release"
)
# # Curl-CPP
# new_local_repository(
#     name = "curlpp",
#     build_file = "third_party/curlpp.BUILD",
#     path = "/home/shuimujie/01.works/3rd/curlpp-0.8.1"
# )

# # YAML-CPP
# #  https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.5.3.zip
# new_local_repository(
#     name = "yaml_cpp",
#     build_file = "third_party/yaml_cpp.BUILD",
#     path = "/home/shuimujie/01.works/3rd/yaml-cpp-yaml-cpp-0.6.3",
# )
#########################################################################################
# 雷达点云功能库
#########################################################################################
# # PCL 1.9
# # =======
# # This requires libpcl-dev to be installed in your Ubuntu/Debian.
# new_local_repository(
#     name = "pcl",
#     build_file = "third_party/pcl.BUILD",
#     path = "/usr/local/include/pcl-1.9",
# )
#########################################################################################
# 数学计算库
#########################################################################################
# # eigen https://github.com/eigenteam/eigen-git-mirror.git
# new_local_repository(
#     name = "eigen",
#     build_file = "third_party/eigen.BUILD",
#     path = "/home/shuimujie/01.works/3rd/eigen-git-mirror-3.2.10",
# )
# # OSQP 算子分解QP解算  https://github.com/oxfordcontrol/osqp.git
# new_local_repository(
#     name = "osqp",
#     build_file = "third_party/osqp.BUILD",
#     path = "/home/shuimujie/01.works/3rd/osqp-0.6.0/",
# )
# # qpOASES来源于  https://www.coin-or.org/download/source/qpOASES/qpOASES-3.2.1.zip ，一种控制策略算法
# new_local_repository(
#     name = "qpOASES",
#     build_file = "third_party/qpOASES.BUILD",
#     # strip_prefix = "qpOASES-3.2.1",
#     path = "/home/shuimujie/01.works/3rd/qpOASES-3.2.1",
# )


#########################################################################################
# 人工智能框架和工具
#########################################################################################
# Cuda
new_local_repository(
    name = "cuda",
    build_file = "third_party/cuda.BUILD",
    path = "/usr/local/cuda",
)
# # Caffe
# new_local_repository(
#     name = "caffe",
#     build_file = "third_party/caffe.BUILD",
#     path = "/home/shuimujie/01.works/3rd/caffe-1.0/build/install",
# )
# # PyTorch
# new_local_repository(
#     name = "pytorch",
#     build_file = "third_party/pytorch.BUILD",
#     path = "/usr/local/lib",
# )

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

# # mkldnn
# new_local_repository(
#     name = "mkldnn",
#     build_file = "third_party/mkldnn.BUILD",
#     path = "/usr/local/apollo/local_third_party/mkldnn",
# )

# # mklml
# new_local_repository(
#     name = "mklml",
#     build_file = "third_party/mklml.BUILD",
#     path = "/usr/local/apollo/local_third_party/mklml",
# )
## tensorrt
# new_local_repository(
#     name = "tensorrt",
#     build_file = "third_party/tensorrt.BUILD",
#     path = "/usr/include/tensorrt",
# )

#########################################################################################
# 其他工具和框架
#########################################################################################
