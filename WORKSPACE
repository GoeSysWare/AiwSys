workspace(name = "GeoSys")
#########################################################################################
#bazel 编译的基础工具链
#########################################################################################
#加载bazel自带的工具链,这样可以运行http_archive，git_repository等函数
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')

#对自带可以bazel构建的开源包，压缩文件就可以，使用时直接调用http_archive规则即可
# strip_prefix 关键字可以解决解压时会发生两层文件名情况

#########################################################################################
#bazel 编译的语言支持
#########################################################################################

#rules_java
http_archive(
    name = "rules_java",
    strip_prefix = "rules_java-master",
    url=  "file:///home/shuimujie/01.works/3rd/rules_java-master.zip",
)
load("@rules_java//java:repositories.bzl", "rules_java_dependencies", "rules_java_toolchains")
rules_java_dependencies()
rules_java_toolchains()
load("@rules_java//java:defs.bzl", "java_library")

#rules_cc
http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-master",
    url=  "file:///home/shuimujie/01.works/3rd/rules_cc-master.zip",
)
load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")
rules_cc_dependencies()
load("@rules_cc//cc:defs.bzl", "cc_library")

#rules_python
http_archive(
    name = "rules_python",
    strip_prefix = "rules_python-master",
    url=  "file:///home/shuimujie/01.works/3rd/rules_python-master.zip",
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



#rules_go
http_archive(
    name = "io_bazel_rules_go",
    strip_prefix = "rules_go-master",
    url = "file:///home/shuimujie/01.works/3rd/rules_go-master.tar.gz",
)
load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies", "go_register_toolchains")
go_rules_dependencies()
go_register_toolchains()
load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_binary","go_test","go_source")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")


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
http_archive(
    name = "gtest",
    strip_prefix = "googletest-release-1.8.1",
    url = "file:///home/shuimujie/01.works/3rd/googletest-release-1.8.1.tar.gz",
)

# gflags
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    url = "file:///home/shuimujie/01.works/3rd/gflags-2.2.2.tar.gz",
)
#bind是用来重命名的，使用时必须使用"//external:"规则使用
bind(
    name = "gflags",
    actual= "@com_github_gflags_gflags//:gflags",
)

# glog
http_archive(
    name = "glog",
    strip_prefix = "glog-0.4.0",
    url = "file:///home/shuimujie/01.works/3rd/glog-0.4.0.tar.gz",
)

# 本地非bazel构建的库，需要自己定义BUILD文件，然后利用bazel规则，像make文件一样去编写
#  注意new_local_repository 是当地文件夹，不是tar.gz压缩文件
# cpplint from google style guide
new_local_repository(
    name = "google_styleguide",
    build_file = "third_party/google_styleguide.BUILD",
    path = "/home/shuimujie/01.works/3rd/styleguide-gh-pages"
)


# protobuf
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-3.9.1",
    url = "file:///home/shuimujie/01.works/3rd/protobuf-3.9.1.tar.gz",
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




