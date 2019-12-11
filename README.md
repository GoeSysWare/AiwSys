# AiwSys
some X things about control system 


# 构建 Cyber
构建cyber是个非常麻烦的过程，需要时间和耐心....
因为是bazel构建工具，最好需要在联网的情况下编译，这样可以自动下载构建需要的工具链
构建源码时发生的错误，绝大多数是因为缺少 'functional' 这个头文件

## C++环境
  以下是在一个最小环境Unbuntu 18.04中构建过程中需要的库,确保或安装以下库:
  + bazel 0.28.0
  + apt install g++    (7.4.0)
  + apt install autoconf automake libtool
  + java 8 
  + apt install python3-dev (3.6)
  + apt install uuid-dev
  + apt install libncurses5-dev 
  + apt install cmake (3.10.2)
  + apt install libmysqlclient-dev (5.7)
  + apt install unixodbc-dev (2.3.4)
  + apt install gdb (8.1)
  + apt install libc6-dbg
  + apt install protobuf-compiler
  + apt install libsqlite3-dev
  + apt install libcurl4-openssl-dev
  + apt install libopencv-core-dev (3.2.0)
  + apt install libavcodec-dev (7.3.4)
  + apt install libswscale-dev (7.3.4)
  + apt install libopencv-highgui-dev(3.2.0)
  + apt install mesa-common-dev （opengl）
  + apt install libgl1-mesa-dev libglu1-mesa-dev
  + apt install libpcl-dev (1.8.1)
  + apt install libpcap0.8-dev

## python 环境
  + Python 版本为3.6
  + protobuf (pip install protobuf)
## Golang 环境
  + @rules_go 里面有一个“rules_go-master\go\private\repositories.bzl”中有“org_golang_x_tools”的仓库定位为google的官方库，某些原因下，是下载不成功的，所以需要修改为github的仓库,然后打包成tar.gz，重新加载:
  ```
      _maybe(
        git_repository,
        name = "org_golang_x_tools",
        remote = "https://github.com/golang/tools.git",
        # remote = "https://go.googlesource.com/tools",
        # "latest", as of 2019-07-08
        commit = "c8855242db9c1762032abe33c2dff50de3ec9d05",
        shallow_since = "1562618051 +0000",
        patches = [
            "@io_bazel_rules_go//third_party:org_golang_x_tools-gazelle.patch",
            "@io_bazel_rules_go//third_party:org_golang_x_tools-extras.patch",
        ],
        patch_args = ["-p1"],
        # gazelle args: -go_prefix golang.org/x/tools
    )
  ```

## 构建清理
bazel clean --expunge


## 构建版本
- Release构建
  + bazel build //model/cyber/...
- Debug构建
  + bazel build --copt='-g' --strip=never //cyber/... 
## 依赖项
- glog 

- gtest  == 1.8.1
  + 需要修改BUILD文件中的的 @gtest//:gtest_main 为 @gtest//:gtest_main

- Bazel     >= 0.28.0 
  + Bazel 版本 与各个rules要匹配，否则会报错
- Fast RTPS == 1.5.0
  + Fast RTPS 是RTPS协议的一种实现，主要是订阅/发布模式的一种实现
  + 此版本必须用1.5.0才能编译通过，否则要修改源码
  + Fast RTPS 编译时依赖 Fast CDR，tinyxml2 asio 几个库，
  + 注意 编译RTPS 1.5.0 版本时，因为1.5.0版本有bug,所以需要补丁,编译参考./scripts/install_fast-rtps.sh的方法
 
- poco  == 1.9.4
  + 类似与boost的C++框架
- protobuf == 3.9.1
  + 由于 @rule_proto限制，版本大于3.8.0
  + 官方"If you're using Bazel 0.21.0 or later, the minimum Protocol Buffer version required is 3.6.1.2. See this pull request for more information."
- gflags
- python 3.6

# 运行
## 运行Cyber
  一种运行是常规方式，把生成的可执行文件和so，拷贝到一处，配置好环境LD_LIBRARY_PATH等环境再执行
  一种是利用bazel run 命令运行。bazel run 无法和vscode一起运行调试
  vscode 可以直接调试
 - 运行examples


# 构建Drivers



