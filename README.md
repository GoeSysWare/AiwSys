# AiwSys
一个集成python、c++ 、java、go等语言的开发平台

# 系统组成
- cyber          通信中间件
- modules  平台基础模块
  + common 基础库，包括数学处理、数据库连接、时间、文件处理等
  + driver 驱动结点，包括照相机、雷达、激光雷达等各类外部设备的驱动
  + percetion 感知结点，人工智能框架接口与人工智能相关的感知处理基本方法、过程和结果
  + transform  坐标变换
  + monitor 系统软硬件监测
- projects 工程项目包，利用平台其他基础库定制开发形成实际产品与项目
  + conductor_rail 三轨检测
  + adas 智能感知系统
  +  ...
# 构建 
## Cyber
构建cyber是个非常麻烦的过程，需要时间和耐心....
因为是bazel构建工具，最好需要在联网的情况下编译，这样可以自动下载构建需要的工具链
构建源码时发生的错误，绝大多数是因为缺少 'functional' 这个头文件

### C++环境
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

### python 环境
  + Python 版本为3.6
  + protobuf (pip install protobuf)
### Golang 环境
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

### 构建清理
bazel clean --expunge


### 构建版本
- Release构建
  + bazel build //model/cyber/...
- Debug构建
  + bazel build --copt='-g' --strip=never //cyber/... 
### 依赖项
- glog 

- gtest  == 1.8.1
  + 需要修改BUILD文件中的的 @gtest//:main 为 @gtest//:gtest_main

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



##  构建CPPLINT
  google_styleguide 内含cpplint.py 工具  
  此脚本目前只支持python2 ,python 官方有支持python3的cpplint 
  需要pip3 install cpplint 然后复制cpplint.py 替换google_styleguide 内的文件

## 构建Drivers


## 构建Opencv 3.4 
  - 可以源码编译
  - 提示:  
  "cudacodec/src/precomp.hpp:60:37: fatal error: dynlink_nvcuvid.h: 没有那个文件或目录#include <dynlink_nvcuvid.h>"  
    错误原因:  
    CUDA 10.0 中 Decode 模块已经被废弃：https://docs.nvidia.com/cuda/video-decoder/index.html   
    该模块和 Encode 模块将作为 NVIDIA VIDEO CODEC SDK 模块独立发行：https://developer.nvidia.com/nvidia-video-codec-sdk  
    解决方法：
    1. 根据上述链接下载安装 NVIDIA VIDEO CODEC SDK 并安装   
    或  
    2. 关闭 CMake 配置中的 BUILD_opencv_cudacodec 标签。  
  - 提示：  
    "home/fychen/install/opencv-3.2.0/modules/core/include/opencv2/core/cuda/vec_math.hpp(203):   
    error: calling a constexpr __host__ function("abs") from a __device__ function("abs") is not allowed.  
    The experimental flag '--expt-relaxed-constexpr' can be used to allow this. " 
    解决方法：
    对vec_math.hpp做如下修改(把203行和205行的 ::abs 也注释掉):  


## 构建AI框架
### 构建Caffe
  - 默认protobuf版本，caffe依赖protobuf，默认去系统路径下寻找，需要与平台使用的一致否则会报:
  "error:  This file was generated by an older version of protoc which is error This file was generated by an older version protoc"
  - 指定protobuf版本，可以修改caffe/cmake/ProtoBuf.cmake 文件(protobuf版本为3.9.1):  
      set(PROTOBUF_INCLUDE_DIR /home/shuimujie/01.works/3rd/protobuf-3.9.1/build/install/include)  
      set(PROTOBUF_LIBRARIES /home/shuimujie/01.works/3rd/protobuf-3.9.1/build/install/lib/libprotobuf.so)  
      set(PROTOBUF_PROTOC_EXECUTABLE /home/shuimujie/01.works/3rd/protobuf-3.9.1/build/install/bin/protoc)  
      #屏蔽原有的默认系统路径  
      #find_package( Protobuf REQUIRED )  
 - caffe 自定义的install_prefix 路径是在build/install下，如果需要修改到系统目录中，需要自行指定 

### 构建TensortRT
 - TensortRT 源码版本为 6.0.1
 - 构建TensortRT 依赖项:
    +  ONNX (1.5.0)
    +  cub (1.7.5)
-  指定protobuf版本，默认为3.0.0 修改cmakelist.txt 中的内容为 3.9.1


### 构建PyTorch  
- [官网下载](https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.3.1.zip)编译好的库  
  要选择cuda、python等版本(推荐)

- git后再源码编译  
  pytorch源码编译需要下载很多的依赖项，最好用git模式下载后再编译  
  git clone --recursive https://github.com/pytorch/pytorch  
  cd pytorch  
  git checkout tags/v1.2.0
  #if you are updating an existing checkout  
  git submodule sync  
  git submodule update --init --recursive  
  安装
  export CMAKE_PREFIX_PATH=${install  path}  
  python setup.py install  
**注意:**  github太慢,所有第三方库都改为gitee  

- 本地源码编译 (非常难编译)  
  'BLAS' 选项可以选择'mkl'  
1.  错误1:  
  "nvcc fatal : redefinition of argument 'std'"  
  原因:  
  CUDA_NVCC_FLAGS的编译参数中重复出现-std=c++11  
  解决方法:  
  找到报错的third_party的模块中，打开cuda相关的cmake文件，寻找 CUDA_NVCC_FLAGS 项  
  例如找到 pytorch/third_party/gloo/cmake/Cuda.cmake文件中  
  gloo_list_append_if_unique(CUDA_NVCC_FLAGS "-std=c++11"),  
  注释掉这一行，编译通过。
2. 错误2:  
"找不到'mkl_vsl.h'文件"  
原因:  
需要Intel的MKL库，MKL是CPU加速库  
解决方法:  
在intel官网下载Math Kernel Library 库，默认安装到"/opt/intel/mkl"目录  
3. 错误3:  
没有"mkldnn::batch_normalization_flag"之类的定义  
原因:  
mkl-dnn的版本过高 . pytorch 依赖 的第三方库 ideep(2.0.0) ,而ideep依赖 mkl-dnn(0.14)   
解决方法:  
删除本机内系统目录存在mkldnn相关文件  
打开 官网的ideep的git库中的mkl-dnn连接，下载对应版本的mkl-dnn
4. 错误4:  
error: pytorch/third_party/ideep/mkl-dnn/src/cpu/ref_rnn.cpp:822:29:‘void cblas_sgemm_free(float*)’ is deprecated [-Wdeprecated-declarations]  
原因:  
cmake 对gcc的配置太严格，只要是警告基本都是错误   
解决办法:  
找到ideep的settings.cmake 和 mkldnn的platform.cmake文件，找出-Werror 并删除  
5. 错误5:  
mkldnn_version_t 未定义  
原因:  
aten需要的mkldnn 版本中根ideep的版本不匹配，导致aten/src/include/ATen/Version.hpp 中的mkldnn_version_t 未定义
解决办法:  
注释掉Version.hpp中的代码
6. 错误6:  
"third_party/fbgemm/src/GenerateKernel.h:107:15: error: ‘asmjit::X86Emitter’ has not been declared"  
原因:  
asmjit 版本不对，asmjit::X86Emitter 在新版中已经换名字了  
解决办法:  
官网下载 asmjit-oldstable 的分支
7. 错误7:  
  "File "/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/pytorch/third_party/python-peachpy/peachpy/  x86_64/function.py", line 16, in <module>  
  import peachpy.x86_64.avx  
  ModuleNotFoundError: No module named 'peachpy.x86_64.avx'"  
原因:  
peachpy 版本不对，peachpy 在新版中缺少x86_64/avx.py文件
解决办法:  
官网下载 PeachPy-pre-generated 的分支

### 构建PadlePadle

- 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

- 切换到较稳定release分支下进行编译，将中括号以及其中的内容替换为目标分支名：
git checkout [分支名]
例如：
git checkout release/1.5
并且请创建并进入一个叫build的目录下：
mkdir build && cd build
- 执行cmake：
具体编译选项含义请参见编译选项表

- 对于需要编译CPU版本PaddlePaddle的用户：
 For Python2: cmake .. -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
 For Python3: cmake .. -DPY_VERSION=3.5 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
对于需要编译GPU版本PaddlePaddle的用户：(仅支持ubuntu16.04/14.04)

- 请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是ubuntu 16.04，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA官方网站:
i. wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
ii. dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
iii. sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0

- 如果您已经正确安装了nccl2，就可以开始cmake了：(For Python3: 请给PY_VERSION参数配置正确的python版本)
 For Python2: cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
 For Python3: cmake .. -DPY_VERSION=3.5 -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
注意：以上涉及Python3的命令，用Python3.5来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7

- 使用以下命令来编译：
make -j$(nproc)

- 编译成功后进入/paddle/build/python/dist目录下找到生成的.whl包： cd /paddle/build/python/dist
在当前机器或目标机器安装编译好的.whl包：
pip install -U（whl包的名字）或pip3 install -U（whl包的名字）
恭喜，至此您已完成PaddlePaddle的编译安装

# 运行
## 环境变量配置  
  "CYBER_PATH"  
  "GLOG_log_dir"  
  "CYBER_DOMAIN_ID"  
## 运行Cyber
  一种运行是常规方式，把生成的可执行文件和so，拷贝到一处，配置好环境LD_LIBRARY_PATH等环境再执行  
  一种是利用bazel run 命令运行。bazel run 无法和vscode一起运行调试  
  vscode 可以直接调试  
 - 运行examples