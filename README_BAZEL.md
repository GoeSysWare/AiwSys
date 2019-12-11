# BAZEL 编译规则

# 常用命令
    bazel clean 
    bazel clean --expunge
    bazel run 
    bazel build

# .bazel.rc 文件
    参考[https://docs.bazel.build/versions/master/guide.html]
    build的参数文件，替代在命令行中输入，也可以重命名参数来缩短命令行的输入
    不同版本默认的位置不一样:
        0.28.0 默认是{workspaceroot},名字是".bazelrc",
        以前的版本是{workspaceroot} 名字是"bazel.rc"
    
    也可以通过参数--bazelrc=file
    
# 参数解释
    --output_filter   这个是对编译的警告进行过滤，regex过滤
    
    --spawn_strategy=s tandalone   
                                    这个参数是将沙盒编译(默认)还是独立子进程编译方式，
                                    注意如果是沙盒编译，所有的头文件必须在 srcs hdrs中指定，否则会报找不到头文件
    --copt                          等同C/C++ 编译时gcc的参数
    --cpu                           =k8|arm
    --jobs                          
    --sandbox_debug =1  编译信息全输出
                                         =0
    --compilation_mode (fastbuild|opt|dbg) (-c)   
                                    编译为release 还是debug


    