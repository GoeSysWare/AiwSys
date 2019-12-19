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

# [WORKSPACE规则(0.28.0)](https://docs.bazel.build/versions/0.28.0/be/workspace.html)
- [依赖bazel工程](https://docs.bazel.build/versions/0.28.0/repo/git.html)
    + [local_repository](https://docs.bazel.build/versions/master/be/workspace.html#local_repository)  
        *path 支持文件相对/绝对目录
    + [git_repository](https://docs.bazel.build/versions/0.28.0/repo/git.html#git_repository)
    + [http_jar](https://docs.bazel.build/versions/1.0.0/repo/http.html#http_jar)
    + [http_file](https://docs.bazel.build/versions/1.0.0/repo/http.html#http_file)
    + [http_archive](https://docs.bazel.build/versions/1.0.0/repo/http.html#http_archive)
        *urls 支持多个文件(tar.gz |zip)
        *url  支持一个(tar.gz |zip)
        *strip_prefix 跳过可能的解压缩重名
-  依赖非bazel工程
    + [new_local_repository](https://docs.bazel.build/versions/master/be/workspace.html#new_local_repository)
    + [new_git_repository](https://docs.bazel.build/versions/0.28.0/repo/git.html#new_git_repository)