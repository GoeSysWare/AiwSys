# BAZEL 编译规则

## 常用命令
    bazel clean 
    bazel clean --expunge
    bazel run 
    bazel build  
    
## 官方全命令
**analyze-profile**	Analyzes build profile data.  
**aquery**	Analyzes the given targets and queries the action graph.  
**build**	Builds the specified targets.  
**canonicalize-flags**	Canonicalizes a list of bazel options.  
**clean**	Removes output files and optionally stops the server.  
**coverage**	Generates code coverage report for specified test targets.  
**cquery**	Loads, analyzes, and queries the specified targets w/ configurations.  
**dump**	Dumps the internal state of the bazel server process.  
**fetch**	Fetches external repositories that are prerequisites to the targets.  
**help**	Prints help for commands, or the index.  
**info**	Displays runtime info about the bazel server.  
**license**	Prints the license of this software.  
**mobile-install**	Installs targets to mobile devices.  
**print_action**	Prints the command line args for compiling a file.  
**query**	Executes a dependency graph query.  
**run**	 Runs the specified target.  
**shutdown**	Stops the bazel server.  
**sync**	Syncs all repositories specified in the workspace file  
**test**	Builds and runs the specified test targets.  
**version**	Prints version information for bazel.  


常用命令
 $bazel help
                           
  用法：bazel <command> <options> ...

  可用命令：
    analyze-profile      分析构建配置文件数据。
    aquery              对分析后的操作图执行查询。
    build                构建指定的目标。

    canonicalize-flags   Canonicalize Bazel flags。
    clean                删除输出文件，并可选择停止服务器。

    cquery               执行分析后依赖图查询。

    dump                 转储Bazel服务器进程的内部状态。

    help                 打印命令或索引的帮助。
    info                 显示有关bazel服务器的运行时信息。

    fetch                获取目标的所有外部依赖项。
    mobile-install       在移动设备上安装应用程序。

    query                执行依赖关系图查询。

    run                  运行指定的目标。
    shutdown             停止Bazel服务器。
    test                 构建并运行指定的测试目标。
    version              打印Bazel的版本信息。
    
## 参数解释
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

## [.bazel.rc 文件](https://docs.bazel.build/versions/master/guide.html)
    build的参数文件，替代在命令行中输入，也可以重命名参数来缩短命令行的输入
    不同版本默认的位置不一样:
        0.28.0 默认是{workspaceroot},名字是".bazelrc",
        以前的版本是{workspaceroot} 名字是"bazel.rc"
    
    也可以通过参数--bazelrc=file
    

# [WORKSPACE规则(0.28.0)](https://docs.bazel.build/versions/0.28.0/be/workspace.html)
- [依赖bazel工程](https://docs.bazel.build/versions/0.28.0/repo/git.html)
    + [local_repository](https://docs.bazel.build/versions/master/be/workspace.html#local_repository)  
        * path 支持文件相对/绝对目录
    + [git_repository](https://docs.bazel.build/versions/0.28.0/repo/git.html#git_repository)
    + [http_jar](https://docs.bazel.build/versions/1.0.0/repo/http.html#http_jar)
    + [http_file](https://docs.bazel.build/versions/1.0.0/repo/http.html#http_file)
    + [http_archive](https://docs.bazel.build/versions/1.0.0/repo/http.html#http_archive)  
        * urls 支持多个网络/本地文件(tar.gz |zip)  
        * url  支持一个网络/本地文件(tar.gz |zip)  
        * strip_prefix 跳过可能的解压缩重名  
-  依赖非bazel工程
    + [new_local_repository](https://docs.bazel.build/versions/master/be/workspace.html#new_local_repository)
    + [new_git_repository](https://docs.bazel.build/versions/0.28.0/repo/git.html#new_git_repository)
-  deps规则  
    + @boost 为@boost//:boost的默认名，当空间与内容名一致，可以缩写  
    + bind() 重命名后必须用 //externel: 来引用  
    + protobuf 引用时，如果需要独立的库则需要用  "@com_google_protobuf//:protobuf_headers",而不是        "@com_google_protobuf//:protobuf",