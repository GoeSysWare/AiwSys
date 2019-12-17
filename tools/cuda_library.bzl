cuda_srcs = ([
    ".cu",
    ".cc",
    ".cpp",
])

cuda_headers = ([
    ".h",
    ".hpp",
])

cuda_arch = " ".join([
    "-arch=sm_30",
    "-gencode=arch=compute_30,code=sm_30",
    "-gencode=arch=compute_50,code=sm_50",
    "-gencode=arch=compute_52,code=sm_52",
    "-gencode=arch=compute_60,code=sm_60",
    "-gencode=arch=compute_61,code=sm_61",
    "-gencode=arch=compute_61,code=compute_61",
])

def cuda_library_impl(ctx):
    flags = ' '.join(ctx.attr.flags)
    output = ctx.outputs.out
     # .cu文件非调试模式
    # lib_flags = ["-std=c++11", "--shared", "--compiler-options -fPIC"]
    # .cu文件调试模式
    lib_flags = ["-std=c++11", "--shared", "-g","--compiler-options -fPIC"]
    args = [f.path for f in ctx.files.srcs]
    deps_flags=[]

    for f in ctx.attr.deps:
      deps_flags += f[CcInfo] .linking_context.user_link_flags
      deps_flags += ["-I" + d for d in f[CcInfo] .compilation_context.includes.to_list()]
      deps_flags += ["-I" + d for d in  f[CcInfo] .compilation_context.quote_includes.to_list()]
      deps_flags += ["-I" + d for d in  f[CcInfo] .compilation_context.system_includes.to_list()]



    ctx.actions.run_shell(
            inputs=ctx.files.srcs + ctx.files.hdrs + ctx.files.deps,
            outputs=[ctx.outputs.out],
            arguments=args,
            env={'PATH':'/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',},
            command="nvcc %s %s %s -I. %s -o %s" % (cuda_arch, ' '.join(lib_flags),  " ".join(args), " ".join(deps_flags), output.path)
     )

def cuda_binary_impl(ctx):
    flags = ' '.join(ctx.attr.flags)
    args = ctx.attr.flags + [f.path for f in ctx.files.srcs] + [f.path for f in ctx.files.hdrs]
    deps_flags=[]
    

    for f in ctx.attr.deps:
      deps_flags += f[CcInfo] .linking_context.user_link_flags
      deps_flags += ["-I" + d for d in f[CcInfo] .compilation_context.includes.to_list()]
      deps_flags += ["-I" + d for d in  f[CcInfo] .compilation_context.quote_includes.to_list()]
      deps_flags += ["-I" + d for d in  f[CcInfo] .compilation_context.system_includes.to_list()]
    output = ctx.outputs.out
    ctx.actions.run_shell(
            inputs=ctx.files.srcs + ctx.files.hdrs + ctx.files.deps,
            outputs=[ctx.outputs.out],
            arguments=args,
            env={ 'PATH':'/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin', },
            command="nvcc %s %s %s -o %s" % (cuda_arch, " ".join(args), " ".join(deps_flags), output.path),
     )
    return DefaultInfo( executable = output)

cuda_library = rule(
    attrs = {
        "hdrs": attr.label_list(allow_files = cuda_headers),
        "srcs": attr.label_list(allow_files = cuda_srcs),
        "deps": attr.label_list(allow_files = False),
        "flags": attr.label_list(allow_files = False),
    },
    outputs = {"out": "lib%{name}.so"},
    implementation = cuda_library_impl,
)

cuda_binary = rule(
    attrs = {
        "hdrs": attr.label_list(allow_files = cuda_headers),
        "srcs": attr.label_list(allow_files = cuda_srcs),
        "deps": attr.label_list(allow_files = False),
        "flags": attr.label_list(allow_files = False),
    },
    executable = True,
    outputs = {"out": "%{name}"},
    implementation = cuda_binary_impl,
)
