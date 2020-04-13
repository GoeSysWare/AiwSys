
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "algorithm_header",
    srcs = [
                    ],
    hdrs = [
        "algorithm_shared_export.h",
        "algorithm_type.h" ,
         "algorithm.h" ,
    ],
    deps = [
      "@pcl",
      "@vtk",    
      "@opencv",    
       "@caffe",            
       "@eigen",     


    ],
)


cc_library(
    name = "algorithm_sdk",
    srcs = [
    ],
    hdrs = [

        
    ],
    deps = [
        ":algorithm_header",
       "//projects/adas/algorithm/autotrain:adas_sdk",
  

    ],
)
