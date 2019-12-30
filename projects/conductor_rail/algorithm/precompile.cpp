


#ifdef _DEBUG


#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "glog.lib")
#pragma comment(lib, "gflags.lib")



#pragma comment(lib, "opencv_world340.lib")
//#pragma comment(lib, "opencv_imgproc310.lib")
//#pragma comment(lib, "opencv_imgcodecs310.lib")
//#pragma comment(lib, "opencv_highgui310.lib")



#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "caffe2_detectron_ops_gpu.lib")


#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cublas_device.lib")
#pragma comment(lib, "cudadevrt.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cudart_static.lib")
#pragma comment(lib, "cudnn.lib")
#pragma comment(lib, "cufft.lib")

#pragma comment(lib, "cufftw.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cusolver.lib")
#pragma comment(lib, "cudadevrt.lib")
#pragma comment(lib, "cusparse.lib")
#pragma comment(lib, "nppc.lib")
#pragma comment(lib, "nppi.lib")
#pragma comment(lib, "nppial.lib")
#pragma comment(lib, "nppicc.lib")
#pragma comment(lib, "nppicom.lib")
#pragma comment(lib, "nppidei.lib")
#pragma comment(lib, "nppif.lib")
#pragma comment(lib, "nppig.lib")
#pragma comment(lib, "nppim.lib")
#pragma comment(lib, "nppisu.lib")
#pragma comment(lib, "nppitc.lib")
#pragma comment(lib, "npps.lib")
#pragma comment(lib, "nvblas.lib")
#pragma comment(lib, "nvcuvid.lib")
#pragma comment(lib, "nvgraph.lib")
#pragma comment(lib, "nvml.lib")
#pragma comment(lib, "nvrtc.lib")
#pragma comment(lib, "OpenCL.lib")




#pragma comment(lib, "caffe.lib")
#pragma comment(lib, "caffeproto.lib")
#pragma comment(lib, "libopenblas.dll.a")
#pragma comment(lib, "caffehdf5.lib")
#pragma comment(lib, "caffehdf5_cpp.lib")
#pragma comment(lib, "caffehdf5_hl.lib")
#pragma comment(lib, "caffehdf5_hl_cpp.lib")

#pragma comment(lib, "caffezlib.lib")
#pragma comment(lib, "caffezlibstatic.lib")
#pragma comment(lib, "leveldb.lib")
#pragma comment(lib, "lmdb.lib")



#pragma comment(lib, "libboost_thread-vc140-mt-1_61.lib")
#pragma comment(lib, "libboost_python-vc140-mt-1_61.lib")

#else


#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "glog.lib")
#pragma comment(lib, "gflags.lib")



#pragma comment(lib, "opencv_world340.lib")
//#pragma comment(lib, "opencv_imgproc310.lib")
//#pragma comment(lib, "opencv_imgcodecs310.lib")
//#pragma comment(lib, "opencv_highgui310.lib")


//pytorch
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "caffe2_module_test_dynamic.lib")
#pragma comment(lib, "caffe2_detectron_ops_gpu.lib")

//cuda
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cublas_device.lib")
#pragma comment(lib, "cudadevrt.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cudart_static.lib")
#pragma comment(lib, "cufft.lib")
#pragma comment(lib, "cufftw.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cusolver.lib")
#pragma comment(lib, "cudadevrt.lib")
#pragma comment(lib, "cusparse.lib")
#pragma comment(lib, "nppc.lib")
#pragma comment(lib, "nppial.lib")
#pragma comment(lib, "nppicc.lib")
#pragma comment(lib, "nppicom.lib")
#pragma comment(lib, "nppidei.lib")
#pragma comment(lib, "nppif.lib")
#pragma comment(lib, "nppig.lib")
#pragma comment(lib, "nppim.lib")
#pragma comment(lib, "nppisu.lib")
#pragma comment(lib, "nppitc.lib")
#pragma comment(lib, "npps.lib")
#pragma comment(lib, "nvblas.lib")
#pragma comment(lib, "nvcuvid.lib")
#pragma comment(lib, "nvgraph.lib")
#pragma comment(lib, "nvml.lib")
#pragma comment(lib, "nvrtc.lib")
#pragma comment(lib, "OpenCL.lib")

//cudnn
#pragma comment(lib, "cudnn.lib")


//caffe
#pragma comment(lib, "caffe.lib")
#pragma comment(lib, "caffeproto.lib")
#pragma comment(lib, "libopenblas.dll.a")
#pragma comment(lib, "caffehdf5.lib")
#pragma comment(lib, "caffehdf5_cpp.lib")
#pragma comment(lib, "caffehdf5_hl.lib")
#pragma comment(lib, "caffehdf5_hl_cpp.lib")
#pragma comment(lib, "caffezlib.lib")
#pragma comment(lib, "leveldb.lib")
#pragma comment(lib, "lmdb.lib")



//boost
#pragma comment(lib, "libboost_thread-vc140-mt-1_61.lib")
#pragma comment(lib, "libboost_python-vc140-mt-1_61.lib")
#endif



#pragma warning(disable: 4700)