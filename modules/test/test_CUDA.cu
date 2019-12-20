#include <stdio.h>
#include <iostream>

__global__ void add(float * x, float *y, float * z, int n){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        
        for (int i = index; i < n; i += stride){
                z[i] = x[i] + y[i];
        }
}

int main(){

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        for(int i=0;i<deviceCount;i++)
        {
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, i);
            std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
            std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
            std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
            std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
            std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
            std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
            std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
            std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
            std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
            std::cout << "======================================================" << std::endl;     
            
        }

        int N = 1 << 20;
        int nBytes = N * sizeof (float);
        float *x, *y, *z;
        // CPU端分配内存
        x = (float*)malloc(nBytes);
        y = (float*)malloc(nBytes);
        z = (float*)malloc(nBytes);
        // 初始化数组
        for (int i = 0; i < N; i++){
                x[i] = 10.0;
                y[i] = 20.0;
        }

        float *d_x, *d_y, *d_z;
         // GPU端分配内存
        cudaMalloc((void**)&d_x, nBytes);
        cudaMalloc((void**)&d_y, nBytes);
        cudaMalloc((void**)&d_z, nBytes);

        // CPU的数据拷贝到GPU端
        cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
        
            // 定义kernel执行配置，（N/256）个block，每个block里面有256个线程

        dim3 blockSize(256);
        // 4096
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
        
        // 执行kernel
        add << < gridSize, blockSize >> >(d_x, d_y, d_z, N);

        // 将在GPU端计算好的结果拷贝回CPU端
        cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

        // 校验误差
        float maxError = 0.0;
        for (int i = 0; i < N; i++){
                maxError = fmax(maxError, (float)(fabs(z[i] - 30.0)));
        }
        printf ("test max default: %.4f\n", maxError);

         // 释放CPU端、GPU端的内存
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        free(x);
        free(y);
        free(z);

        return 0;
}