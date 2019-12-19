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

        int dev = 0;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, dev);
        std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;


        int N = 1 << 20;
        int nBytes = N * sizeof (float);
        float *x, *y, *z;
        x = (float*)malloc(nBytes);
        y = (float*)malloc(nBytes);
        z = (float*)malloc(nBytes);

        for (int i = 0; i < N; i++){
                x[i] = 10.0;
                y[i] = 20.0;
        }

        float *d_x, *d_y, *d_z;
        cudaMalloc((void**)&d_x, nBytes);
        cudaMalloc((void**)&d_y, nBytes);
        cudaMalloc((void**)&d_z, nBytes);

        cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
        
        dim3 blockSize(256);
        // 4096
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
        
        add << < gridSize, blockSize >> >(d_x, d_y, d_z, N);

        cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

        float maxError = 0.0;
        for (int i = 0; i < N; i++){
                maxError = fmax(maxError, (float)(fabs(z[i] - 30.0)));
        }
        printf ("test max default: %.4f\n", maxError);

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        free(x);
        free(y);
        free(z);

        return 0;
}