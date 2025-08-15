#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", 
               static_cast<int>(error_id), cudaGetErrorString(error_id));
        return 1;
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
        return 1;
    }

    printf("Detected %d CUDA Capable device(s)\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
               static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
               (unsigned long long)deviceProp.totalGlobalMem);
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",
               deviceProp.memoryBusWidth);
        
        if (deviceProp.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                   deviceProp.l2CacheSize);
        }

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d, %d) x %d\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);
        printf("  Total amount of constant memory:               %zu bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size (x,y,z):    (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n",
               deviceProp.memPitch);
    }

    printf("\nTest Result: PASS\n");
    return 0;
}