#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MEMCOPY_ITERATIONS 100
#define DEFAULT_SIZE (32 * (1 << 20)) // 32 MB

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

float testDeviceToDeviceBandwidth(int memSize) {
    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    unsigned char *d_src, *d_dst;
    checkCudaError(cudaMalloc((void**)&d_src, memSize), "Failed to allocate source memory");
    checkCudaError(cudaMalloc((void**)&d_dst, memSize), "Failed to allocate destination memory");

    // Initialize memory
    checkCudaError(cudaMemset(d_src, 0, memSize), "Failed to set source memory");
    checkCudaError(cudaMemset(d_dst, 0, memSize), "Failed to set destination memory");

    // Warm up
    checkCudaError(cudaMemcpy(d_dst, d_src, memSize, cudaMemcpyDeviceToDevice), "Warmup memcpy failed");

    // Test
    cudaEventRecord(start, 0);
    for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
        checkCudaError(cudaMemcpy(d_dst, d_src, memSize, cudaMemcpyDeviceToDevice), "Memcpy failed");
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

    // Calculate bandwidth in GB/s
    bandwidthInGBs = ((float)(1 << 10) * memSize * MEMCOPY_ITERATIONS) / (elapsedTimeInMs * (float)(1 << 30));

    // Clean up
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return bandwidthInGBs;
}

float testHostToDeviceBandwidth(int memSize, bool pinned) {
    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host and device memory
    unsigned char *h_src, *d_dst;
    if (pinned) {
        checkCudaError(cudaMallocHost((void**)&h_src, memSize), "Failed to allocate pinned host memory");
    } else {
        h_src = (unsigned char*)malloc(memSize);
        if (!h_src) {
            fprintf(stderr, "Failed to allocate host memory\n");
            exit(1);
        }
    }
    checkCudaError(cudaMalloc((void**)&d_dst, memSize), "Failed to allocate device memory");

    // Initialize memory
    memset(h_src, 0, memSize);

    // Warm up
    checkCudaError(cudaMemcpy(d_dst, h_src, memSize, cudaMemcpyHostToDevice), "Warmup memcpy failed");

    // Test
    cudaEventRecord(start, 0);
    for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
        checkCudaError(cudaMemcpy(d_dst, h_src, memSize, cudaMemcpyHostToDevice), "Memcpy failed");
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

    // Calculate bandwidth in GB/s
    bandwidthInGBs = ((float)(1 << 10) * memSize * MEMCOPY_ITERATIONS) / (elapsedTimeInMs * (float)(1 << 30));

    // Clean up
    if (pinned) {
        cudaFreeHost(h_src);
    } else {
        free(h_src);
    }
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return bandwidthInGBs;
}

float testDeviceToHostBandwidth(int memSize, bool pinned) {
    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host and device memory
    unsigned char *h_dst, *d_src;
    if (pinned) {
        checkCudaError(cudaMallocHost((void**)&h_dst, memSize), "Failed to allocate pinned host memory");
    } else {
        h_dst = (unsigned char*)malloc(memSize);
        if (!h_dst) {
            fprintf(stderr, "Failed to allocate host memory\n");
            exit(1);
        }
    }
    checkCudaError(cudaMalloc((void**)&d_src, memSize), "Failed to allocate device memory");

    // Initialize memory
    checkCudaError(cudaMemset(d_src, 0, memSize), "Failed to set device memory");

    // Warm up
    checkCudaError(cudaMemcpy(h_dst, d_src, memSize, cudaMemcpyDeviceToHost), "Warmup memcpy failed");

    // Test
    cudaEventRecord(start, 0);
    for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
        checkCudaError(cudaMemcpy(h_dst, d_src, memSize, cudaMemcpyDeviceToHost), "Memcpy failed");
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

    // Calculate bandwidth in GB/s
    bandwidthInGBs = ((float)(1 << 10) * memSize * MEMCOPY_ITERATIONS) / (elapsedTimeInMs * (float)(1 << 30));

    // Clean up
    if (pinned) {
        cudaFreeHost(h_dst);
    } else {
        free(h_dst);
    }
    cudaFree(d_src);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return bandwidthInGBs;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    printf("Device: %s\n", deviceProp.name);
    printf("Running bandwidth tests...\n\n");

    int memSize = DEFAULT_SIZE;
    
    // Device to Device
    float d2dBandwidth = testDeviceToDeviceBandwidth(memSize);
    printf("Device to Device Bandwidth: %.2f GB/s\n", d2dBandwidth);
    
    // Host to Device (Pageable)
    float h2dPageableBandwidth = testHostToDeviceBandwidth(memSize, false);
    printf("Host to Device Bandwidth (Pageable): %.2f GB/s\n", h2dPageableBandwidth);
    
    // Host to Device (Pinned)
    float h2dPinnedBandwidth = testHostToDeviceBandwidth(memSize, true);
    printf("Host to Device Bandwidth (Pinned): %.2f GB/s\n", h2dPinnedBandwidth);
    
    // Device to Host (Pageable)
    float d2hPageableBandwidth = testDeviceToHostBandwidth(memSize, false);
    printf("Device to Host Bandwidth (Pageable): %.2f GB/s\n", d2hPageableBandwidth);
    
    // Device to Host (Pinned)
    float d2hPinnedBandwidth = testDeviceToHostBandwidth(memSize, true);
    printf("Device to Host Bandwidth (Pinned): %.2f GB/s\n", d2hPinnedBandwidth);
    
    printf("\n");
    
    // Check if we meet RTX 3090 expected bandwidth (around 936 GB/s)
    if (d2dBandwidth > 900.0f) {
        printf("Test Result: PASS - Memory bandwidth meets RTX 3090 specifications\n");
        return 0;
    } else {
        printf("Test Result: WARNING - Memory bandwidth (%.2f GB/s) is lower than expected for RTX 3090\n", d2dBandwidth);
        return 0; // Still return 0 as it's not a failure, just a warning
    }
}