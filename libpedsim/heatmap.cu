#include "ped_model.h"
#include "ped_agent.h"
#include "cuda_runtime.h"
#include "ped_agents.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cmath>

using namespace std;


__global__ void scaleHeatmapKernel(const int *d_heatmap, int *d_scaled, int size, int cellsize) {
    int scaled_size = size * cellsize;
    int scaled_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = scaled_size * scaled_size;
    
    if (scaled_index < total_elements) {
        // Determine (x, y) coordinate in the scaled heatmap
        int scaled_y = scaled_index / scaled_size;
        int scaled_x = scaled_index % scaled_size;
        
        // Map back to the original heatmap coordinate
        int orig_x = scaled_x / cellsize;
        int orig_y = scaled_y / cellsize;
        int orig_index = orig_y * size + orig_x;
        
        // Assign the value from the original heatmap to the scaled heatmap
        d_scaled[scaled_index] = d_heatmap[orig_index];
    }
}

__global__ void blurHeatmapKernel(const int *d_scaled_heatmap, int *d_blurred_heatmap, int width, const int *d_weights) {
    __shared__ int tile[32][32];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 2;
    int ty = threadIdx.y + 2;

    if (x < width && y < width) {
        // Load central pixel
        tile[ty][tx] = d_scaled_heatmap[y * width + x];

        // Load halos
        if (threadIdx.x < 2 && x >= 2)
            tile[ty][tx - 2] = d_scaled_heatmap[y * width + x - 2];
        if (threadIdx.x >= blockDim.x - 2 && x + 2 < width)
            tile[ty][tx + 2] = d_scaled_heatmap[y * width + x + 2];
        if (threadIdx.y < 2 && y >= 2)
            tile[ty - 2][tx] = d_scaled_heatmap[(y - 2) * width + x];
        if (threadIdx.y >= blockDim.y - 2 && y + 2 < width)
            tile[ty + 2][tx] = d_scaled_heatmap[(y + 2) * width + x];
    }

    __syncthreads();

    if (x >= 2 && y >= 2 && x < width - 2 && y < width - 2) {
        int sum = 0;
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                sum += d_weights[(ky + 2) * 5 + (kx + 2)] * tile[ty + ky][tx + kx];
            }
        }
        d_blurred_heatmap[y * width + x] = 0x00FF0000 | ((sum / 273) << 24);
    }
}


// Error checking for CUDA calls
inline void safe_call(cudaError_t err) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void initializeHeatmap(int *hm, int *shm, int *bhm, int length, int scaled_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        hm[idx] = 0;
    }
    if (idx < scaled_length) {
        shm[idx] = 0;
        bhm[idx] = 0;
    }
}

void Ped::Model::setupHeatmap() {
    int *d_hm, *d_shm, *d_bhm;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int length = SIZE * SIZE;
    int scaled_length = SCALED_SIZE * SCALED_SIZE;

    safe_call(cudaMallocAsync(&d_hm, length * sizeof(int), stream));
    safe_call(cudaMallocAsync(&d_shm, scaled_length * sizeof(int), stream));
    safe_call(cudaMallocAsync(&d_bhm, scaled_length * sizeof(int), stream));

    int threadsPerBlock = THREADSPERBLOCK;
    int totalElements = max(length, scaled_length);
    int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    initializeHeatmap<<<numBlocks, threadsPerBlock, 0, stream>>>(d_hm, d_shm, d_bhm, length, scaled_length);

    int *hm = (int*)malloc(length * sizeof(int));
    int *shm = (int*)malloc(scaled_length * sizeof(int));
    int *bhm = (int*)malloc(scaled_length * sizeof(int));

    heatmap = (int**)malloc(SIZE * sizeof(int*));
    scaled_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));
    blurred_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));

    safe_call(cudaMemcpyAsync(hm, d_hm, length * sizeof(int), cudaMemcpyDeviceToHost, stream));
    safe_call(cudaMemcpyAsync(shm, d_shm, scaled_length * sizeof(int), cudaMemcpyDeviceToHost, stream));
    safe_call(cudaMemcpyAsync(bhm, d_bhm, scaled_length * sizeof(int), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    for (int i = 0; i < SIZE; i++) {
        heatmap[i] = hm + SIZE * i;
    }
    for (int i = 0; i < SCALED_SIZE; i++) {
        scaled_heatmap[i] = shm + SCALED_SIZE * i;
        blurred_heatmap[i] = bhm + SCALED_SIZE * i;
    }

    safe_call(cudaFreeAsync(d_hm, stream));
    safe_call(cudaFreeAsync(d_shm, stream));
    safe_call(cudaFreeAsync(d_bhm, stream));

    cudaStreamDestroy(stream);
}

void Ped::Model::scaleHeatmapCUDA() {
    int numOrig = SIZE * SIZE;
    int numScaled = SCALED_SIZE * SCALED_SIZE;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int *hm_flat = (int*)malloc(numOrig * sizeof(int));
    for (int i = 0; i < SIZE; i++) {
        memcpy(hm_flat + i * SIZE, heatmap[i], SIZE * sizeof(int));
    }

    int *d_heatmap, *d_scaled;
    safe_call(cudaMallocAsync(&d_heatmap, numOrig * sizeof(int), stream));
    safe_call(cudaMallocAsync(&d_scaled, numScaled * sizeof(int), stream));

    safe_call(cudaMemcpyAsync(d_heatmap, hm_flat, numOrig * sizeof(int), cudaMemcpyHostToDevice, stream));

    int threadsPerBlock = THREADSPERBLOCK;
    int blocks = (numScaled + threadsPerBlock - 1) / threadsPerBlock;
    scaleHeatmapKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_heatmap, d_scaled, SIZE, CELLSIZE);

    int *shm_flat = (int*)malloc(numScaled * sizeof(int));
    safe_call(cudaMemcpyAsync(shm_flat, d_scaled, numScaled * sizeof(int), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    for (int i = 0; i < SCALED_SIZE; i++) {
        scaled_heatmap[i] = shm_flat + i * SCALED_SIZE;
    }

    free(hm_flat);
    safe_call(cudaFreeAsync(d_heatmap, stream));
    safe_call(cudaFreeAsync(d_scaled, stream));

    cudaStreamDestroy(stream);
}

void Ped::Model::applyBlurFilterCUDA() {
    int numElements = SCALED_SIZE * SCALED_SIZE;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int *d_scaled_heatmap, *d_blurred_heatmap, *d_weights;
    safe_call(cudaMallocAsync(&d_scaled_heatmap, numElements * sizeof(int), stream));
    safe_call(cudaMallocAsync(&d_blurred_heatmap, numElements * sizeof(int), stream));
    safe_call(cudaMallocAsync(&d_weights, 25 * sizeof(int), stream));

    int weights[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };

    safe_call(cudaMemcpyAsync(d_weights, weights, 25 * sizeof(int), cudaMemcpyHostToDevice, stream));

    dim3 blockDim(16, 16);
    dim3 gridDim((SCALED_SIZE + blockDim.x - 1) / blockDim.x, (SCALED_SIZE + blockDim.y - 1) / blockDim.y);
    blurHeatmapKernel<<<gridDim, blockDim, 0, stream>>>(d_scaled_heatmap, d_blurred_heatmap, SCALED_SIZE, d_weights);

    int *h_blurred_heatmap = new int[numElements];
    safe_call(cudaMemcpyAsync(h_blurred_heatmap, d_blurred_heatmap, numElements * sizeof(int), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    for (int i = 0; i < SCALED_SIZE; i++) {
        memcpy(blurred_heatmap[i], h_blurred_heatmap + i * SCALED_SIZE, SCALED_SIZE * sizeof(int));
    }
    

    delete[] h_blurred_heatmap;
    safe_call(cudaFreeAsync(d_scaled_heatmap, stream));
    safe_call(cudaFreeAsync(d_blurred_heatmap, stream));
    safe_call(cudaFreeAsync(d_weights, stream));

    cudaStreamDestroy(stream);
}
void Ped::Model::updateHeatmap() {
    // Your CUDA-specific update logic here
    scaleHeatmapCUDA();
    applyBlurFilterCUDA();
}
