// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//

#include "ped_model.h"
#include "ped_agent.h"
#include "cuda_runtime.h"
#include "ped_agents.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// #define SIZE 256
// #define SCALED_SIZE (SIZE * 2)

// CUDA kernel for scaling the heatmap.
// The input 'd_heatmap' is a 1D array representing the original heatmap of size SIZE x SIZE.
// The output 'd_scaled' is a 1D array for the scaled heatmap of size SCALED_SIZE x SCALED_SIZE,
// where SCALED_SIZE = SIZE * CELLSIZE.

inline void safe_call(cudaError_t err) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
    }
}

__global__ void fadeKernel(int *d_hm, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        d_hm[idx] = (int)round(d_hm[idx] * 0.80); //TODO rounding
        // d_heatmap[idx] = (int)(d_heatmap[idx]*0.80f+0.5f); // 0.5f is for rounding. 8.1+0.5=8.6 -> 8, but 8.6+0.5=9
    }
}

__global__ void addAgentHeatKernel(int* d_hm, int size,
    const int* d_agentDesiredX,
    const int* d_agentDesiredY,
    int numAgents) {

     // printf("CUDA_AgentHeat ");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numAgents) {
        int x = d_agentDesiredX[idx];
        int y = d_agentDesiredY[idx];
        // printf("Agent %d at (%d, %d)\n", idx, x, y);
        if (x >= 0 && x < size && y >= 0 && y < size) {
            atomicAdd(&d_hm[y*size+x], 40); //TODO checka atomicAdd om korrekt
            // printf("Heatmap[%d][%d] = %d\n", x, y, d_heatmap[y * size + x]);
        }
    }
}

__global__ void limitHeatmapValueKernel(int* d_hm, int length) {
    // printf("CUDA_Limit ");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length && d_hm[idx] > 255) {
        d_hm[idx]=255;
    }
}


__global__ void scaleHeatmapKernel(const int *d_hm, int *d_shm, int size, int cellsize) {
    int scaled_size = size * cellsize;
    int scaled_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = scaled_size * scaled_size;
    
    if (scaled_index < total_elements) {
        // Determine the (x, y) coordinate in the scaled heatmap.
        int scaled_y = scaled_index / scaled_size;
        int scaled_x = scaled_index % scaled_size;
        
        // Map back to the original heatmap coordinate.
        int orig_x = scaled_x / cellsize;
        int orig_y = scaled_y / cellsize;
        int orig_index = orig_y * size + orig_x;
        
        // Each pixel in the scaled image gets the value from the corresponding original cell.
        d_shm[scaled_index] = d_hm[orig_index];
    }
}

// 1D blur kernel: No shared memory tiles, each thread handles one pixel
__global__ void blurHeatmapKernel(const int* d_in, int* d_out, int width, const int* d_weights)
{
    // Compute the global 1D index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't read/write out of bounds
    int total = width * width;
    if (idx >= total) return;

    // Convert 1D index -> (x, y)
    int x = idx % width;
    int y = idx / width;

    // Only blur if we can safely access a 5×5 region around (x, y)
    if (x >= 2 && x < width - 2 && y >= 2 && y < width - 2) {
        int sum = 0;
        // Accumulate weighted sum from 5×5 neighborhood
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int neighborVal = d_in[(y + dy) * width + (x + dx)];
                int weight      = d_weights[(dy + 2) * 5 + (dx + 2)];
                sum += neighborVal * weight;
            }
        }
        // Store ARGB result (red channel, alpha == sum/273)
        d_out[idx] = 0x00FF0000 | ((sum / 273) << 24);
    }
}

__global__ void initializeHeatmap(int *d_hm, int *d_shm, int *d_bhm, int length, int scaled_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        d_hm[idx] = 0;
    }
    if (idx < scaled_length) {
        d_shm[idx] = 0;
        d_bhm[idx] = 0;
    }
}

void Ped::Model::setupHeatmap() {
    int *d_hm, *d_shm, *d_bhm;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int length = SIZE * SIZE;
    int scaled_length = SCALED_SIZE * SCALED_SIZE;

    // Allocate GPU memory
    safe_call(cudaMalloc((void**)&d_hm, SIZE * SIZE * sizeof(int)));
    safe_call(cudaMalloc((void**)&d_shm, SCALED_SIZE * SCALED_SIZE * sizeof(int)));
    safe_call(cudaMalloc((void**)&d_bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int)));

    // Define thread block and grid sizes
    int threadsPerBlock = THREADSPERBLOCK;
    int totalElements = max(length, scaled_length);
    int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    initializeHeatmap<<<numBlocks, threadsPerBlock>>>(d_hm, d_shm, d_bhm, length, scaled_length);
    safe_call(cudaDeviceSynchronize());
    // Allocate host memory and set up pointers
    int *hm = (int*)malloc(SIZE * SIZE * sizeof(int));
    int *shm = (int*)malloc(SCALED_SIZE * SCALED_SIZE * sizeof(int));
    int *bhm = (int*)malloc(SCALED_SIZE * SCALED_SIZE * sizeof(int));

    heatmap = (int**)malloc(SIZE * sizeof(int*));
    scaled_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));
    blurred_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));

    // Copy initialized data from GPU to CPU
    safe_call(cudaMemcpy(hm, d_hm, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    safe_call(cudaMemcpy(shm, d_shm, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    safe_call(cudaMemcpy(bhm, d_bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Set up 2D pointers
    for (int i = 0; i < SIZE; i++) {
        heatmap[i] = hm + SIZE * i;
    }
    for (int i = 0; i < SCALED_SIZE; i++) {
        scaled_heatmap[i] = shm + SCALED_SIZE * i;
        blurred_heatmap[i] = bhm + SCALED_SIZE * i;
    }

    // Free GPU memory
    safe_call(cudaFree(d_hm));
    safe_call(cudaFree(d_shm));
    safe_call(cudaFree(d_bhm));
    cudaStreamDestroy(stream);
}


void Ped::Model::updateHeatmap() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int length = SIZE * SIZE;
    int scaled_length = SCALED_SIZE * SCALED_SIZE;
    int numAgents = agents->x.size();

    size_t hmSize = length * sizeof(int);
    size_t shmSize = scaled_length * sizeof(int);
    size_t agentSize = numAgents * sizeof(int);

    // Device pointers
    int *d_hm, *d_shm, *d_bhm, *d_agentDesiredX, *d_agentDesiredY;

    // Allocate memory for integer positions
    std::vector<int> agentDesiredX(agents->desiredX.size());
    std::vector<int> agentDesiredY(agents->desiredY.size());

    // Convert float to int before copying to the GPU
    for (size_t i = 0; i < agents->desiredX.size(); i++) {
        agentDesiredX[i] = static_cast<int>(roundf(agents->desiredX[i])); // Use roundf() for better accuracy
        agentDesiredY[i] = static_cast<int>(roundf(agents->desiredY[i]));
    }

    // float *agentDesiredX = agents->desiredX.data();
    // float *agentDesiredY = agents->desiredY.data();

    int *hm = (int*)malloc(hmSize);
    for (int i = 0; i < SIZE; i++) {
        memcpy(hm + i * SIZE, heatmap[i], SIZE * sizeof(int));
    }


    int *shm = (int*)malloc(shmSize);
    int *bhm = (int*)malloc(shmSize);


    cudaMallocAsync((void**)&d_hm, hmSize, stream);
    cudaMallocAsync((void**)&d_shm, shmSize, stream);
    cudaMallocAsync((void**)&d_bhm, shmSize, stream);
    cudaMallocAsync((void**)&d_agentDesiredX, agentSize, stream);
    cudaMallocAsync((void**)&d_agentDesiredY, agentSize, stream);
    cudaStreamSynchronize(stream);



    cudaMemcpyAsync(d_hm, hm, hmSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_agentDesiredX, agentDesiredX.data(), numAgents * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_agentDesiredY, agentDesiredY.data(), numAgents * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    int threadsPerBlock = THREADSPERBLOCK; // divisible by 32 (warp size)
    int blocksForFade = (length+threadsPerBlock-1) / threadsPerBlock;
    int blocksForAgents = (numAgents+threadsPerBlock-1) / threadsPerBlock;
    int blocks = (scaled_length+threadsPerBlock-1) / threadsPerBlock;
    dim3 blockDim2D(32, 32); 
    dim3 gridDim2D((SCALED_SIZE+blockDim2D.x -1) / blockDim2D.x,(SCALED_SIZE+blockDim2D.y -1) / blockDim2D.y);
    // determine the number of grids by SCALED_SIZE/blockDim2D.x and SCALED_SIZE/blockDim2D.y
    // (SCALED_SIZE + blockDim2D.x - 1) / blockDim2D.x to allow for partial blocks
    size_t sharedMemSize = (blockDim2D.x + 4) * (blockDim2D.y + 4) * sizeof(int); // +4 for halo, 2 on each side

    fadeKernel<<<blocksForFade, threadsPerBlock, 0, stream>>>(d_hm, length);
    // for (int x = 0; x < SIZE; x++) {
    //     for (int y = 0; y < SIZE; y++) {
    //         // heat fades
    //         heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
    //     }
    // }

    addAgentHeatKernel<<<blocksForAgents, threadsPerBlock, 0, stream>>>(d_hm, SIZE, d_agentDesiredX, d_agentDesiredY, numAgents);
    // // Count how many agents want to go to each location
    // for (int i = 0; i < agents->x.size(); i++) {
    //     int x = agents->desiredX[i];
    //     int y = agents->desiredY[i];

    //     if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
    //         continue;
    //     }

    //     // Intensify heat for better color results
    //     heatmap[y][x] += 40;
    // }

    limitHeatmapValueKernel<<<blocksForFade, threadsPerBlock, 0, stream>>>(d_hm, length);
    // for (int x = 0; x < SIZE; x++) {
    //     for (int y = 0; y < SIZE; y++) {
    //         heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
    //     }
    // }

    scaleHeatmapKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_hm, d_shm, SIZE, CELLSIZE);
    // Scale the heatmap using existing CUDA scaling
    // scaleHeatmapCUDA();
    int *d_weights;

    int weights[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };

    // Flatten the weights
    int h_weights[25];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            h_weights[i * 5 + j] = weights[i][j];

    safe_call(cudaMallocAsync(&d_weights, 25 * sizeof(int), stream));
    cudaMemcpyAsync(d_weights, h_weights, 25 * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    blurHeatmapKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_shm, d_bhm, SCALED_SIZE, d_weights);
    // Apply the parallelized CUDA blur filter
    // applyBlurFilterCUDA();
    // std::cout << "HEjHEJ" << endl;
    // Copy the final heatmap and blurred heatmap back to host memory asynchronously.
    // for(int i = 0; i < SIZE; i++){
    cudaMemcpyAsync(heatmap[0], d_hm, hmSize, cudaMemcpyDeviceToHost, stream);
    // }

    // for(int i = 0; i < SCALED_SIZE; i++){
    cudaMemcpyAsync(blurred_heatmap[0], d_bhm, shmSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(scaled_heatmap[0], d_shm, shmSize, cudaMemcpyDeviceToHost, stream);
    // }
    // cudaMemcpyAsync(shm, d_shm, shmSize, cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(bhm, d_bhm, shmSize, cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(hm, d_hm, hmSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    // std::cout << "Byebye" << endl;


    // for (int i = 0; i < SCALED_SIZE; i++) {
    //     memcpy(blurred_heatmap[i], bhm + i * SCALED_SIZE, SCALED_SIZE * sizeof(int));
    // }
    // for (int i = 0; i < SCALED_SIZE; i++) {
    //     memcpy(scaled_heatmap[i], shm + i * SCALED_SIZE, SCALED_SIZE * sizeof(int));
    // }
    // for (int i = 0; i < SIZE; i++) {
    //     memcpy(heatmap[i], hm + i * SIZE, SIZE * sizeof(int));
    // }

    // cout << "Blurred Heatmap Color Values (Hexadecimal):" << endl;
    // for (int i = 240; i < 250; i++) {
    //     for (int j = 240; j < 250; j++) {
    //         cout << heatmap[i][j] << " ";
    //         if(blurred_heatmap[i][j] != 0){
    //             cout << blurred_heatmap[i][j] << " ";
    //         }
    //         cout << endl;
    //     }
    // }

    // for (int i = 50; i < 60; i++) {
    //     for (int j = 70; j < 80; j++) {
    //         cout << blurred_heatmap[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    cudaStreamSynchronize(stream);
    // Free device memory.
    // cudaStreamSynchronize(stream); // CPU waits for GPU to finish before CPU moves on to the next step.
    // printf("---------Async kernel execution complete-----------------\n");
    cudaFreeAsync(d_hm, stream);
    cudaFreeAsync(d_shm, stream);
    cudaFreeAsync(d_bhm, stream);
    cudaFreeAsync(d_agentDesiredX, stream);
    cudaFreeAsync(d_agentDesiredY, stream);
    cudaFreeAsync(d_weights, stream);
    cudaStreamDestroy(stream);
    free(hm);
    free(shm);
    free(bhm);
}




void Ped::Model::applyBlurFilterCUDA() {
    int numElements = SCALED_SIZE * SCALED_SIZE;

    int *d_scaled_heatmap, *d_blurred_heatmap, *d_weights;
    int weights[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };

    // Flatten the weights
    int h_weights[25];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            h_weights[i * 5 + j] = weights[i][j];

    safe_call(cudaMalloc(&d_scaled_heatmap, numElements * sizeof(int)));
    safe_call(cudaMalloc(&d_blurred_heatmap, numElements * sizeof(int)));
    safe_call(cudaMalloc(&d_weights, 25 * sizeof(int)));

    // Flatten the heatmap
    int *h_scaled_heatmap = new int[numElements];
    for (int i = 0; i < SCALED_SIZE; i++)
        memcpy(h_scaled_heatmap + i * SCALED_SIZE, scaled_heatmap[i], SCALED_SIZE * sizeof(int));

    cudaMemcpy(d_scaled_heatmap, h_scaled_heatmap, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, 25 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((SCALED_SIZE + blockDim.x - 1) / blockDim.x, (SCALED_SIZE + blockDim.y - 1) / blockDim.y);

    blurHeatmapKernel<<<gridDim, blockDim>>>(d_scaled_heatmap, d_blurred_heatmap, SCALED_SIZE, d_weights);
    cudaDeviceSynchronize();

    // Copy the result back to host
    int *h_blurred_heatmap = new int[numElements];
    cudaMemcpy(h_blurred_heatmap, d_blurred_heatmap, numElements * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < SCALED_SIZE; i++) {
        memcpy(blurred_heatmap[i], h_blurred_heatmap + i * SCALED_SIZE, SCALED_SIZE * sizeof(int));
    }

    delete[] h_scaled_heatmap;
    delete[] h_blurred_heatmap;

    cudaFree(d_scaled_heatmap);
    cudaFree(d_blurred_heatmap);
    cudaFree(d_weights);
}


void Ped::Model::scaleHeatmapCUDA() {
    // Assume SIZE and CELLSIZE are defined, and SCALED_SIZE = SIZE * CELLSIZE.
    int numOrig = SIZE * SIZE;
    int numScaled = SCALED_SIZE * SCALED_SIZE;

    // Flatten the 2D heatmap into a contiguous array.
    int *hm_flat = (int*)malloc(numOrig * sizeof(int));
    for (int i = 0; i < SIZE; i++) {
        memcpy(hm_flat + i * SIZE, heatmap[i], SIZE * sizeof(int));
    }

    // Allocate GPU memory.
    int *d_heatmap, *d_scaled;
    safe_call(cudaMalloc((void**)&d_heatmap, numOrig * sizeof(int)));
    safe_call(cudaMalloc((void**)&d_scaled, numScaled * sizeof(int)));

    // Copy the original heatmap to device.
    // cudaMemcpy(d_heatmap, hm_flat, numOrig * sizeof(int), cudaMemcpyHostToDevice);
    safe_call(cudaMemcpy(d_heatmap, hm_flat, numOrig * sizeof(int), cudaMemcpyHostToDevice));


    // Launch the scaling kernel.
    int threadsPerBlock = THREADSPERBLOCK;
    int blocks = (numScaled + threadsPerBlock - 1) / threadsPerBlock;
    scaleHeatmapKernel<<<blocks, threadsPerBlock>>>(d_heatmap, d_scaled, SIZE, CELLSIZE);
    cudaDeviceSynchronize();

    // Allocate a contiguous host array for the scaled heatmap.
    int *shm_flat = (int*)malloc(numScaled * sizeof(int));
    safe_call(cudaMemcpy(shm_flat, d_scaled, numScaled * sizeof(int), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    // Reassign the 'scaled_heatmap' pointer-to-pointers using the flat array.
    // (Note: You must ensure that later code uses the same layout.)
    for (int i = 0; i < SCALED_SIZE; i++) {
        scaled_heatmap[i] = shm_flat + i * SCALED_SIZE;
    }

    // Free temporary memory.
    free(hm_flat);
    // Do not free shm_flat here because scaled_heatmap[i] pointers refer into it.
    safe_call(cudaFree(d_heatmap));
    safe_call(cudaFree(d_scaled));
}