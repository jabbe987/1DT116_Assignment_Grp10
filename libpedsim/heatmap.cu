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
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl
}

// CUDA kernel for Gaussian blur filter using shared memory
__global__ void blurHeatmapKernel(const int *d_scaled_heatmap, int *d_blurred_heatmap, int width, const int *d_weights) {
    __shared__ int tile[32][32];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 2;
    int ty = threadIdx.y + 2;

    if (x < width && y < width) {
        tile[ty][tx] = d_scaled_heatmap[y * width + x];

        if (threadIdx.x < 2 && x >= 2)
            tile[ty][tx - 2] = d_scaled_heatmap[y * width + x - 2];
        if (threadIdx.x >= blockDim.x - 2 && x + 2 < width)
            tile[ty][tx + 2] = d_scaled_heatmap[y * width + x + 2];
        if (threadIdx.y < 2 && y >= 2)
            tile[ty - 2][tx] = d_scaled_heatmap[(y - 2) * width + x];
        if (threadIdx.y >= blockDim.y - 2 && y + 2 < width)
            tile[ty + 2][tx] = d_scaled_heatmap[(y + 2) * width + x];

        __syncthreads();

        if (x >= 2 && y >= 2 && x < width - 2 && y < width - 2) {
            int sum = 0;
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    sum += d_weights[(i + 2) * 5 + (j + 2)] * tile[ty + i][tx + j];
                }
            }
            d_blurred_heatmap[y * width + x] = 0x00FF0000 | ((sum / 273) << 24);
        }
    }
}




__global__ void scaleHeatmapKernel(const int *d_heatmap, int *d_scaled, int size, int cellsize) {
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
        d_scaled[scaled_index] = d_heatmap[orig_index];
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


// Updates the heatmap according to the agent positions
// void Ped::Model::updateHeatmap()
// {
// 	for (int x = 0; x < SIZE; x++)
// 	{
// 		for (int y = 0; y < SIZE; y++)
// 		{
// 			// heat fades
// 			heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
// 		}
// 	}

// 	// Count how many agents want to go to each location
// 	for (int i = 0; i < agents->x.size(); i++)
// 	{
// 		//Ped::Tagent* agent = agents->desiredX[i];
// 		int x = agents->desiredX[i];
// 		int y = agents->desiredY[i];

// 		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
// 		{
// 			continue;
// 		}

// 		// intensify heat for better color results
// 		heatmap[y][x] += 40;

// 	}

// 	for (int x = 0; x < SIZE; x++)
// 	{
// 		for (int y = 0; y < SIZE; y++)
// 		{
// 			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
// 		}
// 	}

// 	// Scale the data for visual representation
// 	// for (int y = 0; y < SIZE; y++)
// 	// {
// 	// 	for (int x = 0; x < SIZE; x++)
// 	// 	{
// 	// 		int value = heatmap[y][x];
// 	// 		for (int cellY = 0; cellY < CELLSIZE; cellY++)
// 	// 		{
// 	// 			for (int cellX = 0; cellX < CELLSIZE; cellX++)
// 	// 			{
// 	// 				scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
// 	// 			}
// 	// 		}
// 	// 	}
// 	// }
//     scaleHeatmapCUDA();
// }

void Ped::Model::updateHeatmap() {
    for (int x = 0; x < SIZE; x++) {
        for (int y = 0; y < SIZE; y++) {
            // heat fades
            heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
        }
    }

    // Count how many agents want to go to each location
    for (int i = 0; i < agents->x.size(); i++) {
        int x = agents->desiredX[i];
        int y = agents->desiredY[i];

        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
            continue;
        }

        // Intensify heat for better color results
        heatmap[y][x] += 40;
    }

    for (int x = 0; x < SIZE; x++) {
        for (int y = 0; y < SIZE; y++) {
            heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
        }
    }

    // Scale the heatmap using existing CUDA scaling
    scaleHeatmapCUDA();

    // Apply the parallelized CUDA blur filter
    applyBlurFilterCUDA();
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

	// Weights for blur filter
	// const int w[5][5] = {
	// 	{ 1, 4, 7, 4, 1 },
	// 	{ 4, 16, 26, 16, 4 },
	// 	{ 7, 26, 41, 26, 7 },
	// 	{ 4, 16, 26, 16, 4 },
	// 	{ 1, 4, 7, 4, 1 }
	// };

// #define WEIGHTSUM 273
// 	// Apply gaussian blurfilter		       
// 	for (int i = 2; i < SCALED_SIZE - 2; i++)
// 	{
// 		for (int j = 2; j < SCALED_SIZE - 2; j++)
// 		{
// 			int sum = 0;
// 			for (int k = -2; k < 3; k++)
// 			{
// 				for (int l = -2; l < 3; l++)
// 				{
// 					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
// 				}
// 			}
// 			int value = sum / WEIGHTSUM;
// 			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
// 		}
// 	}
// }

// int Ped::Model::getHeatmapSize() const {
// 	return SCALED_SIZE;
// }
