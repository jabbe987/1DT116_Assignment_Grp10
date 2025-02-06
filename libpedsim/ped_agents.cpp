#include "ped_agents.h"
#include <iostream>
#include <immintrin.h>

void Ped::Tagents::addAgent(float posX, float posY, const std::vector<Ped::Twaypoint*>& agentWaypoints) {
    x.push_back(posX);
    y.push_back(posY);
    desiredX.push_back(posX);
    desiredY.push_back(posY);
    destinations.push_back(agentWaypoints.empty() ? nullptr : agentWaypoints.front());
    waypoints.push_back(agentWaypoints);
}

void Ped::Tagents::computeNextDesiredPositions() {
	
	for (int i = 0; i<x.size();i+=8) {
	std::cout << "X[" << i << "] = " << x[i] << std::endl;
	__m256i x = _mm256_loadu_si256((__m256i*)&x[i]); //måste fixa namngivning
	__m256i y = _mm256_loadu_si256((__m256i*)&y[i]);
	
	// __m256 dx = _mm256_loadu_ps(&destinations[i]->getx());
	// __m256 dy = _mm256_loadu_ps(&destinations[i]->gety());
	//alignas(32) float x[8], y[8], dx[8], dy[8];

	// Load agent data into arrays
	// if we have trouble we might need this solution below
	int dx[8], dy[8]; 
	for (int j = 0; j < 8; j++) {
		dx[j] = destinations[i+j]->getx();
		dy[j] = destinations[i+j]->gety();
	}

	
	__m256i destX = _mm256_loadu_si256((__m256i*)dx);
	__m256i destY = _mm256_loadu_si256((__m256i*)dy);
	

	//Calculate distance to next waypoint m
	__m256i diffX = _mm256_sub_epi32(x, destX);
	__m256i diffY = _mm256_sub_epi32(y, destY);
	__m256i mulY = _mm256_mul_epi32(diffY,diffY);
	__m256i mulX = _mm256_mul_epi32(diffX,diffX);
	__m256i addxy = _mm256_add_epi32(mulX, mulY);
	__m256 len = _mm256_sqrt_ps(_mm256_cvtepi32_ps(addxy));

	__m256i xdiffx = _mm256_add_epi32(diffX, x);
	__m256 desPosX = _mm256_div_ps(_mm256_cvtepi32_ps(xdiffx), len);
	__m256i ydiffy = _mm256_add_epi32(diffY, y);
	__m256 desPosY = _mm256_div_ps(_mm256_cvtepi32_ps(ydiffy), len);
	
	_mm256_storeu_ps((float*)&x, desPosX);
	_mm256_storeu_ps((float*)&y, desPosY);
}

size_t numAgents = x.size();
    
    for (size_t i = 0; i < numAgents; i += 8) {
        __m256 xVec = _mm256_loadu_ps(&x[i]);
        __m256 yVec = _mm256_loadu_ps(&y[i]);

        __m256 destXVec, destYVec;
        float destXArr[8], destYArr[8];

        // Gather destination points
        for (int j = 0; j < 8; ++j) {
            if (i + j < numAgents && destinations[i + j]) {
                destXArr[j] = destinations[i + j]->getx();
                destYArr[j] = destinations[i + j]->gety();
            } else {
                destXArr[j] = x[i + j];  // Keep position if no destination
                destYArr[j] = y[i + j];
            }
        }

        destXVec = _mm256_loadu_ps(destXArr);
        destYVec = _mm256_loadu_ps(destYArr);

        // Compute movement vector
        __m256 diffX = _mm256_sub_ps(destXVec, xVec);
        __m256 diffY = _mm256_sub_ps(destYVec, yVec);
        __m256 len = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(diffX, diffX), _mm256_mul_ps(diffY, diffY)));

        // Prevent division by zero
        __m256 mask = _mm256_cmp_ps(len, _mm256_set1_ps(0.0001f), _CMP_GT_OQ);
        len = _mm256_blendv_ps(_mm256_set1_ps(1.0f), len, mask);

        // Compute new desired positions
        __m256 nextX = _mm256_add_ps(xVec, _mm256_div_ps(diffX, len));
        __m256 nextY = _mm256_add_ps(yVec, _mm256_div_ps(diffY, len));

        // Store results
        _mm256_storeu_ps(&desiredX[i], nextX);
        _mm256_storeu_ps(&desiredY[i], nextY);
    }
}