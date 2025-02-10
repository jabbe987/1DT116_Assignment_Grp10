#include "ped_agents.h"
#include <iostream>
#include <immintrin.h>
#include <cmath>

void Ped::Tagents::addAgent(float posX, float posY, const std::vector<Ped::Twaypoint*>& agentWaypoints) {
    x.push_back(posX);
    y.push_back(posY);
    desiredX.push_back(posX);
    desiredY.push_back(posY);
    destinationX.push_back(agentWaypoints.empty() ? posX : agentWaypoints.front()->getx());
    destinationY.push_back(agentWaypoints.empty() ? posY : agentWaypoints.front()->gety());
    destinations.push_back(agentWaypoints.empty() ? nullptr : agentWaypoints.front());
    waypoints.push_back(agentWaypoints);
}

void Ped::Tagents::computeNextDesiredPositions() {
	
    size_t numAgents = x.size();
    size_t simdLimit = numAgents / 8 * 8;

    for (size_t i = 0; i < numAgents; i += 8) {
        // std::cout << "i: " << i << std::endl;
        // Load position vectors
        __m256 xVec = _mm256_loadu_ps(&x[i]);
        __m256 yVec = _mm256_loadu_ps(&y[i]);

        // Destination vectors
        __m256 destXVec = _mm256_loadu_ps(&destinationX[i]);
        __m256 destYVec = _mm256_loadu_ps(&destinationY[i]);

        // Compute direction vectors
        __m256 diffX = _mm256_sub_ps(destXVec, xVec);
        __m256 diffY = _mm256_sub_ps(destYVec, yVec);

        // Compute length (Euclidean distance)
        __m256 distSquared = _mm256_add_ps(
            _mm256_mul_ps(diffX, diffX),
            _mm256_mul_ps(diffY, diffY)
        );
        __m256 len = _mm256_sqrt_ps(distSquared);

        // Prevent division by zero
        __m256 zeroMask = _mm256_cmp_ps(len, _mm256_set1_ps(0.0001f), _CMP_GT_OQ);
        len = _mm256_blendv_ps(_mm256_set1_ps(1.0f), len, zeroMask);

        // Normalize direction
        __m256 dirX = _mm256_div_ps(diffX, len);
        __m256 dirY = _mm256_div_ps(diffY, len);

        // Compute new desired positions
        __m256 nextX = _mm256_add_ps(xVec, dirX);
        __m256 nextY = _mm256_add_ps(yVec, dirY);

        // Store results back
        _mm256_storeu_ps(&x[i], nextX);
        _mm256_storeu_ps(&y[i], nextY);
    }

    for (size_t i = simdLimit; i < numAgents; i++) {
        // std::cout << "i: " << i << std::endl;
        float diffX = destinationX[i] - x[i];
        float diffY = destinationY[i] - y[i];

        float len = sqrtf(diffX * diffX + diffY * diffY);
        if (len < 0.0001f) len = 1.0f; // Prevent division by zero

        x[i] = x[i] + (diffX / len);
        y[i] = y[i] + (diffY / len);
    }
}