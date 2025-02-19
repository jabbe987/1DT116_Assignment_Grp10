#include "ped_agents.h"
#include <iostream>
#include <immintrin.h>
#include <cmath>
#include <omp.h>

void Ped::Tagents::addAgent(float posX, float posY, const std::vector<Ped::Twaypoint*>& agentWaypoints) {
    x.push_back(posX);
    y.push_back(posY);
    desiredX.push_back(posX);
    desiredY.push_back(posY);

    destinationX.push_back(agentWaypoints.empty() ? posX : agentWaypoints.front()->getx());
    destinationY.push_back(agentWaypoints.empty() ? posY : agentWaypoints.front()->gety());
    destinationR.push_back(agentWaypoints.empty() ? 0.1f : agentWaypoints.front()->getr());

    destinationX2.push_back(agentWaypoints.empty() ? posX : agentWaypoints.back()->getx());
    destinationY2.push_back(agentWaypoints.empty() ? posY : agentWaypoints.back()->gety());
    destinationR2.push_back(agentWaypoints.empty() ? 0.1f : agentWaypoints.back()->getr());
    // destinations.push_back(agentWaypoints.empty() ? nullptr : agentWaypoints.front());
    // waypoints.push_back(agentWaypoints);
}

void Ped::Tagents::computeNextDesiredPositions(int i) {

    __m256 xVec = _mm256_loadu_ps(&x[i]);
    __m256 yVec = _mm256_loadu_ps(&y[i]);
    // Destination vectors
    __m256 destXVec = _mm256_loadu_ps(&destinationX[i]);
    __m256 destYVec = _mm256_loadu_ps(&destinationY[i]);
    __m256 destRVec = _mm256_loadu_ps(&destinationR[i]);
    // Compute direction vectors
    __m256 diffX = _mm256_sub_ps(destXVec, xVec);
    __m256 diffY = _mm256_sub_ps(destYVec, yVec);

    // Compute length (Euclidean distance)
    __m256 distSquared = _mm256_add_ps(
        _mm256_mul_ps(diffX, diffX),
        _mm256_mul_ps(diffY, diffY)
    );
    __m256 len = _mm256_sqrt_ps(distSquared);
    
    
    __m256 mask = _mm256_cmp_ps(len, destRVec, _CMP_LT_OQ); 
    // Check each agent separately
    
    __m256 destX2Vec = _mm256_loadu_ps(&destinationX2[i]);
    __m256 destY2Vec = _mm256_loadu_ps(&destinationY2[i]);
    __m256 destR2Vec = _mm256_loadu_ps(&destinationR2[i]);

    __m256 destXVecUp = _mm256_blendv_ps(destXVec, destX2Vec, mask);
    __m256 destYVecUp = _mm256_blendv_ps(destYVec, destY2Vec, mask);
    __m256 destRVecUp = _mm256_blendv_ps(destRVec, destR2Vec, mask);

    __m256 destX2VecUp = _mm256_blendv_ps(destX2Vec, destXVec, mask);
    __m256 destY2VecUp = _mm256_blendv_ps(destY2Vec, destYVec, mask);
    __m256 destR2VecUp = _mm256_blendv_ps(destR2Vec, destRVec, mask);

    _mm256_storeu_ps(&destinationX[i], destXVecUp);
    _mm256_storeu_ps(&destinationY[i], destYVecUp);
    _mm256_storeu_ps(&destinationR[i], destRVecUp);

    _mm256_storeu_ps(&destinationX2[i], destX2VecUp);
    _mm256_storeu_ps(&destinationY2[i], destY2VecUp);
    _mm256_storeu_ps(&destinationR2[i], destR2VecUp);

    // int maskArr = _mm256_movemask_ps(mask);
    
    // if (maskArr) {
    //     // Case 1: agent has reached destination (or has no current destination);
    //     // get next destination if available
    //     __m256 destX2Vec = _mm256_loadu_ps(&destinationX2[i]);
    //     __m256 destY2Vec = _mm256_loadu_ps(&destinationY2[i]);
    //     __m256 destR2Vec = _mm256_loadu_ps(&destinationR2[i]);
        
    //     __m256 tempx = destXVec;
    //     _mm256_storeu_ps(&destinationX[i], destX2Vec);
    //     _mm256_storeu_ps(&destinationX2[i], tempx);
    //     destXVec = destX2Vec;

    //     __m256 tempy = destYVec;
    //     _mm256_storeu_ps(&destinationY[i], destY2Vec);
    //     _mm256_storeu_ps(&destinationY2[i], tempy);
    //     destYVec = destY2Vec;

    //     __m256 tempr = destRVec;
    //     _mm256_storeu_ps(&destinationR[i], destR2Vec);
    //     _mm256_storeu_ps(&destinationR2[i], tempr);
    //     destRVec = destR2Vec;
    // }

    diffX = _mm256_sub_ps(destXVecUp, xVec);
    diffY = _mm256_sub_ps(destYVecUp, yVec);

    // Compute length (Euclidean distance)
    distSquared = _mm256_add_ps(
        _mm256_mul_ps(diffX, diffX),
        _mm256_mul_ps(diffY, diffY)
    );
    len = _mm256_sqrt_ps(distSquared);
    

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
    _mm256_storeu_ps(&desiredX[i], nextX);
    _mm256_storeu_ps(&desiredY[i], nextY);
    
}


void Ped::Tagents::getNextDestinationSeq(int i) {
    // Compute if the agent has reached its current destination
    float diffX = destinationX[i] - x[i];
    float diffY = destinationY[i] - y[i];
    float length = sqrtf(diffX * diffX + diffY * diffY); // Compute distance
    // std::cout << "length: " << length << "  i: " << i << "  r:  "<< destinationR[i] << " destination: " << destinationX[i] << "destination2: " << destinationX2[i]<<std::endl;

    if (length < destinationR[i]) {

        // std::cout << "swap nr :" << i << std::endl;
        std::swap(destinationX[i], destinationX2[i]);
        std::swap(destinationY[i], destinationY2[i]);
        std::swap(destinationR[i], destinationR2[i]);
        // std::cout << "Agent " << i <<"swapped location!"<< std::endl;

    }
}