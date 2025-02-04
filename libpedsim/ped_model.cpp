//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <atomic>
#include <immintrin.h>
#include <xmmintrin.h>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
    std::cout << "Not compiled for CUDA" << std::endl;
#endif
	for (const auto& agent : agentsInScenario) {
        posX.push_back(agent->getX());
        posY.push_back(agent->getY());
        desiredPosX.push_back(agent->getDesiredX());
        desiredPosY.push_back(agent->getDesiredY());
    }
	// destinations = destinationsInScenario;
    this->implementation = implementation;
	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// // Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::tick()
{
    if (implementation == OMP) {
		
        // omp_set_num_threads(6);
		
		// #pragma omp parallel for schedule(static)
        // for(size_t i = 0; i < agents.size(); i++) {
        //     agents[i]->computeNextDesiredPosition();
        //     agents[i]->setX(agents[i]->getDesiredX());
        //     agents[i]->setY(agents[i]->getDesiredY());
        // }
    }
    else if (implementation == PTHREAD) {
		// int numThreads = 4;
		// size_t numAgents = agents.size();
		// std::vector<std::thread> threads;

		// // Determine workload boundaries for each thread
		// size_t chunkSize = numAgents / numThreads;
		// size_t remainder = numAgents % numThreads;

		// size_t start = 0;
		
		// // Worker function: Each thread gets a pre-determined chunk
		// auto worker = [&](size_t start, size_t end) {
        //     for (size_t i = start; i < end; i++) {
        //         agents[i]->computeNextDesiredPosition();
        //         agents[i]->setX(agents[i]->getDesiredX());
        //         agents[i]->setY(agents[i]->getDesiredY());
        //     }
        // };

		// // Launch threads
		// for (size_t i = 0; i < numThreads; i++) {
        //     size_t end = start + chunkSize + (i < remainder ? 1 : 0); // Distribute extra agents to first few threads
        //     if (start < numAgents) {  // Avoid empty threads
        //         threads.push_back(std::thread(worker, start, end));
        //     }
        //     start = end;
        // }

		// // Join threads
		// for (auto& t : threads) {
		// 	t.join();
    	// }
	}
	/*
    else if (implementation == SEQ) {  // Default to serial
        for (Ped::Tagent* agent : agents) {
            agent->computeNextDesiredPosition();
            agent->setX(agent->getDesiredX());
            agent->setY(agent->getDesiredY());
        }*/
	else if (implementation == SEQ) {  // simd_avx implementation
		std::cout << "Inside SEQ implementation" << std::endl;
        for (int i = 0; i<posX.size();i+=8) {
			std::cout << "posX[" << i << "] = " << posX[i] << std::endl;
			__m256i x = _mm256_loadu_si256((__m256i*)&posX[i]);
        	__m256i y = _mm256_loadu_si256((__m256i*)&posY[i]);
			
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
			
			_mm256_storeu_ps((float*)&posX, desPosX);
			_mm256_storeu_ps((float*)&posY, desPosY);

			/*agent->computeNextDesiredPosition();
            agent->setX(agent->getDesiredX());
            agent->setY(agent->getDesiredY());*/
			
        }   
	
}

}







////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
