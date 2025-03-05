//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_agents.h"
#include "ped_agent.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <atomic>
#include <cmath>
#include <immintrin.h>
#include <xmmintrin.h>
#include <mutex>
// std::mutex regionMutex;  // Global mutex for region updates

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
	agent_old = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());
	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());
	
	if (!agents) {  // Prevent double allocation
		std::cout << "Initialize" << std::endl;
		this->agents = new Ped::Tagents(agentsInScenario.size());
	}
	
	std::cout << "Number of agents 1111: " << agent_old.size() << std::endl;
    for (const auto& a : agent_old) {
        this->agents->addAgent(a->getX(), a->getY(), a->getWaypoints(), getRegion(a->getX(), a->getY())); // if not good take from agent_old
    }
	std::cout << "Number of agents: " << this->agents->x.size() << std::endl;
	
	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

std::vector<std::pair<int, int>> regionBuffer;  // Stores (agentIndex, new_region)
std::mutex bufferMutex;  // Mutex for the buffer

void Ped::Model::tick(){
	
	if (implementation == OMP) {
		computeNext(0, agents->x.size());
		forceMove();
	}
    else if (implementation == OMPSIMD) {
		
		size_t numAgents = agents->x.size();
		size_t simdLimit = numAgents / 8 * 8;
		omp_set_num_threads(6);
		#pragma omp parallel for schedule(static)
        for(size_t i = 0; i < simdLimit; i+=8) {
            agents->computeNextSimd(i);
        }

		computeNext(simdLimit, numAgents);

		forceMove();
	}
    
	else if (implementation == OMPMOVE) {		
		computeNext(0, agents->x.size());
		// updateHeatmapSeq();
		omp_set_num_threads(4);
		#pragma omp parallel for schedule(static)
		for (size_t region = 0; region < agents->regions.size(); region++) {  
			for (size_t j = 0; j < agents->regions[region].size(); j++) {
				int agentIndex = agents->regions[region][j];  
				move(agentIndex, region, j);
			}
		}
	}
	
	else if (implementation == OMPSIMDMOVE){
		size_t numAgents = agents->x.size();
		size_t simdLimit = numAgents / 8 * 8;
		omp_set_num_threads(6);
		#pragma omp parallel for schedule(static)
        for(size_t i = 0; i < simdLimit; i+=8) {
            agents->computeNextSimd(i);
        }

		computeNext(simdLimit, numAgents);

		for (size_t region = 0; region < agents->regions.size(); region++) {  
			for (size_t j = 0; j < agents->regions[region].size(); j++) {
				int agentIndex = agents->regions[region][j];  
				move(agentIndex, region, j);
			}
		}
		
	}

	else if (implementation == PTHREAD) {
		computeNext(0, agents->x.size()); 
		forceMove();
	}
    else if (implementation == PTHREADSIMD) {
		int numThreads = 4;
		size_t numAgents = agents->x.size();
		size_t simdLimit = numAgents / 8 * 8;
		std::vector<std::thread> threads;

		// Determine workload boundaries for each thread
		size_t chunkSize = (numAgents / numThreads) & ~7;
		// std::cout << "Chunksize: " << chunkSize <<std::endl;
		size_t remainder = numAgents - (chunkSize * numThreads);

		size_t start = 0;
		
		// Worker function: Each thread gets a pre-determined chunk
		auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i+=8) {
                agents->computeNextSimd(i);
            }
        };

		// Launch threads
		for (size_t i = 0; i < numThreads; i++) {
            size_t end = start + chunkSize; // Distribute extra agents to first few threads
			if (remainder > 7) {
				end += 8;
				remainder -= 8;
			}
            if (start < numAgents) {  // Avoid empty threads
                threads.push_back(std::thread(worker, start, end));
            }
            start = end;
        }

		// Join threads
		for (auto& t : threads) {
			t.join();
    	}
		computeNext(numAgents - remainder, numAgents);

		forceMove();
	}

	// else if (implementation == PTHREADSIMDMOVE){
	// 	int numThreads = 4;
	// 	size_t numAgents = agents->x.size();
	// 	size_t simdLimit = numAgents / 8 * 8;
	// 	std::vector<std::thread> threads;

	// 	// Determine workload boundaries for each thread
	// 	size_t chunkSize = (numAgents / numThreads) & ~7;
	// 	// std::cout << "Chunksize: " << chunkSize <<std::endl;
	// 	size_t remainder = numAgents - (chunkSize * numThreads);

	// 	size_t start = 0;
		
	// 	// Worker function: Each thread gets a pre-determined chunk
	// 	auto worker = [&](size_t start, size_t end) {
    //         for (size_t i = start; i < end; i+=8) {
    //             agents->computeNextSimd(i);
    //         }
    //     };

	// 	// Launch threads
	// 	for (size_t i = 0; i < numThreads; i++) {
    //         size_t end = start + chunkSize; // Distribute extra agents to first few threads
	// 		if (remainder > 7) {
	// 			end += 8;
	// 			remainder -= 8;
	// 		}
    //         if (start < numAgents) {  // Avoid empty threads
    //             threads.push_back(std::thread(worker, start, end));
    //         }
    //         start = end;
    //     }

	// 	// Join threads
	// 	for (auto& t : threads) {
	// 		t.join();
    // 	}
	// 	computeNext(numAgents - remainder, numAgents);

		
	// }
	
	else if (implementation == VECTOR) {  // SIMD serial
		size_t numAgents = agents->x.size();
		size_t simdLimit = numAgents / 8 * 8;

        for(size_t i = 0; i < simdLimit; i+=8) {
            agents->computeNextSimd(i);
        }
		computeNext(simdLimit, numAgents);
		
		forceMove();
	}

	else if (implementation == SEQMOVE) {
		computeNext(0, agents->x.size()); //struct of arrays version

		for(size_t i = 0; i < agents->x.size(); i++) {
			moveSeq(i);
		}
	}
    else if (implementation == SEQ) {  // Default to serial
		computeNext(0, agents->x.size()); //struct of arrays version
		// updateHeatmapSeq();

		for(size_t i = 0; i < agents->x.size(); i++) {
			agents->x[i] = agents->desiredX[i];
			agents->y[i] = agents->desiredY[i];
		}
	}
	
	std::lock_guard<std::mutex> lock(agents->agentsMutex);  // Lock global mutex for safe update
    for (const auto& change : regionBuffer) {
        int agentIndex = change.first;
        int new_region = change.second;
        int old_region = getRegion(agents->x[agentIndex], agents->y[agentIndex]);

        // Move agent to new region in agents->regions
        agents->regions[new_region].push_back(agentIndex);
        agents->regions[old_region].erase(
            std::remove(agents->regions[old_region].begin(), agents->regions[old_region].end(), agentIndex),
            agents->regions[old_region].end()
        );
    }
    regionBuffer.clear();  // Clear buffer for next tick
}


void Ped::Model::forceMove() {
	if (implementation == OMP){
		#pragma omp parallel for schedule(static)
		for(size_t i = 0; i < agents->x.size(); i++) {
			agents->x[i] = agents->desiredX[i];
			agents->y[i] = agents->desiredY[i];
		}
	}
	else if (implementation == PTHREAD) {
		int numThreads = 4;
		size_t numAgents = agents->x.size();
		std::vector<std::thread> threads;

		// Determine workload boundaries for each thread
		size_t chunkSize = numAgents / numThreads;
		size_t remainder = numAgents % numThreads;

		size_t start = 0;

		// Worker function: Each thread gets a pre-determined chunk
		auto worker = [&](size_t start, size_t end) {
			for (size_t i = start; i < end; i++) {
				agents->x[i] = agents->desiredX[i];
				agents->y[i] = agents->desiredY[i];
			}
		};

		// Launch threads
		for (size_t i = 0; i < numThreads; i++) {
			size_t end = start + chunkSize + (i < remainder ? 1 : 0); // Distribute extra agents to first few threads
			if (start < numAgents) {  // Avoid empty threads
				threads.push_back(std::thread(worker, start, end));
			}
			start = end;
		}

		// Join threads
		for (auto& t : threads) {
			t.join();
		}
	}

	else{
		for(size_t i = 0; i < agents->x.size(); i++) {
			agents->x[i] = agents->desiredX[i];
			agents->y[i] = agents->desiredY[i];
		}
	}
}

void Ped::Model::computeNext(size_t start, size_t end) {
	if (implementation == OMP || implementation == OMPMOVE) {
		#pragma omp parallel for schedule(static)
		for (size_t i = start; i < end; i++) {
			agents->getNextDestinationSeq(i);
			double diffX = agents->destinationX[i] - agents->x[i];
			double diffY = agents->destinationY[i] - agents->y[i];
			double length = sqrtf(diffX * diffX + diffY * diffY);
			diffX = agents->destinationX[i] - agents->x[i];
			diffY = agents->destinationY[i] - agents->y[i];
			length = sqrtf(diffX * diffX + diffY * diffY);
			if(length < 0.0001) {
				length = 0.1;
			}
			agents->desiredX[i] = (int)round(agents->x[i] + (diffX / length));
			agents->desiredY[i] = (int)round(agents->y[i] + (diffY / length));
		}
	}
	else if (implementation == PTHREAD || implementation == PTHREADMOVE) {
		size_t numAgents = agents->x.size();
		int numThreads = 4;
		std::vector<std::thread> threads;
		size_t chunkSize = (end - start) / numThreads;
		size_t remainder = (end - start) % numThreads;
		size_t s = start;
		auto worker = [&](size_t start, size_t end) {
			for (size_t i = start; i < end; i++) {
				agents->getNextDestinationSeq(i);
				double diffX = agents->destinationX[i] - agents->x[i];
				double diffY = agents->destinationY[i] - agents->y[i];
				double length = sqrtf(diffX * diffX + diffY * diffY);
				diffX = agents->destinationX[i] - agents->x[i];
				diffY = agents->destinationY[i] - agents->y[i];
				length = sqrtf(diffX * diffX + diffY * diffY);
				if(length < 0.0001) {
					length = 0.1;
				}
				agents->desiredX[i] = (int)round(agents->x[i] + (diffX / length));
				agents->desiredY[i] = (int)round(agents->y[i] + (diffY / length));
			}
		};
		for (int i = 0; i < numThreads; i++) {
			size_t e = s + chunkSize;
			if (remainder > 0) {
				end++;
				remainder--;
			}
			if (start < numAgents) {
				threads.push_back(std::thread(worker, s, e));
			}
			s = e;
		}
		for (auto& t : threads) {
			t.join();
		}
	}
	else {
		for (size_t i = start; i < end; i++) {
			agents->getNextDestinationSeq(i);
			double diffX = agents->destinationX[i] - agents->x[i];
			double diffY = agents->destinationY[i] - agents->y[i];
			double length = sqrtf(diffX * diffX + diffY * diffY);
			diffX = agents->destinationX[i] - agents->x[i];
			diffY = agents->destinationY[i] - agents->y[i];
			length = sqrtf(diffX * diffX + diffY * diffY);
			if(length < 0.0001) {
				length = 0.1;
			}
			agents->desiredX[i] = (int)round(agents->x[i] + (diffX / length));
			agents->desiredY[i] = (int)round(agents->y[i] + (diffY / length));
		}
	}
	
}

// std::vector<std::vector<int>> Ped::Model::getAgentsByRegion() {
// 	std::vector<std::vector<int>> regions(4);  // Assuming 4 regions (1 to 4)

// 	for (size_t i = 0; i < agents->x.size(); ++i) {
// 		int region = agents->regions[i];
		
// 		regions[region - 1].push_back(i);  // Store agent index, region - 1 for 0-based indexing
// 	}

// 	return regions;
// }

int Ped::Model::getRegion(float x, float y) {
	if (x >= 80 && y >= 60) {
		return 0;  // First quadrant (Region 1)
	} else if (x < 80 && y >= 60) {
		return 1;  // Second quadrant (Region 2)
	} else if (x < 80 && y < 60) {
		return 2;  // Third quadrant (Region 3)
	} else {
		return 3;  // Fourth quadrant (Region 4)
	}
}
// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(int i, int region, int regionIndex)
{	
	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agents->desiredX[i], agents->desiredY[i]);
	prioritizedAlternatives.push_back(pDesired);
	
	int diffX = pDesired.first - agents->x[i]; //feels wrong 
	int diffY = pDesired.second - agents->y[i];
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agents->y[i]);
		p2 = std::make_pair(agents->x[i], pDesired.second);
	}
	
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);
	
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// int old_region = getRegion(agents->x[i],agents->y[i]);
		int new_region = getRegion(((*it).first), ((*it).second));

		if (region != new_region){
			// Lock both old and new regions
			std::lock_guard<std::mutex> lock(agents->agentsMutex); 
			std::vector<std::pair<int, int> > takenPositionsLock;
			for (size_t j = 0; j < agents->x.size(); j++) {
				if (i == j) continue;
				std::pair<int, int> position(agents->x[j], agents->y[j]);
				takenPositionsLock.push_back(position);
			}

			if (std::find(takenPositionsLock.begin(), takenPositionsLock.end(), *it) == takenPositionsLock.end()) {
				// agents->regions[new_region].push_back(i);
				// agents->regions[region].erase(agents->regions[region].begin() + regionIndex);

				std::lock_guard<std::mutex> bufLock(bufferMutex);
				regionBuffer.emplace_back(i, new_region);

				agents->x[i] = ((*it).first);
				agents->y[i] = ((*it).second);
				break;

			}
		}

		else{
			std::vector<std::pair<int, int> > takenPositionsNew;
			for (size_t j = 0; j < agents->x.size(); j++) {
				if (i == j) continue;
				std::pair<int, int> position(agents->x[j], agents->y[j]);
				takenPositionsNew.push_back(position);
			}

			if (std::find(takenPositionsNew.begin(), takenPositionsNew.end(), *it) == takenPositionsNew.end()) {
				agents->x[i] = ((*it).first);
				agents->y[i] = ((*it).second);
				break;
			}
		}
	}
}

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::moveSeq(int i)
{	
	std::vector<std::pair<int, int> > takenPositionsNew;
	for (size_t j = 0; j < agents->x.size(); j++) {
		if (i == j) continue;
		std::pair<int, int> position(agents->x[j], agents->y[j]);
		takenPositionsNew.push_back(position);
	}
	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agents->desiredX[i], agents->desiredY[i]);
	prioritizedAlternatives.push_back(pDesired);
	
	int diffX = pDesired.first - agents->x[i]; //feels wrong 
	int diffY = pDesired.second - agents->y[i];
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agents->y[i]);
		p2 = std::make_pair(agents->x[i], pDesired.second);
	}
	
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);
	
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		if (std::find(takenPositionsNew.begin(), takenPositionsNew.end(), *it) == takenPositionsNew.end()) {
			agents->x[i] = ((*it).first);
			agents->y[i] = ((*it).second);
			break;
		}
	}
}

// std::vector<std::pair<int, int> > Ped::Model::getNeighbors(int x, int y, int region, int index) {

// 	// create the output list
// 	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
// 		// Retrieve their positions
// 	// Retrieve their positions
// 	std::vector<std::vector<int>> agentsInRegion = getAgentsByRegion();

// 	std::vector<std::pair<int, int> > takenPositions;

// 	for (size_t j = 0; j < agentsInRegion[region].size(); j++) {
// 		int i = agentsInRegion[region][j];
// 		if (index == i) continue;
// 		std::pair<int, int> position(agents->x[i], agents->y[i]);
// 		takenPositions.push_back(position);
// 	}

// 	return takenPositions;
// }
////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move_old(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors_old(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
		for(int i = 0; i < 10; i++) {
			std::cout << "takenPositions: " << position.first << " " << position.second << std::endl;
			
		}	
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
set<const Ped::Tagent*> Ped::Model::getNeighbors_old(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agent_old.begin(), agent_old.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agent_old.begin(), agent_old.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
