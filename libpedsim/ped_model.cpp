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
	agent_old = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());
	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());
	
	if (!agents) {  // Prevent double allocation
		std::cout << "Initialize" << std::endl;
		this->agents = new Ped::Tagents(agentsInScenario.size());
	}
	
	// std::cout << "blabla: " << agentsInScenario.size() << std::endl;
	// std::cout << "blabla: " << numAgents << std::endl;
	std::cout << "Number of agents 1111: " << agent_old.size() << std::endl;
    for (const auto& a : agent_old) {
		// std::cout << "a->getX() : " << a->getX() << std::endl;
		// std::cout << "a->getY() : " << a->getY() << std::endl;
        this->agents->addAgent(a->getX(), a->getY(), a->getWaypoints()); // if not good take from agent_old
    }
	std::cout << "Number of agents: " << this->agents->x.size() << std::endl;


	
	/*for (size_t i = 0; i < agents->x.size(); ++i) {
		std::cout << "Agent " << i << " - x: " << agents->x[i] << ", y: " << agents->y[i] << std::endl;
	}*/

	
	// destinations = destinations	InScenario;
	
	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::tick()
{
	/*
	std::cout << "Number of agents: " << agents->x.size() << std::endl;
	std::cout << "Number of destinations: " << agents->destinations.size() << std::endl;
	std::cout << "Number of agents: " << agent_old.size() << std::endl;
	

	/*std::cout << "Agents:" << std::endl;
	for (const auto& agent : agent_old) {
		std::cout << "Agent at (" << agent->getX() << ", " << agent->getY() << ")" << std::endl;
	}

	std::cout << "Destinations:" << std::endl;
	for (const auto& destination : destinations) {
		std::cout << "Destination at (" << destination->getX() << ", " << destination->getY() << ")" << std::endl;
	}*/
    if (implementation == OMP) {
		agents->computeNextDesiredPositions();
		// std::cout << "x pos: " << agents->x[100] <<std::endl;
		// std::cout << "x old: " << agent_old[100]->getX() <<std::endl;
		
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
	
    else if (implementation == SEQ) {  // Default to serial
        for (Ped::Tagent* agent : agent_old) {
            agent->computeNextDesiredPosition();
            agent->setX(agent->getDesiredX());
            agent->setY(agent->getDesiredY());
        }
	// else if (implementation == SEQ) {  // simd_avx implementation
	// 	//std::cout << "Inside SEQ implementation" << std::endl;
	// 		//compute next blablabla
			
	// 		agents->computeNextDesiredPositions();
	// 		// std::cout << "Completed compute desired positions" << std::endl;
			
    //         /*agent->setX(agent->getDesiredX());
    //         agent->setY(agent->getDesiredY());*/
			
    //     }   
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
