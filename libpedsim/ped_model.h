//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>

#include "ped_agent.h"
#include "ped_agents.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, OMPMOVE, OMPSIMDMOVE, PTHREAD, SEQ, OMPSIMD, PTHREADSIMD, PTHREADMOVE, SEQSIMD, SEQMOVE};

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation);
		
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		void computeNext(size_t start, size_t end);

		std::vector<std::vector<int>> getAgentsByRegion();

		int getRegion(int x, int y);

		bool validateRegions();

		// Returns the agents of this scenario
		const std::vector<Tagent*>& getAgents() const { return agent_old; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

		const Ped::Tagents* getAgentsSoA() const { return agents; }

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agent_old; //before SIMD
		Ped::Tagents* agents = nullptr; //with SIMD

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void move_old(Ped::Tagent *agent);

		void move(int i, int region, int regionIndex);

		std::vector<int> closeRegions(int x, int y, int region);

		void moveSeq(int i);

		void forceMove();
		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors_old(int x, int y, int dist) const;
		std::vector<std::pair<int, int> > getNeighbors(int x, int y, int region, int index);

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE
#define THREADSPERBLOCK 256

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapSeq();
		
		
		void setupHeatmap();
		void updateHeatmap();
		// void createStream();
		// void destroyStream();
		// void syncHeatmap();
		// void scaleHeatmapCUDA();
		// void applyBlurFilterCUDA();

		

	};
}
#endif
