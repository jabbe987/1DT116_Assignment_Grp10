#ifndef PED_AGENTS_H
#define PED_AGENTS_H

#include <vector>
#include <immintrin.h>  // For SIMD intrinsics
#include "ped_waypoint.h"

namespace Ped {
    class Twaypoint;
    
    struct Tagents {
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> desiredX;
        std::vector<float> desiredY;
        std::vector<float> destinationX;
        std::vector<float> destinationY;
        std::vector<float> destinationR;
        std::vector<float> destinationX2;
        std::vector<float> destinationY2;
        std::vector<float> destinationR2;
        std::vector<int> regions; // 11, 27,103 
        // std::vector<Ped::Twaypoint*> destinations; //current destination for all agents
        // std::vector<std::vector<Ped::Twaypoint*>> waypoints; //all possible destinations for all agents

        Tagents(size_t numAgents) {}

        
        void addAgent(int posX, int posY, const std::vector<Ped::Twaypoint*>& agentWaypoints, int region);
        void computeNextDesiredPositions(int i);
        void getNextDestinationSeq(int i);
    };
}

#endif
