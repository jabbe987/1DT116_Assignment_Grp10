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
        std::vector<Ped::Twaypoint*> destinations; //current destination for all agents
        std::vector<std::vector<Ped::Twaypoint*>> waypoints; //all possible destinations for all agents

        Tagents(size_t numAgents) {
            // x.resize(numAgents, 1.337);
            // y.resize(numAgents, 0.0f);
            // desiredX.resize(numAgents, 0.0f);
            // desiredY.resize(numAgents, 0.0f);
            // destinations.resize(numAgents, nullptr);
            // waypoints.resize(numAgents);
        }

        
        void addAgent(float posX, float posY, const std::vector<Ped::Twaypoint*>& agentWaypoints);
        void computeNextDesiredPositions();
    };
}

#endif
