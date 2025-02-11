#include "ExportSimulation.h"

#include <cstdint> // for int16_t and int32_t

using namespace std;

ExportSimulation::ExportSimulation(Ped::Model &model_, int maxSteps,
        std::string outputFilename_) : Simulation(model_, maxSteps), outputFilename(outputFilename_)
{
    file = std::ofstream(outputFilename.c_str(), std::ios::binary);
    file.write(reinterpret_cast<const char*>(&maxSimulationSteps), sizeof(maxSimulationSteps));
}

ExportSimulation::~ExportSimulation() {
    file.seekp(0, std::ios::beg);
    file.write(reinterpret_cast<const char*>(&tickCounter), sizeof(tickCounter));
    file.close();
}

void ExportSimulation::serialize()
{
    const Ped::Tagents* agents = model.getAgentsSoA();
    size_t num_agents = agents->x.size();
    file.write(reinterpret_cast<const char*>(&num_agents), sizeof(num_agents));

    for (size_t i = 0; i < num_agents; i++) {
        int16_t x = static_cast<int16_t>(agents->x[i]);
        int16_t y = static_cast<int16_t>(agents->y[i]);

        file.write(reinterpret_cast<const char *>(&x), sizeof(x));
        file.write(reinterpret_cast<const char *>(&y), sizeof(y));
    }

    //size_t heatmap_elements = model.getHeatmapSize();
    //file.write(reinterpret_cast<const char*>(&heatmap_elements), sizeof(heatmap_elements));
    int16_t height = HEATMAP_HEIGHT;
    int16_t width = HEATMAP_WIDTH;
    const int* const* heatmap = model.getHeatmap();

    unsigned long heatmap_start = 0xFFFF0000FFFF0000;
    file.write(reinterpret_cast<const char*>(&heatmap_start), sizeof(heatmap_start));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int ARGBvalue = heatmap[i][j];
            int8_t Avalue = (ARGBvalue >> 24) & ((1 << 8)-1);
            file.write(reinterpret_cast<const char*>(&Avalue), sizeof(Avalue));
        }
    }
    file.flush();
}

void ExportSimulation::runSimulation()
{
    for (int i = 0; i < maxSimulationSteps; i++) {
        tickCounter++;
        model.tick();
        serialize();
    }
}
