#pragma once

#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"

using namespace std;

class CasterNesterov : public CasterCPU {
public:
    CasterNesterov(int n, function<void(float)> onErr,
                   function<void(vector<std::pair<float, float>> &)> onPos)
            : CasterCPU(n, onErr, onPos), v(n, {0, 0}), f(n, {0, 0}) {}

    virtual void simul_step_cpu() override;

protected:
    vector<std::pair<float, float>> v;
    vector<std::pair<float, float>> f;

private:
    std::pair<float, float> force(DistElem distance);

    float a_factor = 0.9;
    float b_factor = 0.002;
    float w_random = 0.01;
};
