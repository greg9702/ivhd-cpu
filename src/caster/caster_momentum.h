#pragma once

#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"

using namespace std;

class CasterMomentum : public CasterCPU {
public:
    CasterMomentum(int n, function<void(float)> onErr, function<void(vector<float2> &)> onPos)
            : CasterCPU(n, onErr, onPos), f(n, {0, 0}), momentum(n, {0, 0}) {}

    virtual void simul_step_cpu() override;

protected:
    vector<float2> f;
    vector<float2> momentum;

private:
    float2 force(DistElem distance);

    float learning_rate = 0.01;
    float momentum_rate = 0.9;
    float w_random = 0.01;
};
