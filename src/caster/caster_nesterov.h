#pragma once

#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"

using namespace std;

class CasterNesterov : public CasterCPU {
public:
    CasterNesterov(int n, function<void(float)> onErr, function<void(vector<float2> &)> onPos)
            : CasterCPU(n, onErr, onPos), velocity(n, {0, 0}), f(n, {0, 0}) {}

    virtual void simul_step_cpu() override;

protected:
    vector<float2> velocity;
    vector<float2> f;

private:
    float2 force(DistElem distance);
};
