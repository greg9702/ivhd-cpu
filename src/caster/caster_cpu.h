#pragma once

#include <cmath>
#include <vector>
#include "caster/caster.h"
#include "distance.h"

using namespace std;

class CasterCPU : public Caster {
public:
    CasterCPU(int n, function<void(float)> onErr, function<void(vector<std::pair<float, float>> &)> onPos)
            : Caster(n, onErr, onPos) {}

    ~CasterCPU() {};

    virtual void simul_step() override {
        simul_step_cpu();
        if (it++ % 100 == 0) {
            float err = 0.0;
            for (auto &dist : distances) {
                float d = dist.r;
                std::pair<float, float> iPos = positions[dist.i];
                std::pair<float, float> jPos = positions[dist.j];
                std::pair<float, float> ij = {iPos.first - jPos.first, jPos.second - jPos.second};
                err += abs(d - sqrt(ij.first * ij.first + ij.second * ij.second));
            }
            onError(err);
        }
    };

    virtual void simul_step_cpu() = 0;

protected:
    unsigned it;
};
