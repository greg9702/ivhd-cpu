#include "caster/caster_adam.h"
#include <cmath>
#include <iostream>

using namespace std;

float2 CasterAdam::force(DistElem distance) {
    float2 posI = positions[distance.i];
    float2 posJ = positions[distance.j];

    float2 rv = {posI.x - posJ.x, posI.y - posJ.y};

    float r = sqrt(rv.x * rv.x + rv.y * rv.y + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.x * energy, rv.y * energy};
}

void CasterAdam::simul_step_cpu() {
    // calculate forces
    for (auto &i : f) {
        i = {0, 0};
    }

    for (auto &distance : distances) {
        float2 df = force(distance);

        if (distance.type == etRandom) {
            df.x *= w_random;
            df.y *= w_random;
        }

        f[distance.i].x += df.x;
        f[distance.i].y += df.y;
        f[distance.j].x -= df.x;
        f[distance.j].y -= df.y;
    }

    // update positions
    for (int i = 0; i < positions.size(); i++) {
        decGrad[i].x = decGrad[i].x * beta1 + (1.0 - beta1) * f[i].x;
        decGrad[i].y = decGrad[i].y * beta1 + (1.0 - beta1) * f[i].y;

        decSquaredGrad[i].x = decSquaredGrad[i].x * beta2 + (1.0 - beta2) * f[i].x * f[i].x;
        decSquaredGrad[i].y = decSquaredGrad[i].y * beta2 + (1.0 - beta2) * f[i].y * f[i].y;

        float adjustedDecGradX = decGrad[i].x / (1 - std::pow(beta1, epoch));
        float adjustedDecGradY = decGrad[i].y / (1 - std::pow(beta1, epoch));
        float adjustedDecSquaredGradX = decSquaredGrad[i].x / (1 - std::pow(beta2, epoch));
        float adjustedDecSquaredGradY = decSquaredGrad[i].y / (1 - std::pow(beta2, epoch));

        positions[i].x += learning_rate / sqrtf(epsilon + adjustedDecSquaredGradX) * adjustedDecGradX;
        positions[i].y += learning_rate / sqrtf(epsilon + adjustedDecSquaredGradY) * adjustedDecGradY;
    }

    epoch++;
}
