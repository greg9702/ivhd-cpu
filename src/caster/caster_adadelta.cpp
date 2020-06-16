#include "caster/caster_adadelta.h"
#include <cmath>
#include "caster/constants.h"

using namespace std;

float2 CasterAdadelta::force(DistElem distance) {
    float2 posI = positions[distance.i];
    float2 posJ = positions[distance.j];

    float2 rv = {posI.x - posJ.x, posI.y - posJ.y};

    float r = sqrt(rv.x * rv.x + rv.y * rv.y + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.x * energy, rv.y * energy};
}

void CasterAdadelta::simul_step_cpu() {
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
        decGrad[i].x = decGrad[i].x * beta + (1.0 - beta) * f[i].x * f[i].x;
        decGrad[i].y = decGrad[i].y * beta + (1.0 - beta) * f[i].y * f[i].y;

        float deltax = f[i].x / sqrtf(epsilon + decGrad[i].x) * sqrtf(epsilon + decDelta[i].x);
        float deltay = f[i].y / sqrtf(epsilon + decGrad[i].y) * sqrtf(epsilon + decDelta[i].y);

        positions[i].x += deltax;
        positions[i].y += deltay;

        decDelta[i].x = decDelta[i].x * beta + (1.0 - beta) * deltax * deltax;
        decDelta[i].y = decDelta[i].y * beta + (1.0 - beta) * deltay * deltay;
    }
}
