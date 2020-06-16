#include "caster/caster_momentum.h"
#include <cmath>
#include "caster/constants.h"

using namespace std;

float2 CasterMomentum::force(DistElem distance) {
    float2 posI = positions[distance.i];
    float2 posJ = positions[distance.j];

    float2 rv = {posI.x - posJ.x, posI.y - posJ.y};

    float r = sqrt(rv.x * rv.x + rv.y * rv.y + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.x * energy, rv.y * energy};
}

void CasterMomentum::simul_step_cpu() {
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
        momentum[i].x = (momentum_rate * momentum[i].x) + (learning_rate * f[i].x);
        momentum[i].y = (momentum_rate * momentum[i].y) + (learning_rate * f[i].y);
        positions[i].x += momentum[i].x;
        positions[i].y += momentum[i].y;
    }
}
