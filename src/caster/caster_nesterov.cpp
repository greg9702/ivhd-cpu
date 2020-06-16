#include "caster/caster_nesterov.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include "caster/constants.h"
#include <iostream>

using namespace std;

float2 CasterNesterov::force(DistElem distance) {
    float2 posI = positions[distance.i];
    float2 posJ = positions[distance.j];

    // estimate next positions with previous velocity
    posI.x += momentum_rate * velocity[distance.i].x;
    posI.y += momentum_rate * velocity[distance.i].y;
    posJ.x += momentum_rate * velocity[distance.j].x;
    posJ.y += momentum_rate * velocity[distance.j].y;

    float2 rv = {posI.x - posJ.x, posI.y - posJ.y};

    float r = sqrt(rv.x * rv.x + rv.y * rv.y + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.x * energy, rv.y * energy};
}

void CasterNesterov::simul_step_cpu() {
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
        velocity[i].x = (momentum_rate * velocity[i].x) + (learning_rate * f[i].x);
        velocity[i].y = (momentum_rate * velocity[i].y) + (learning_rate * f[i].y);
        positions[i].x += velocity[i].x;
        positions[i].y += velocity[i].y;
    }
}
