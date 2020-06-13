#include "caster/caster_sgd.h"
#include "caster_sgd.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

float2 CasterSGD::force(DistElem distance) {
    float2 posI = positions[distance.i];
    float2 posJ = positions[distance.j];

    float2 rv = {posI.x - posJ.x, posI.y - posJ.y};

    float r = sqrt(rv.x * rv.x + rv.y * rv.y + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.x * energy, rv.y * energy};
}

void CasterSGD::simul_step_cpu() {
    // calculate forces
    for (int i = 0; i < f.size(); i++) {
        f[i] = {0, 0};
    }

    for (int i = 0; i < distances.size(); i++) {
        float2 df = force(distances[i]);

        f[distances[i].i].x += df.x;
        f[distances[i].i].y += df.y;
        f[distances[i].j].x -= df.x;
        f[distances[i].j].y -= df.y;
    }

    // update positions
    for (int i = 0; i < positions.size(); i++) {
        positions[i].x += f[i].x * learning_rate;
        positions[i].y += f[i].y * learning_rate;
    }
}
