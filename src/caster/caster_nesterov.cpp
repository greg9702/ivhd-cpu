#include "caster/caster_nesterov.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

std::pair<float, float> CasterNesterov::force(DistElem distance) {
    std::pair<float, float> posI = positions[distance.i];
    std::pair<float, float> posJ = positions[distance.j];

    // estimate next positions with previous velocity
    posI.first += v[distance.i].first;
    posI.second += v[distance.i].second;
    posJ.first += v[distance.j].first;
    posJ.second += v[distance.j].second;

    std::pair<float, float> rv = {posI.first - posJ.first, posI.second - posJ.second};

    float r = sqrt(rv.first * rv.first + rv.second * rv.second + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.first * energy, rv.second * energy};
}

void CasterNesterov::simul_step_cpu() {
    // calculate forces
    for (int i = 0; i < f.size(); i++) {
        f[i] = {0, 0};
    }

    for (int i = 0; i < distances.size(); i++) {
        std::pair<float, float> df = force(distances[i]);

        if (distances[i].type == etRandom) {
            df.first *= w_random;
            df.second *= w_random;
        }

        f[distances[i].i].first += df.first;
        f[distances[i].i].second += df.second;
        f[distances[i].j].first -= df.first;
        f[distances[i].j].second -= df.second;
    }

    // update velicities and positions
    for (int i = 0; i < positions.size(); i++) {
        v[i].first = v[i].first * a_factor + f[i].first * b_factor;
        v[i].second = v[i].second * a_factor + f[i].second * b_factor;
        positions[i].first += v[i].first;
        positions[i].second += v[i].second;
    }
}
