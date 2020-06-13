#include "caster/caster_ab.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

std::pair<float, float> CasterAB::force(DistElem distance) {
    std::pair<float, float> rv = {positions[distance.i].first - positions[distance.j].first,
                                  positions[distance.i].second - positions[distance.j].second};

    float r = sqrt(rv.first * rv.first + rv.second * rv.second + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.first * energy, rv.second * energy};
}

void CasterAB::simul_step_cpu() {
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
