#include "caster/caster_adadelta_sync.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

#define DECAYING_PARAM 0.9
#define EPS 0.00000001

std::pair<float, float> CasterAdadeltaSync::force(DistElem distance) {
    std::pair<float, float> rv = {positions[distance.i].first - positions[distance.j].first,
                                  positions[distance.i].second - positions[distance.j].second};

    float r = sqrt(rv.first * rv.first + rv.second * rv.second + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.first * energy, rv.second * energy};
}

std::pair<float, float> CasterAdadeltaSync::calcForce(int i) {
    std::pair<float, float> df = {0, 0};
    for (int j = 0; j < neighbours[i].size(); j++) {
        std::pair<float, float> dfcomponent = force(neighbours[i][j]);
        if (neighbours[i][j].type == etRandom) {
            dfcomponent.first *= w_random;
            dfcomponent.second *= w_random;
        }
        if (distances[i].i == i) {
            df = {df.first + dfcomponent.first, df.second + dfcomponent.second};
        } else {
            df = {df.first - dfcomponent.first, df.second - dfcomponent.second};
        }
    }

    return df;
}

void CasterAdadeltaSync::simul_step_cpu() {
    // update velicities and positions
    for (int i = 0; i < positions.size(); i++) {
        std::pair<float, float> force = calcForce(i);

        decGrad[i].first = decGrad[i].first * DECAYING_PARAM +
                           (1.0 - DECAYING_PARAM) * force.first * force.first;
        decGrad[i].second = decGrad[i].second * DECAYING_PARAM +
                            (1.0 - DECAYING_PARAM) * force.second * force.second;

        float deltax =
                force.first / sqrtf(EPS + decGrad[i].first) * sqrtf(EPS + decDelta[i].first);
        float deltay =
                force.second / sqrtf(EPS + decGrad[i].second) * sqrtf(EPS + decDelta[i].second);

        positions[i].first += deltax;
        positions[i].second += deltay;

        decDelta[i].first = decDelta[i].first * DECAYING_PARAM +
                            (1.0 - DECAYING_PARAM) * deltax * deltax;
        decDelta[i].second = decDelta[i].second * DECAYING_PARAM +
                             (1.0 - DECAYING_PARAM) * deltay * deltay;
    }
}

void CasterAdadeltaSync::prepare(vector<int> &labels) {
    // initialize nn array
    for (int i = 0; i < distances.size(); i++) {
        neighbours[distances[i].i].push_back(distances[i]);
        neighbours[distances[i].j].push_back(distances[i]);
    }
}
