#include "caster/caster_adadelta_async.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

#define DECAYING_PARAM 0.9
#define EPS 0.00000001

std::pair<float, float> CasterAdadeltaAsync::force(DistElem distance) {
    std::pair<float, float> rv = {positions[distance.i].first - positions[distance.j].first,
                                  positions[distance.i].second - positions[distance.j].second};

    float r = sqrt(rv.first * rv.first + rv.second * rv.second + 0.00001f);
    float D = distance.r;

    float energy = (D - r) / r;

    return {rv.first * energy, rv.second * energy};
}

void CasterAdadeltaAsync::simul_step_cpu() {
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
        decGrad[i].first = decGrad[i].first * DECAYING_PARAM +
                           (1.0 - DECAYING_PARAM) * f[i].first * f[i].first;
        decGrad[i].second = decGrad[i].second * DECAYING_PARAM +
                            (1.0 - DECAYING_PARAM) * f[i].second * f[i].second;

        float deltax =
                f[i].first / sqrtf(EPS + decGrad[i].first) * sqrtf(EPS + decDelta[i].first);
        float deltay =
                f[i].second / sqrtf(EPS + decGrad[i].second) * sqrtf(EPS + decDelta[i].second);

        positions[i].first += deltax;
        positions[i].second += deltay;

        decDelta[i].first = decDelta[i].first * DECAYING_PARAM +
                            (1.0 - DECAYING_PARAM) * deltax * deltax;
        decDelta[i].second = decDelta[i].second * DECAYING_PARAM +
                             (1.0 - DECAYING_PARAM) * deltay * deltay;
    }
}
