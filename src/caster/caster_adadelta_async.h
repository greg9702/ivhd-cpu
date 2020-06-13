#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"

using namespace std;

class CasterAdadeltaAsync : public CasterCPU {
public:
    CasterAdadeltaAsync(int n, function<void(float)> onErr,
                        function<void(vector<std::pair<float, float>> &)> onPos)
            : CasterCPU(n, onErr, onPos),
              v(n, {0, 0}),
              f(n, {0, 0}),
              decGrad(n, {0, 0}),
              decDelta(n, {0, 0}) {}

    virtual void simul_step_cpu() override;

protected:
    vector<std::pair<float, float>> v;
    vector<std::pair<float, float>> f;
    vector<std::pair<float, float>> decGrad;
    vector<std::pair<float, float>> decDelta;

private:
    std::pair<float, float> force(DistElem distance);

    float a_factor = 0.9;
    float b_factor = 0.002;
    float w_random = 0.01;
};
