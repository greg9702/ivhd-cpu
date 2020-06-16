#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"

using namespace std;

class CasterAdadelta : public CasterCPU {
public:
    CasterAdadelta(int n, function<void(float)> onErr, function<void(vector<float2> &)> onPos)
            : CasterCPU(n, onErr, onPos),
              f(n, {0, 0}),
              decGrad(n, {0, 0}),
              decDelta(n, {0, 0}) {}

    virtual void simul_step_cpu() override;

protected:
    vector<float2> f;
    vector<float2> decGrad;
    vector<float2> decDelta;

private:
    float2 force(DistElem distance);

    float epsilon = 0.00000001;
    float beta = 0.9;
};
