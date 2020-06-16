#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"

using namespace std;

class CasterAdam : public CasterCPU {
public:
    CasterAdam(int n, function<void(float)> onErr, function<void(vector<float2> &)> onPos)
            : CasterCPU(n, onErr, onPos),
              f(n, {0, 0}),
              decGrad(n, {0, 0}),
              decSquaredGrad(n, {0, 0}) {}

    virtual void simul_step_cpu() override;

protected:
    vector<float2> f;
    vector<float2> decGrad;
    vector<float2> decSquaredGrad;

private:
    float2 force(DistElem distance);

    float epoch = 1;
    float epsilon = 0.00000001;
    float beta1 = 0.9;
    float beta2 = 0.999;
};
