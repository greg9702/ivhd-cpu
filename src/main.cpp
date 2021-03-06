#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include "caster/caster_momentum.h"
#include "caster/caster_adadelta.h"
#include "caster/caster_nesterov.h"
#include "caster/caster_adam.h"
#include "caster/caster_sgd.h"
#include "data.h"

using namespace std;
using namespace std::chrono;

string dataset_file;
string knn_file;
string experiment_name;
string algorithm_name;

unsigned iterations;
unsigned seed;
unsigned neighbours;
unsigned random_neighbours;

void parseArg(int argc, char *argv[]) {
    if (argc != 9) {
        cerr << "Expected 8 arguments:\n";
        cerr << "./ivhd dataset_file knn_file iterations neighbours random_neighbours experiment_name "
                "algorithm_name seed\n";
        exit(-1);
    }

    dataset_file = argv[1];
    knn_file = argv[2];
    iterations = stoi(argv[3]);
    neighbours = stoi(argv[4]);
    random_neighbours = stoi(argv[5]);
    experiment_name = argv[6];
    algorithm_name = argv[7];
    seed = stoi(argv[8]);
}

Caster *getCaster(int n, function<void(float)> onError,
                  function<void(vector<float2> &)> onPos) {

    if (algorithm_name == "sgd") {
        return new CasterSGD(n, onError, onPos);
    } else if (algorithm_name == "momentum") {
        return new CasterMomentum(n, onError, onPos);
    } else if (algorithm_name == "nesterov") {
        return new CasterNesterov(n, onError, onPos);
    } else if (algorithm_name == "adadelta") {
        return new CasterAdadelta(n, onError, onPos);
    } else if (algorithm_name == "adam") {
        return new CasterAdam(n, onError, onPos);
    } else {
        cerr << "Invalid algorithm_name. Expected one of: 'sgd', 'momentum', ";
        cerr << "'nesterov', 'adadelta', 'adam'\n";
        exit(-1);
    }
}

int main(int argc, char *argv[]) {
    parseArg(argc, argv);
    srand(seed);

    Data data;
    int n = data.load_mnist(dataset_file);

    system_clock::time_point now = system_clock::now();
    long start = time_point_cast<milliseconds>(now).time_since_epoch().count();
    long offset = 0;

    ofstream errFile;
    errFile.open(experiment_name + "_error");
    float minError = std::numeric_limits<float>::max();
    auto onError = [&](float err) -> void {
        now = system_clock::now();
        minError = min(minError, err);
        auto time = time_point_cast<milliseconds>(now).time_since_epoch().count() -
                    start - offset;

        errFile << time << " " << err << endl;
    };

    auto onPos = [&](vector<float2> &positions) -> void {
        now = system_clock::now();
        auto time = time_point_cast<milliseconds>(now).time_since_epoch().count() -
                    start - offset;

        ofstream posFile;
        posFile.open(experiment_name + "_" + to_string(time) + "_positions");

        for (unsigned i = 0; i < positions.size(); i++) {
            posFile << positions[i].x << " " << positions[i].y << " "
                    << data.labels[i] << endl;
        }
        posFile.close();

        system_clock::time_point end = system_clock::now();
        offset += time_point_cast<milliseconds>(end).time_since_epoch().count() -
                  time_point_cast<milliseconds>(now).time_since_epoch().count();
    };

    Caster *casterPtr = getCaster(n, onError, onPos);
    Caster &caster = *casterPtr;

    data.generateNearestDistances(caster, n, knn_file, neighbours);
    data.generateRandomDistances(caster, n, random_neighbours);
    for (int i = 0; i < n; i++) {
        caster.positions[i].x = rand() % 100000 / 100000.0;
        caster.positions[i].y = rand() % 100000 / 100000.0;
    }

    caster.prepare(data.labelsRef());

    now = system_clock::now();
    start = time_point_cast<milliseconds>(now).time_since_epoch().count();

    for (unsigned i = 0; i < iterations; i++) {
        if (i % 100 == 0) {
            cout << i << endl;
            ofstream results;
            results.open(experiment_name + "_result_" + std::to_string(i));
            for (int j = 0; j < n; j++) {
                results << caster.positions[j].x << " " << caster.positions[j].y << " "
                        << data.labels[j] << endl;
            }
            results.close();
        }
        caster.simul_step();
    }

    now = system_clock::now();
    auto totalTime =
            time_point_cast<milliseconds>(now).time_since_epoch().count() - start -
            offset;
    cerr << totalTime << endl;
    cerr << "minError: " << minError << endl;

    caster.finish();

    ofstream results;
    results.open(experiment_name + "_result");
    for (int i = 0; i < n; i++) {
        results << caster.positions[i].x << " " << caster.positions[i].y << " "
                << data.labels[i] << endl;
    }

    results.close();
    errFile.close();
    return 0;
}
