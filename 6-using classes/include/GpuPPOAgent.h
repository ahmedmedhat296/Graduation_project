#ifndef GPUPPOAGENT_H
#define GPUPPOAGENT_H

#include "GpuNet.h"
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>

using namespace std;

// GPU trajectory step (state/action on CPU, GPU does the heavy math)
struct GpuTrajStep {
    float* state;       // [stateDim]
    float* action;      // [actionDim] softmaxed weights
    float  reward;
    float  logProb;
    float  value;
    float  advantage;
    float  returnToGo;
};

class GpuPPOAgent {
public:
    int stateDim, actionDim;

    GpuNet* actor;   // stateDim -> 512 -> 256 -> actionDim
    GpuNet* critic;  // stateDim -> 512 -> 256 -> 1

    // PPO hyperparams
    float gamma, lambda, clipEps, entropyCoeff, noiseStd;
    int   ppoEpochs;

    // GPU buffer for gradient injection (actionDim & 1)
    float* d_actorGrad;
    float* d_criticGrad;
    float* d_probs;     // for softmax result
    float* d_noiseIn;   // GPU noise buffer

    // Host-side softmax probs (pulled from GPU)
    float* h_probs;
    float* h_logits;
    float* h_criticOut;

    vector<GpuTrajStep> trajectory;

    GpuPPOAgent(int sDim, int aDim);
    ~GpuPPOAgent();

    // Select action: FF actor, add noise, softmax, return logProb
    float selectAction(float* state, float* actionOut);

    // Get V(s)
    float getValue(float* state);

    // Store step
    void storeStep(float* state, float* action,
                   float reward, float logProb, float value);

    // Compute GAE advantages
    void computeGAE(float lastValue);

    // PPO update
    void update();

    // Clear trajectory
    void clearTrajectory();

    // Save/load
    void saveModels(const string& actorF, const string& criticF);
    void loadModels(const string& actorF, const string& criticF);

private:
    void cpuSoftmax(float* logits, float* probs, int n);
    float computeLogProb(float* logits, float* chosenAction, int n);
    void updateActor(float* state, float* oldAction,
                     float oldLogProb, float advantage);
    void updateCritic(float* state, float target);
    float gaussNoise();
};

#endif // GPUPPOAGENT_H
