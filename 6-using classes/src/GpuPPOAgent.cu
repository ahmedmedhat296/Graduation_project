// GpuPPOAgent.cu — PPO agent backed by GpuNet (cuBLAS + CUDA)
// Actor/Critic run entirely on GPU; trajectory buffer on CPU.

#include "GpuPPOAgent.h"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <ctime>

#define PI_F 3.14159265f

// FIX 3: Temperature > 1 flattens softmax output, preventing winner-takes-all collapse.
// At T=1 (original), a logit just 2 units higher gets ~7x more weight.
// At T=2, that same gap only gives ~2.7x more weight — far less extreme.
static const float SOFTMAX_TEMPERATURE = 2.0f;

// ============================================================
//  Constructor
// ============================================================
GpuPPOAgent::GpuPPOAgent(int sDim, int aDim)
    : stateDim(sDim), actionDim(aDim)
{
    gamma        = 0.99f;
    lambda       = 0.95f;
    clipEps      = 0.2f;
    entropyCoeff = 0.3f;   // FIX B: raised from 0.05 — must compete with PPO gradient
    noiseStd     = 0.5f;   // already good
    ppoEpochs    = 4;

    // --- Actor: stateDim -> 1024 -> 512 -> 256 -> actionDim ---
    int actorTopo[] = {1024, 512, 256, actionDim};
    actor = new GpuNet(actorTopo, 4, stateDim, 0.0003f);

    // --- Critic: stateDim -> 1024 -> 512 -> 256 -> 1 ---
    int criticTopo[] = {1024, 512, 256, 1};
    critic = new GpuNet(criticTopo, 4, stateDim, 0.001f);

    // GPU gradient buffers
    CUDA_CHECK(cudaMalloc(&d_actorGrad,  actionDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_criticGrad, 1          * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs,      actionDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_noiseIn,    actionDim * sizeof(float)));

    // CPU host buffers
    h_probs     = new float[actionDim];
    h_logits    = new float[actionDim];
    h_criticOut = new float[1];

    // Wire last-layer gradient pointers
    actor->Ls[actor->nLayers   - 1]->d_dOut = d_actorGrad;
    critic->Ls[critic->nLayers - 1]->d_dOut = d_criticGrad;

    printf("GpuPPOAgent: actor[%d->1024->512->256->%d] critic[%d->1024->512->256->1]\n",
           stateDim, actionDim, stateDim);
    printf("  entropyCoeff=%.2f  noiseStd=%.2f  temperature=%.1f\n",
           entropyCoeff, noiseStd, SOFTMAX_TEMPERATURE);
}

GpuPPOAgent::~GpuPPOAgent() {
    clearTrajectory();
    delete actor;
    delete critic;
    cudaFree(d_actorGrad);
    cudaFree(d_criticGrad);
    cudaFree(d_probs);
    cudaFree(d_noiseIn);
    delete[] h_probs;
    delete[] h_logits;
    delete[] h_criticOut;
}

// ============================================================
//  Utility: Gaussian noise (Box-Muller)
// ============================================================
float GpuPPOAgent::gaussNoise() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_F * u2);
}

// ============================================================
//  CPU softmax (member — used internally)
// ============================================================
void GpuPPOAgent::cpuSoftmax(float* logits, float* probs, int n) {
    float maxL = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxL) maxL = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - maxL); sum += probs[i]; }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

// ============================================================
//  FIX 3: temperature-scaled softmax — plain static helper,
//  no header change needed.
//  Divides raw logits by SOFTMAX_TEMPERATURE before softmax
//  so no single stock dominates the output distribution.
// ============================================================
static void softmaxTempered(float* rawLogits, float* probs, int n,
                             float temperature) {
    float maxL = rawLogits[0];
    for (int i = 1; i < n; i++) if (rawLogits[i] > maxL) maxL = rawLogits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((rawLogits[i] - maxL) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

// ============================================================
//  Log probability under Gaussian policy around softmax mean
//  Uses tempered softmax to match how actions were sampled.
// ============================================================
float GpuPPOAgent::computeLogProb(float* logits, float* chosenAction, int n) {
    float* probs = new float[n];
    softmaxTempered(logits, probs, n, SOFTMAX_TEMPERATURE);
    float logP = 0;
    for (int i = 0; i < n; i++) {
        float diff = chosenAction[i] - probs[i];
        logP += -0.5f * (diff * diff) / (noiseStd * noiseStd);
        logP += -logf(noiseStd) - 0.9189385f;
    }
    delete[] probs;
    return logP;
}

// ============================================================
//  selectAction: FF on GPU, pull logits, tempered softmax + noise
// ============================================================
float GpuPPOAgent::selectAction(float* state, float* actionOut) {
    actor->FF(state);
    actor->getOutput(h_logits, actionDim);

    // FIX 3: Apply temperature scaling + noise, THEN softmax
    // Without temperature, one slightly-higher logit captures ~99% of weight.
    float* noisyLogits = new float[actionDim];
    for (int i = 0; i < actionDim; i++)
        noisyLogits[i] = (h_logits[i] + noiseStd * gaussNoise()) / SOFTMAX_TEMPERATURE;

    cpuSoftmax(noisyLogits, actionOut, actionDim);

    // Log probability (uses tempered distribution to match sampling)
    float logProb = computeLogProb(h_logits, actionOut, actionDim);

    delete[] noisyLogits;
    return logProb;
}

// ============================================================
//  getValue: FF critic on GPU, pull scalar
// ============================================================
float GpuPPOAgent::getValue(float* state) {
    critic->FF(state);
    critic->getOutput(h_criticOut, 1);
    return h_criticOut[0];
}

// ============================================================
//  storeStep
// ============================================================
void GpuPPOAgent::storeStep(float* state, float* action,
                             float reward, float logProb, float value) {
    GpuTrajStep step;
    step.state  = new float[stateDim];
    step.action = new float[actionDim];
    memcpy(step.state,  state,  stateDim  * sizeof(float));
    memcpy(step.action, action, actionDim * sizeof(float));
    step.reward     = reward;
    step.logProb    = logProb;
    step.value      = value;
    step.advantage  = 0;
    step.returnToGo = 0;
    trajectory.push_back(step);
}

// ============================================================
//  computeGAE: Generalized Advantage Estimation
// ============================================================
void GpuPPOAgent::computeGAE(float lastValue) {
    int T = (int)trajectory.size();
    if (T == 0) return;

    float gae = 0;
    for (int t = T - 1; t >= 0; t--) {
        float nextVal = (t == T - 1) ? lastValue : trajectory[t + 1].value;
        float delta = trajectory[t].reward + gamma * nextVal - trajectory[t].value;
        gae = delta + gamma * lambda * gae;
        trajectory[t].advantage  = gae;
        trajectory[t].returnToGo = gae + trajectory[t].value;
    }

    // Normalise advantages
    float mean = 0, var = 0;
    for (int t = 0; t < T; t++) mean += trajectory[t].advantage;
    mean /= T;
    for (int t = 0; t < T; t++) {
        float d = trajectory[t].advantage - mean; var += d * d;
    }
    var /= T;
    float std = sqrtf(var + 1e-8f);
    for (int t = 0; t < T; t++)
        trajectory[t].advantage = (trajectory[t].advantage - mean) / std;
}

// ============================================================
//  updateActor: PPO-Clip gradient injection for one step
// ============================================================
void GpuPPOAgent::updateActor(float* state, float* oldAction,
                               float oldLogProb, float advantage) {
    actor->FF(state);
    actor->getOutput(h_logits, actionDim);

    float newLogProb = computeLogProb(h_logits, oldAction, actionDim);
    float ratio      = expf(newLogProb - oldLogProb);
    float clip       = (ratio < 1.0f - clipEps) ? 1.0f - clipEps :
                       (ratio > 1.0f + clipEps) ? 1.0f + clipEps : ratio;

    // FIX 3: Use tempered softmax here too — must match selectAction
    // If we compute probs without temperature, the gradient points the wrong direction
    float* probs = new float[actionDim];
    softmaxTempered(h_logits, probs, actionDim, SOFTMAX_TEMPERATURE);

    float* gradH = new float[actionDim];
    for (int i = 0; i < actionDim; i++) {
        // PPO policy gradient: pushes distribution toward oldAction
        float ppoG = advantage * (oldAction[i] - probs[i]) / (noiseStd * noiseStd);

        // FIX B: Entropy gradient (high coefficient means this genuinely competes
        // with ppoG instead of being drowned out by it).
        // -d/d(logit) of -sum(p*log(p)) = -(log(p) + 1) per output
        float entG = -entropyCoeff * (logf(probs[i] + 1e-10f) + 1.0f);

        gradH[i] = ppoG + entG;
    }

    CUDA_CHECK(cudaMemcpy(d_actorGrad, gradH, actionDim * sizeof(float), cudaMemcpyHostToDevice));
    actor->BP();

    delete[] probs;
    delete[] gradH;
}

// ============================================================
//  updateCritic: MSE gradient for one step
// ============================================================
void GpuPPOAgent::updateCritic(float* state, float target) {
    critic->FF(state);
    critic->getOutput(h_criticOut, 1);
    float grad = target - h_criticOut[0];
    CUDA_CHECK(cudaMemcpy(d_criticGrad, &grad, sizeof(float), cudaMemcpyHostToDevice));
    critic->BP();
}

// ============================================================
//  update: K PPO epochs over collected trajectory
// ============================================================
void GpuPPOAgent::update() {
    int T = (int)trajectory.size();
    if (T == 0) return;

    for (int epoch = 0; epoch < ppoEpochs; epoch++) {
        actor->clearGrads();
        critic->clearGrads();
        for (int t = 0; t < T; t++) {
            updateActor(trajectory[t].state, trajectory[t].action,
                        trajectory[t].logProb, trajectory[t].advantage);
            updateCritic(trajectory[t].state, trajectory[t].returnToGo);
        }
        actor->update(T);
        critic->update(T);
    }
    cudaDeviceSynchronize();
}

// ============================================================
//  clearTrajectory
// ============================================================
void GpuPPOAgent::clearTrajectory() {
    for (auto& s : trajectory) {
        delete[] s.state;
        delete[] s.action;
    }
    trajectory.clear();
}

// ============================================================
//  save / load
// ============================================================
void GpuPPOAgent::saveModels(const string& actorF, const string& criticF) {
    actor->save(actorF);
    critic->save(criticF);
}

void GpuPPOAgent::loadModels(const string& actorF, const string& criticF) {
    actor->load(actorF);
    critic->load(criticF);
}
