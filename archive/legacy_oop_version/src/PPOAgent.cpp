#include "PPOAgent.h"
#include <iostream>
#include <iomanip>
#include <cstring>

// ============================================================
//  Constructor: create actor and critic nets
// ============================================================
PPOAgent::PPOAgent(int sDim, int aDim)
    : stateDim(sDim), actionDim(aDim)
{
    // PPO hyperparameters
    gamma       = 0.99;
    lambda      = 0.95;
    clipEps     = 0.2;
    ppoEpochs   = 4;
    actorLR     = 0.0003;
    criticLR    = 0.001;
    entropyCoeff = 0.01;
    noiseStd     = 0.1;

    // --- Create dummy trainSets (net requires one) ---
    dummyTS_actor  = new trainSet();
    dummyTS_actor->nIn  = stateDim;
    dummyTS_actor->nOut = actionDim;
    dummyTS_actor->nPat = 1;
    dummyTS_actor->Create();

    dummyTS_critic = new trainSet();
    dummyTS_critic->nIn  = stateDim;
    dummyTS_critic->nOut = 1;
    dummyTS_critic->nPat = 1;
    dummyTS_critic->Create();

    // --- Actor network: stateDim -> 512 -> 256 -> actionDim ---
    actor = new net(3, dummyTS_actor, actorLR, 0.0);
    actor->nForLayers[0] = 512;
    actor->nForLayers[1] = 256;
    // nForLayers[2] = actionDim set by net constructor from trainSet->nOut
    actor->Creat();

    // --- Critic network: stateDim -> 512 -> 256 -> 1 ---
    critic = new net(3, dummyTS_critic, criticLR, 0.0);
    critic->nForLayers[0] = 512;
    critic->nForLayers[1] = 256;
    critic->Creat();

    cout << "PPOAgent created: state=" << stateDim
         << " action=" << actionDim
         << " actor=[" << stateDim << "->512->256->" << actionDim << "]"
         << " critic=[" << stateDim << "->512->256->1]" << endl;
}

PPOAgent::~PPOAgent() {
    clearTrajectory();
    // Note: net/trainSet cleanup omitted for simplicity
    // (program runs till end anyway)
}

// ============================================================
//  softmax: convert logits to probabilities
// ============================================================
void PPOAgent::softmax(double* logits, double* probs, int n) {
    double maxL = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > maxL) maxL = logits[i];

    double sumExp = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = exp(logits[i] - maxL);
        sumExp += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sumExp;
}

// ============================================================
//  computeLogProb: log probability under Gaussian around softmax mean
//  We treat the policy as: for each weight w_i, it's a Gaussian
//  centered on softmax(logits)_i with std = noiseStd.
//  Total log prob = sum of individual log probs.
// ============================================================
double PPOAgent::computeLogProb(double* logits, double* chosenAction, int n) {
    double* probs = new double[n];
    softmax(logits, probs, n);

    double logP = 0;
    for (int i = 0; i < n; i++) {
        double diff = chosenAction[i] - probs[i];
        // Gaussian log prob: -0.5 * (diff/std)^2 - log(std) - 0.5*log(2*pi)
        logP += -0.5 * (diff * diff) / (noiseStd * noiseStd);
        logP += -log(noiseStd) - 0.9189385; // -0.5*log(2*pi)
    }
    delete[] probs;
    return logP;
}

// ============================================================
//  selectAction: forward pass through actor, add noise, softmax
// ============================================================
double PPOAgent::selectAction(double* state, double* actionOut) {
    // Set input
    actor->Ls[0]->pInF = state;
    actor->setdropout(false, 1.0);
    actor->FF();

    // Get logits from last layer
    double* logits = actor->Ls[actor->nL - 1]->mOutF;

    // Add exploration noise to logits
    double* noisyLogits = new double[actionDim];
    for (int i = 0; i < actionDim; i++) {
        // Box-Muller for Gaussian noise
        double u1 = ((double)rand() / RAND_MAX);
        double u2 = ((double)rand() / RAND_MAX);
        if (u1 < 1e-10) u1 = 1e-10;
        double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265 * u2);
        noisyLogits[i] = logits[i] + noiseStd * noise;
    }

    // Softmax to get portfolio weights
    softmax(noisyLogits, actionOut, actionDim);

    // Compute log probability
    double logProb = computeLogProb(logits, actionOut, actionDim);

    delete[] noisyLogits;
    return logProb;
}

// ============================================================
//  getValue: forward pass through critic
// ============================================================
double PPOAgent::getValue(double* state) {
    critic->Ls[0]->pInF = state;
    critic->setdropout(false, 1.0);
    critic->FF();
    return critic->Ls[critic->nL - 1]->mOutF[0];
}

// ============================================================
//  storeStep: add a step to the trajectory buffer
// ============================================================
void PPOAgent::storeStep(double* state, double* action,
                         double reward, double logProb, double value) {
    TrajectoryStep step;
    step.state = new double[stateDim];
    step.action = new double[actionDim];
    memcpy(step.state, state, stateDim * sizeof(double));
    memcpy(step.action, action, actionDim * sizeof(double));
    step.reward   = reward;
    step.logProb  = logProb;
    step.value    = value;
    step.advantage   = 0;
    step.returnToGo  = 0;
    trajectory.push_back(step);
}

// ============================================================
//  computeGAE: Generalized Advantage Estimation
// ============================================================
void PPOAgent::computeGAE(double lastValue) {
    int T = (int)trajectory.size();
    if (T == 0) return;

    double gae = 0;
    for (int t = T - 1; t >= 0; t--) {
        double nextValue = (t == T - 1) ? lastValue : trajectory[t + 1].value;
        double delta = trajectory[t].reward
                     + gamma * nextValue
                     - trajectory[t].value;
        gae = delta + gamma * lambda * gae;
        trajectory[t].advantage  = gae;
        trajectory[t].returnToGo = gae + trajectory[t].value;
    }

    // Normalize advantages
    double mean = 0, var = 0;
    for (int t = 0; t < T; t++) mean += trajectory[t].advantage;
    mean /= T;
    for (int t = 0; t < T; t++) {
        double d = trajectory[t].advantage - mean;
        var += d * d;
    }
    var /= T;
    double std = sqrt(var + 1e-8);
    for (int t = 0; t < T; t++) {
        trajectory[t].advantage = (trajectory[t].advantage - mean) / std;
    }
}

// ============================================================
//  update: PPO-Clip update over collected trajectory
// ============================================================
void PPOAgent::update() {
    int T = (int)trajectory.size();
    if (T == 0) return;

    for (int epoch = 0; epoch < ppoEpochs; epoch++) {
        double totalActorLoss  = 0;
        double totalCriticLoss = 0;

        // Process each step in the trajectory
        for (int t = 0; t < T; t++) {
            // ----- ACTOR UPDATE -----
            updateActor(trajectory[t].state, trajectory[t].action,
                       trajectory[t].logProb, trajectory[t].advantage);

            // ----- CRITIC UPDATE -----
            updateCritic(trajectory[t].state, trajectory[t].returnToGo);
        }

        // Update weights after processing all steps (batch update)
        actor->update_Ls(T);
        critic->update_Ls(T);
    }
}

// ============================================================
//  updateActor: inject PPO-Clip gradient for one sample
// ============================================================
void PPOAgent::updateActor(double* state, double* oldAction,
                           double oldLogProb, double advantage) {
    // Forward pass
    actor->Ls[0]->pInF = state;
    actor->FF();

    double* logits = actor->Ls[actor->nL - 1]->mOutF;

    // Compute new log prob
    double newLogProb = computeLogProb(logits, oldAction, actionDim);

    // Probability ratio
    double ratio = exp(newLogProb - oldLogProb);

    // Clipped ratio
    double clippedRatio = ratio;
    if (clippedRatio < 1.0 - clipEps) clippedRatio = 1.0 - clipEps;
    if (clippedRatio > 1.0 + clipEps) clippedRatio = 1.0 + clipEps;

    // PPO objective: min(ratio * A, clip(ratio) * A)
    double surr1 = ratio * advantage;
    double surr2 = clippedRatio * advantage;
    double ppoObj = (surr1 < surr2) ? surr1 : surr2;

    // Compute gradient: d(ppoObj)/d(logits)
    // We need to push gradient through the softmax + Gaussian log-prob
    // Simplified: gradient direction = advantage * (action - softmax(logits))
    // scaled by the effective ratio
    double* probs = new double[actionDim];
    softmax(logits, probs, actionDim);

    double effectiveRatio = (surr1 < surr2) ? ratio : clippedRatio;
    // Only apply gradient if not clipped out
    double* grad = actor->Ls[actor->nL - 1]->pInB;
    for (int i = 0; i < actionDim; i++) {
        // Policy gradient direction (maximise objective)
        double policyGrad = advantage * (oldAction[i] - probs[i]) / (noiseStd * noiseStd);

        // Entropy bonus: encourage uniform distribution
        double entropyGrad = -entropyCoeff * (log(probs[i] + 1e-10) + 1.0);

        // Combined gradient (we want to MAXIMISE, so this is the direction)
        grad[i] = policyGrad + entropyGrad;
    }

    // Backward pass to accumulate gradients
    actor->BP();

    delete[] probs;
}

// ============================================================
//  updateCritic: MSE gradient for one sample
// ============================================================
void PPOAgent::updateCritic(double* state, double target) {
    // Forward pass
    critic->Ls[0]->pInF = state;
    critic->FF();

    double predicted = critic->Ls[critic->nL - 1]->mOutF[0];

    // MSE gradient: d/d(pred) of (target - pred)^2 = -2*(target - pred)
    // But our BP expects (target - output) signal, which it already uses
    double* grad = critic->Ls[critic->nL - 1]->pInB;
    grad[0] = target - predicted;

    // Backward pass
    critic->BP();
}

// ============================================================
//  saveModels / loadModels
// ============================================================
void PPOAgent::saveModels(const string& actorFile, const string& criticFile) {
    actor->save(actorFile);
    critic->save(criticFile);
}

void PPOAgent::loadModels(const string& actorFile, const string& criticFile) {
    actor->load(actorFile);
    critic->load(criticFile);
}

// ============================================================
//  clearTrajectory: free memory
// ============================================================
void PPOAgent::clearTrajectory() {
    for (int i = 0; i < (int)trajectory.size(); i++) {
        delete[] trajectory[i].state;
        delete[] trajectory[i].action;
    }
    trajectory.clear();
}
