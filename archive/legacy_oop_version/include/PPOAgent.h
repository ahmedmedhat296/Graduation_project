#ifndef PPOAGENT_H
#define PPOAGENT_H

#include "net.h"
#include "trainSet.h"
#include "MarketData.h"
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;

// ============================================================
//  Trajectory step: stores one (s, a, r, logprob, value)
// ============================================================
struct TrajectoryStep {
    double* state;       // [stateDim]
    double* action;      // [actionDim] — the softmaxed portfolio weights
    double  reward;
    double  logProb;     // log probability of the action
    double  value;       // V(s) from critic
    double  advantage;   // GAE advantage
    double  returnToGo;  // discounted return
};

// ============================================================
//  PPOAgent: actor-critic PPO with the existing net framework
// ============================================================
class PPOAgent {
public:
    int stateDim;
    int actionDim;

    // Networks (use trainSet as a dummy — net requires one)
    trainSet* dummyTS_actor;
    trainSet* dummyTS_critic;
    net* actor;   // outputs actionDim logits
    net* critic;  // outputs 1 value

    // PPO hyperparameters
    double gamma;       // discount factor
    double lambda;      // GAE lambda
    double clipEps;     // PPO clip epsilon
    int    ppoEpochs;   // K update epochs per batch
    double actorLR;
    double criticLR;
    double entropyCoeff;

    // Noise for exploration
    double noiseStd;

    // Trajectory buffer
    vector<TrajectoryStep> trajectory;

    PPOAgent(int sDim, int aDim);
    ~PPOAgent();

    // Select action given state, returns log probability
    double selectAction(double* state, double* actionOut);

    // Get value V(s)
    double getValue(double* state);

    // Store a trajectory step
    void storeStep(double* state, double* action,
                   double reward, double logProb, double value);

    // Compute GAE advantages and returns
    void computeGAE(double lastValue);

    // PPO update on collected trajectory
    void update();

    // Save/load actor and critic
    void saveModels(const string& actorFile, const string& criticFile);
    void loadModels(const string& actorFile, const string& criticFile);

    // Clear trajectory buffer
    void clearTrajectory();

private:
    // Softmax over buffer of size n
    void softmax(double* logits, double* probs, int n);

    // Log probability of the chosen action under the softmaxed distribution
    // (treating each weight as a Gaussian with small std around the softmax output)
    double computeLogProb(double* actionLogits, double* chosenAction, int n);

    // Manual forward pass and gradient injection for PPO objectives
    void updateActor(double* state, double* oldAction,
                     double oldLogProb, double advantage);
    void updateCritic(double* state, double target);
};

#endif // PPOAGENT_H
