#ifndef AGENT_H
#define AGENT_H

#include <iostream>
#include <vector>
#include <thread>
#include "environment.h"
#include "net.h"
using namespace std;

class agent{
public:

    net* actor;
    net* critic;
    int nlayers;
    int steps;
    double* action;
    double* state;
    double** states;
    double** sampled_actions;
    double** mean_actions;
    double* log_probs;
    double std_dev;
    tradingenv* env;
    int num_threads;
    std::vector<tradingenv*> thread_envs;
    std::vector<net*> thread_actors;
    std::vector<net*> thread_critics;
    int  episode;
    double gamma;
    double lambda;
    double clip;
    double value_clip;          // Value function clipping range
    double reward_clip_max;     // Max absolute reward (reward clipping)
    double reward_scale;        // Reward scaling factor
    int ppo_epochs;
    int action_size;
    int state_size;
    int feature_size;           // total_features from env (num_stocks * 10)
    int concat_after_layer;     // which hidden layer to concat after (configurable)
    int trajs_batch;
    double* rewards;
    double avg_rwd;
    double avg_traj_return;
    double* traj_returns_exact;
    double c_loss;
    double* advs;
    double* softmaxed_act;
    double* values;
    double* old_values;         // V(s) from rollout, for value function clipping
    double* returns;            // Discounted returns (V_target = adv + V_old)
    int*actor_layers;
    int*critic_layers;
    double** stored_features;   // [steps x feature_size] stored features for PPO replay
    double* act_dist();
    double* sample_action(double *m, int action_size);
    double compute_value();
    void softmax_action(double* samp_act);
    double log_prob(double* act);
    void play_batch();
    void calc_advs();
    void norm_advs();
    void update_critic();
    double gaussian_noise();
    void update_actor();
    void train(int epochs);
    void compute_critic_loss();
    void compute_avg_rwd();
    agent(tradingenv* _env);
    ~agent();
};

#endif // AGENT_H
