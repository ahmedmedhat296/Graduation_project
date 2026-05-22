#include "agent.h"
agent::agent(tradingenv* _env)
{
    env=_env;
    trajs_batch=250;

    num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4;

    // Adjust trajs_batch so it divides evenly by num_threads
    if (trajs_batch % num_threads != 0) {
        trajs_batch = (trajs_batch / num_threads + 1) * num_threads;
    }

    episode=50;
    steps=trajs_batch*episode;
    action_size=env->portfolio_size;
    state_size=env->state_size;
    feature_size=env->total_features;
    gamma=0.95;
    lambda=0.99;
    concat_after_layer= -1;
    action=matD(action_size);
    sampled_actions=matD(steps,action_size);
    mean_actions=matD(steps,action_size);
    states=matD(steps,state_size);
    stored_features=matD(steps,feature_size);
    values=matD(steps);
    old_values=matD(steps);
    returns=matD(steps);
    log_probs=matD(steps);
    rewards=matD(steps);
    traj_returns_exact=matD(trajs_batch);
    advs=matD(steps);
    softmaxed_act=matD(action_size);
    nlayers=4;
    std_dev=1;
    ppo_epochs = 3;
    clip = 0.2;
    value_clip = 0.2;         // Value function clipping range (same as policy clip)
    reward_clip_max = 5.0;    // Clip extreme outlier rewards only (rewards are ~±0.5 to ±2)
    reward_scale = 1.0;       // No scaling needed — rewards are already small (log returns * 100)
    actor_layers=new int[nlayers];
    critic_layers=new int[nlayers];
    actor_layers[0]=32;
    actor_layers[1]=16;
    actor_layers[2]=8;
    actor_layers[3]=action_size;
    critic_layers[0]=32;
    critic_layers[1]=16;
    critic_layers[2]=8;
    critic_layers[3]=1;
    actor= new net(nlayers,state_size,actor_layers, concat_after_layer, feature_size);
    actor->Ls[nlayers-1]->pInB = new double[action_size];
    actor->Ls[nlayers-1]->isLinear=true;
    critic= new net(nlayers,state_size,critic_layers, concat_after_layer, feature_size);
    critic->Ls[nlayers-1]->pInB = new double[1];
    critic->Ls[nlayers-1]->isLinear=true;

    for(int i = 0; i < num_threads; i++){
        thread_envs.push_back(new tradingenv(env->data, env->start_idx, env->end_idx));

        net* l_actor = new net(nlayers, state_size, actor_layers, concat_after_layer, feature_size);
        l_actor->Ls[nlayers-1]->pInB = new double[action_size];
        l_actor->Ls[nlayers-1]->isLinear = true;
        thread_actors.push_back(l_actor);

        net* l_critic = new net(nlayers, state_size, critic_layers, concat_after_layer, feature_size);
        l_critic->Ls[nlayers-1]->pInB = new double[1];
        l_critic->Ls[nlayers-1]->isLinear = true;
        thread_critics.push_back(l_critic);
    }
}
double* agent::act_dist()
{
    state=env->state;
    actor->concat_features=env->features;
    actor->Ls[0]->pInF=state;
    actor->FF();
    return actor->Ls[nlayers-1]->mOutF;
}
double agent::gaussian_noise()
{
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}
double* agent::sample_action(double *m, int action_size)
{
    double noise;
    for(int i = 0;i<action_size;i++)
    {
         noise = gaussian_noise();
         action[i] = m[i]+std_dev*noise;
    }
    return action;
}
double agent::compute_value()
{
    state=env->state;
    critic->concat_features=env->features;
    critic->Ls[0]->pInF=state;
    critic->FF();
    return critic->Ls[nlayers-1]->mOutF[0];
}
double agent::log_prob(double* act)
{
    double log_p = 0.0;
    double* mu = actor->Ls[nlayers-1]->mOutF;
    for(int i = 0; i < action_size; i++)
    {
        double diff = act[i] - mu[i];
        log_p -= 0.5 * ((diff * diff) / (std_dev * std_dev) + log(2.0 * M_PI * std_dev * std_dev));
    }
    return log_p;
}
void agent::softmax_action(double* samp_act)
{
    // Find max value for numerical stability
    double max_val = samp_act[0];
    for(int a = 1; a < action_size; a++) {
        if(samp_act[a] > max_val) max_val = samp_act[a];
    }

    // Calculate exponentials and their sum
    double sum_exp = 0.0;
    for(int a = 0; a < action_size; a++) {
        softmaxed_act[a] = exp(samp_act[a] - max_val);
        sum_exp += softmaxed_act[a];
    }

    // Normalize to get valid probabilities (portfolio weights)
    for(int a = 0; a < action_size; a++) {
        softmaxed_act[a] /= sum_exp;
    }
}
void agent::play_batch()
{
    actor->setdropout(false);
    critic->setdropout(false);

    // Sync local networks with the current global weights
    for(int i = 0; i < num_threads; i++) {
        thread_actors[i]->copy_weights_from(actor);
        thread_critics[i]->copy_weights_from(critic);
        thread_actors[i]->setdropout(false);
        thread_critics[i]->setdropout(false);
    }

    int trajs_per_thread = trajs_batch / num_threads;
    std::vector<std::thread> workers;

    for (int t_id = 0; t_id < num_threads; t_id++) {
        workers.push_back(std::thread([this, t_id, trajs_per_thread]() {
            tradingenv* local_env = thread_envs[t_id];
            net* local_actor = thread_actors[t_id];
            net* local_critic = thread_critics[t_id];

            double* local_samp_act = new double[action_size];
            double* local_softmaxed_act = new double[action_size];

            for (int t = 0; t < trajs_per_thread; t++) {
                int traj = t_id * trajs_per_thread + t;
                local_env->reset(episode);
                double initial_cash = local_env->cash;

                for (int j = 0; j < episode; j++) {
                    int step_idx = traj * episode + j;

                    // 1. Record the current state and features
                    for (int s = 0; s < state_size; s++) {
                        states[step_idx][s] = local_env->state[s];
                    }
                    for (int f = 0; f < feature_size; f++) {
                        stored_features[step_idx][f] = local_env->features[f];
                    }

                    // 2. Forward pass actor (thread-safe on local_actor)
                    local_actor->concat_features = local_env->features;
                    local_actor->Ls[0]->pInF = local_env->state;
                    local_actor->FF();
                    double* mu = local_actor->Ls[nlayers - 1]->mOutF;

                    for (int a = 0; a < action_size; a++) {
                        mean_actions[step_idx][a] = mu[a];
                    }

                    // 3. Sample an action
                    for (int a = 0; a < action_size; a++) {
                        local_samp_act[a] = mu[a] + std_dev * gaussian_noise();
                        sampled_actions[step_idx][a] = local_samp_act[a];
                    }

                    // 4. Forward pass critic (thread-safe on local_critic)
                    local_critic->concat_features = local_env->features;
                    local_critic->Ls[0]->pInF = local_env->state;
                    local_critic->FF();
                    values[step_idx] = local_critic->Ls[nlayers - 1]->mOutF[0];

                    // 5. Compute log probability
                    double log_p = 0.0;
                    for (int i = 0; i < action_size; i++) {
                        double diff = local_samp_act[i] - mu[i];
                        log_p -= 0.5 * ((diff * diff) / (std_dev * std_dev) + log(2.0 * M_PI * std_dev * std_dev));
                    }
                    log_probs[step_idx] = log_p;

                    // Compute softmax
                    double max_val = local_samp_act[0];
                    for (int a = 1; a < action_size; a++) {
                        if (local_samp_act[a] > max_val) max_val = local_samp_act[a];
                    }
                    double sum_exp = 0.0;
                    for (int a = 0; a < action_size; a++) {
                        local_softmaxed_act[a] = exp(local_samp_act[a] - max_val);
                        sum_exp += local_softmaxed_act[a];
                    }
                    for (int a = 0; a < action_size; a++) {
                        local_softmaxed_act[a] /= sum_exp;
                    }

                    // To populate the global softmaxed_act array for printing at end of train()
                    if (t_id == 0 && t == trajs_per_thread - 1 && j == episode - 1) {
                        for (int a = 0; a < action_size; a++) {
                            softmaxed_act[a] = local_softmaxed_act[a];
                        }
                    }

                    // 6. Step the environment forward
                    local_env->step(local_softmaxed_act);

                    // 7. Record the reward (with clipping and scaling)
                    double raw_reward = local_env->reward;
                    // Reward clipping
                    if (raw_reward > reward_clip_max) raw_reward = reward_clip_max;
                    if (raw_reward < -reward_clip_max) raw_reward = -reward_clip_max;
                    // Reward scaling
                    rewards[step_idx] = raw_reward * reward_scale;

                    if (j == episode - 1) {
                        traj_returns_exact[traj] = (local_env->portfolio_value - initial_cash) / initial_cash;
                    }
                }
            }

            delete[] local_samp_act;
            delete[] local_softmaxed_act;
        }));
    }

    for (int t_id = 0; t_id < num_threads; t_id++) {
        workers[t_id].join();
    }

    double total_exact_ret = 0.0;
    for (int traj = 0; traj < trajs_batch; traj++) {
        total_exact_ret += traj_returns_exact[traj];
    }
    avg_traj_return = total_exact_ret / trajs_batch;
}
void agent::calc_advs()
{
    // Iterate over each trajectory in the batch
    for(int traj = 0; traj < trajs_batch; traj++)
    {
        double gae_running = 0.0;

        // Calculate the starting index of the current trajectory in the flat arrays
        int offset = traj * episode;

        // Iterate backwards through the episode
        for(int t = episode - 1; t >= 0; t--)
        {
            int step_idx = offset + t;

            // If it's the last step of the episode, the next state value is typically 0
            // Otherwise, it's the value of the next step in the same trajectory
            double next_value = (t == episode - 1) ? 0.0 : values[step_idx + 1];

            // Calculate TD error: delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            double delta = rewards[step_idx] + gamma * next_value - values[step_idx];

            // Calculate GAE: A_t = delta + gamma * lambda * A_{t+1}
            gae_running = delta + gamma * lambda * gae_running;

            // Store the computed advantage
            advs[step_idx] = gae_running;

            // Compute value targets: returns = advantages + old values
            returns[step_idx] = advs[step_idx] + values[step_idx];

            // Store old values for value function clipping
            old_values[step_idx] = values[step_idx];
        }
    }
}
void agent::norm_advs()
{
    double sum = 0.0;

    // 1. Calculate the mean of the advantages
    for(int i = 0; i < steps; i++)
    {
        sum += advs[i];
    }
    double mean = sum / steps;

    // 2. Calculate the standard deviation
    double sq_sum = 0.0;
    for(int i = 0; i < steps; i++)
    {
        double diff = advs[i] - mean;
        sq_sum += diff * diff;
    }
    double variance = sq_sum / steps;
    double std_dev_adv = sqrt(variance);

    // Small epsilon to prevent division by zero
    double epsilon = 1e-8;

    // 3. Normalize the advantages in-place
    for(int i = 0; i < steps; i++)
    {
        advs[i] = (advs[i] - mean) / (std_dev_adv + epsilon);
    }
}
void agent::update_critic()
{
    // 1. Clear previous accumulated gradients
    critic->clear();
    critic->setdropout(true,0.8);

    for(int i = 0; i < num_threads; i++) {
        thread_critics[i]->copy_weights_from(critic);
        thread_critics[i]->copy_mask_from(critic);
        thread_critics[i]->clear();
    }

    int steps_per_thread = steps / num_threads;
    std::vector<std::thread> workers;

    for (int t_id = 0; t_id < num_threads; t_id++) {
        workers.push_back(std::thread([this, t_id, steps_per_thread]() {
            net* local_critic = thread_critics[t_id];

            int start_idx = t_id * steps_per_thread;
            int end_idx = (t_id == num_threads - 1) ? steps : start_idx + steps_per_thread;

            // 2. Accumulate gradients for the thread's chunk
            for(int i = start_idx; i < end_idx; i++)
            {
                // A. Forward pass with stored features
                local_critic->concat_features = stored_features[i];
                local_critic->Ls[0]->pInF = states[i];
                local_critic->FF();

                // B. Inject the error gradient at the output layer
                local_critic->Ls[nlayers - 1]->pInB[0] = advs[i];

                // C. Backpropagate the error
                local_critic->BP();
            }
        }));
    }

    for (int t_id = 0; t_id < num_threads; t_id++) {
        workers[t_id].join();
        critic->add_gradients_from(thread_critics[t_id]);
    }

    // 3. Apply the Adam optimizer update using the accumulated gradients
    critic->update_Ls(steps);
    critic->setdropout(false);
}
void agent::update_actor()
{
    actor->setdropout(false);

    for (int epoch = 0; epoch < ppo_epochs; epoch++)
    {
        actor->clear();

        for(int i = 0; i < num_threads; i++) {
            thread_actors[i]->copy_weights_from(actor);
            thread_actors[i]->setdropout(false);
            thread_actors[i]->clear();
        }

        int steps_per_thread = steps / num_threads;
        std::vector<std::thread> workers;

        for (int t_id = 0; t_id < num_threads; t_id++) {
            workers.push_back(std::thread([this, t_id, steps_per_thread]() {
                net* local_actor = thread_actors[t_id];

                int start_idx = t_id * steps_per_thread;
                int end_idx = (t_id == num_threads - 1) ? steps : start_idx + steps_per_thread;

                for (int i = start_idx; i < end_idx; i++)
                {
                    // 1. Forward pass with stored features
                    local_actor->concat_features = stored_features[i];
                    local_actor->Ls[0]->pInF = states[i];
                    local_actor->FF();

                    // 2. Get current log probability
                    double current_log_p = 0.0;
                    double* local_mu = local_actor->Ls[nlayers - 1]->mOutF;
                    for (int j = 0; j < action_size; j++) {
                        double diff = sampled_actions[i][j] - local_mu[j];
                        current_log_p -= 0.5 * ((diff * diff) / (std_dev * std_dev) + log(2.0 * M_PI * std_dev * std_dev));
                    }

                    double old_log_p = log_probs[i];

                    // 3. Calculate the PPO Ratio
                    double r = exp(current_log_p - old_log_p);
                    double adv = advs[i]; // MUST be normalized!

                    // 4. Determine if clipped
                    bool clipped = false;
                    if (adv > 0 && r > 1.0 + clip) clipped = true;
                    if (adv < 0 && r < 1.0 - clip) clipped = true;

                    // 5. Inject gradients
                    for (int j = 0; j < action_size; j++)
                    {
                        double mu_val = local_mu[j];
                        double act = sampled_actions[i][j];

                        double grad = 0.0;
                        if (!clipped) {
                            grad = adv * r * (act - mu_val) / (std_dev * std_dev);
                        }

                        local_actor->Ls[nlayers - 1]->pInB[j] = grad;
                    }

                    local_actor->BP();
                }
            }));
        }

        for (int t_id = 0; t_id < num_threads; t_id++) {
            workers[t_id].join();
            actor->add_gradients_from(thread_actors[t_id]);
        }

        // Apply Adam update
        actor->update_Ls(steps);
    }
}
void agent::compute_critic_loss()
{
    c_loss=0.0;
    for (int i = 0; i < steps; i++)
        {
            double diff = returns[i] - values[i];
            c_loss += 0.5 * diff * diff;
        }
        c_loss /= steps;
}
void agent::compute_avg_rwd()
{
    avg_rwd = 0.0;
    for (int i = 0; i < steps; i++) {
        avg_rwd += rewards[i];
    }
    avg_rwd /= steps;

}
void agent::train(int epochs)
{
    for (int e = 0; e < epochs; e++)
    {
        actor->setdropout(false);
        critic->setdropout(false);
        if(e<700)std_dev *= 0.999;
        play_batch();
        calc_advs();
        compute_critic_loss();
        update_critic();
        norm_advs();
        compute_avg_rwd();
        update_actor();
        cout << "Training Epoch: " << e + 1 << " / " << epochs
             << " | Avg Return per Trajectory: " << (avg_traj_return * 100) << "%"
             << " | Avg Step Rwd: " << avg_rwd
             << " | closs: " << c_loss << endl;

        if ((e + 1) % 10 == 0)
        {
            cout << "--- Portfolio Weights at Epoch " << e + 1 << " ---" << endl;
            for (int a = 0; a < action_size; a++) {
                if (a < action_size - 1)
                    cout << "Stock " << a << ": " << softmaxed_act[a] * 100 << "%\n";
                else
                    cout << "Cash: " << softmaxed_act[a] * 100 << "%" << endl;
            }
            cout << "---------------------------------------" << endl;
        }
    }
}
agent::~agent()
{
    // 1. Deallocate 2D arrays
    for(int i = 0; i < steps; i++)
    {
        delete[] sampled_actions[i];
        delete[] mean_actions[i];
        delete[] states[i];
        delete[] stored_features[i];
    }
    delete[] sampled_actions;
    delete[] mean_actions;
    delete[] states;
    delete[] stored_features;

    // 2. Deallocate 1D arrays
    delete[] action;
    delete[] values;
    delete[] old_values;
    delete[] returns;
    delete[] log_probs;
    delete[] rewards;
    delete[] traj_returns_exact;
    delete[] advs;
    delete[] softmaxed_act;
    delete[] actor_layers;
    delete[] critic_layers;
    delete[] actor->Ls[actor->nL-1]->pInB;   // Clean up the manual allocation
    delete[] critic->Ls[critic->nL-1]->pInB; // Clean up the manual allocation
    // 3. Deallocate network objects
    delete actor;
    delete critic;

    for(int i=0; i<num_threads; i++) {
        delete thread_envs[i];
        delete[] thread_actors[i]->Ls[nlayers-1]->pInB;
        delete thread_actors[i];
        delete[] thread_critics[i]->Ls[nlayers-1]->pInB;
        delete thread_critics[i];
    }

    // Note: Since 'env' is passed in as a pointer from outside,
    // the agent usually doesn't own it, so we don't delete it here.
    // If the agent IS supposed to destroy the environment, add: delete env;
}


