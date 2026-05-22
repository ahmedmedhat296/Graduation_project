#include <iostream>
#include <vector>
#include <string>
#include "MarketData.h"
#include "environment.h"
#include "agent.h"

#include <cmath>
#include <map>
#include <iomanip>

using namespace std;

double run_test_episode(tradingenv& test_env, agent& ppoAgent, MarketDataLoader& marketData,
                        const vector<string>& fileNames, int test_episode_length,
                        const map<string, double>& benchClose) {
    tradingenv* train_env = ppoAgent.env;
    ppoAgent.env = &test_env;

    test_env.reset(test_episode_length);

    int test_start_step = test_env.current_step;
    double initial_cash = test_env.cash;
    double total_test_reward = 0.0;

    int num_samples = 10;
    int sample_interval = test_episode_length / num_samples;
    if (sample_interval < 1) sample_interval = 1;
    // Store portfolio snapshots: [sample_idx][stock_idx]
    vector<vector<double>> portfolio_snapshots;
    vector<int> snapshot_steps;

    int actual_steps = 0;
    for (int step = 0; step < test_episode_length; step++) {
        double* mu = ppoAgent.act_dist();
        ppoAgent.softmax_action(mu);
        bool done = test_env.step(ppoAgent.softmaxed_act);
        total_test_reward += test_env.reward;
        actual_steps++;

        // Capture portfolio snapshot at evenly-spaced points
        if (step % sample_interval == 0 || step == test_episode_length - 1) {
            vector<double> snapshot(ppoAgent.action_size);
            for (int a = 0; a < ppoAgent.action_size; a++)
                snapshot[a] = ppoAgent.softmaxed_act[a] * 100.0;
            portfolio_snapshots.push_back(snapshot);
            snapshot_steps.push_back(step);
        }

        if (done) break;
    }

    int final_step = test_env.current_step;

    cout << "\n--- Test Episode Results ---" << endl;
    double exact_return_fraction = (test_env.portfolio_value - initial_cash) / initial_cash;
    cout << "Total Agent Test Return: " << (exact_return_fraction * 100.0) << "%" << endl;
    cout << "Final Agent Portfolio Value: $" << test_env.portfolio_value << endl;

    double agent_return = ((test_env.portfolio_value - initial_cash) / initial_cash) * 100.0;
    cout << "Agent Absolute Return: " << agent_return << "%" << endl;

    // --- Buy-and-Hold benchmark for each stock ---
    cout << "--- Benchmark: Buy & Hold per Stock ---" << endl;
    double avg_stock_return = 0.0;
    for (int i = 0; i < marketData.numStocks; i++) {
        double start_price = marketData.stock_prices[i][test_start_step];
        double end_price = marketData.stock_prices[i][final_step];
        double stock_return = ((end_price - start_price) / start_price) * 100.0;
        avg_stock_return += stock_return;
        cout << "  " << fileNames[i] << " | Buy&Hold Return: " << stock_return << "%" << endl;
    }
    avg_stock_return /= marketData.numStocks;

    if (marketData.numStocks > 1) {
        cout << "  Equal-Weight Avg Return: " << avg_stock_return << "%" << endl;
    }

    // --- EGX30 benchmark ---
    if (!benchClose.empty()) {
        string startDate = (test_start_step < (int)marketData.commonDates.size()) ?
                           marketData.commonDates[test_start_step] : "";
        string endDate   = (final_step < (int)marketData.commonDates.size()) ?
                           marketData.commonDates[final_step] : "";

        auto it_start = benchClose.find(startDate);
        auto it_end   = benchClose.find(endDate);
        if (it_start != benchClose.end() && it_end != benchClose.end() && it_start->second > 0) {
            double egx_return = ((it_end->second - it_start->second) / it_start->second) * 100.0;
            cout << "  EGX30 Return (" << startDate << " -> " << endDate << "): " << egx_return << "%" << endl;
        } else {
            cout << "  EGX30 Return: N/A (dates not found in benchmark)" << endl;
        }
    }

    cout << "  Agent vs Equal-Weight: " << (agent_return - avg_stock_return) << "%" << endl;

    // --- Portfolio allocation over the episode (10 snapshots) ---
    cout << "--- Portfolio Weights Along Episode ---" << endl;
    // Header row
    cout << "  Step  |";
    for (int i = 0; i < marketData.numStocks; i++) {
        // Truncate filename to 10 chars for compact display
        string name = fileNames[i];
        if (name.size() > 10) name = name.substr(0, 10);
        cout << " " << name;
    }
    cout << " | Cash" << endl;

    // Data rows
    for (int s = 0; s < (int)portfolio_snapshots.size(); s++) {
        cout << "  " << setw(5) << snapshot_steps[s] << " |";
        for (int a = 0; a < ppoAgent.action_size; a++) {
            if (a < ppoAgent.action_size - 1)
                cout << " " << fixed << setprecision(1) << setw(6) << portfolio_snapshots[s][a] << "%";
            else
                cout << " | " << fixed << setprecision(1) << portfolio_snapshots[s][a] << "%";
        }
        cout << endl;
    }
    cout << "----------------------------\n" << endl;
    cout << defaultfloat << setprecision(6);  // Reset so training prints aren't affected

    ppoAgent.env = train_env;
    return exact_return_fraction;
}

int main() {
    cout << "Starting PPO Agent Training..." << endl;

    // 1. Define files to load
    // Selected a diverse set of stocks for a good portfolio
    vector<string> fileNames = {
        "MASR.CA.csv"
    };
    string dataFolder = "data";

    // 2. Load Market Data
    MarketDataLoader marketData;
    bool success = marketData.load(dataFolder, fileNames);
    if (!success) {
        cerr << "Failed to load market data!" << endl;
        return 1;
    }

    // 3. Load EGX30 benchmark
    map<string, double> benchClose = MarketDataLoader::loadBenchmark(dataFolder + "/EGX30.csv");
    if (benchClose.empty()) {
        cout << "Warning: EGX30 benchmark not loaded (file not found). Continuing without it." << endl;
    } else {
        cout << "Loaded EGX30 benchmark with " << benchClose.size() << " dates." << endl;
    }

    // 4. Setup Environment Parameters
    // Need at least start_idx + lookback + episode length
    // From environment.cpp, lookback = 80
    // agent.cpp episode length = 250
    int start_index = 0;
    int test_episode_length = 250;
    int train_end_index = marketData.numDates - test_episode_length;

    cout << "Setting up training environment from index " << start_index << " to " << train_end_index << endl;
    tradingenv env(&marketData, start_index, train_end_index);

    // 5. Initialize PPO Agent (concat_after_layer = 1 by default, configurable)
    int concat_after = 1;  // Concatenate features after layer 1 (the first 64-neuron hidden layer)
    cout << "Initializing PPO Agent with late-fusion after layer " << concat_after << "..." << endl;
    cout << "Feature size: " << env.total_features << " (" << env.num_stocks << " stocks x "
         << env.features_per_stock << " indicators)" << endl;
    agent ppoAgent(&env);

    // 6. Train Agent
    int epochs = 4000;
    cout << "Beginning training for " << epochs << " epochs..." << endl;

    // Create a dedicated test environment that operates on the left-out test episode
    tradingenv test_env(&marketData, train_end_index - env.lookback, marketData.numDates);

    double best_test_return = -1e9; // Track best return

    for (int e = 0; e < epochs; e += 10) {
        int epochs_to_run = std::min(10, epochs - e);
        cout << "\n--- Starting Training Chunk: Epoch " << e + 1 << " to " << e + epochs_to_run << " ---" << endl;
        ppoAgent.train(epochs_to_run);

        cout << "\nRunning interim test at epoch " << e + epochs_to_run << "..." << endl;
        double test_return = run_test_episode(test_env, ppoAgent, marketData, fileNames,
                                              test_episode_length, benchClose);

        if (test_return > best_test_return) {
            best_test_return = test_return;
            cout << ">>> New best test return found: " << (best_test_return * 100.0) << "%! Saving networks..." << endl;
            ppoAgent.actor->save("best_actor.txt");
            ppoAgent.critic->save("best_critic.txt");
        }
    }

    cout << "Training completed successfully." << endl;

    return 0;
}
