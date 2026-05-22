#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include<iostream>
#include "MarketData.h"
#include "matrix.h"
#include <cmath>
#include <random>
 class tradingenv{
    public:
    double cash;
    int current_step;
    int lookback;
    int portfolio_size;
    int history_size;
    int state_size;
    int num_stocks;
    int features_per_stock;  // 10 indicators per stock
    int total_features;      // num_stocks * features_per_stock
    int start_idx;
    int end_idx;
    double *prev_shares;
    double *shares;
    double reward;
    double prev_portfolio_value;
    double portfolio_value;
    float tx_cost;
    double tx_plenty;
    bool done;
    MarketDataLoader* data;
    double* state;
    double* portfolio;
    double* features;   // technical indicator features for late-fusion
    bool step(double* action);
    void update_state(double* action);
    void compute_reward();
    void process_portfolio(double* action);
    void init_portfolio();
    void init_state();
    void compute_features();  // compute technical indicators at current_step
    void reset(int ep);
    int getRandom(int min, int max);
    tradingenv(MarketDataLoader* marketdata, int start_index, int end_index);
    ~tradingenv();

 };

#endif // ENVIRONMENT_H
