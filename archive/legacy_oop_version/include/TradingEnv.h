#ifndef TRADINGENV_H
#define TRADINGENV_H

#include "MarketData.h"
#include <cmath>
#include <cstdlib>
#include <vector>

struct Transaction {
    int day;
    int stockIdx;
    bool isBuy;
    double shares;
    double price;
    double value;
};

// ============================================================
//  TradingEnv: portfolio-based trading environment
// ============================================================
class TradingEnv {
public:
    MarketDataLoader* data;

    // Dimensions
    int numStocks;
    int stateDim;
    int actionDim;

    // Episode state
    int startDay;       // first tradeable day of episode (after lookback)
    int currentDay;     // current day index into commonDates
    int stepCount;
    bool done;

    // Transaction log
    std::vector<Transaction> transactionLog;

    // Portfolio state
    double  portfolioValue;
    double  cashAmount;
    double* shares;          // shares[numStocks]
    double* currentWeights;  // current portfolio weights

    // Transaction cost
    double txCost;

    TradingEnv(MarketDataLoader* loader);
    ~TradingEnv();

    // Reset to a random episode window, returns initial state
    void reset(double* stateOut);

    // Take action (portfolio weights), returns reward
    double step(double* weights, double* nextStateOut);

    // Build state vector into buffer
    void getState(double* stateOut);

    // Get current portfolio value
    double getPortfolioValue();

private:
    void computePortfolioValue();
};

#endif // TRADINGENV_H
