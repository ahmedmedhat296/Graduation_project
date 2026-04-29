#include "TradingEnv.h"

// --- Diversification parameters ---
static const double MAX_STOCK_WEIGHT      = 0.40;  // Hard cap: max 20% per stock
static const double DIVERSIFICATION_COEFF = 0.05;  // HHI penalty (raised from 0.01) (FIX 1)
static const double ENTROPY_BONUS_COEFF   = 0.005; // Reward spreading across stocks (FIX C)

TradingEnv::TradingEnv(MarketDataLoader* loader)
    : data(loader), done(false), stepCount(0), startDay(0), currentDay(0)
{
    numStocks = data->numStocks;
    stateDim  = data->stateDim;
    actionDim = data->actionDim;
    txCost    = 0.0015; // 0.15% per trade

    shares         = new double[numStocks];
    currentWeights = new double[actionDim];

    portfolioValue = 0;
    cashAmount     = 0;

    for (int i = 0; i < numStocks; i++) shares[i] = 0;
    for (int i = 0; i < actionDim; i++) currentWeights[i] = 0;
}

TradingEnv::~TradingEnv() {
    delete[] shares;
    delete[] currentWeights;
}

// ---------------------------------------------------------------
//  computePortfolioValue: sum of cash + market value of all shares
// ---------------------------------------------------------------
void TradingEnv::computePortfolioValue() {
    portfolioValue = cashAmount;
    for (int i = 0; i < numStocks; i++) {
        portfolioValue += shares[i] * data->close_aligned[i][currentDay];
    }
}

// ---------------------------------------------------------------
//  getState: price history + current portfolio weights
//  Layout: [numStocks * LOOKBACK normalised returns] [actionDim weights]
// ---------------------------------------------------------------
void TradingEnv::getState(double* stateOut) {
    int base = numStocks * LOOKBACK;
    for (int s = 0; s < numStocks; s++) {
        double curPrice = data->close_aligned[s][currentDay];
        if (curPrice <= 0) curPrice = 1.0;

        for (int d = 0; d < LOOKBACK; d++) {
            int pastDay = currentDay - d;
            double pastPrice = data->close_aligned[s][pastDay];
            stateOut[s * LOOKBACK + d] = (pastPrice - curPrice) / curPrice;
        }
    }

    computePortfolioValue();
    double pv = (portfolioValue > 0) ? portfolioValue : 1.0;
    for (int i = 0; i < numStocks; i++) {
        double price = data->close_aligned[i][currentDay];
        if (price <= 0) price = 1.0;
        stateOut[base + i] = (shares[i] * price) / pv;
    }
    stateOut[base + numStocks] = cashAmount / pv;
}

// ---------------------------------------------------------------
//  reset: pick random window, start with 100% cash
// ---------------------------------------------------------------
void TradingEnv::reset(double* stateOut) {
    int maxTrainDates = data->numDates - WINDOW_NEED;
    int maxStart = maxTrainDates - WINDOW_NEED;
    if (maxStart < 1) maxStart = 1;

    int windowStart = rand() % maxStart;

    startDay   = windowStart + LOOKBACK;
    currentDay = startDay;
    stepCount  = 0;
    done       = false;

    transactionLog.clear();

    portfolioValue = 1000000.0;
    cashAmount     = 1000000.0;
    for (int i = 0; i < numStocks; i++) shares[i] = 0;
    for (int i = 0; i < actionDim; i++) currentWeights[i] = 0;
    currentWeights[actionDim - 1] = 1.0;

    getState(stateOut);
}

// ---------------------------------------------------------------
//  step: delta-based rebalancing to target weights, advance one day
// ---------------------------------------------------------------
double TradingEnv::step(double* targetWeights, double* nextStateOut) {
    if (done) {
        getState(nextStateOut);
        return 0.0;
    }

    // --- Mark-to-market portfolio at current prices ---
    computePortfolioValue();
    double prevValue = portfolioValue;

    if (!std::isfinite(prevValue) || prevValue <= 0) {
        portfolioValue = 1000000.0;
        cashAmount     = 1000000.0;
        for (int i = 0; i < numStocks; i++) shares[i] = 0;
        prevValue = portfolioValue;
    }

    double totalPV = portfolioValue;

    // --- Normalise target weights to sum to 1 ---
    double wSum = 0;
    for (int i = 0; i < actionDim; i++) {
        if (targetWeights[i] > 0) wSum += targetWeights[i];
    }
    if (wSum < 1e-8) wSum = 1.0;

    double* normW = new double[actionDim];
    for (int i = 0; i < actionDim; i++) {
        normW[i] = (targetWeights[i] > 0 ? targetWeights[i] : 0) / wSum;
    }

    // --- FIX A+1: Cap each stock at MAX, route excess to cash ---
    double excess = 0.0;
    for (int i = 0; i < numStocks; i++) {
        if (normW[i] > MAX_STOCK_WEIGHT) {
            excess += (normW[i] - MAX_STOCK_WEIGHT);
            normW[i] = MAX_STOCK_WEIGHT;
        }
    }
    normW[actionDim - 1] += excess;

    // --- FIX 2: Count how many stocks were capped (penalty applied to reward later) ---
    double clipPenalty = 0.0;
    for (int i = 0; i < numStocks; i++) {
        if (normW[i] == MAX_STOCK_WEIGHT)
            clipPenalty += 0.02;
    }

    // --- Compute HHI from stock weights only (excluding cash) ---
    double hhi = 0.0;
    for (int i = 0; i < numStocks; i++) {
        hhi += normW[i] * normW[i];
    }

    // --- FIX C: Compute portfolio entropy for reward bonus ---
    // H = -sum(w * log(w)), max is log(numStocks) for equal weights
    double portfolioEntropy = 0.0;
    for (int i = 0; i < numStocks; i++) {
        if (normW[i] > 1e-8)
            portfolioEntropy -= normW[i] * log(normW[i]);
    }
    double maxEntropy = log((double)numStocks);

    // --- Pass 1: SELL overweight positions ---
    for (int i = 0; i < numStocks; i++) {
        double price = data->close_aligned[i][currentDay];
        if (price <= 0) price = 1.0;

        double targetValue  = normW[i] * totalPV;
        double currentValue = shares[i] * price;
        double delta = targetValue - currentValue;

        if (delta < -1.0) {
            double sellValue = -delta;
            double sharesToSell = sellValue / price;
            if (sharesToSell > shares[i]) sharesToSell = shares[i];
            shares[i] -= sharesToSell;
            cashAmount += sharesToSell * price * (1.0 - txCost);

            if (sharesToSell > 0) {
                transactionLog.push_back({currentDay, i, false, sharesToSell, price, sharesToSell * price});
            }
        }
    }

    // --- Pass 2: BUY underweight positions ---
    for (int i = 0; i < numStocks; i++) {
        double price = data->close_aligned[i][currentDay];
        if (price <= 0) price = 1.0;

        double targetValue  = normW[i] * totalPV;
        double currentValue = shares[i] * price;
        double delta = targetValue - currentValue;

        if (delta > 1.0) {
            double buyValue = delta;
            double totalCost = buyValue * (1.0 + txCost);
            if (totalCost > cashAmount) {
                buyValue = cashAmount / (1.0 + txCost);
                totalCost = cashAmount;
            }
            if (buyValue > 0) {
                double sharesBought = buyValue / price;
                shares[i] += sharesBought;
                cashAmount -= totalCost;

                transactionLog.push_back({currentDay, i, true, sharesBought, price, buyValue});
            }
        }
    }

    if (cashAmount < 0) cashAmount = 0;

    for (int i = 0; i < actionDim; i++)
        currentWeights[i] = targetWeights[i];

    delete[] normW;

    // --- Advance to next day ---
    currentDay++;
    stepCount++;

    // --- Revalue portfolio ---
    computePortfolioValue();

    if (!std::isfinite(portfolioValue) || portfolioValue > 1e8) {
        portfolioValue = 1e8;
        cashAmount = portfolioValue;
        for (int i = 0; i < numStocks; i++) shares[i] = 0;
    }
    if (portfolioValue < 0) portfolioValue = cashAmount;

    // --- Log return reward, clamped to ±0.1 ---
    double reward = 0.0;
    if (prevValue > 0 && portfolioValue > 0 && std::isfinite(portfolioValue)) {
        reward = log(portfolioValue / prevValue);
        if (reward >  0.1) reward =  0.1;
        if (reward < -0.1) reward = -0.1;
    }

    // --- FIX 1: HHI penalty (raised coefficient) ---
    reward -= DIVERSIFICATION_COEFF * hhi;

    // --- FIX 2: Clip penalty (agent feels the cap, not just silently corrected) ---
    reward -= clipPenalty;

    // --- FIX C: Portfolio entropy bonus (direct reward for spreading weights) ---
    if (maxEntropy > 0)
        reward += ENTROPY_BONUS_COEFF * (portfolioEntropy / maxEntropy);

    // --- Episode termination ---
    if (stepCount >= EPISODE_LEN || currentDay >= data->numDates - 1) {
        done = true;
    }

    getState(nextStateOut);
    return reward;
}

double TradingEnv::getPortfolioValue() {
    computePortfolioValue();
    return portfolioValue;
}
