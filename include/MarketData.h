#ifndef MARKETDATA_H
#define MARKETDATA_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>

using namespace std;

// ============================================================
//  MarketDataLoader: loads all CSVs directly into 2D matrix
// ============================================================
class MarketDataLoader {
public:
    int numStocks;
    int numDates;         // number of common trading dates
    int numFeatures;      // features per stock (e.g., RSI, MACD on current day)
    vector<string> tickers;
    vector<string> commonDates;

    // aligned OHLCV data: [stock][day]
    double** open_prices;
    double** high_prices;
    double** low_prices;
    double** stock_prices;  // close prices
    double** volume;

    MarketDataLoader();
    ~MarketDataLoader();

    bool load(const string& folder, const vector<string>& fileNames);

    // Static helper to load benchmark to replace MarketData class usage
    static map<string, double> loadBenchmark(const string& path);
};

#endif // MARKETDATA_H
