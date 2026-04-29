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
//  Bar: one row from a CSV (Date,Open,High,Low,Close,Volume)
// ============================================================
struct Bar {
    string date;
    double open, high, low, close;
    double volume;
};

// ============================================================
//  MarketData: holds all bars for a single stock
// ============================================================
class MarketData {
public:
    string ticker;
    vector<Bar> bars;

    bool loadCSV(const string& path, const string& name) {
        ticker = name;
        ifstream f(path);
        if (!f.is_open()) {
            cout << "ERROR: Cannot open " << path << endl;
            return false;
        }
        string line;
        getline(f, line); // skip header
        while (getline(f, line)) {
            if (line.empty()) continue;
            Bar b;
            stringstream ss(line);
            string token;
            getline(ss, b.date, ',');
            getline(ss, token, ','); b.open   = atof(token.c_str());
            getline(ss, token, ','); b.high   = atof(token.c_str());
            getline(ss, token, ','); b.low    = atof(token.c_str());
            getline(ss, token, ','); b.close  = atof(token.c_str());
            getline(ss, token, ','); b.volume = atof(token.c_str());
            if (b.close > 0) bars.push_back(b);
        }
        f.close();
        return !bars.empty();
    }
    int size() const { return (int)bars.size(); }
};

// ============================================================
//  Constants
// ============================================================
static const int LOOKBACK   = 120;  // 6 months of trading days
static const int EPISODE_LEN = 200; // trading days per episode
static const int WINDOW_NEED = LOOKBACK + EPISODE_LEN; // 320

// ============================================================
//  MarketDataLoader: loads all CSVs, aligns by common dates
// ============================================================
class MarketDataLoader {
public:
    int numStocks;
    int numDates;         // number of common trading dates
    int stateDim;         // numStocks * LOOKBACK
    int actionDim;        // numStocks + 1 (cash)
    vector<string> tickers;
    vector<string> commonDates;

    // aligned close prices: close_aligned[stock][day]
    double** close_aligned;
    // aligned volumes: vol_aligned[stock][day]
    double** vol_aligned;

    MarketDataLoader() : numStocks(0), numDates(0),
        stateDim(0), actionDim(0),
        close_aligned(nullptr), vol_aligned(nullptr) {}

    ~MarketDataLoader() {
        if (close_aligned) {
            for (int i = 0; i < numStocks; i++) delete[] close_aligned[i];
            delete[] close_aligned;
        }
        if (vol_aligned) {
            for (int i = 0; i < numStocks; i++) delete[] vol_aligned[i];
            delete[] vol_aligned;
        }
    }

    bool load(const string& folder, const vector<string>& fileNames) {
        // 1. Load all CSVs
        vector<MarketData> stocks(fileNames.size());
        for (int i = 0; i < (int)fileNames.size(); i++) {
            string path = folder + "/" + fileNames[i];
            if (!stocks[i].loadCSV(path, fileNames[i])) {
                cout << "Warning: skipping " << fileNames[i] << endl;
            }
        }

        // 2. Build date frequency map — find dates present in ALL stocks
        map<string, int> dateCount;
        for (int i = 0; i < (int)stocks.size(); i++) {
            // Use a set to avoid counting duplicate dates in the same stock
            map<string, bool> seen;
            for (int j = 0; j < stocks[i].size(); j++) {
                if (!seen[stocks[i].bars[j].date]) {
                    dateCount[stocks[i].bars[j].date]++;
                    seen[stocks[i].bars[j].date] = true;
                }
            }
        }

        // Collect dates present in at least 80% of stocks (to handle delisted/new stocks)
        int threshold = (int)(stocks.size() * 0.8);
        vector<string> validDates;
        for (auto& kv : dateCount) {
            if (kv.second >= threshold) {
                validDates.push_back(kv.first);
            }
        }
        sort(validDates.begin(), validDates.end()); // chronological

        if ((int)validDates.size() < WINDOW_NEED) {
            cout << "ERROR: Only " << validDates.size()
                 << " common dates, need " << WINDOW_NEED << endl;
            return false;
        }

        // 3. Build the aligned matrices
        numStocks = (int)stocks.size();
        numDates  = (int)validDates.size();
        actionDim = numStocks + 1; // +1 for cash
        stateDim  = numStocks * LOOKBACK + actionDim; // price history + portfolio weights

        commonDates = validDates;
        tickers.resize(numStocks);
        for (int i = 0; i < numStocks; i++)
            tickers[i] = stocks[i].ticker;

        // Allocate
        close_aligned = new double*[numStocks];
        vol_aligned   = new double*[numStocks];
        for (int i = 0; i < numStocks; i++) {
            close_aligned[i] = new double[numDates];
            vol_aligned[i]   = new double[numDates];
            for (int d = 0; d < numDates; d++) {
                close_aligned[i][d] = 0;
                vol_aligned[i][d] = 0;
            }
        }

        // 4. Fill aligned data with forward-fill for missing dates
        for (int i = 0; i < numStocks; i++) {
            // Build a date→bar map for this stock
            map<string, int> dateToBar;
            for (int j = 0; j < stocks[i].size(); j++)
                dateToBar[stocks[i].bars[j].date] = j;

            double lastClose = 0;
            double lastVol   = 0;
            for (int d = 0; d < numDates; d++) {
                auto it = dateToBar.find(validDates[d]);
                if (it != dateToBar.end()) {
                    lastClose = stocks[i].bars[it->second].close;
                    lastVol   = stocks[i].bars[it->second].volume;
                }
                close_aligned[i][d] = (lastClose > 0) ? lastClose : 1.0;
                vol_aligned[i][d]   = lastVol;
            }
        }

        cout << "Loaded " << numStocks << " stocks, "
             << numDates << " common dates, "
             << "STATE_DIM=" << stateDim
             << ", ACTION_DIM=" << actionDim << endl;

        // 5. Sanitise: clamp daily returns at ±20% (EGX daily limit)
        //    This neutralises corrupted data (e.g. alternating split/unsplit prices).
        int clampCount = 0;
        for (int i = 0; i < numStocks; i++) {
            for (int d = 1; d < numDates; d++) {
                double prev = close_aligned[i][d - 1];
                double cur  = close_aligned[i][d];
                if (prev > 0) {
                    double ret = (cur - prev) / prev;
                    if (ret > 0.20) {
                        close_aligned[i][d] = prev * 1.20;
                        clampCount++;
                    } else if (ret < -0.20) {
                        close_aligned[i][d] = prev * 0.80;
                        clampCount++;
                    }
                }
            }
        }
        if (clampCount > 0) {
            cout << "Sanitised " << clampCount << " anomalous daily price jumps (clamped to +/-20%)" << endl;
        }

        return true;
    }
};

#endif // MARKETDATA_H
