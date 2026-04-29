// ============================================================
//  PPO Portfolio Allocation Agent — GPU-Accelerated Main
//  Usage:
//    ppo_gpu.exe --train        (train from scratch / resume)
//    ppo_gpu.exe --test         (evaluate on most recent EGX data)
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <conio.h>
#include <cuda_runtime.h>

#include "MarketData.h"
#include "TradingEnv.h"
#include "GpuPPOAgent.h"

using namespace std;

// ============================================================
//  Enumerate all CSV files in the data directory
// ============================================================
vector<string> getStockFiles() {
    vector<string> files;
    files.push_back("ABUK.CA.csv"); files.push_back("ADIB.CA.csv");
    files.push_back("ALCN.CA.csv"); files.push_back("AMOC.CA.csv");
    files.push_back("ARCC.CA.csv"); files.push_back("ASCM.CA.csv");
    files.push_back("ATLC.CA.csv"); files.push_back("BINV.CA.csv");
    files.push_back("BTFH.CA.csv"); files.push_back("CCAP.CA.csv");
    files.push_back("CERA.CA.csv"); files.push_back("CIEB.CA.csv");
    files.push_back("CIRA.CA.csv"); files.push_back("COMI.CA.csv");
    files.push_back("CSAG.CA.csv"); files.push_back("DCRC.CA.csv");
    files.push_back("DOMT.CA.csv"); files.push_back("DSCW.CA.csv");
    files.push_back("EAST.CA.csv"); files.push_back("EFIC.CA.csv");
    files.push_back("EFID.CA.csv"); files.push_back("EGAL.CA.csv");
    files.push_back("EGBE.CA.csv"); files.push_back("EGCH.CA.csv");
    files.push_back("EGTS.CA.csv"); files.push_back("EIPH.CA.csv");
    files.push_back("EKHO.CA.csv"); files.push_back("ELSH.CA.csv");
    files.push_back("EMFD.CA.csv"); files.push_back("ENGC.CA.csv");
    files.push_back("ETEL.CA.csv"); files.push_back("ETRS.CA.csv");
    files.push_back("EXPA.CA.csv"); files.push_back("FAIT.CA.csv");
    files.push_back("FWRY.CA.csv");
    files.push_back("HDBK.CA.csv"); files.push_back("HELI.CA.csv");
    files.push_back("HRHO.CA.csv"); files.push_back("IDRE.CA.csv");
    files.push_back("ISPH.CA.csv"); files.push_back("JUFO.CA.csv");
    files.push_back("KIMA.CA.csv");
    files.push_back("MBSC.CA.csv"); files.push_back("MCQE.CA.csv");
    files.push_back("MFPC.CA.csv"); files.push_back("MICH.CA.csv");
    files.push_back("MOIL.CA.csv"); files.push_back("MPRC.CA.csv");
    files.push_back("MTIE.CA.csv"); files.push_back("OCDI.CA.csv");
    files.push_back("ORAS.CA.csv"); files.push_back("ORHD.CA.csv");
    files.push_back("ORWE.CA.csv"); files.push_back("PHDC.CA.csv");
    files.push_back("POUL.CA.csv"); files.push_back("PRCL.CA.csv");
    files.push_back("QNBA.CA.csv"); files.push_back("RAYA.CA.csv");
    files.push_back("RMDA.CA.csv"); files.push_back("SAUD.CA.csv");
    files.push_back("SCEM.CA.csv"); files.push_back("SKPC.CA.csv");
    files.push_back("SNFI.CA.csv"); files.push_back("SPMD.CA.csv");
    files.push_back("SUGR.CA.csv"); files.push_back("SWDY.CA.csv");
    files.push_back("TMGH.CA.csv"); files.push_back("ZMID.CA.csv");
    return files;
}

// ============================================================
//  GPU info
// ============================================================
void printGpuInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s | VRAM: %.0f MB | Compute: %d.%d\n",
           prop.name,
           (double)prop.totalGlobalMem / (1024 * 1024),
           prop.major, prop.minor);
}

// ============================================================
//  TRAIN MODE
// ============================================================
void runTraining(MarketDataLoader& loader, const string& actorBest,
                 const string& criticBest, const string& actorCkpt,
                 const string& criticCkpt)
{
    TradingEnv env(&loader);
    GpuPPOAgent agent(loader.stateDim, loader.actionDim);

    // Resume from checkpoint if it exists
    ifstream chk(actorCkpt, ios::binary);
    if (chk) {
        chk.close();
        cout << "Resuming from checkpoint..." << endl;
        agent.loadModels(actorCkpt, criticCkpt);
    }

    float* state     = new float[loader.stateDim];
    float* nextState = new float[loader.stateDim];
    float* action    = new float[loader.actionDim];

    cout << fixed << setprecision(4);
    cout << "\n--- Training (press ESC to stop) ---\n";
    cout << setw(8)  << "Ep"
         << setw(13) << "Return%"
         << setw(14) << "FinalValue"
         << setw(13) << "AvgReward"
         << setw(10) << "Ep/s"
         << "  Top-3 Holdings"
         << "\n" << string(95, '-') << "\n";

    int    totalEpisodes = 5000;
    int    saveInterval  = 50;
    float  bestReturn    = -1e9f;
    double runReturn     = 0;
    auto   t0 = chrono::high_resolution_clock::now();

    for (int ep = 1; ep <= totalEpisodes; ep++) {
        if (_kbhit()) {
            char ch = _getch();
            if (ch == 27) {
                cout << "\n>>> Stopped at episode " << ep << " <<<\n";
                agent.saveModels(actorCkpt, criticCkpt);
                break;
            }
        }

        // Collect trajectory
        double* dStateInit = new double[loader.stateDim];
        env.reset(dStateInit);
        for(int i=0; i<loader.stateDim; i++) state[i] = (float)dStateInit[i];
        delete[] dStateInit;
        agent.clearTrajectory();
        float totalReward = 0;

        for (int t = 0; t < EPISODE_LEN; t++) {
            float logProb = agent.selectAction(state, action);
            float value   = agent.getValue(state);
            // Convert float action to double for env
            double* dAction = new double[loader.actionDim];
            double* dNext   = new double[loader.stateDim];
            double* dState  = new double[loader.stateDim];
            for (int i = 0; i < loader.actionDim; i++) dAction[i] = (double)action[i];
            for (int i = 0; i < loader.stateDim;  i++) dState[i]  = (double)state[i];

            double reward = env.step(dAction, dNext);

            // Convert back
            for (int i = 0; i < loader.stateDim; i++) nextState[i] = (float)dNext[i];

            agent.storeStep(state, action, (float)reward, logProb, value);
            totalReward += (float)reward;
            memcpy(state, nextState, loader.stateDim * sizeof(float));
            delete[] dAction; delete[] dNext; delete[] dState;
            if (env.done) break;
        }

        float lastVal = agent.getValue(state);
        agent.computeGAE(lastVal);
        agent.update();

        // Stats — use ACTUAL portfolio value for return calculation
        auto now = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(now - t0).count();
        float epsPerSec = ep / (float)elapsed;
        float finalVal  = (float)env.getPortfolioValue();
        float totalRet  = (finalVal - 1000000.0f) / 1000000.0f; // actual % return
        float avgRew    = totalReward / EPISODE_LEN;
        if (std::isfinite(totalRet))
            runReturn = 0.95 * runReturn + 0.05 * totalRet;

        if (ep % 10 == 0 || ep <= 5) {
            // Build top-3 holdings from ACTUAL portfolio weights
            vector<pair<float,int>> wt(loader.actionDim);
            int wBase = loader.stateDim - loader.actionDim;
            for (int i = 0; i < loader.actionDim; i++) wt[i] = {nextState[wBase + i], i};
            sort(wt.begin(), wt.end(), [](auto&a, auto&b){ return a.first > b.first; });

            cout << setw(8)  << ep
                 << setw(12) << totalRet * 100.0f << "%"
                 << setw(14) << finalVal
                 << setw(13) << avgRew
                 << setw(10) << epsPerSec
                 << "  ";
            for (int k = 0; k < 3 && k < (int)wt.size(); k++) {
                int idx = wt[k].second;
                string name = (idx < loader.numStocks) ? loader.tickers[idx] : "CASH";
                cout << name << "(" << setprecision(1) << wt[k].first * 100.0f << "%) ";
                cout << setprecision(4);
            }
            cout << " avg=" << runReturn * 100.0 << "%"
                 << "\n";
        }

        // Save best model based on RUNNING AVERAGE, not a single lucky episode
        if (ep >= 20 && runReturn > bestReturn) {
            bestReturn = runReturn;
            agent.saveModels(actorBest, criticBest);
        }
        if (ep % saveInterval == 0) {
            agent.saveModels(actorCkpt, criticCkpt);
            cout << "  [Checkpoint ep=" << ep
                 << " bestAvg=" << bestReturn * 100.0f << "%]\n";
        }
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "\nDone. Time: "
         << chrono::duration<double>(end - t0).count() << "s"
         << " | Best avg return: " << bestReturn * 100.0f << "%" << endl;
    agent.saveModels(actorCkpt, criticCkpt);

    delete[] state; delete[] nextState; delete[] action;
}

// ============================================================
//  TEST MODE: evaluate on most recent data + EGX30 benchmark
// ============================================================
void runTest(MarketDataLoader& loader, const string& actorF, const string& criticF) {
    GpuPPOAgent agent(loader.stateDim, loader.actionDim);
    agent.loadModels(actorF, criticF);
    agent.noiseStd = 0.0f; // no exploration noise during test

    // Use the most recent WINDOW_NEED days as test window
    int testStart = loader.numDates - WINDOW_NEED;
    if (testStart < 0) {
        cout << "ERROR: Not enough data for test." << endl;
        return;
    }

    // --- Load EGX30 Benchmark ---
    MarketData egx30;
    string benchPath = "InvestingCom_EGX70_6Years_Aligned/EGX30_Benchmark.csv";
    bool hasBenchmark = egx30.loadCSV(benchPath, "EGX30");

    // Build a date->close map for the benchmark
    map<string, double> benchClose;
    if (hasBenchmark) {
        for (int i = 0; i < egx30.size(); i++) {
            benchClose[egx30.bars[i].date] = egx30.bars[i].close;
        }
        cout << "Loaded EGX30 benchmark (" << egx30.size() << " bars)\n";
    } else {
        cout << "WARNING: EGX30_Benchmark.csv not found, skipping benchmark comparison\n";
    }

    // Find the benchmark start price at the test start date
    int firstTestDay = testStart + LOOKBACK;
    string firstDate = loader.commonDates[firstTestDay];
    double benchStartPrice = 0;
    if (hasBenchmark && benchClose.count(firstDate)) {
        benchStartPrice = benchClose[firstDate];
    }

    // Manually construct a test env episode at the most recent window
    TradingEnv env(&loader);

    // Allocate
    float* state     = new float[loader.stateDim];
    float* nextState = new float[loader.stateDim];
    float* action    = new float[loader.actionDim];

    // Reset to the latest window
    env.startDay   = firstTestDay;
    env.currentDay = env.startDay;
    env.stepCount  = 0;
    env.done       = false;
    env.portfolioValue = 1000000.0;
    env.cashAmount     = 1000000.0;
    for (int i = 0; i < loader.numStocks; i++) env.shares[i] = 0;

    // Build initial state (double, then cast)
    double* ds = new double[loader.stateDim];
    env.getState(ds);
    for (int i = 0; i < loader.stateDim; i++) state[i] = (float)ds[i];
    delete[] ds;

    // CSV output
    ofstream csv("test_results.csv");
    csv << "Date,PortfolioValue,AgentReturn%,EGX30Return%,Alpha%";
    // Top-5 stock header
    for (int k = 0; k < 5; k++) csv << ",Top" << (k+1) << "_Ticker,Top" << (k+1) << "_Weight%";
    csv << "\n";

    cout << "\n=== TEST MODE: Most Recent " << EPISODE_LEN
         << " Trading Days ===\n";
    cout << setw(12) << "Date"
         << setw(14) << "Portfolio"
         << setw(12) << "Agent%"
         << setw(12) << "EGX30%"
         << setw(10) << "Alpha%"
         << "  Top-3 Holdings\n"
         << string(85, '-') << "\n";

    double initValue = 1000000.0;
    float  totalReward = 0;

    for (int t = 0; t < EPISODE_LEN; t++) {
        agent.selectAction(state, action);

        // Step env
        double* dAction = new double[loader.actionDim];
        double* dNext   = new double[loader.stateDim];
        for (int i = 0; i < loader.actionDim; i++) dAction[i] = (double)action[i];
        double reward = env.step(dAction, dNext);
        for (int i = 0; i < loader.stateDim; i++) nextState[i] = (float)dNext[i];
        delete[] dAction; delete[] dNext;
        totalReward += (float)reward;

        // Build top-5 positions from ACTUAL portfolio weights
        vector<pair<float,int>> wt(loader.actionDim);
        int wBase = loader.stateDim - loader.actionDim;
        for (int i = 0; i < loader.actionDim; i++) wt[i] = {nextState[wBase + i], i};
        sort(wt.begin(), wt.end(), [](auto&a, auto&b){ return a.first > b.first; });

        double pv    = env.getPortfolioValue();
        double agentRet = (pv - initValue) / initValue * 100.0;
        string date  = (env.currentDay > 0 && env.currentDay <= loader.numDates)
                       ? loader.commonDates[env.currentDay - 1] : "---";

        // Compute benchmark return for this date
        double benchRet = 0;
        if (hasBenchmark && benchStartPrice > 0 && benchClose.count(date)) {
            benchRet = (benchClose[date] - benchStartPrice) / benchStartPrice * 100.0;
        }
        double alpha = agentRet - benchRet;

        // Console print every 10 days
        if (t % 10 == 0) {
            cout << setw(12) << date
                 << setw(14) << fixed << setprecision(0) << pv
                 << setw(11) << setprecision(2) << agentRet << "%"
                 << setw(11) << benchRet << "%"
                 << setw(9)  << alpha << "% ";
            // Top-3 holdings
            for (int k = 0; k < 3 && k < (int)wt.size(); k++) {
                int idx = wt[k].second;
                string name = (idx < loader.numStocks) ? loader.tickers[idx] : "CASH";
                cout << name << "(" << setprecision(1) << wt[k].first * 100.0f << "%) ";
            }
            cout << "\n";
        }

        // CSV row
        csv << date << "," << fixed << setprecision(2) << pv
            << "," << agentRet << "," << benchRet << "," << alpha;
        for (int k = 0; k < 5 && k < (int)wt.size(); k++) {
            int idx = wt[k].second;
            string name = (idx < loader.numStocks) ? loader.tickers[idx] : "CASH";
            csv << "," << name << "," << wt[k].first * 100.0f;
        }
        csv << "\n";

        memcpy(state, nextState, loader.stateDim * sizeof(float));
        if (env.done) break;
    }
    csv.close();

    double finalVal = env.getPortfolioValue();
    double agentFinalRet = (finalVal - initValue) / initValue * 100.0;

    // Get benchmark final return
    string lastDate = loader.commonDates[env.currentDay > 0 ? env.currentDay - 1 : loader.numDates - 1];
    double benchFinalRet = 0;
    if (hasBenchmark && benchStartPrice > 0 && benchClose.count(lastDate)) {
        benchFinalRet = (benchClose[lastDate] - benchStartPrice) / benchStartPrice * 100.0;
    }

    cout << "\n--- Test Summary ---\n";
    cout << "Start:          $1,000,000\n";
    cout << "Final:          $" << fixed << setprecision(2) << finalVal << "\n";
    cout << "Agent Return:   " << agentFinalRet << "%\n";
    if (hasBenchmark) {
        cout << "EGX30 Return:   " << benchFinalRet << "%\n";
        cout << "Alpha (excess): " << agentFinalRet - benchFinalRet << "%\n";
    }
    // Write detailed transaction report
    ofstream tsv("transaction_report.csv");
    tsv << "Date,Action,Ticker,Shares,Price,TotalValue\n";
    for (auto& txn : env.transactionLog) {
        string date = (txn.day >= 0 && txn.day < loader.numDates) ? loader.commonDates[txn.day] : "---";
        string ticker = txn.stockIdx < loader.numStocks ? loader.tickers[txn.stockIdx] : "UNKNOWN";
        string actionType = txn.isBuy ? "BUY" : "SELL";
        tsv << date << "," << actionType << "," << ticker << "," 
            << fixed << setprecision(2) << txn.shares << ","
            << txn.price << "," << txn.value << "\n";
    }
    tsv.close();
    cout << "Detailed transaction report saved to: transaction_report.csv\n";

    delete[] state; delete[] nextState; delete[] action;
}

// ============================================================
//  Main
// ============================================================
int main(int argc, char* argv[]) {
    printGpuInfo();

    bool doTest  = false;
    bool doTrain = true;
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--test")  { doTest = true;  doTrain = false; }
        if (string(argv[i]) == "--train") { doTrain = true; doTest  = false; }
    }

    srand((unsigned)time(NULL));

    cout << "\n========================================\n";
    cout << "  PPO Portfolio Agent (GPU Edition)\n";
    cout << "  Mode: " << (doTrain ? "TRAINING" : "TESTING") << "\n";
    cout << "  Folder: InvestingCom_EGX70_6Years_Aligned\n";
    cout << "========================================\n";

    // Load data
    cout << "\nLoading market data..." << endl;
    string dataFolder = "InvestingCom_EGX70_6Years_Aligned";
    MarketDataLoader loader;
    if (!loader.load(dataFolder, getStockFiles())) {
        cout << "FATAL: data load failed." << endl; return 1;
    }

    const string actorBest  = "gpu_actor_best.bin";
    const string criticBest = "gpu_critic_best.bin";
    const string actorCkpt  = "gpu_actor_ckpt.bin";
    const string criticCkpt = "gpu_critic_ckpt.bin";

    if (doTrain) {
        runTraining(loader, actorBest, criticBest, actorCkpt, criticCkpt);
    } else {
        runTest(loader, actorBest, criticBest);
    }

    cout << "\nPress any key to exit..." << endl;
    _getch();
    return 0;
}
