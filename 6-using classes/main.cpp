#include <iostream>
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <fstream>
#include <string>
#include "matrix.h"
#include "trainSet.h"
#include "layer.h"
#include "net.h"
#include "trainer.h"

using namespace std;


void analyzeActivations(net *N, trainSet *TS)
{
    int randPat = rand() % TS->nPat;
    N->Ls[0]->pInF = TS->x[randPat];
    N->FF();
    cout << "\n=== NEURON SATURATION REPORT (Init Check) ===" << endl;
    cout << "Pattern Index: " << randPat << endl;
    cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < N->nL; i++)
    {
        int nTotal = N->Ls[i]->nOut;
        int nSaturated = 0; // |y| > 0.9
        int nLinear = 0;    // |y| < 0.1
        double sumMag = 0;
        for (int j = 0; j < nTotal; j++)
        {
            double val = fabs(N->Ls[i]->mOutF[j]);
            sumMag += val;
            if (val > 0.9)
            nSaturated++;
            if (val < 0.1)
                nLinear++;
        }
        double avgMag = sumMag / nTotal;
        cout << "Layer " << i << " (" << nTotal << " neurons): "
        << "Avg |y| = " << avgMag << " | "
        << "Linear (<0.1): " << (nLinear * 100.0 / nTotal) << "% | "
        << "Saturated (>0.9): " << (nSaturated * 100.0 / nTotal) << "%" << endl;
    }
    cout << "=============================================\n"
    << endl;
}
int main()
{
    // ---------------------------------------------------------
    // 1. CONFIGURATION VARIABLES
    // ---------------------------------------------------------
    int nLayer = 2;
    int nIn = 15;
    int nOut = 1;
    int nPat = 32768;
    float alpha = 0.4;
    float beta = 0.8;
    int epochs = 100;
    int batchSize = 8192;
    int initMode = 0;
    //float p_mask=1; // 0 = Random, 1 = Load from File
    string modelFileName;
    int *hiddenLayers = NULL;
    
    // ---------------------------------------------------------
    // 2. READ CONFIG FILE
    // ---------------------------------------------------------
    ifstream configFile("config.txt");
    if (!configFile.is_open())
    {
        cout << "Warning: config.txt not found! Using hardcoded defaults." << endl;
        hiddenLayers = new int[1];
        hiddenLayers[0] = 48;
    }
    else
    {
        string label;
        // Order must match file exactly
        configFile >> label >> nLayer;
        configFile >> label >> nIn;
        configFile >> label >> nOut;
        configFile >> label >> nPat;
        configFile >> label >> alpha;
        configFile >> label >> beta;
        configFile >> label >> epochs;
        configFile >> label >> batchSize;
        // Read Hidden Layers
        configFile >> label; // "hiddenNodes"
        hiddenLayers = new int[nLayer - 1];
        for (int k = 0; k < nLayer - 1; k++)
        {
            configFile >> hiddenLayers[k];
        }
        // Read Init Mode & Filename
        configFile >> label >> initMode;      // Read "initMode"
        configFile >> label >> modelFileName; // Read "modelFile"
        configFile.close();
            cout << "Configuration loaded. Mode: " << (initMode == 1 ? "LOAD FILE" : "RANDOM INIT") << endl;
        }
        // ---------------------------------------------------------
        // 3. NETWORK SETUP
        // ---------------------------------------------------------
        srand(time(NULL));
        trainSet *TestS = new trainSet();
        TestS->nIn = nIn;
        TestS->nOut = nOut;
        TestS->nPat = nPat;
        TestS->Create();
        TestS->Fill_minst("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
        trainSet *TS = new trainSet();
        TS->nIn = nIn;
        TS->nOut = nOut;
        TS->nPat = nPat;
        TS->Create();
        TS->Fill_minst("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        // Create Net
        net *N = new net(nLayer, TS, alpha, beta );
        
        for (int k = 0; k < nLayer - 1; k++)
        {
            N->nForLayers[k] = hiddenLayers[k];
        }
        
        N->Creat(); // Weights are randomized here by default
        // ---------------------------------------------------------
        // === INITIALIZATION LOGIC ===
        if (initMode == 1)
        {
            cout << "Loading weights from " << modelFileName << "..." << endl;
            N->load(modelFileName); // Overwrite random weights with file data
        }
        else
        {
            cout << "Initializing weights randomly..." << endl;
            // No action needed; N->Creat() already did this.
        }
        // ============================
        cout << "Checking initialization state..." << endl;
        analyzeActivations(N, TS);
        // ---------------------------------------------------------
        // 4. TRAINING
        // ---------------------------------------------------------
    trainer *tr = new trainer(N, TS);
    
        cout << "Starting training..." << endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        int actual_epochs = tr->train(epochs, batchSize,modelFileName,TestS);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        cout << "Training Time: " << elapsed_seconds.count() << " seconds" << endl;
        cout << "Final Stats -> Epoch: " << actual_epochs
             << ", Loss: " << tr->Loss
             << ", Errors: " << tr->errorCount << endl;

    // ---------------------------------------------------------
    // 5. SAVE RESULT (Optional auto-save)
    // ---------------------------------------------------------
    // Automatically save the new weights after training finishes
    tr->validate(N, TestS);
    getche();
    cout << "Saving updated model to " << modelFileName << "..." << endl;
    N->save(modelFileName);
    delete[] hiddenLayers;
    return 0;
}
