#include "trainer.h"

trainer::trainer(net *theNet, trainSet *myts) : ts(myts), Net(theNet)
{
    nIn = (ts->nIn);
    nOut = (ts->nOut);
    nPat = (ts->nPat);
    mda = matD(nOut); // de/do
    px = (ts->x);
    py = (ts->y);
    // to connect trainer to the network and training set.
    Net->Ls[Net->nL - 1]->pInB = mda; // connect input for last layer for BP
    pa = Net->Ls[Net->nL - 1]->mOutF;
}
////////////////////////////////////////////////////////////////////////
void trainer::update_Err(int pat)
{
    for (int i = 0; i < nOut; i++)
    {
        mda[i] = py[pat][i] - pa[i];
    }
}
////////////////////////////////////////////////////////////////
void trainer::printTs_out()
{
    cout << "\n=======================================================" << endl;
    cout << " FINAL RESULTS CHECK " << endl;
    cout << "=======================================================" << endl;
    cout << " Pat  |   Input (Subset)  |   Target   |   Output   |  Status" << endl;
    cout << "-------------------------------------------------------" << endl;

    int correct = 0;

    for (int k = 0; k < nPat; k++)
    {
        // 1. Point Network to this pattern
        Net->Ls[0]->pInF = px[k];

        // 2. Run Forward Pass Only
        Net->FF();

        // 3. Logic to check if correct (Signs match)
        // Note: Assuming output is accessed via pa pointer set in constructor
        bool isCorrect = false;
        if ((py[k][0] > 0 && pa[0] > 0) || (py[k][0] < 0 && pa[0] < 0))
        {
            isCorrect = true;
            correct++;
        }

        // 4. Print (Show first 10 and last 5 to keep console clean)
        if (k < 10 || k >= nPat - 5)
        {
            cout << setw(5) << k << " | ";

            // Print first 2 inputs just for reference
            cout << fixed << setprecision(2) << px[k][0] << " " << px[k][1] << ".. | ";

            cout << setw(10) << py[k][0] << " | ";
            cout << setw(10) << pa[0] << " | ";
            cout << (isCorrect ? "OK" : "XX") << endl;
        }
    }

    cout << "-------------------------------------------------------" << endl;
    cout << "Final Accuracy: " << ((double)correct / nPat) * 100.0 << "%" << endl;
    cout << "=======================================================" << endl;
}
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
int trainer::train(int cycles, int B_size, string modelname, trainSet *testset)
{
   // bool d_out=false;
    double current_acc, max_acc = 0.0;
    double avgLoss;
    int patienceCount = 0;
    int patienceLimit = 10;
    double beta_loss = 0.9;
    int epoch, i, j, k;
    double sumSqError;
    int maxOutIdx, maxTgtIdx;
    double maxOutVal, maxTgtVal;
    double minDelta = 0.0001;
    double best_loss = 1000;
    // Outer Loop: Epochs (Renamed 'j' to 'epoch' to avoid conflict with inner loop)
    for (epoch = 0; epoch < cycles; epoch++)
    {
        if (_kbhit()) // Check if any key was pushed
        {
            char ch = _getch(); // Read the key
            if (ch == 27)       // 'q' or ESC (ASCII 27)
            {
                cout << "\n\n>>> INTERRUPTED BY USER <<<" << endl;
                cout << "Stopping at epoch " << epoch << "..." << endl;
               current_acc = validate(Net, testset);
            if (current_acc > max_acc)
            {
                max_acc = current_acc;
                Net->save(modelname);
            }
                return epoch; // Exit function immediately, returning current epoch
            }
        }
        ts->shuffle();
        // Reset metrics at the start of every epoch
        Loss = 0;
        MaxError = 0;
        errorCount = 0;

        // Middle Loop: Batches
        for (i = 0; i < nPat; i += B_size)
        {
            Net->clear(); // Clear gradients before batch starts

            // Calculate limit to protect against crash on last partial batch
            int limit = (i + B_size < nPat) ? (i + B_size) : nPat;

            // Inner Loop: Patterns in Batch
            for (j = i; j < limit; j++)
            {
                Net->Ls[0]->pInF = px[j]; // Inject input
                Net->FF();                // Forward Pass
                update_Err(j);            // Calculate error into mda[]

                // --- INJECTED ACCURACY MEASURES ---
                sumSqError = 0;
                for (k = 0; k < nOut; k++)
                {
                    sumSqError += mda[k] * mda[k];
                }
                Loss += sumSqError / nOut; // Average error per neuron
                // ----------------------------------
                maxOutIdx = -1;
                maxOutVal = -99999.0;
                for (k = 0; k < nOut; k++)
                {
                    if (pa[k] > maxOutVal)
                    {
                        maxOutVal = pa[k];
                        maxOutIdx = k;
                    }
                }

                // Find index of highest target neuron (The correct answer)
                maxTgtIdx = -1;
                maxTgtVal = -99999.0;
                for (k = 0; k < nOut; k++)
                {
                    if (py[j][k] > maxTgtVal)
                    {
                        maxTgtVal = py[j][k];
                        maxTgtIdx = k;
                    }
                }

                // If guess doesn't match target, count as error
                if (maxOutIdx != maxTgtIdx)
                    errorCount++;
                // ---------------------------------------

                Net->BP(); // Back Propagation
            }

            // Update weights using actual batch size (for correct averaging)
            Net->update_Ls(B_size);
        }

        // Calculate average loss for the epoch
        Loss /= nPat;
        if (avgLoss == 0)
            avgLoss = Loss;
        else
            avgLoss = (beta_loss * avgLoss) + ((1.0 - beta_loss) * Loss);
        if (avgLoss < best_loss - minDelta)
        {
            best_loss = avgLoss;
            patienceCount = 0; // Reset counter if we improved
            // Optional: Save "best_model_so_far.txt" here
        }
        else
        {
            patienceCount++; // No significant improvement
        }

        // 2. Trigger the "Rescue" if patience runs out
        if (patienceCount >= patienceLimit)
        {
            cout << "\n>>> PLATEAU DETECTED (Epoch " << epoch << ") <<<" << endl;
            Net->alfa *= 0.8;
            Net->beta *= 0.95;
            Net->clear_mom();
            patienceCount=0;
        }
        if (Net->alfa<0.001)
        {
            cout << "Learning rate too small. Stopping training." << endl;
            return epoch;
        }

        // Print status every 10 epochs (more frequent for MNIST)
        
        if(epoch%2 == 0){
        cout << "Epoch: " << epoch
             << " | Loss: " << Loss
             << " | Wrong Patterns: " << errorCount
             << " | Acc: " << (1.0 - (double)errorCount / nPat) * 100.0 << "%" << endl;
        }
        if (epoch % 10 == 0 && epoch != 0)
        {
            current_acc = validate(Net, testset);
            if (current_acc > max_acc)
            {
                max_acc = current_acc;
                Net->save(modelname);
            }
        }

        // Stop early if perfect (unlikely for MNIST but good practice)
        if (errorCount == 0)
        {
            cout << "-> Converged to 0 errors at epoch " << epoch << endl;
            return epoch;
        }
    }
    return epoch;
}
double trainer::validate(net *net_test, trainSet *test_set)
{
    double totalLoss = 0.0;
    int correctCount = 0;

    // Local shortcuts
    int nPatterns = test_set->nPat;
    int nOutputs = test_set->nOut;

    // Variables for Argmax/MSE logic
    int maxOutIdx, maxTgtIdx;
    double maxOutVal, maxTgtVal;
    double sumSqError;

    // Loop through the entire test set
    for (int k = 0; k < nPatterns; k++)
    {
        // 1. Inject Input from the test set
        net_test->Ls[0]->pInF = test_set->x[k];

        // 2. Forward Pass (No Backprop)
        net_test->FF();

        // 3. Get Pointers to current output and target
        double *currentOutput = net_test->Ls[net_test->nL - 1]->mOutF;
        double *currentTarget = test_set->y[k];

        // --- Calculate Loss (MSE) for reporting ---
        sumSqError = 0;
        for (int i = 0; i < nOutputs; i++)
        {
            double err = currentTarget[i] - currentOutput[i];
            sumSqError += err * err;
        }
        totalLoss += sumSqError / nOutputs;

        // --- Calculate Accuracy (Argmax) ---

        // Find Network's Prediction (Index of highest neuron)
        maxOutIdx = -1;
        maxOutVal = -99999.0;
        for (int i = 0; i < nOutputs; i++)
        {
            if (currentOutput[i] > maxOutVal)
            {
                maxOutVal = currentOutput[i];
                maxOutIdx = i;
            }
        }

        // Find Actual Target (Index of highest label)
        maxTgtIdx = -1;
        maxTgtVal = -99999.0;
        for (int i = 0; i < nOutputs; i++)
        {
            if (currentTarget[i] > maxTgtVal)
            {
                maxTgtVal = currentTarget[i];
                maxTgtIdx = i;
            }
        }

        // Check if correct
        if (maxOutIdx == maxTgtIdx)
        {
            correctCount++;
        }
    }

    // Calculate final metrics
    double avgLoss = totalLoss / nPatterns;
    double accuracy = (double)correctCount / nPatterns * 100.0;

    // Print Report
    cout << "\n>>> VALIDATION RESULTS <<<" << endl;
    cout << "Test Patterns: " << nPatterns << endl;
    cout << "Test Loss:     " << avgLoss << endl;
    cout << "Test Accuracy: " << accuracy << "%" << endl;
    cout << "==========================" << endl;

    return accuracy;
}
