#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <conio.h>
#include <iomanip>
using namespace std;

// ===================================================
// Function Descriptions
// ===================================================

// Function 0: Weighted AND
// Fires (+1) only when most inputs are positive (>=3 of 4 inputs are +1).
// Effectively implements a "strict AND" logic where all/most inputs must be active to trigger.

// Function 1: Weighted OR
// Fires (+1) if at least one input is positive.
// Implements a "loose OR" logic where any single active input is enough to trigger the output.

// Function 2: Weighted NAND
// Fires (+1) unless the majority of inputs are positive.
// This is the inverse of the Weighted AND function, useful for inhibitory logic.

// ===================================================
// Batch Learning Output Function
// ===================================================
void printTS(double *x, double *y, double *OutF, int nPat, int nIn, int nOut)
{
    cout << "\n========= Network Output Test =========\n";
    cout << "Pattern | Inputs\t\t\t| Target Outputs\t| Network Output\t| Correct?\n";
    cout << "--------------------------------------------------------------------------\n";

    int correctCount = 0;

    for (int i = 0; i < nPat; i++)
    {
        cout << setw(3) << i << "     | ";

        // Print inputs
        for (int j = 0; j < nIn; j++)
            cout << setw(3) << x[i * nIn + j] << " ";
        cout << "\t| ";

        // Print target outputs
        for (int j = 0; j < nOut; j++)
            cout << setw(3) << y[i * nOut + j] << " ";
        cout << "\t| ";

        // Print network outputs as -1 / +1
        for (int j = 0; j < nOut; j++)
        {
            int signOut = (OutF[i * nOut + j] >= 0) ? 1 : -1;
            cout << setw(3) << signOut << " ";
            if (signOut == (int)y[i * nOut + j])
                correctCount++;
        }
        cout << "\t|\n";
    }

    cout << "--------------------------------------------------------------------------\n";
    double accuracy = 100.0 * correctCount / (nPat * nOut);
    cout << "Network Accuracy: " << fixed << setprecision(2) << accuracy << " %\n";
    cout << "========================================\n\n";
}

// ===================================================
// Print Network Weights and Biases
// ===================================================
void printWeight(double *w, int nOut, int nIn, double *b)
{
    cout << "\n========= Network Weights and Biases =========\n";
    for (int i = 0; i < nOut; i++)
    {
        cout << "Output neuron " << i << ":\n";
        for (int j = 0; j < nIn; j++)
            cout << "  w[" << i << "][" << j << "] = "
                 << setw(8) << fixed << setprecision(4) << w[i * nIn + j] << "\n";
        cout << "  b[" << i << "] = "
             << setw(8) << fixed << setprecision(4) << b[i] << "\n\n";
    }
    cout << "=============================================\n\n";
}

// ===================================================
// Main Function
// ===================================================
int main()
{
    const int nIn = 4;     // Number of inputs
    const int nOut = 3;    // Number of outputs
    const int nPat = 16;   // Number of patterns
    const int nCycles = 500;
    const double alfa = 0.2; // Learning rate

    double x[nPat][nIn], y[nPat][nOut];   // Input, Output
    double w[nOut][nIn], dw[nIn];         // Weights and weight changes
    double b[nOut], db;                   // Bias and bias change
    double OutF[nPat][nOut];              // Network output
    double z, dz, dedo;                   // Temp variables

    srand(time(NULL)); // Random seed

    // ===================================================
    // Training Set
    // ===================================================
    // ===== Inputs x[16][4] =====
    x[0][0] = -1; x[0][1] = -1; x[0][2] = -1; x[0][3] = -1;
    x[1][0] =  1; x[1][1] = -1; x[1][2] = -1; x[1][3] = -1;
    x[2][0] = -1; x[2][1] =  1; x[2][2] = -1; x[2][3] = -1;
    x[3][0] = -1; x[3][1] = -1; x[3][2] =  1; x[3][3] = -1;
    x[4][0] = -1; x[4][1] = -1; x[4][2] = -1; x[4][3] =  1;
    x[5][0] =  1; x[5][1] =  1; x[5][2] = -1; x[5][3] = -1;
    x[6][0] =  1; x[6][1] = -1; x[6][2] =  1; x[6][3] = -1;
    x[7][0] =  1; x[7][1] = -1; x[7][2] = -1; x[7][3] =  1;
    x[8][0] = -1; x[8][1] =  1; x[8][2] =  1; x[8][3] = -1;
    x[9][0] = -1; x[9][1] =  1; x[9][2] = -1; x[9][3] =  1;
    x[10][0] = -1; x[10][1] = -1; x[10][2] =  1; x[10][3] =  1;
    x[11][0] =  1; x[11][1] =  1; x[11][2] =  1; x[11][3] = -1;
    x[12][0] =  1; x[12][1] =  1; x[12][2] = -1; x[12][3] =  1;
    x[13][0] =  1; x[13][1] = -1; x[13][2] =  1; x[13][3] =  1;
    x[14][0] = -1; x[14][1] =  1; x[14][2] =  1; x[14][3] =  1;
    x[15][0] =  1; x[15][1] =  1; x[15][2] =  1; x[15][3] =  1;

    // ===== Outputs y[pattern][function] =====
    // Columns: {Weighted AND, Weighted OR, Weighted NAND}
    y[0][0] = -1; y[0][1] = -1; y[0][2] =  1;
    y[1][0] = -1; y[1][1] =  1; y[1][2] =  1;
    y[2][0] = -1; y[2][1] =  1; y[2][2] =  1;
    y[3][0] = -1; y[3][1] =  1; y[3][2] =  1;
    y[4][0] = -1; y[4][1] =  1; y[4][2] =  1;
    y[5][0] = -1; y[5][1] =  1; y[5][2] =  1;
    y[6][0] = -1; y[6][1] =  1; y[6][2] =  1;
    y[7][0] = -1; y[7][1] =  1; y[7][2] =  1;
    y[8][0] = -1; y[8][1] =  1; y[8][2] =  1;
    y[9][0] = -1; y[9][1] =  1; y[9][2] =  1;
    y[10][0] = -1; y[10][1] =  1; y[10][2] =  1;
    y[11][0] =  1; y[11][1] =  1; y[11][2] = -1;
    y[12][0] =  1; y[12][1] =  1; y[12][2] = -1;
    y[13][0] =  1; y[13][1] =  1; y[13][2] = -1;
    y[14][0] =  1; y[14][1] =  1; y[14][2] = -1;
    y[15][0] =  1; y[15][1] =  1; y[15][2] = -1;

    // ===================================================
    // Network Initialization
    // ===================================================
    for (int j = 0; j < nOut; j++)
    {
        for (int i = 0; i < nIn; i++)
            w[j][i] = ((rand() % 100) - 50) / 200.0;
        b[j] = ((rand() % 100) - 50) / 200.0;
    }

    printWeight((double *)w, nOut, nIn, b);
    getche();

    // ===================================================
    // Training
    // ===================================================
    for (int k = 0; k < nCycles; k++)
    {
        for (int f = 0; f < nOut; f++)
        {
            db = 0.0;
            for (int n = 0; n < nIn; n++)
                dw[n] = 0.0;

            // Full batch learning
            for (int i = 0; i < nPat; i++)
            {
                // Feed Forward
                z = b[f];
                for (int j = 0; j < nIn; j++)
                    z += w[f][j] * x[i][j];
                OutF[i][f] = tanh(z);

                // Error
                dedo = y[i][f] - OutF[i][f];

                // Backpropagation
                dz = dedo * (1 - (OutF[i][f] * OutF[i][f]));
                db += dz;
                for (int j = 0; j < nIn; j++)
                    dw[j] += dz * x[i][j];
            }

            // Update weights and biases
            b[f] += db * alfa / nPat;
            for (int j = 0; j < nIn; j++)
                w[f][j] += dw[j] * alfa / nPat;
        }
    }

    // ===================================================
    // Display Results
    // ===================================================
    printTS((double *)x, (double *)y, (double *)OutF, nPat, nIn, nOut);
    printWeight((double *)w, nOut, nIn, (double *)b);
    return 0;
}
