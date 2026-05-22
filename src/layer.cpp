#include "layer.h"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace std;

layer::layer(int myin, int myout)
    : nIn(myin), nOut(myout)
{
    int i, j;
    w = matD(nOut, nIn);
    dw = matD(nOut, nIn);
    b = matD(nOut);
    db = matD(nOut);
    mOutF = matD(nOut);
    mOutB = matD(nIn);
    mask = matD(nOut);
    m_w = matD(nOut, nIn);
    v_w = matD(nOut, nIn);
    m_b = matD(nOut);
    v_b = matD(nOut);

    alfa  = 0.0003;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-5;
    t = 0;
    isLinear = false;

    // Orthogonal initialization via Gram-Schmidt QR decomposition
    // Step 1: Fill w with random Gaussian values
    for (i = 0; i < nOut; i++)
    {
        for (j = 0; j < nIn; j++)
        {
            // Box-Muller transform for Gaussian random
            double u1 = ((double)rand() / RAND_MAX);
            double u2 = ((double)rand() / RAND_MAX);
            if (u1 < 1e-10) u1 = 1e-10;
            w[i][j] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        }
    }

    // Step 2: Gram-Schmidt orthogonalization (on the rows of w)
    // We orthogonalize min(nOut, nIn) rows
    int num_ortho = (nOut < nIn) ? nOut : nIn;
    for (i = 0; i < num_ortho; i++)
    {
        // Subtract projections onto all previous rows
        for (int k = 0; k < i; k++)
        {
            double dot = 0.0;
            double norm_k = 0.0;
            for (j = 0; j < nIn; j++)
            {
                dot += w[i][j] * w[k][j];
                norm_k += w[k][j] * w[k][j];
            }
            if (norm_k > 1e-12)
            {
                for (j = 0; j < nIn; j++)
                    w[i][j] -= (dot / norm_k) * w[k][j];
            }
        }
        // Normalize the row
        double norm = 0.0;
        for (j = 0; j < nIn; j++)
            norm += w[i][j] * w[i][j];
        norm = sqrt(norm);
        if (norm > 1e-12)
        {
            for (j = 0; j < nIn; j++)
                w[i][j] /= norm;
        }
    }

    // Step 3: Scale by gain (sqrt(2) is standard for tanh, 1.0 for linear)
    double gain = sqrt(2.0);
    for (i = 0; i < nOut; i++)
    {
        for (j = 0; j < nIn; j++)
            w[i][j] *= gain;
    }

    // Initialize remaining state
    for (i = 0; i < nOut; i++)
    {
        m_b[i] = 0;
        v_b[i] = 0;
        mask[i] = 1;
        for (j = 0; j < nIn; j++)
        {
            m_w[i][j] = 0;
            v_w[i][j] = 0;
        }
        b[i] = 0;
    }
}

// Added Destructor to prevent memory leaks
layer::~layer()
{
    for (int i = 0; i < nOut; i++) {
        delete[] w[i];
        delete[] dw[i];
        delete[] m_w[i];
        delete[] v_w[i];
    }
    delete[] w;
    delete[] dw;
    delete[] b;
    delete[] db;
    delete[] mOutF;
    delete[] mOutB;
    delete[] mask;
    delete[] m_w;
    delete[] v_w;
    delete[] m_b;
    delete[] v_b;
}

////////////////////////////////////////////////////////////////////
void layer::BP()
{
    int i, j;
    double dz;
    double d;
    //-------------------initialization----------------------
    for (i = 0; i < nIn; i++)
        mOutB[i] = 0;
    //-----------------------main loops----------------------
    for (j = 0; j < nOut; j++)
    {
        // Recover the original tanh(z) value by dividing by the mask scalar
        // mask[j] is either 0 or 1/p.
        double raw_activation = (mask[j] > 0) ? (mOutF[j] / mask[j]) : 0.0;

        d = isLinear ? 1.0 : (1.1 - raw_activation * raw_activation);
        dz = pInB[j] * d * mask[j];
        db[j] += dz;
        for (i = 0; i < nIn; i++)
        {
            dw[j][i] += dz * pInF[i];
            mOutB[i] += w[j][i] * dz;
        }
    }
}

void layer::clear_mom()
{
    // 1. Reset Adam Time Step (Critical for bias correction)
    t = 0;

    // 2. Clear Weight Buffers
    for(int j=0; j<nOut; j++)
    {
        for(int i=0; i<nIn; i++)
        {
            m_w[j][i] = 0; // Clear 1st moment (Momentum)
            v_w[j][i] = 0; // Clear 2nd moment (RMSProp)
        }

        // 3. Clear Bias Buffers
        m_b[j] = 0;
        v_b[j] = 0;
    }
}
////////////////////////////////////////////////////////////////////
void layer::FF() // to get layer output
{
    int i, j;
    double z;
    // each pattern
    for (j = 0; j < nOut; j++)
    {
        z = 0.0;
        for (i = 0; i < nIn; i++)
        {
            z += w[j][i] * pInF[i];
        }
        z += b[j];
       if (isLinear) {
           mOutF[j] = z; // Linear activation for output layers
       } else {
           mOutF[j] = tanh(z);
       }
       mOutF[j]*=mask[j];
    }
}

void layer::clear_grads()
{
    for (int i = 0; i < nOut; i++)
    {
        db[i] = 0;
        for (int j = 0; j < nIn; j++)
        {
            dw[i][j] = 0;
        }
    }
}

void layer::change_mask(float p)
{
    for(int i=0;i<nOut;i++)
    {
        double r = (double)rand() / RAND_MAX;

        if (r < p)
        {
            // KEEP: Set to scaling factor
            mask[i] = 1.0/p;
        }
        else
        {
            // DROP: Set to 0
            mask[i] = 0.0;
        }
    }
}
//////////////////////////////////////////////////////////////////////
void layer::update(int B_size)
{
    // 1. Increment Time Step (globally for the layer)
    t++;

    // 2. Calculate Bias Correction Factors (Scalar math, do it once)
    double correction1 = 1.0 - pow(beta1, t);
    double correction2 = 1.0 - pow(beta2, t);

    double lr_t = alfa * sqrt(correction2) / correction1;

    // Configurable weight decay parameter (L2 regularization proxy for AdamW-style update)
    double weight_decay = 0.0001;

    // --- Gradient Clipping (by global norm, max_grad_norm = 0.5) ---
    double max_grad_norm = 0.5;
    double grad_norm_sq = 0.0;
    for (int j = 0; j < nOut; j++)
    {
        double g = db[j] / B_size;
        grad_norm_sq += g * g;
        for (int i = 0; i < nIn; i++)
        {
            g = dw[j][i] / B_size;
            grad_norm_sq += g * g;
        }
    }
    double grad_norm = sqrt(grad_norm_sq);
    double clip_coef = (grad_norm > max_grad_norm) ? (max_grad_norm / grad_norm) : 1.0;

    for (int j = 0; j < nOut; j++)
    {
        // --- UPDATE BIASES ---
        double g = (db[j] / B_size) * clip_coef; // Clipped gradient

        // Update Moments
        m_b[j] = beta1 * m_b[j] + (1 - beta1) * g;
        v_b[j] = beta2 * v_b[j] + (1 - beta2) * g * g;

        // Bias Correction
        double m_hat = m_b[j] / correction1;
        double v_hat = v_b[j] / correction2;

        // Apply Update (Removed bias decay here, as decaying biases is bad practice)
        b[j] += lr_t * m_hat / (sqrt(v_hat) + epsilon);

        // Clear gradient for next batch
        db[j] = 0;

        // --- UPDATE WEIGHTS ---
        for (int i = 0; i < nIn; i++)
        {
            g = (dw[j][i] / B_size) * clip_coef; // Clipped gradient

            // Update Moments
            m_w[j][i] = beta1 * m_w[j][i] + (1 - beta1) * g;
            v_w[j][i] = beta2 * v_w[j][i] + (1 - beta2) * g * g;

            // Bias Correction
            m_hat = m_w[j][i] / correction1;
            v_hat = v_w[j][i] / correction2;

            // Apply Update with Weight Decay (Proper AdamW style)
            w[j][i] *= (1.0 - weight_decay);
            w[j][i] += lr_t * m_hat / (sqrt(v_hat) + epsilon);

            // Clear gradient
            dw[j][i] = 0;
        }
    }
}
//////////////////////////////////////////////////////////////////////
void layer::print()
{
    int i, j;
    cout << "No of Input    =" << nIn << endl;
    cout << "No of Output   =" << nOut << endl;
    cout << "Alfa Value     =" << alfa << endl;

    for (i = 0; i < nOut; i++)
    {
        cout << "w[" << i + 1 << "] = ";
        for (j = 0; j < nIn; j++)
            cout << w[i][j] << " , ";
        cout << "b[" << i + 1 << "] = " << b[i] << endl;
    }
}
void layer::printOut()
{
    for (int j = 0; j < nOut; j++)
    {
        cout << mOutF[j] << "  ";
    }
    cout << endl;
}
void layer::save(ofstream &file)
{
    // 1. Write Layer Dimensions
    file << nIn << " " << nOut << endl;

    // 2. Write Weights (Row by Row)
    for (int j = 0; j < nOut; j++)
    {
        for (int i = 0; i < nIn; i++)
        {
            file << setprecision(15) << w[j][i] << " ";
        }
        file << endl;
    }

    // 3. Write Biases
    for (int j = 0; j < nOut; j++)
    {
        file << setprecision(15) << b[j] << " ";
    }
    file << endl;

    // 4. Write Adam optimizer state
    file << t << endl; // Adam timestep
    for (int j = 0; j < nOut; j++) {
        for (int i = 0; i < nIn; i++) {
            file << setprecision(15) << m_w[j][i] << " " << v_w[j][i] << " ";
        }
        file << endl;
    }
    for (int j = 0; j < nOut; j++) {
        file << setprecision(15) << m_b[j] << " " << v_b[j] << " ";
    }
    file << endl;
}

void layer::load(ifstream &file)
{
    int tempIn, tempOut;
    // 1. Read and Check Dimensions
    file >> tempIn >> tempOut;

    if (tempIn != nIn || tempOut != nOut)
    {
        cout << "ERROR: Layer Loading Mismatch!" << endl;
        cout << "File expects: " << tempIn << "->" << tempOut << endl;
        cout << "Layer is:     " << nIn << "->" << nOut << endl;
        return;
    }

    // 2. Read Weights
    for (int j = 0; j < nOut; j++)
    {
        for (int i = 0; i < nIn; i++)
        {
            file >> w[j][i];
        }
    }

    // 3. Read Biases
    for (int j = 0; j < nOut; j++)
    {
        file >> b[j];
    }

    // 4. Read Adam optimizer state
    file >> t;
    for (int j = 0; j < nOut; j++) {
        for (int i = 0; i < nIn; i++) {
            file >> m_w[j][i] >> v_w[j][i];
        }
    }
    for (int j = 0; j < nOut; j++) {
        file >> m_b[j] >> v_b[j];
    }
}
