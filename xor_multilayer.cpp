#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm> // For std::swap

using namespace std;

const int NUM_PATTERNS = 32768;
const int NUM_INPUTS = 15;
const int BATCH = 2048;

// Allocate arrays
double inputs[NUM_PATTERNS][NUM_INPUTS];
double targets[NUM_PATTERNS][1];
const double ALPHA = 0.4; // Learning Rate
const double BETA = 0.9;

class layer
{
    int nIn;
    int nOut;
    int nPat;
    double **W;
    double **pre_dW;
    double **Input;
    double **output;
    double **err;
    double **delta;
    double *b;
    double *pre_db;
    double alpha;
    double db, dW;

public:
    layer(int nin, int nout, int npat, double **input, float a) : nIn(nin), nOut(nout), nPat(npat), Input(input), alpha(a)
    {
        output = new double *[nPat];
        delta = new double *[nPat];
        err = new double *[nPat];
        W = new double *[nOut];
        pre_dW = new double *[nOut];
        for (int i = 0; i < nPat; i++)
        {
            output[i] = new double[nOut];
            delta[i] = new double[nOut];
            err[i] = new double[nOut];
        }
        for (int i = 0; i < nOut; i++)
        {
            W[i] = new double[nIn];
            pre_dW[i] = new double[nIn];
        }
        b = new double[nOut];
        pre_db = new double[nOut];
    }

    void init_Wb()
    {
        for (int i = 0; i < nOut; i++)
        {
            for (int j = 0; j < nIn; j++)
            {
                W[i][j] = ((rand() % 100) - 50) / 50.0;
                pre_dW[i][j] = 0.0;
            }
            b[i] = ((rand() % 100) - 50) / 50.0;
            pre_db[i] = 0.0;
        }
    }

    void calc_Out()
    {
        for (int i = 0; i < nOut; i++)
        {
            for (int j = 0; j < nPat; j++)
            {
                float sum = b[i];
                for (int k = 0; k < nIn; k++)
                {
                    sum += Input[j][k] * W[i][k];
                }
                float O = tanh(sum);
                output[j][i] = O;
            }
        }
    }

    void calc_err(double **y_train)
    {
        for (int i = 0; i < nOut; i++)
        {
            for (int j = 0; j < nPat; j++)
            {
                err[j][i] = y_train[j][i] - output[j][i];
            }
        }
    }

    void calc_delta()
    {
        for (int i = 0; i < nOut; i++)
        {
            for (int j = 0; j < nPat; j++)
            {
                delta[j][i] = err[j][i] * (1 - output[j][i] * output[j][i]);
            }
        }
    }

    double **back_err()
    {
        double **b_err = new double *[nPat];
        for (int i = 0; i < nPat; i++)
        {
            b_err[i] = new double[nIn];
        }

        for (int i = 0; i < nPat; i++)
        {
            for (int j = 0; j < nIn; j++)
            {
                b_err[i][j] = 0;
                for (int k = 0; k < nOut; k++)
                {
                    b_err[i][j] += delta[i][k] * W[k][j];
                }
            }
        }
        return b_err;
    }

    double **get_out() { return output; }

    void set_err(layer *next)
    {
        // Prevent memory leak
        if (err != nullptr)
        {
            for (int i = 0; i < nPat; i++)
                delete[] err[i];
            delete[] err;
        }
        err = next->back_err();
    }
    void setInput(double **newInput) {
        Input = newInput;
    }

    void update()
    {
        for (int i = 0; i < nOut; i++)
        {
            for (int j = 0; j < nIn; j++)
            {
                dW = 0;
                for (int k = 0; k < nPat; k++)
                {
                    dW += delta[k][i] * Input[k][j];
                    
                }
                dW /= nPat;
                double new_dW = BETA * pre_dW[i][j] + ALPHA * dW;
                pre_dW[i][j] = new_dW;
                W[i][j] += new_dW;
            }
            db = 0;
            for (int j = 0; j < nPat; j++)
            {
                db += delta[j][i];
            }
            db /= nPat;
            double new_db = BETA * pre_db[i] + ALPHA * db;
            pre_db[i] = new_db;
            b[i] += new_db;
        }
    }
    ~layer()
    {
        for (int j = 0; j < nPat; j++)
        {
            delete[] output[j];
            delete[] delta[j];
            if (err)
                delete[] err[j];
        }
        delete[] output;
        delete[] delta;
        if (err)
            delete[] err;

        for (int i = 0; i < nOut; i++)
        {
            delete[] W[i];
            delete[] pre_dW[i];
        }
        delete[] W;
        delete[] pre_dW;
        delete[] b;
        delete[] pre_db;
    }
};

void shuffle_data(double **in_ptrs, double (*targets)[1], int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(in_ptrs[i], in_ptrs[j]);
        double temp = targets[i][0];
        targets[i][0] = targets[j][0];
        targets[j][0] = temp;
    }
}

double **FF(layer **net, int L)
{
    for (int i = 0; i < L; i++)
    {
        net[i]->calc_Out();
    }
    return net[L - 1]->get_out();
}

void BB(layer **net, int L, double **y_train)
{
    net[L - 1]->calc_err(y_train);
    net[L - 1]->calc_delta();
    for (int l = L - 2; l >= 0; l--)
    {
        net[l]->set_err(net[l + 1]);
        net[l]->calc_delta();
    }
    net[L - 1]->update();
    for (int l = L - 2; l >= 0; l--)
    {
        net[l]->update();
    }
}

// --- MOVED UP SO TRAIN_OUT CAN SEE IT ---
int check_full_accuracy(layer **layers, int L, double **input_ptrs, double **target_ptrs) {
    int correct_count = 0;
    
    // We must loop through the data in chunks of BATCH
    for(int b = 0; b < NUM_PATTERNS; b += BATCH) {
        
        // 1. Slide the window
        layers[0]->setInput(input_ptrs + b);
        
        // 2. Predict (Feed Forward only)
        double** batch_out = FF(layers, L);
        
        // 3. Check this batch
        for(int k = 0; k < BATCH; k++) {
            // Be careful not to go past 1024 if NUM_PATTERNS isn't a multiple of BATCH
            if (b + k >= NUM_PATTERNS) break;

            double raw_out = batch_out[k][0];
            double target = target_ptrs[b + k][0]; 
            
            int predicted = (raw_out > 0) ? 1 : -1;
            int actual = (target > 0) ? 1 : -1;
            
            if (predicted == actual) correct_count++;
        }
    }
    return correct_count;
}

double **train_out(double **Out, double **input, layer **layers, int nIn, int nOut, int npat, int L, int cycles)
{
    for (int i = 0; i < cycles; i++)
    {
        // 1. Shuffle at start of epoch
        shuffle_data(input, targets, NUM_PATTERNS);

        // 2. Loop through Mini-Batches
        for (int b = 0; b < NUM_PATTERNS; b += BATCH) 
        {
            layers[0]->setInput(input + b);
            FF(layers, L);
            BB(layers, L, Out + b); // Pass shifted targets
        }

        // --- NEW LOGIC: PRINT WRONG PATTERNS EVERY 100 CYCLES ---
        if ((i + 1) % 100 == 0) 
        {
            // We can pass the current shuffled pointers; accuracy check works regardless of order
            int correct = check_full_accuracy(layers, L, input, Out);
            int wrong = NUM_PATTERNS - correct;
            cout << "Cycle " << (i + 1) << " | Wrong Patterns: " << wrong << endl;
        }
    }
    return FF(layers, L);
}

void generate_Nbit_dataset()
{
    for (int i = 0; i < NUM_PATTERNS; i++)
    {
        int count_ones = 0;

        // Generate binary pattern and count ones
        for (int bit = 0; bit < NUM_INPUTS; bit++)
        {
            // Check if bit is set in pattern number i
            if (i & (1 << bit))
            {
                inputs[i][bit] = 1.0; // Logic 1
                count_ones++;
            }
            else
            {
                inputs[i][bit] = -1.0; // Logic 0
            }
        }

        // Set target: Odd parity = 1.0, Even parity = -1.0
        targets[i][0] = (count_ones % 2 == 1) ? 1.0 : -1.0;
    }
}

int main()
{
    srand(time(0));
    generate_Nbit_dataset();

    double *input_ptrs[NUM_PATTERNS];
    for (int i = 0; i < NUM_PATTERNS; ++i) input_ptrs[i] = inputs[i];

    double *output_ptrs[NUM_PATTERNS];
    for (int i = 0; i < NUM_PATTERNS; ++i) output_ptrs[i] = targets[i];

    layer l1(NUM_INPUTS, 30, BATCH, input_ptrs, ALPHA); 
    l1.init_Wb();
    layer l2(30,1 , BATCH, l1.get_out(), ALPHA); 
    l2.init_Wb();
    
    layer *layers[2];
    layers[0] = &l1;
    layers[1] = &l2;

    // Train
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Train for 6000 cycles
    train_out(output_ptrs, input_ptrs, layers, NUM_INPUTS, 1, NUM_PATTERNS, 2, 5000);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Training Time: " << elapsed_seconds.count() << " seconds" << std::endl;

    // Final Check
    std::cout << "---------------------------------------------------" << std::endl;
    int correct_count = check_full_accuracy(layers, 2, input_ptrs, output_ptrs);
    
     l1.init_Wb();
     l2.init_Wb();
    
    double percentage = (100.0 * correct_count / NUM_PATTERNS);
    std::cout << "Final Accuracy: " << correct_count << " / " << NUM_PATTERNS << std::endl;
    std::cout << "Percentage: " << percentage << "%" << std::endl;
    
    return 0;
}