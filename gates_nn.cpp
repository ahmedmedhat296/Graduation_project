#include <iostream>
#include <math.h>   // for tanh
using namespace std;
float sigmoid( float X){return 1.0/(1.0+exp(-X));}
const int training_set_size = 8;
const int input_size = 4; // 3 inputs + bias
const float alpha = 0.4;

int main() {
    const int training_set_size = 8;
const int input_size = 4; // 3 inputs + bias

int input[training_set_size][input_size] = {
    {0, 0, 0, 1},
    {0, 0, 1, 1},
    {0, 1, 0, 1},
    {0, 1, 1, 1},
    {1, 0, 0, 1},
    {1, 0, 1, 1},
    {1, 1, 0, 1},
    {1, 1, 1, 1}
};

int and_output[] = {0, 0, 0, 0, 0, 0, 0, 1};
int or_output[]  = {0, 1, 1, 1, 1, 1, 1, 1};
int xor_output[] = {0, 1, 1, 0, 1, 0, 0, 1};


    // Initial random weights (unlearned)
    float W_and[] = {1.2, -1.6, -1.3,  -0.9}; // w1, w2, w3, bias weight
    float W_or[]={-1.8,1.3,1.5,-1.9};
    float W_xor[]={-1.2,1.3,-1.4,-1.8};

    cout << "Testing untrained perceptron:\n";
    
    for(int g=0; g<3; ++g) {
        float *const W =(g==0? W_and:g==1?W_or:W_xor);
        cout << "Weights: ";
        for(int i=0;i<input_size;i++) cout << W[i] << " ";
        cout << "\n\n";
        cout << (g==0 ? "AND Gate" : g==1 ? "OR Gate" : "XOR Gate") << " results:\n";
        for(int i=0; i<training_set_size; ++i) {
            // Forward pass (weighted sum)
            float sum = 0;
            for(int j=0; j<input_size; ++j)
                sum += input[i][j] * W[j];
            int y_pred = tanh(sum)>0 ? 1:0 ;
            int y_true = (g==0 ? and_output[i] : g==1 ? or_output[i] : xor_output[i]);
            cout << "Input (" 
                 << input[i][0] << "," << input[i][1] << "," << input[i][2] 
                 << ") -> Pred: " << y_pred << "  |  Target: " << y_true << "\n";
        }
        cout << "\n";
    }
    // the learning process
    cout<<"Training perceptron...\n";
    for(int g=0;g<3;g++)
    {
        float *const W =(g==0? W_and:g==1?W_or:W_xor);
        int *const output =(g==0? and_output:g==1?or_output:xor_output);
        float dW[input_size];
        float O;
        for(int epoch=0;epoch<10000;epoch++)
        {
            bool no_error=true;
            for(int k=0;k<input_size;k++)
            { 
               dW[k] =0;
            }
        for(int i=0;i<training_set_size;i++)
        {
            float Z=0;
            for(int j=0;j<input_size;j++)
            {
                Z+=input[i][j]*W[j];
            }
            float O_train =(tanh(Z)+1)/2;  // U1se tanh output
            no_error &= (O_train-output[i]>0.02 ? false:true);

            for(int k=0;k<input_size;k++)
            { 
               dW[k] += (output[i]-O_train) * ((1-O_train*O_train)/2) * input[i][k];
            }
        }
        for(int k=0;k<4;k++)
        W[k]=W[k]+alpha*dW[k];
        if(no_error){cout<<"true after" << epoch;break;}
    }
}
    cout << "Testing trained perceptron:\n";
    
    for(int g=0; g<3; ++g) {
        float *const W =(g==0? W_and:g==1?W_or:W_xor);
        cout << "Weights: ";
        for(int i=0;i<input_size;i++) cout << W[i] << " ";
        cout << "\n\n";
        cout << (g==0 ? "AND Gate" : g==1 ? "OR Gate" : "XOR Gate") << " results:\n";
        for(int i=0; i<training_set_size; ++i) {
            // Forward pass (weighted sum)    
            float sum = 0;
            for(int j=0; j<input_size; ++j)
                sum += input[i][j] * W[j];

            // Activation function (step function)
            int y_pred = tanh(sum)>0 ? 1:0 ;

            int y_true = (g==0 ? and_output[i] : g==1 ? or_output[i] : xor_output[i]);

            cout << "Input (" 
                 << input[i][0] << "," << input[i][1] << "," << input[i][2] 
                 << ") -> Pred: " << y_pred << "  |  Target: " << y_true << "\n";
        }
        cout << "\n";
    }
    return 0;
}
