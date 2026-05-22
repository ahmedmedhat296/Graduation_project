#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream> 
#include <conio.h>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "trainSet.h"
#include "trainer.h"
using namespace std;
class trainSet;
class trainer;
class layer
{
    public:
    int nIn;         //number of inputs to this layer
    int nOut;        //number of outputs to this layer
    double alfa,beta;
    double** w ;
    double** dw; 
    double*  b;
    double*  db; 
    double* mOutF;  // output o 
    double* mOutB;  // de/do from second layer or from training set.
    double* pInF;   // pointer to input (forward)
    double* pInB;
    double* mask;
    double** m_w;    // 1st moment for weights (Mean)
    double** v_w;    // 2nd moment for weights (Variance)
    double* m_b;    // 1st moment for biases
    double* v_b;    // 2nd moment for biases
    int t; 
    bool isLinear; // true = linear activation (for output layers)
    double beta1;
    double beta2;
    double epsilon;
    trainer * tr;    // trainer pointer
    layer(int myin, int myout);
    void print();
    void printOut();
    void FF();
    void BP();
    void clear_grads();
    void clear_mom();
    void update(int B_size);
    void save(ofstream& file);
    void load(ifstream& file);
    void change_mask(float p);
    ~layer();

};


#endif // LAYER_H
