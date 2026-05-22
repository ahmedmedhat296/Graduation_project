#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <cstdlib>
#include <cmath>
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
    double alfa;
    double** w ;
    double** dw; //we can give up the two dimensional array with just a double variable
    double*  b;
    double*  db; //we can give up the two dimensional array with just a double varibale
    double** mOutF;  // output o 
    double** mOutB;  // de/do from second layer or from training set.
    double** pInF;   // pointer to input (forward)
    double** pInB;   // de/do from output or de/do from next to previous
    int* pnPat;      // pointer to number of patterns
    trainer * tr;    // trainer pointer
    layer(int myin, int myout, double myalfa,int* myPnPatern);
    void makeBefore(layer* L); //connect phantom before L
    void print();
    void printOut();
    void FF();
    void BP();
};

#endif // LAYER_H
