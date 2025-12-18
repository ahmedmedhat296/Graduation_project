#ifndef TRAINER_H
#define TRAINER_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "layer.h"
#include "trainSet.h"
#include "net.h"
using namespace std;
class trainSet;
class layer;
class net;

const int Cont=0;  // normal
const int Done=1;  // we are done

class trainer
{
public:
    trainSet* ts;       // pointer to training set.
    net* Net;           // the net
    double MaxError;    // maximum error
    double Loss;        // sum of all error
    int errorCount;     // number of pattern in error
    double ** pa;       // pointer to the output of the last layer
    double ** mda;      // de/do
    int mode;           // cont or done

    // pointer for other variables to improve computation
    int* pnIn;           // pointer input,
    int* pnOut;          // pointer output,
    int* pnPat;          // pointer patterns
    double ** px;        // pointer to input
    double ** py;        // pointer to output

    trainer(net* theNet,trainSet* ts);
    void   NFF ();       // network feed forward
    void   NBP ();       // network back propagation
    void   update();     // to update de/do and others
    void   printTs_out();
    int    train (int cycles);
};

#endif // TRAINER_H
