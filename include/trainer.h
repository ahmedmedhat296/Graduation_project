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
    double * pa;       // pointer to the output of the last layer
    double * mda;  
    double * pI;
    int Batch_Size;    // de/do
    int mode;           // cont or done
    //double MaxError;    // maximum error
    double Loss;        // sum of all error
    //int errorCount;     // number of pattern in error

    // pointer for other variables to improve computation
    int nIn;           // pointer input,
    int nOut;          // pointer output,
    int nPat;          // pointer patterns
    double ** px, **py; // pointer to input and output

    trainer(net* theNet,trainSet* ts);
    void   update_Err(int pat);     // to update de/do and others
    void   printTs_out();
    int    train (int cycles,int Batch_size,string modelname,trainSet* test_set);
    double validate(net* test_net,trainSet* test_set);

};

#endif // TRAINER_H
