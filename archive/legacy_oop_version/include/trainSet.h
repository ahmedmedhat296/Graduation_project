#ifndef TRAINSET_H
#define TRAINSET_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "layer.h"
using namespace std;
class trainSet
{
public:
    int nIn;            // number of input,
    int nOut;           // number of output,
    int nPat;           // number of patterns
    double ** x;        // input
    double ** y;        // output

    trainSet();         // create empty be filled
    void Create ();      // to actually construct the TS.
    void printTs();
    void XfillRand( int v);  // to fill x with random variable
    void YfillParity ();     // to fill y with parity.
};

#endif // TRAINSET_H
