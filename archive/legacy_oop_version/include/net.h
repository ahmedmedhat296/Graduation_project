#ifndef NET_H
#define NET_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "layer.h"
#include "trainSet.h"
using namespace std;
class trainSet;
class trainer;
class layer;
//==================================================
// we assume the input is not a layer but output is
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class net
{
public:
    int  nL;           // number of layers
    int* pnIn;         // pointer to number of inputs.
    int* nForLayers;   // number of neurons in each layer
    int* pnPatterns;   // pointer yo number of patterns
    float alfa;
    trainSet* pts;
    layer** Ls;

    net(int mynL,trainSet* myts);
    layer* operator[](int i); // to return a pointer to a layer.
    void Creat();
    void print();
};
#endif // NET_H
