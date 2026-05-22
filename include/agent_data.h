#ifndef AGENT_DATA_H
#define AGENT_DATA_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "layer.h"
#include <string>

using namespace std;

class agent_data
{
public:
    int nIn;    // number of input,
    int nOut;   // number of output,
    int nPat;   // number of patterns
    double **x; // input
    double **y; // output

    agent_data(int _nIn, int _nOut, int _nPat); // create empty be filled
    void printTs();
};

#endif // AGENT_DATA_H
