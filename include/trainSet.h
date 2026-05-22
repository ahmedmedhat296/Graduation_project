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
#include <string>
using namespace std;
class trainSet
{
public:
    int nIn;    // number of input,
    int nOut;   // number of output,
    int nPat;   // number of patterns
    double **x; // input
    double **y; // output

    trainSet(int _nIn, int _nOut, int _nPat); // create empty be filled
    void printTs();
    //void Create(); // to actually construct the TS.
    //void Fill_minst(const char *imFile = "TRIMG", const char *lbFile = "LABEL");
    //double **readLABEL(unsigned int &r, unsigned int &w, const char *name);
    //double **readIm(unsigned int &r, unsigned int &w, unsigned int &patternLen, unsigned int &nPat, const char *imFile);
    //unsigned int readI(ifstream *inDataFile);
    //void displayIm(double *P, double *L);
    //void shuffle(); // to fill x and y with all the patterns needed
    //void shiftPattern(double *src, double *dest, int dx, int dy, double background);
};

#endif // TRAINSET_H
