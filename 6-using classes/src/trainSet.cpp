#include "trainSet.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iomanip>

using namespace std;

trainSet::trainSet()
{
}

////////////////////////////////////////////////////////////////////////
void trainSet::Create()
{   
    x = matD(nPat,nIn);       // full input data
    y = matD(nPat,nOut);      // full output data
}

////////////////////////////////////////////////////////////////////////
unsigned int readI(ifstream* inDataFile)
{   
    unsigned char a,b,c,d;
    unsigned int r=0;
    inDataFile->read((char*)(&a), sizeof(char));
    inDataFile->read((char*)(&b), sizeof(char));
    inDataFile->read((char*)(&c), sizeof(char));
    inDataFile->read((char*)(&d), sizeof(char));
    r=d+256*c+65536*b+16777216*a;
    return r;
}

/////////////////////////////readIm/////////////////////////////////////
double ** readIm(unsigned int& r, unsigned int& w,unsigned int& patternLen,
                 unsigned int& nPat, const char* imFile)
{   
    unsigned int mn;
    unsigned char* p;

    ifstream imF;
    imF.open(imFile, ios::binary|ios::in);
    
    if (!imF)
    {   
        cout << "Unable to open file: " << imFile << endl;
        exit(1);   // call system to stop
    }

    mn = readI(&imF);
    nPat = readI(&imF);
    r = readI(&imF);
    w = readI(&imF);
    patternLen = r * w;
    
    // Allocate raw buffer
    p = (unsigned char *) malloc(patternLen * nPat);
    imF.read((char*)(p), patternLen * nPat);
    imF.close();
    
    cout << "Loaded Images: " << imFile << " | Count: " << nPat << endl;

    //====================all data flat but double======================
    double * pat = (double *) malloc(sizeof(double) * patternLen * nPat);
    unsigned int i;
    for (i = 0; i < patternLen * nPat; i++)
    {
        // Normalize 0..255 to -1.0..1.0
        pat[i] = (p[i] / 127.5) - 1.0;
    }
        
    //==================== as a two dimensional array====================
    double ** patMat = (double**) malloc(sizeof(double*) * nPat);
    for (i = 0; i < nPat; i++)
    {
        // Point each row to the correct place in the flat array
        patMat[i] = pat + (i * patternLen);
    }

    free(p);
    // never do "delete [] pat;" it is the only place with pattern data
    return patMat;
}

////////////////////////////////readLABEL/////////////////////////////////
double ** readLABEL(unsigned int& r, unsigned int& w, const char* name)
{   
    unsigned int mn, nPat = 0;
    unsigned char* imL;

    ifstream labF;
    labF.open(name, ios::binary|ios::in);
    
    if (!labF)
    {   
        cout << "Unable to open file: " << name << endl;
        exit(1);   // call system to stop
    }
    
    mn = readI(&labF);
    nPat = readI(&labF);
    imL = (unsigned char*) malloc(nPat);
    labF.read((char*)(imL), sizeof(char) * nPat);
    labF.close();

    cout << "Loaded Labels: " << name << " | Count: " << nPat << endl;

    //====================all data flat but double======================
    double * label = (double *) malloc(sizeof(double) * 10 * nPat);
    
    unsigned int i, j;
    for (i = 0; i < nPat; i++)
    {
        for (j = 0; j < 10; j++)
        {   
            // One-Hot Encoding: 1.0 for correct class, -1.0 for others
            if (j == imL[i]) 
                label[10 * i + j] = 1.0;
            else 
                label[10 * i + j] = -1.0;
        }
    }

    //==================== as a two dimensional array====================
    double **labMat = (double**) malloc(sizeof(double*) * nPat);
    for (i = 0; i < nPat; i++)
    {
        labMat[i] = label + (10 * i);
    }

    free(imL);
    // never do "delete [] label;" it is the only place with pattern data

    return labMat;
}

////////////////////////////////////////////////////////////////////////
void trainSet::Fill_minst(const char* imFile, const char* lbFile) 
{   
    unsigned int r, w;
    unsigned int patternLen = 0, nPatLocal = 0;
    
    // Pass the arguments directly to the reading functions
    x = readIm(r, w, patternLen, nPatLocal, imFile);
    cout << "<><><><><><><><><><><><>><><><>><><>" << endl;
    
    y = readLABEL(r, w, lbFile);
    cout << "<><><><><><><><><><><><>><><><>><><>" << endl;
    
    // IMPORTANT: Update class member so other functions know the size
    this->nPat = nPatLocal; 
}

////////////////////////////////////////////////////////////////
void trainSet::printTs()
{   
    int i, j;
    cout << "Ts=" << nPat << endl;
    for(i = 0; i < nPat; i++)
    {   
        for(j = 0; j < nIn; j++)
            cout << x[i][j] << " ";
        cout << "=" << endl;
        
        for(j = 0; j < nOut; j++)
            cout << y[i][j] << " ";
        cout << endl;
    }
}

////////////////////////////////////////////////////////////////
void trainSet::shuffle()
{
    int k, r;
    double* tempPtr;
    for (k = nPat - 1; k > 0; k--)
    {
        r = rand() % (k + 1); 
        tempPtr = x[k];
        x[k] = x[r];
        x[r] = tempPtr;
        
        tempPtr = y[k];
        y[k] = y[r];
        y[r] = tempPtr;
    }
}
