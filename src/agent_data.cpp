#include "agent_data.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iomanip>
using namespace std;

agent_data::agent_data(int _nIn, int _nOut, int _nPat)
{
    nIn=_nIn;
    nOut=_nOut;
    nPat=_nPat;
    x=matD(_nPat,_nIn);
    y=matD(_nPat,_nOut);
}

void agent_data::printTs()
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
