#include "trainSet.h"

trainSet::trainSet()
{
}
////////////////////////////////////////////////////////////////////////
void trainSet::Create()
{   x    = matD(nPat,nIn);       // full input data
    y    = matD(nPat,nOut);      // full output data
}
////////////////////////////////////////////////////////////////////////
void trainSet::XfillRand( int p) // to fill x with random variable
{   for(int j=0; j<nIn; j++)
        for(int n=0; n<nPat; n++)
        {   double  r = rand() % 100;
            if(r>p) r = 1;
            else    r = -1;
            x[n][j]   = r;
        }
}
////////////////////////////////////////////////////////////////////////
void trainSet::YfillParity () // to fill y with parity.
{   int i,j,pluss;
    for(j=0; j<nPat; j++)
    {   pluss=0;
        for(i=0; i<nIn; i++)
            if(x[j][i]==1)   pluss+=1;
        y[j][0]=-1;
        for(i=1; i<=nIn; i+=2)
            if  (pluss==i)  y[j][0]=1;
    }
}
////////////////////////////////////////////////////////////////
void  trainSet::printTs()
{   int i,j;
    cout<<"Ts="<<endl;
    for(i=0; i<nPat; i++)
    {   for(j=0; j<nIn; j++)
            cout<<setw(3)<<x[i][j]<<" ";
        cout<<"="<<setw(9);
        for(j=0; j<nOut; j++)   cout<<setw(3)<<y[i][j];
        cout<<endl;
    }
}
