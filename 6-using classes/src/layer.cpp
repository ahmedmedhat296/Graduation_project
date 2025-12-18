#include "layer.h"

layer:: layer(int myin, int myout, double myalfa,int* myPnPatern)
    :nIn(myin), nOut(myout),alfa(myalfa), pnPat(myPnPatern)
{   int i, j;
    w     = matD(nOut,nIn);
    dw    = matD(nOut,nIn);
    b     = matD(nOut);
    db    = matD(nOut);
    mOutF = matD(*pnPat,nOut);
    mOutB = matD(*pnPat,nIn);
    for(i=0; i<nOut; i++)
    {   for(j=0; j<nIn; j++)
            w[i][j]=((rand()%100)-50)/1000.00;
        b[i]=((rand()%100)-50)/1000.00;
    }
}
////////////////////////////////////////////////////////////////////
void layer::BP ()
{   int i,j,k;
    double dz;
//-------------------initialization----------------------
    for(k=0; k<*pnPat; k++) for(i=0; i<nIn; i++) mOutB[k][i]=0;
    for(i=0; i<nIn; i++)   for(j=0; j<nOut; j++) dw[j][i]=0;
    for(j=0; j<nOut; j++)   db[j]=0;

//-----------------------main loops----------------------
    for (k=0; k<(*pnPat); k++)//for each pattern
    {   for(j=0; j<nOut; j++)
        {   dz=pInB[k][j]*(1.2-(mOutF[k][j]*mOutF[k][j]));   //should be 1-o^2
            db[j]+= dz;
            for(i=0; i<nIn; i++)
            {   dw[j][i]+= dz*pInF[k][i];
                mOutB[k][i]+=w[j][i]*dz;
            }
        }
    }
//--------------------------update----------------------------
    for(j=0; j<nOut; j++)
    {   b[j]+=db[j]*alfa/(*pnPat);
        for(i=0; i<nIn; i++) w[j][i]+=dw[j][i]*alfa/(*pnPat);
    }
}
////////////////////////////////////////////////////////////////////
void layer::FF() // to get layer output
{   int i,j,k;
    double z,a;
    for (k=0; k<(*pnPat); k++)  //each pattern
    {   for(j=0; j<nOut; j++)
        {   z=0.0;
            for(i=0; i<nIn; i++)
            {   z+=w[j][i]*pInF[k][i];
            }
            z+=b[j];
            //should be  mOutF[k][j]=tanh(z);
            // but the following is better
            if (z>1)a=1;
            else if(z<-1)a=-1;
            else a=z;
            mOutF[k][j]=a;
        }
    }
}
//////////////////////////////////////////////////////////////////////
void layer::makeBefore(layer* L) //connect phantom before L
{   pInB=L->mOutB;
    L->pInF= mOutF;
}
//////////////////////////////////////////////////////////////////////
void layer::print()
{   int i,j;
    //int nPat=*tr->pnPat;
    cout << "No of Input    ="<< nIn  <<  endl;
    cout << "No of Output   ="<< nOut <<  endl;
    cout << "Alfa Value     ="<< alfa <<  endl;

    for(i=0; i<nOut  ; i++)
    {   cout<<"w["<<i+1<<"] = ";
        for(j=0; j<nIn; j++) cout<<w[i][j]<<" , ";
        cout<< "b["<<i+1<<"] = "<<b[i]<<endl;
    }
}
