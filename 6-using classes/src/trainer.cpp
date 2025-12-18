#include "trainer.h"

trainer::trainer(net* theNet,trainSet* myts): ts(myts), Net(theNet)
{   pnIn  = &(ts->nIn);
    pnOut = &(ts->nOut);
    pnPat = &(ts->nPat);
    mda   =  matD(*pnPat,*pnOut);   // de/do
    px = (ts->x);
    py = (ts->y);
    Net->Ls[0]->pInF = px;
    // to connect trainer to the network and training set.
    Net->Ls[Net->nL-1]->pInB  = mda;      // connect input for last layer for BP
    pa = Net->Ls[Net->nL-1]->mOutF;
}
////////////////////////////////////////////////////////////////////////
void trainer::update()
{   int j,k;
    int nError;      // number of outputs with error
    double error=0;
    MaxError=0;
    errorCount=0;
    Loss=0;
    k=0;
    while(k<(*pnPat))    //for each location in the batch index table
    {   nError=0;
        for(j=0; j<*pnOut; j++) // for all outputs per pattern
        {   mda[k][j] = (py[k][j]-pa[k][j]);
            error = abs(mda[k][j]);
            Loss += error;
            if (error>MaxError) MaxError=error;
            nError += error>0.05;
        }
        k++;
        if (nError>0)    //  pattern still has error
        {   errorCount++;
        }
    }
//-----------------------------------------------------
    Loss/=(*pnPat);

    if(errorCount== 0)
    {   mode= Done;     // we are done
    }
    return;
}
////////////////////////////////////////////////////////////////
void  trainer::printTs_out()
{   int i,j;
    cout<<"Ts="<<endl;
    for(i=0; i<*pnPat; i++)
    {   for(j=0; j<*pnIn; j++)
            cout<<setw(3)<<px[i][j]<<" ";
        cout<<"=";
        for(j=0; j<*pnOut; j++)   cout<<setw(3)<<py[i][j];
        for(j=0; j<*pnOut; j++)   cout<<setprecision (3)<<setw(5)<<pa[i][j]<<" E=";
        for(j=0; j<*pnOut; j++)   cout<<setprecision (3)<<setw(5)<<abs(pa[i][j]-py[i][j]);
        cout<<endl;
    }
}
////////////////////////////////////////////////////////////////
void  trainer:: NFF ()
{   int i;
    // Net->nL : number of layers in the net
    for (i=0; i<Net->nL; i++ ) (*Net)[i]->FF();
}
////////////////////////////////////////////////////////////////
void  trainer:: NBP ()
{   int i;
    // Net->nL : number of layers in the net
    for (i=Net->nL-1; i>=0; i-- )  (*Net)[i]->BP();
}
////////////////////////////////////////////////////////////////
int   trainer:: train (int cycles)
{   int i;
    for (i=0; i<cycles; i++)
    {   NFF();
        update();
        if (i%50==0) cout<<"i="<<i<<"  errorCount= "<<errorCount<<endl;
        if (mode==Done)  break;      // we are done :}
        NBP();
    }
    return i;
}
