#include "net.h"
//==================================================
// we assume the input is not a layer but output is
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
////////////////////////////////////////////////////////////////////
net::net(int mynL,trainSet* myts): nL(mynL), pts(myts)
{   Ls= new layer*[nL];
    nForLayers= new int[nL];
    pnIn= &(pts->nIn);              // number of inputs per pattern
    nForLayers[nL-1]=pts->nOut;     // number of outputs per pattern
    pnPatterns=&(pts->nPat);        // number of patterns in the training set
}
////////////////////////////////////////////////////////////////////
void net::Creat()
{   Ls[0]= new layer(*pnIn,nForLayers[0],alfa,pnPatterns); // input to first layer
    for (int i=1; i<nL; i++)
    {   Ls[i]= new layer(nForLayers[i-1],nForLayers[i],alfa,pnPatterns);
        Ls[i-1]->makeBefore(Ls[i]);
    }
}
//////////////////////////////////////////////////////////////
layer* net::operator [] (int i) // to return a pointer to layer.
{   return Ls[i];
}
//////////////////////////////////////////////////////////////
void net::print()
{   int i;

    cout<<endl<< "====Network Information start======"<<endl;
    cout<<"number of layers= "<<nL<< endl;
    cout<<"number of inputs"<< *pnIn<<endl;
    for (i=0; i<nL; i++) cout<<"number of neurons in layer "<<i<<"= " <<nForLayers[i]<<endl;
    for (i=0; i<nL; i++)
    {   cout <<"layer number "<<i<<endl;
        Ls[i]->print();
    }
    cout<< "====Network Information end======"<<endl;
}
