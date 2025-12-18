#include <iostream>
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <iomanip>
#include<ctime>
#include "matrix.h"
#include "trainSet.h"
#include "layer.h"
#include "net.h"
#include "trainer.h"
using namespace std;
//===========================================================
// Using Classes
//===========================================================
//==================================================
// we assume the input is not a layer but output is
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
////////////////////////////////////////////////////////////////
int main()
{
    int i=0;
    int nLayer=2;   // number of layers
    srand(time(NULL));
    trainSet* TS = new trainSet();
    TS->nIn=8;         // number of inputs per pattern
    TS->nOut=1;        // number of output per pattern
    TS->nPat=2000;     // number of patterns in the training set
    TS->Create();      // to create empty training set
    TS->XfillRand(50); // fill x with random numbers.
    TS->YfillParity(); // fill y
    //TS->printTs(); getch();

    net* N= new net(nLayer,TS);
    N->alfa =.5;           // learning rate
    N->nForLayers[0]= 50;   // number of neurons in hidden layer
    N->Creat();
    N->print();
    trainer*  tr = new trainer(N, TS);
    i=tr->train(30000);
    //tr->printTs_out();
    cout<<endl<<i<<", "<<tr->Loss<<", "<< tr->MaxError<<", "<< tr->errorCount<<endl;

    return 0;
}

