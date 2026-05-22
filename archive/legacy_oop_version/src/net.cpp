#include "net.h"
//==================================================
// we assume the input is not a layer but output is
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
////////////////////////////////////////////////////////////////////
net::net(int mynL,trainSet* myts,float myalfa,float mybeta) :nL(mynL), pts(myts),alfa(myalfa),beta(mybeta)
{   Ls= new layer*[nL];
    nForLayers= new int[nL];
    nIn= (pts->nIn);              // number of inputs per pattern
    nForLayers[nL-1]=pts->nOut;     // number of outputs per pattern
}
void net::FF()
{
    for(int i=0;i<nL;i++)
    {
        Ls[i]->FF();
    }
}
void net::BP()
{
    for(int i=nL-1;i>=0;i--)
    {
        Ls[i]->BP();
    }
}
void net::update_Ls(int B_size)
{
    for(int i=0;i<nL;i++)
    {

        Ls[i]->update(B_size);
    }
}
////////////////////////////////////////////////////////////////////
void net::Creat()
{   Ls[0]= new layer(nIn,nForLayers[0],alfa,beta); // input to first layer
    for (int i=1; i<nL; i++)
    {   
        Ls[i]= new layer(nForLayers[i-1],nForLayers[i],alfa,beta);
        Ls[i-1]->pInB=Ls[i]->mOutB;
        Ls[i]->pInF=Ls[i-1]->mOutF;

    }
}
void net::clear()
{
    for(int i=0;i<nL;i++)
    {
        Ls[i]->clear_grads();
    }
}
void net::clear_mom()
{
    for(int i=0;i<nL;i++)
    {
        Ls[i]->clear_mom();
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
    cout<<"number of inputs"<< nIn<<endl;
    for (i=0; i<nL; i++) cout<<"number of neurons in layer "<<i<<"= " <<nForLayers[i]<<endl;
    for (i=0; i<nL; i++)
    {   cout <<"layer number "<<i<<endl;
        Ls[i]->print();
    }
    cout<< "====Network Information end======"<<endl;
}
void net::save(string filename)
{
    ofstream file(filename.c_str());
    
    if(!file.is_open())
    {
        cout << "Error: Could not open file " << filename << " for saving." << endl;
        return;
    }

    // Write number of layers as a header
    file << nL << endl;

    for(int i=0; i<nL; i++)
    {
        Ls[i]->save(file);
    }

    file.close();
    cout << "Network saved to " << filename << " successfully." << endl;
}

void net::load(string filename)
{
    ifstream file(filename.c_str());
    
    if(!file.is_open())
    {
        cout << "Error: Could not open file " << filename << " for loading." << endl;
        return;
    }

    int tempNL;
    file >> tempNL;

    if(tempNL != nL)
    {
        cout << "Error: Architecture mismatch. File has " << tempNL 
             << " layers, Net has " << nL << "." << endl;
        file.close();
        return;
    }

    for(int i=0; i<nL; i++)
    {
        Ls[i]->load(file);
    }

    file.close();
    cout << "Network loaded from " << filename << " successfully." << endl;
}
