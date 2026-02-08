#include "layer.h"

layer:: layer(int myin, int myout, double myalfa,double mybeta )
    :nIn(myin), nOut(myout),alfa(myalfa),beta(mybeta)
{   int i, j;
    w     = matD(nOut,nIn);
    dw    = matD(nOut,nIn);
    pre_dw=matD(nOut,nIn);
    b     = matD(nOut);
    db    = matD(nOut);
    pre_db=matD(nOut);
    mOutF = matD(nOut);
    mOutB = matD(nIn);
   // mask= matI(nOut);
     double limit = sqrt(6.0 / (nIn + nOut));
    for(i=0; i<nOut; i++)
    {   for(j=0; j<nIn; j++)
           {
             w[i][j]=-limit+(((double)rand()/RAND_MAX)*2*limit);
             pre_dw[i][j]=0;
           }
        b[i]=0;
        pre_db[i]=0;
    }
}
////////////////////////////////////////////////////////////////////
void layer::BP ()
{   int i,j,k;
    double dz;
//-------------------initialization----------------------
    for(i=0; i<nIn; i++) mOutB[i]=0;
//-----------------------main loops----------------------
   for(j=0; j<nOut; j++)
        {   dz=pInB[j]*(1.0-(mOutF[j]*mOutF[j]));   //should be 1-o^2 
            db[j]+= dz; 
            for(i=0; i<nIn; i++)
            {   dw[j][i]+= dz*pInF[i];
                mOutB[i]+=w[j][i]*dz;
            }
        }
    
}
void layer::clear_mom()
{
    for(int i=0;i<nOut;i++)
    {
        for(int j=0;j<nIn;j++)
        {
            pre_dw[i][j]=0;
        }
        pre_db[i]=0;
    }
}
////////////////////////////////////////////////////////////////////
void layer::FF() // to get layer output
{   int i,j,k;
    double z;
     //each pattern
      for(j=0; j<nOut; j++)
        {   z=0.0;
            for(i=0; i<nIn; i++)
            {   z+=w[j][i]*pInF[i];
            }
            z+=b[j];
            mOutF[j]=tanh(z);
        }

}
void layer::clear_grads()
{
    for(int i=0;i<nOut;i++)
    {
        db[i]=0;
        for(int j=0;j<nIn;j++)
        {
            dw[i][j]=0;
        }
    }
}
//////////////////////////////////////////////////////////////////////
void layer::update(int B_size)
{
 double decay=1;
 double v_b,v_w;   //--------------------------update----------------------------
for(int j=0; j<nOut; j++)
{  
    v_b = beta * pre_db[j] + alfa * db[j]/B_size;
    b[j]*=decay;
    b[j] += v_b;
    pre_db[j] = v_b;
    
    // Correct weight update
    for(int i=0; i<nIn; i++) 
    {
        v_w = beta * pre_dw[j][i] + alfa * dw[j][i]/B_size ;
        w[j][i]*=decay;
        w[j][i] += v_w;
        pre_dw[j][i] = v_w;
    }
}
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
void layer::printOut()
{
   
        for(int j=0;j<nOut;j++)
        {
            cout<<mOutF[j]<<"  ";
        }
        cout<<endl;
    
}
void layer::save(ofstream& file)
{
    // 1. Write Layer Dimensions
    file << nIn << " " << nOut << endl;

    // 2. Write Weights (Row by Row)
    for(int j=0; j<nOut; j++)
    {
        for(int i=0; i<nIn; i++)
        {
            file << setprecision(15) << w[j][i] << " ";
        }
        file << endl;
    }

    // 3. Write Biases
    for(int j=0; j<nOut; j++)
    {
        file << setprecision(15) << b[j] << " ";
    }
    file << endl;
}

void layer::load(ifstream& file)
{
    int tempIn, tempOut;
    
    // 1. Read and Check Dimensions
    file >> tempIn >> tempOut;
    
    if(tempIn != nIn || tempOut != nOut)
    {
        cout << "ERROR: Layer Loading Mismatch!" << endl;
        cout << "File expects: " << tempIn << "->" << tempOut << endl;
        cout << "Layer is:     " << nIn << "->" << nOut << endl;
        return;
    }

    // 2. Read Weights
    for(int j=0; j<nOut; j++)
    {
        for(int i=0; i<nIn; i++)
        {
            file >> w[j][i];
            // Reset momentum when loading new weights
            pre_dw[j][i] = 0; 
        }
    }

    // 3. Read Biases
    for(int j=0; j<nOut; j++)
    {
        file >> b[j];
        // Reset momentum
        pre_db[j] = 0;
    }
}


