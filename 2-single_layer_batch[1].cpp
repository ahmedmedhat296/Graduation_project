#include <iostream>
#include <cmath>
#include <cstdlib>
#include<ctime>
#include <conio.h>
#include <math.h>
#include <iomanip>
using namespace std;
//===================================================
//batch learning
//===================================================
void printTS( double x[4][2],double y[4],double OutF [4])
{  int i;
   for (i=0; i<4; i++)
      cout<<setw(3)<< x[i][0]<<", "<<setw(3)<<x[i][1]
          <<"="<<setw(4)<< y[i]<<setw(10)<<OutF[i]<<endl;
   cout<<"_____________________________"<<endl;
}
///////////////////////////////////////////////////////////
 void printWeight(double  w[4],double  b)
 { int i;
   for(i=0; i<2; i++)     cout<<setprecision(3)<<w[i]<<", " ;
   cout<<setprecision(3)<<"    "<<b<<endl;
 }
///////////////////////////////////////////////////////////
int main()
{ int i,j,k,s;
  double z,dz;             // just temps
  const int nIn=2;         // input,
  const int nOut=1;        // output,
  const int nPat=4;        // patterns
  double  x[4][2],y[4];    // input ,output
  double  w[nIn],dw[nIn];  // weights and changes
  double  b,db;            // bias and change
  double  OutF [nPat];     // output
  double  dedo[nPat];      // output dedo
  int nCycles=500;         // training cycle
  double  alfa =0.2;       // learning rate
  srand(time(NULL));       // to start from a random seed
  //-------------------training set-------------------
  x[0][0]=-1; x[0][1]=-1;   y[0]=  -1;
  x[1][0]= 1; x[1][1]=-1;   y[1]=  -1;
  x[2][0]=-1; x[2][1]= 1;   y[2]=  -1;
  x[3][0]= 1; x[3][1]= 1;   y[3]=   1;
  //--------------- network initialization-------------
   for(j=0;j<nIn;j++)
       w[j]=((rand()%100)-50)/200.00;
   b=((rand()%100)-50)/200.00;
   printWeight(w,b); getche();
   //-----------------training-------------------------
   for(k=0; k<nCycles; k++)
   { // ------------------Initialization-------------
     db =0.0;
     for(s=0;s<nIn;s++) dw[s]= 0.0;
     //-------------- full batch learning ----------------
     for (i =0; i<nPat; i++)  // for all pattern in the set
      { // ------------------Feed Forward-------------
          z=b;
          for(j=0;j<nIn;j++) z+=w[j]*x[i][j];
          OutF[i]=tanh(z);
         //------------------------ error--------------------------
          dedo[i] = y[i]- OutF[i];
         //------------------------ Back propagation-------------
          dz=dedo[i]* (1-(OutF[i]*OutF[i]));
          db += dz;
          for(j=0;j<nIn;j++) dw[j]+= dz*x[i][j];
      }
      //----------------------------update-----------------------
          b+=db*alfa/nPat;
          for(j=0;j<nIn;j++) w[j]+=dw[j]*alfa/nPat;
        }
    printTS(x,y,OutF);
    printWeight(w,b);
  return 0;
}
