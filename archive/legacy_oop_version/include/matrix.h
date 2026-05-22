#ifndef MATRIX_H
#define MATRIX_H
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <iomanip>
#include<ctime>
class matrix
   {
    private:
    public:
    double **m;
    const int r; // row
    const int c;  //colum
    matrix (int colum);
    matrix (int row,int colum);
   ~matrix () ;
    void Emult (matrix* x,matrix* z);
    void mult(matrix* x,matrix* z);
    void add (matrix* x,matrix* z);
    void sub(matrix* x,matrix* z);
    void sub(matrix* x);
    void eqaul(matrix* x);
    void fil();
    void print(char * str);
    void print2mat(matrix* x);
    void square(matrix* x);
    void Esquare(matrix* x);
    void transpose(matrix* x);
    void filrandnum(int p) ;
    void filrandfrac() ;
    void filone();
    void sub(int p ,matrix* x);
    double sumofmat();
    double sumofrow (int num);
    double sumofcolum (int num);
    void Ediv(double d, matrix * x);
    void Emult(double d ,matrix * x);
    void geta(matrix * x, matrix * w,matrix * b);
    void getdldz2(matrix * y,matrix * a);
    matrix* operator =(matrix* x);

////////////
};
 double ** matD(int r, int c);
 double * matD(int r);
 double** matDP(int r);
 int    ** matI(int r, int c);
 int     * matI(int r);

#endif // MATRIX_H
