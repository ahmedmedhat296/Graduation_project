#ifndef NET_H
#define NET_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <fstream> // NEW: Required for file I/O
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "layer.h"
#include "agent_data.h"
using namespace std;
class agent_data;
class trainer_trade;
class layer;
//==================================================
// we assume the input is not a layer but output is
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class net
{
public:
    int  nL;           // number of layers
    int nIn;         // pointer to number of inputs.
    int* nForLayers;   // number of neurons in each layer
    // float alfa;  // alpha and beta for each layer
    layer** Ls;
    // Late-fusion support
    int concat_after_layer;    // layer index after which to inject features (-1 = disabled)
    int concat_feature_size;   // number of extra features to concatenate
    double* concat_features;   // pointer to the external feature buffer (not owned)
    double* concat_buffer;     // internal buffer: [layer output | extra features]
    net(int mynL,int _nIn, int* layers, int _concat_after = -1, int _concat_size = 0);
    void FF();
    void BP();
    void update_Ls(int B_size);
    void clear();
    void clear_mom();
    layer* operator[](int i); // to return a pointer to a layer.
    void print();
    void copy_weights_from(net* src);
    void add_gradients_from(net* src);
    void copy_mask_from(net* src);
    void save(string filename);
    void load(string filename);
    void change_mask(float p);
    void setdropout(bool d=false, float p=1);
    ~net();
};
#endif // NET_H
