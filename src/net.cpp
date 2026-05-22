#include "net.h"
//==================================================
// we assume the input is not a layer but output is
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
////////////////////////////////////////////////////////////////////
net::net(int mynL,int _nIn, int* layers, int _concat_after, int _concat_size)
    : nL(mynL), concat_after_layer(_concat_after), concat_feature_size(_concat_size)
{
    concat_features = nullptr;
    concat_buffer = nullptr;
    Ls = new layer*[nL];
    nForLayers = new int[nL];
    nIn = _nIn;

    // Copy layer sizes
    for (int i = 0; i < nL; i++)
    {
        nForLayers[i] = layers[i];
    }

    // Create layers, widening the fusion layer's input
    Ls[0] = new layer(nIn, nForLayers[0]); // input to first layer
    for (int i = 1; i < nL; i++)
    {
        int prev_out = nForLayers[i-1];
        // If this layer comes right after the concat point, widen its input
        if (concat_after_layer >= 0 && i == concat_after_layer + 1)
            prev_out += concat_feature_size;
        Ls[i] = new layer(prev_out, nForLayers[i]);
        Ls[i]->pInF = Ls[i-1]->mOutF;
        Ls[i-1]->pInB = Ls[i]->mOutB;
    }

    // Allocate concat buffer and fix pointers for the fusion layer
    if (concat_after_layer >= 0 && concat_after_layer < nL - 1)
    {
        int buf_size = nForLayers[concat_after_layer] + concat_feature_size;
        concat_buffer = matD(buf_size);
        // The layer after the concat point reads from the merged buffer
        int next = concat_after_layer + 1;
        Ls[next]->pInF = concat_buffer;
        // Fix the backward pointer: the concat layer's pInB must point to
        // just the first part of mOutB of the next layer (handled in BP())
    }
}
void net::FF()
{
    for(int i=0;i<nL;i++)
    {
        Ls[i]->FF();

        // After the concat layer, build the merged input for the next layer
        if (concat_after_layer >= 0 && i == concat_after_layer && concat_features != nullptr)
        {
            int base = nForLayers[i];
            // Copy layer output into first part of concat_buffer
            for (int k = 0; k < base; k++)
                concat_buffer[k] = Ls[i]->mOutF[k];
            // Append extra features into second part
            for (int k = 0; k < concat_feature_size; k++)
                concat_buffer[base + k] = concat_features[k];
        }
    }
}
void net::BP()
{
    for(int i=nL-1;i>=0;i--)
    {
        Ls[i]->BP();

        // After backprop through the fusion layer, route gradients correctly
        if (concat_after_layer >= 0 && i == concat_after_layer + 1)
        {
            // Ls[i]->mOutB has size = nForLayers[concat_after_layer] + concat_feature_size
            // The first nForLayers[concat_after_layer] values are the gradient for the previous layer
            // The remaining concat_feature_size values are gradients for the side features (discarded)
            int base = nForLayers[concat_after_layer];
            for (int k = 0; k < base; k++)
                Ls[concat_after_layer]->pInB[k] = Ls[i]->mOutB[k];
        }
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
// Creat function removed as it is now integrated into the constructor
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
void net::change_mask(float p)
{
    for(int i=0;i<nL-1;i++)
    {
        Ls[i]->change_mask(p);
    }

}
void net::setdropout(bool d, float p)
{
    if(d==false)
    {change_mask(1);}
    if(d==true)
    {change_mask(p);}
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

void net::copy_weights_from(net* src)
{
    if(src->nL != this->nL) return;
    for(int i=0; i<nL; i++)
    {
        for(int j=0; j<Ls[i]->nOut; j++)
        {
            Ls[i]->b[j] = src->Ls[i]->b[j];
            for(int k=0; k<Ls[i]->nIn; k++)
            {
                Ls[i]->w[j][k] = src->Ls[i]->w[j][k];
            }
        }
    }
}

void net::add_gradients_from(net* src)
{
    if(src->nL != this->nL) return;
    for(int i=0; i<nL; i++)
    {
        for(int j=0; j<Ls[i]->nOut; j++)
        {
            Ls[i]->db[j] += src->Ls[i]->db[j];
            for(int k=0; k<Ls[i]->nIn; k++)
            {
                Ls[i]->dw[j][k] += src->Ls[i]->dw[j][k];
            }
        }
    }
}

void net::copy_mask_from(net* src)
{
    if(src->nL != this->nL) return;
    for(int i=0; i<nL; i++)
    {
        for(int j=0; j<Ls[i]->nOut; j++)
        {
            Ls[i]->mask[j] = src->Ls[i]->mask[j];
        }
    }
}

net::~net()
{
    delete[] concat_buffer;
    for (int i = 0; i < nL; i++)
        delete Ls[i];
    delete[] Ls;
    delete[] nForLayers;
}
