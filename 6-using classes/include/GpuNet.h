#ifndef GPUNET_H
#define GPUNET_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;

// ============================================================
//  Helpers: CUDA / cuBLAS error checking
// ============================================================
#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t st = (x); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %d at %s:%d\n", (int)st, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================
//  CUDA kernels (declared here, defined in GpuNet.cu)
// ============================================================
__global__ void kLeakyReLU(float* x, int n);
__global__ void kLeakyReLUGrad(float* grad, float* act, int n);
__global__ void kAdamUpdate(float* W, float* dW, float* mW, float* vW,
                             int n, float lr, float beta1, float beta2,
                             float eps, int t, int batchSize);
__global__ void kAddBias(float* out, float* b, int n);
__global__ void kAccumBiasGrad(float* db, float* dout, int n);
__global__ void kSoftmax(float* logits, float* probs, int n);
__global__ void kAddNoise(float* logits, float* noise, float std, int n);

// ============================================================
//  GpuLayer: one FC layer, all data in GPU memory
// ============================================================
class GpuLayer {
public:
    int nIn, nOut;
    float lr;

    // GPU buffers
    float* d_W;     // [nOut × nIn] row-major
    float* d_b;     // [nOut]
    float* d_dW;    // gradient accumulators
    float* d_db;
    float* d_mW;    // Adam first moment (weights)
    float* d_vW;    // Adam second moment (weights)
    float* d_mb;    // Adam first moment (bias)
    float* d_vb;    // Adam second moment (bias)
    float* d_out;   // [nOut] activations (output)
    float* d_dIn;   // [nIn] gradient flowing back

    // Pointers wired by GpuNet
    float* d_inAct;  // points to previous layer's d_out  (or d_hostIn for layer 0)
    float* d_dOut;   // points to next layer's d_dIn      (or user-injected gradient)

    int adamt;       // Adam time step

    GpuLayer(int in, int out, float learningRate);
    ~GpuLayer();

    void FF(cublasHandle_t handle);
    void BP(cublasHandle_t handle);
    void update(cublasHandle_t handle, int batchSize);
    void clearGrads();
    void save(ofstream& f);
    void load(ifstream& f);
};

// ============================================================
//  GpuNet: a stack of GpuLayers
// ============================================================
class GpuNet {
public:
    int nLayers;
    GpuLayer** Ls;
    cublasHandle_t cublas;

    // Pinned host buffer for state input (fast H2D transfer)
    float* h_input;
    float* d_input;  // device copy of state
    int    inputDim;

    GpuNet(int* topology, int nL, int inputDim, float lr);
    ~GpuNet();

    // Forward: copies state from host, runs FF through all layers
    void FF(float* hostState);

    // Backward: runs BP from last layer, uses d_dOut on last layer as seed
    void BP();

    // Weight update (Adam)
    void update(int batchSize);

    // Clear gradient accumulators
    void clearGrads();

    // Get output to host (last layer d_out → host buffer)
    void getOutput(float* hostOut, int n);

    // Save / load all weights
    void save(const string& path);
    void load(const string& path);

    // Expose last-layer gradient input for PPO gradient injection
    float* getGradIn() { return Ls[nLayers - 1]->d_dOut; }
    float* getOutput() { return Ls[nLayers - 1]->d_out; }
};

#endif // GPUNET_H
