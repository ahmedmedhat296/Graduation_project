// GpuNet.cu — GPU-native neural network: cuBLAS matrix ops + CUDA kernels
// GTX 1650, Compute 7.5, CUDA 13.1

#include "GpuNet.h"
#include <cstring>
#include <ctime>
#include <cstdio>

// ============================================================
//  CUDA Kernels
// ============================================================

// LeakyReLU activation: f(x) = x > 0 ? x : 0.1x
__global__ void kLeakyReLU(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] > 0.0f ? x[i] : 0.1f * x[i];
}

// LeakyReLU gradient: d = 1 if act > 0, else 0.1
__global__ void kLeakyReLUGrad(float* grad, float* act, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= (act[i] > 0.0f ? 1.0f : 0.1f);
}

// Add bias vector to output
__global__ void kAddBias(float* out, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += b[i];
}

// Accumulate bias gradients: db += dout
__global__ void kAccumBiasGrad(float* db, float* dout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) db[i] += dout[i];
}

// Adam parameter update (fused kernel for efficiency)
__global__ void kAdamUpdate(float* W, float* dW, float* mW, float* vW,
                             int n, float lr, float beta1, float beta2,
                             float eps, int t, int batchSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = dW[i] / (float)batchSize;
        mW[i] = beta1 * mW[i] + (1.0f - beta1) * g;
        vW[i] = beta2 * vW[i] + (1.0f - beta2) * g * g;
        float mHat = mW[i] / (1.0f - powf(beta1, (float)t));
        float vHat = vW[i] / (1.0f - powf(beta2, (float)t));
        W[i] += lr * mHat / (sqrtf(vHat) + eps);
        dW[i] = 0.0f; // clear gradient
    }
}

// Numerically stable softmax (single-block, shared memory)
__global__ void kSoftmax(float* logits, float* probs, int n) {
    __shared__ float maxVal;
    __shared__ float sumExp;
    int i = threadIdx.x;

    // Find max (sequential in one thread for simplicity at n=64)
    if (i == 0) {
        maxVal = logits[0];
        for (int j = 1; j < n; j++)
            if (logits[j] > maxVal) maxVal = logits[j];
        sumExp = 0.0f;
        for (int j = 0; j < n; j++) {
            probs[j] = expf(logits[j] - maxVal);
            sumExp += probs[j];
        }
        for (int j = 0; j < n; j++) probs[j] /= sumExp;
    }
}

// Add Gaussian noise to logits (noise pre-generated on CPU, passed in)
__global__ void kAddNoise(float* logits, float* noise, float std, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) logits[i] += std * noise[i];
}

// ============================================================
//  GpuLayer implementation
// ============================================================
GpuLayer::GpuLayer(int in, int out, float learningRate)
    : nIn(in), nOut(out), lr(learningRate), adamt(0)
{
    size_t wSz = (size_t)nOut * nIn * sizeof(float);
    size_t bSz = (size_t)nOut * sizeof(float);
    size_t iSz = (size_t)nIn  * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_W,   wSz));
    CUDA_CHECK(cudaMalloc(&d_b,   bSz));
    CUDA_CHECK(cudaMalloc(&d_dW,  wSz));
    CUDA_CHECK(cudaMalloc(&d_db,  bSz));
    CUDA_CHECK(cudaMalloc(&d_mW,  wSz));
    CUDA_CHECK(cudaMalloc(&d_vW,  wSz));
    CUDA_CHECK(cudaMalloc(&d_mb,  bSz));
    CUDA_CHECK(cudaMalloc(&d_vb,  bSz));
    CUDA_CHECK(cudaMalloc(&d_out, bSz));
    CUDA_CHECK(cudaMalloc(&d_dIn, iSz));

    // Zero all buffers
    CUDA_CHECK(cudaMemset(d_W,   0, wSz));
    CUDA_CHECK(cudaMemset(d_b,   0, bSz));
    CUDA_CHECK(cudaMemset(d_dW,  0, wSz));
    CUDA_CHECK(cudaMemset(d_db,  0, bSz));
    CUDA_CHECK(cudaMemset(d_mW,  0, wSz));
    CUDA_CHECK(cudaMemset(d_vW,  0, wSz));
    CUDA_CHECK(cudaMemset(d_mb,  0, bSz));
    CUDA_CHECK(cudaMemset(d_vb,  0, bSz));
    CUDA_CHECK(cudaMemset(d_out, 0, bSz));
    CUDA_CHECK(cudaMemset(d_dIn, 0, iSz));

    // Xavier uniform init on CPU then copy to GPU
    float* hW = new float[nOut * nIn];
    float limit = sqrtf(6.0f / (nIn + nOut));
    for (int j = 0; j < nOut * nIn; j++)
        hW[j] = -limit + ((float)rand() / RAND_MAX) * 2.0f * limit;
    CUDA_CHECK(cudaMemcpy(d_W, hW, wSz, cudaMemcpyHostToDevice));
    delete[] hW;

    d_inAct = nullptr;
    d_dOut  = nullptr;
}

GpuLayer::~GpuLayer() {
    cudaFree(d_W);  cudaFree(d_b);
    cudaFree(d_dW); cudaFree(d_db);
    cudaFree(d_mW); cudaFree(d_vW);
    cudaFree(d_mb); cudaFree(d_vb);
    cudaFree(d_out); cudaFree(d_dIn);
}

void GpuLayer::FF(cublasHandle_t handle) {
    // d_out = W * d_inAct + b
    // cuBLAS is column-major; W is [nOut×nIn] row-major = [nIn×nOut] col-major
    float alpha = 1.0f, beta = 0.0f;
    // y = alpha * A * x + beta * y
    // A = d_W [nOut×nIn row-major], x = d_inAct [nIn], y = d_out [nOut]
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                             nIn, nOut,
                             &alpha, d_W, nIn,
                             d_inAct, 1,
                             &beta, d_out, 1));
    // Add bias
    int blocks = (nOut + 255) / 256;
    kAddBias<<<blocks, 256>>>(d_out, d_b, nOut);
    // LeakyReLU
    kLeakyReLU<<<blocks, 256>>>(d_out, nOut);
}

void GpuLayer::BP(cublasHandle_t handle) {
    // Apply activation gradient to d_dOut in-place
    int blocks = (nOut + 255) / 256;
    kLeakyReLUGrad<<<blocks, 256>>>(d_dOut, d_out, nOut);

    // Accumulate bias grad: d_db += d_dOut
    kAccumBiasGrad<<<blocks, 256>>>(d_db, d_dOut, nOut);

    // Accumulate weight grad: d_dW += d_dOut ⊗ d_inAct (outer product)
    float alpha = 1.0f;
    // dW [nOut×nIn row-major] += d_dOut[nOut] * d_inAct[nIn]^T
    // In cuBLAS col-major: A [nIn×nOut] += x[nIn] * y[nOut]^T → Sger
    CUBLAS_CHECK(cublasSger(handle,
                            nIn, nOut,
                            &alpha,
                            d_inAct, 1,
                            d_dOut, 1,
                            d_dW, nIn));

    // Propagate gradient: d_dIn = W^T * d_dOut
    float beta = 0.0f;
    // W row-major [nOut×nIn], treat as col-major [nIn×nOut]
    // d_dIn[nIn] = W[nIn×nOut col-maj] * d_dOut[nOut]  →  CUBLAS_OP_N
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                             nIn, nOut,
                             &alpha, d_W, nIn,
                             d_dOut, 1,
                             &beta, d_dIn, 1));
}

void GpuLayer::update(cublasHandle_t handle, int batchSize) {
    adamt++;
    int wTotal = nOut * nIn;
    // Weights
    int blocks = (wTotal + 255) / 256;
    kAdamUpdate<<<blocks, 256>>>(d_W, d_dW, d_mW, d_vW,
                                  wTotal, lr, 0.9f, 0.999f, 1e-8f, adamt, batchSize);
    // Biases
    blocks = (nOut + 255) / 256;
    kAdamUpdate<<<blocks, 256>>>(d_b, d_db, d_mb, d_vb,
                                  nOut, lr, 0.9f, 0.999f, 1e-8f, adamt, batchSize);
}

void GpuLayer::clearGrads() {
    CUDA_CHECK(cudaMemset(d_dW, 0, (size_t)nOut * nIn * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db, 0, (size_t)nOut * sizeof(float)));
}

void GpuLayer::save(ofstream& f) {
    int wTotal = nOut * nIn;
    float* hW = new float[wTotal];
    float* hb = new float[nOut];
    CUDA_CHECK(cudaMemcpy(hW, d_W, wTotal * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb, d_b, nOut * sizeof(float),   cudaMemcpyDeviceToHost));
    f.write((char*)hW, wTotal * sizeof(float));
    f.write((char*)hb, nOut * sizeof(float));
    delete[] hW; delete[] hb;
}

void GpuLayer::load(ifstream& f) {
    int wTotal = nOut * nIn;
    float* hW = new float[wTotal];
    float* hb = new float[nOut];
    f.read((char*)hW, wTotal * sizeof(float));
    f.read((char*)hb, nOut * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_W, hW, wTotal * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, hb, nOut * sizeof(float),   cudaMemcpyHostToDevice));
    delete[] hW; delete[] hb;
}

// ============================================================
//  GpuNet implementation
// ============================================================
GpuNet::GpuNet(int* topology, int nL, int inDim, float lr)
    : nLayers(nL), inputDim(inDim)
{
    CUBLAS_CHECK(cublasCreate(&cublas));

    Ls = new GpuLayer*[nLayers];
    Ls[0] = new GpuLayer(inputDim, topology[0], lr);
    for (int i = 1; i < nLayers; i++)
        Ls[i] = new GpuLayer(topology[i-1], topology[i], lr);

    // Wire activation/gradient pointers
    // Pinned host buffer for state
    CUDA_CHECK(cudaMallocHost(&h_input, inputDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, inputDim * sizeof(float)));

    Ls[0]->d_inAct = d_input;  // layer 0 reads from device input copy
    for (int i = 1; i < nLayers; i++) {
        Ls[i]->d_inAct   = Ls[i-1]->d_out;  // input = prev output
        Ls[i-1]->d_dOut  = Ls[i]->d_dIn;    // prev receives grad from current
    }
    // Last layer's d_dOut is set externally by GpuPPOAgent
}

GpuNet::~GpuNet() {
    for (int i = 0; i < nLayers; i++) delete Ls[i];
    delete[] Ls;
    cudaFreeHost(h_input);
    cudaFree(d_input);
    cublasDestroy(cublas);
}

void GpuNet::FF(float* hostState) {
    // Copy state to device (async via pinned memory)
    memcpy(h_input, hostState, inputDim * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, inputDim * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < nLayers; i++) Ls[i]->FF(cublas);
}

void GpuNet::BP() {
    for (int i = nLayers - 1; i >= 0; i--) Ls[i]->BP(cublas);
}

void GpuNet::update(int batchSize) {
    for (int i = 0; i < nLayers; i++) Ls[i]->update(cublas, batchSize);
}

void GpuNet::clearGrads() {
    for (int i = 0; i < nLayers; i++) Ls[i]->clearGrads();
}

void GpuNet::getOutput(float* hostOut, int n) {
    CUDA_CHECK(cudaMemcpy(hostOut, Ls[nLayers-1]->d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
}

void GpuNet::save(const string& path) {
    ofstream f(path, ios::binary);
    if (!f) { cout << "ERROR: Cannot save to " << path << endl; return; }
    for (int i = 0; i < nLayers; i++) Ls[i]->save(f);
    f.close();
    cout << "Saved: " << path << endl;
}

void GpuNet::load(const string& path) {
    ifstream f(path, ios::binary);
    if (!f) { cout << "WARNING: Cannot load " << path << " — using random weights." << endl; return; }
    for (int i = 0; i < nLayers; i++) Ls[i]->load(f);
    f.close();
    cout << "Loaded: " << path << endl;
}
