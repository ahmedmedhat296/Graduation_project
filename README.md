# Graduation Project — Neural Network to PPO Trading Agent

> **From a single-layer perceptron to a GPU-accelerated Reinforcement Learning trading agent for the Egyptian Stock Exchange.**

This repository documents the full evolution of a C++ neural network project — starting from the most basic building blocks and growing into a production-grade, CUDA-accelerated PPO (Proximal Policy Optimization) agent that autonomously manages a stock portfolio on the EGX70 index.

---

## 📂 Repository Structure

```
Graduation_project/
│
├── 1-single_layer[1].cpp                     ← Stage 1: Single neuron, manual math
├── 2-single_layer_batch[1].cpp               ← Stage 2: Batch learning, weight updates
├── 3-single_layer_batch_4inputs_3outputs.cpp ← Stage 3: Multi-input/output layer
├── gates_nn.cpp                              ← Stage 4: Logic gate learning (AND/OR/XOR)
├── xor_multilayer.cpp                        ← Stage 5: Multi-layer backprop for XOR
│
└── 6-using classes/                          ← Stage 6: Full PPO GPU Trading Agent ★
    ├── README.md                             ← Detailed documentation for Stage 6
    ├── ppo_main.cpp                          ← Main: train/test entry point
    ├── build_gpu.bat                         ← NVCC build script (CUDA + cuBLAS)
    ├── fix_data.py                           ← Market data anomaly fixer
    ├── include/                              ← Header files
    │   ├── MarketData.h                      ← Multi-stock CSV loader & alignment
    │   ├── TradingEnv.h                      ← Gym-style trading environment
    │   ├── GpuNet.h                          ← GPU neural network (cuBLAS)
    │   ├── GpuPPOAgent.h                     ← GPU PPO actor-critic agent
    │   ├── PPOAgent.h                        ← CPU PPO agent (legacy)
    │   ├── layer.h / net.h / matrix.h        ← Original NN framework headers
    │   └── trainSet.h / trainer.h
    └── src/                                  ← Source implementations
        ├── TradingEnv.cpp                    ← Portfolio rebalancing + reward shaping
        ├── GpuNet.cu                         ← CUDA kernels + cuBLAS matrix ops
        ├── GpuPPOAgent.cu                    ← PPO: GAE, clip, entropy, temperature
        ├── PPOAgent.cpp                      ← CPU PPO (legacy reference)
        ├── layer.cpp / net.cpp / matrix.cpp  ← Original NN framework
        └── trainSet.cpp / trainer.cpp
```

---

## 🧠 Project Evolution — Stage by Stage

### Stage 1–3: Building the Perceptron from Scratch
Starting with raw math: a single neuron computing `output = Σ(wᵢ × xᵢ) + b`, then adding batch training, and extending to multiple inputs and outputs. No libraries — pure C++ to understand the fundamentals.

### Stage 4–5: Logic Gates and Backpropagation
Trained the network to learn logic gates (AND, OR, NAND) and then tackled XOR — the classic problem that requires a hidden layer. Implemented full backpropagation with chain rule derivatives manually.

### Stage 6: GPU-Accelerated PPO Trading Agent ★
The major leap. The original network framework was extended and then a completely new GPU-native system was built on top of it:

| What Changed | Detail |
|---|---|
| **Task** | From MNIST/logic gates → Stock portfolio allocation |
| **Algorithm** | From supervised backprop → PPO (Reinforcement Learning) |
| **Hardware** | From CPU `double` → GPU `float` with CUDA + cuBLAS |
| **Data** | From static datasets → 6 years of live EGX market data |
| **Network** | From 2-layer CPU net → 4-layer GPU net (1024→512→256→output) |
| **Output** | From class labels → Portfolio weights (softmax distribution) |

**See [`6-using classes/README.md`](6-using%20classes/README.md) for the full technical breakdown.**

---

## 🚀 The PPO Trading Agent (Stage 6)

The agent learns to allocate a $1,000,000 simulated portfolio across **70 EGX-listed stocks + cash** by:

1. **Observing** 120 days of normalized price history for all 70 stocks + current portfolio weights (~7,271 input features)
2. **Deciding** target allocation weights via a 4-layer GPU neural network (Actor)
3. **Receiving a reward** based on portfolio return, penalized for over-concentration (HHI) and rewarded for diversification (entropy bonus)
4. **Updating** using PPO-Clip with Generalized Advantage Estimation (GAE)

**Test mode** benchmarks the agent against the EGX30 index and reports daily **alpha** (excess return).

```bat
# Build
.\6-using classes\build_gpu.bat

# Train (GTX 1650+ required)
ppo_gpu.exe --train

# Test against EGX30
ppo_gpu.exe --test
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| C++17 | Core language |
| CUDA (Compute 7.5) | GPU kernel execution |
| cuBLAS | GPU matrix multiply (forward/backward pass) |
| Adam Optimizer | Fused GPU weight update kernel |
| PPO (RL) | Policy optimization algorithm |
| Python + pandas | Data preparation pipeline |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
