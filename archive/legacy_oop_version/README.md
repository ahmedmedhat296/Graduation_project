# PPO Portfolio Allocation Agent — GPU Accelerated
### Egyptian Stock Exchange (EGX70) | CUDA + cuBLAS | C++17

> **Graduation Project** — Evolution from a simple single-layer neural network to a full GPU-accelerated Proximal Policy Optimization (PPO) agent for autonomous stock portfolio management on the Egyptian market.

---

## 📌 What Was Built

This folder contains the complete source code for a **reinforcement learning trading agent** that:

- Loads 6 years of daily OHLCV data for **70 EGX stocks**
- Trains a **PPO actor-critic agent** using a GPU-native neural network
- Outputs **portfolio allocation weights** (how much of the portfolio to assign to each stock + cash)
- Tests against the **EGX30 benchmark** and tracks **alpha** (excess return)
- Logs all **buy/sell transactions** with prices, shares, and trade values

---

## 🏗️ Project Architecture

```
6-using classes/
│
├── ppo_main.cpp              ← Main entry point (--train / --test modes)
├── build_gpu.bat             ← NVCC build script (GTX 1650, sm_75, cuBLAS)
├── fix_data.py               ← Data anomaly fixer (price spike cleaner)
│
├── include/
│   ├── MarketData.h          ← CSV loader + multi-stock alignment engine
│   ├── TradingEnv.h          ← Gym-style trading environment header
│   ├── GpuNet.h              ← GPU neural network layer + net header
│   ├── GpuPPOAgent.h         ← GPU PPO actor-critic agent header
│   └── PPOAgent.h            ← CPU PPO agent header (legacy reference)
│
└── src/
    ├── TradingEnv.cpp        ← Portfolio rebalancing, reward shaping, transaction log
    ├── GpuNet.cu             ← CUDA kernels + cuBLAS matrix ops
    ├── GpuPPOAgent.cu        ← Full PPO: GAE, PPO-clip, entropy, temperature softmax
    └── PPOAgent.cpp          ← CPU PPO agent (legacy, before GPU migration)
```

---

## 🔧 What Each File Does

### `ppo_main.cpp` — Main Controller
**What was added:** Full training loop with ESC-to-stop, best-model saving based on a running average return (not a single lucky episode), and a complete test mode with EGX30 benchmark comparison.

**Key features:**
- `--train` mode: runs up to 5,000 episodes, saves checkpoints every 50 episodes, saves the best model based on a smoothed 95/5 exponential moving average of returns
- `--test` mode: loads the best model, runs on the most recent market data window, outputs `test_results.csv` (daily portfolio values, agent return %, EGX30 %, alpha) and `transaction_report.csv` (every buy/sell with ticker, shares, price, value)
- Console prints top-3 holdings per episode with their weight %

---

### `build_gpu.bat` — NVCC Build Script
**What was added:** A single-command build script for the CUDA project.

Compiles all `.cpp` and `.cu` files with:
- `nvcc` targeting **CUDA Compute 7.5** (GTX 1650)
- `-O3 / /O2` optimization flags
- Links `cublas` for GPU matrix operations
- MSVC toolchain via `vcvars64.bat`

```bat
.\build_gpu.bat
ppo_gpu.exe --train     # Start/resume training
ppo_gpu.exe --test      # Evaluate on latest data
```

---

### `include/MarketData.h` — Market Data Engine
**What was added:** A complete multi-stock CSV loading and alignment system.

**Key design decisions:**
- Loads all 70 stock CSVs from a folder
- Finds **common trading dates** (dates present in ≥80% of stocks) — handles delisted/new stocks gracefully
- Applies **forward-fill** for missing dates (holidays, suspensions)
- **Sanitizes data**: clamps daily returns to ±20% (EGX daily circuit-breaker limit) to remove corrupted split-adjusted prices
- Computes `stateDim = numStocks × LOOKBACK + actionDim` (price history + current weights)

**Constants:**
```cpp
LOOKBACK    = 120  // ~6 months of trading days
EPISODE_LEN = 200  // trading days per training episode
WINDOW_NEED = 320  // minimum data required
```

---

### `include/TradingEnv.h` + `src/TradingEnv.cpp` — Trading Environment
**What was added:** A full OpenAI Gym-style environment for portfolio trading.

**State space:** `[numStocks × LOOKBACK normalized returns] + [portfolio weights]`
Each price feature is normalized as: `(pastPrice - currentPrice) / currentPrice`

**Action space:** Portfolio target weights (one per stock + 1 for cash), softmax-normalized to sum to 1

**Reward shaping (multiple improvements applied):**

| Component | Formula | Purpose |
|-----------|---------|---------|
| Log return | `log(newPV / oldPV)`, clamped ±0.1 | Main signal |
| HHI penalty | `-0.05 × Σ(wᵢ²)` | Penalizes over-concentration |
| Clip penalty | `-0.02 × n_capped` | Agent feels the 40% per-stock cap |
| Entropy bonus | `+0.005 × H(w) / H_max` | Rewards spreading across stocks |

**Transaction cost:** 0.15% per trade (realistic EGX brokerage fee)

**Transaction logging:** Every buy/sell is recorded with `{day, stockIdx, isBuy, shares, price, value}` — exported to `transaction_report.csv` in test mode

---

### `include/GpuNet.h` + `src/GpuNet.cu` — GPU Neural Network
**What was added:** A fully GPU-native neural network using **cuBLAS for matrix multiply** and custom **CUDA kernels** for activations and optimizers.

**Architecture:** `inputDim → [hidden layers] → outputDim`, all weights live on the GPU.

**CUDA Kernels implemented:**
```
kLeakyReLU       — f(x) = x > 0 ? x : 0.1x
kLeakyReLUGrad   — Backprop gradient gate
kAdamUpdate      — Fused Adam optimizer (β₁=0.9, β₂=0.999, ε=1e-8)
kAddBias         — GPU bias addition
kAccumBiasGrad   — Gradient accumulation for biases
kSoftmax         — Numerically stable softmax (shared-memory single-block)
kAddNoise        — Gaussian noise injection for exploration
```

**Weight init:** Xavier uniform: `limit = sqrt(6 / (nIn + nOut))`

**Memory layout:** Weights stored row-major on GPU; cuBLAS called with `CUBLAS_OP_T` to handle row/column-major transposition correctly.

---

### `include/GpuPPOAgent.h` + `src/GpuPPOAgent.cu` — GPU PPO Agent
**What was added:** The core reinforcement learning algorithm — Proximal Policy Optimization — running actor and critic entirely on the GPU.

**Network topology:**
```
Actor:  stateDim → 1024 → 512 → 256 → actionDim
Critic: stateDim → 1024 → 512 → 256 → 1
```

**PPO hyperparameters:**
| Param | Value | Reason |
|-------|-------|--------|
| γ (discount) | 0.99 | Long-horizon trading |
| λ (GAE) | 0.95 | Bias-variance tradeoff |
| ε (clip) | 0.2 | Standard PPO |
| Entropy coeff | 0.3 | **Raised from 0.05** — must compete with policy gradient to prevent portfolio collapse |
| Noise std | 0.5 | Exploration during training |
| PPO epochs | 4 | Updates per trajectory |

**Key improvements over naive PPO:**

**1. Temperature Softmax (`SOFTMAX_TEMPERATURE = 2.0`)**
Without temperature, one logit even slightly higher captures ~99% of portfolio weight ("winner-takes-all collapse"). Temperature divides logits before softmax, smoothing the distribution.

**2. High Entropy Coefficient (0.3)**
Entropy regularization is what forces the agent to spread portfolio weight. At the original 0.05, the PPO policy gradient overpowers entropy and the agent converges to holding 1-2 stocks. At 0.3, entropy genuinely competes.

**3. GAE Advantage Normalization**
Advantages are normalized to zero mean and unit variance before the update — stabilizes learning across very different episode returns.

**4. Running Average Best Model**
Best model is saved based on a smoothed EMA (`0.95 × prev + 0.05 × current`) instead of raw episode return — prevents saving on lucky outlier episodes.

---

### `include/PPOAgent.h` + `src/PPOAgent.cpp` — CPU PPO Agent (Legacy)
**What this is:** The original CPU-only PPO agent, kept as a reference implementation. Uses the existing `net` / `layer` / `matrix` framework (the original graduation project codebase).

**Why it's kept:** Shows the evolution from CPU to GPU. Same PPO algorithm, but runs on the CPU using `double` precision instead of GPU `float`.

---

### `fix_data.py` — Data Anomaly Fixer
**What was added:** A post-download Python script that scans all 70 stock CSVs for price anomalies and forward-fills suspicious zero/negative prices.

---

## 📊 State Space Design

The input to the neural network for each step is:

```
State = [normalized_price_history | current_portfolio_weights]
         └─ numStocks × LOOKBACK ─┘  └──── actionDim ────┘
              (7,200 features)          (71 features)
         Total: ~7,271 floats per state
```

Price features are computed as: `(price_t-d - price_now) / price_now` for d = 0..119

This gives the network a **relative price history** — invariant to absolute price levels (handles EGP vs USD, stock splits, etc.).

---

## 🚀 How to Build and Run

### Prerequisites
- NVIDIA GPU with CUDA Compute ≥ 7.5 (GTX 1650 or better)
- CUDA Toolkit 11+ with cuBLAS
- MSVC 2022 (Visual Studio)
- Python 3 + pandas (for data preparation)

### Build
```bat
.\build_gpu.bat
```

### Prepare Data
The data folder `InvestingCom_EGX70_6Years_Aligned/` must contain 70 CSV files named `TICKER.CA.csv` and an `EGX30_Benchmark.csv` file.

### Train
```bat
ppo_gpu.exe --train
```
- Press **ESC** to stop and save a checkpoint
- Checkpoint saved every 50 episodes to `gpu_actor_ckpt.bin` / `gpu_critic_ckpt.bin`
- Best model saved to `gpu_actor_best.bin` / `gpu_critic_best.bin`

### Test
```bat
ppo_gpu.exe --test
```
Outputs:
- `test_results.csv` — daily portfolio value, agent return %, EGX30 return %, alpha
- `transaction_report.csv` — all buy/sell trades with ticker, shares, price, value

---

## 📈 What Was Improved Over the Base Neural Network

| Area | Original (simple NN) | This Project |
|------|---------------------|-------------|
| **Task** | MNIST digit classification | Portfolio allocation on EGX70 |
| **Algorithm** | Backprop (supervised) | PPO (reinforcement learning) |
| **Hardware** | CPU only | GPU (cuBLAS + CUDA kernels) |
| **Network** | 2-3 layers, CPU double | 4 layers (1024-512-256), GPU float |
| **Optimizer** | SGD | Adam (fused GPU kernel) |
| **Data** | Static MNIST | Live 6-year EGX market data |
| **Output** | Class label | Portfolio weights (softmax) |
| **Exploration** | None | Gaussian noise + temperature softmax |
| **Reward** | Cross-entropy loss | Log return + HHI penalty + entropy bonus |

---

## 📁 Output Files

| File | Contents |
|------|---------|
| `gpu_actor_best.bin` | Best actor network weights (binary) |
| `gpu_critic_best.bin` | Best critic network weights (binary) |
| `gpu_actor_ckpt.bin` | Latest training checkpoint (actor) |
| `gpu_critic_ckpt.bin` | Latest training checkpoint (critic) |
| `test_results.csv` | Per-day test results vs EGX30 benchmark |
| `transaction_report.csv` | Full trade log (every buy/sell) |

---

## 🔬 Key Technical Decisions

1. **Why PPO?** Policy gradient methods naturally output probability distributions over actions — perfect for portfolio weights that must sum to 1.

2. **Why GPU?** The state dimension is ~7,271 floats. With a 4-layer 1024-512-256 network, each forward pass is thousands of floating-point operations. GPU execution via cuBLAS is ~10-50× faster for these matrix sizes.

3. **Why temperature softmax?** The EGX has 70 stocks. Without temperature scaling, the network learns to concentrate >95% in 1-2 stocks (local optimum that avoids transaction costs). Temperature forces the distribution to stay wider.

4. **Why HHI penalty?** The Herfindahl-Hirschman Index measures portfolio concentration. Penalizing it directly in the reward signal teaches the agent to diversify without needing a hard-coded rule.

5. **Why ±20% daily return clamp in data loading?** The EGX has a ±20% daily limit-up/limit-down rule. Any price move beyond this is a data error (usually from split-adjusted vs unadjusted prices in the same feed) — clamping prevents corrupted data from poisoning the training signal.
