#include "environment.h"
using namespace std;

tradingenv::tradingenv(MarketDataLoader* marketdata, int start_index, int end_index){
data = marketdata;
start_idx = start_index;
end_idx = end_index;
lookback = 80;
num_stocks = marketdata->numStocks;
history_size = num_stocks*lookback;
portfolio_size = num_stocks+1;
state_size = history_size+portfolio_size;
features_per_stock = 10;
total_features = num_stocks * features_per_stock;
state = matD(state_size);
portfolio = matD(portfolio_size);
features = matD(total_features);
current_step = lookback+start_idx;
tx_cost=0.0015;
reward=0.0;
shares=matD(num_stocks);
prev_shares=matD(num_stocks);
init_portfolio();
init_state();
done = false;
}
void tradingenv:: init_portfolio()
{
    cash=10000;
    portfolio_value=cash;
for(int i=0;i<num_stocks;i++){
    portfolio[i] = 0.0;
}
portfolio[num_stocks] = 1.0;
for(int i=0;i<num_stocks;i++)
{
    prev_shares[i]=0.0;
    shares[i]=0.0;
}
prev_portfolio_value=0;
tx_plenty=0.0;
}
void tradingenv:: init_state()
{
for(int i=0;i<num_stocks;i++)
{
    for(int j=0; j<lookback; j++)
    {
        int time_idx = current_step - lookback + j;
        state[i*lookback + j] = log(data->stock_prices[i][time_idx + 1] / data->stock_prices[i][time_idx]);
    }
}
for(int i=0;i<portfolio_size;i++)
{
    state[history_size+i] = portfolio[i];
}
compute_features();
}
void tradingenv::process_portfolio(double* action) {
    prev_portfolio_value = portfolio_value;
    double new_stock_value = 0.0;
    tx_plenty = 0.0;

    // 1. Calculate the raw target values based on the agent's action
    for(int i = 0; i < num_stocks; i++) {
        double today_price = data->stock_prices[i][current_step];

        // The exact dollar amount the agent WANTS to hold in this stock
        double target_value = action[i] * prev_portfolio_value;

        // Calculate how much stock we currently hold at today's prices
        double current_value = shares[i] * today_price;

        // Transaction cost is based on the difference (what we are buying/selling)
        double trade_value = fabs(target_value - current_value);
        double fee = trade_value * tx_cost;

        tx_plenty += fee;

        // Update shares mathematically (we subtract the fee from the target capital)
        // This ensures fees eat into our purchasing power, preventing negative cash
        shares[i] = (target_value - fee) / today_price;

        // Accumulate total stock value for portfolio calculation
        new_stock_value += (shares[i] * today_price);
    }

    // 2. The remaining cash is the total value minus what we put in stocks
    cash = prev_portfolio_value - new_stock_value - tx_plenty;

    // 3. Final portfolio value (should exactly equal prev_value if no prices moved, minus fees)
    portfolio_value = new_stock_value + cash;
}
void tradingenv::compute_reward()
{
    reward = log(portfolio_value/prev_portfolio_value)*100;
    //if (reward<0)reward*=2.5;
    /*double bh_growth = 0.0;
    for(int i=0;i<num_stocks;i++)
    {
        double today_price = data->stock_prices[i][current_step];
        double prev_price = data->stock_prices[i][current_step - 1];
        bh_growth += log(today_price/prev_price)*100;
    }
    bh_growth /= num_stocks;

    // Reward is the excess return (alpha) over Buy & Hold
    reward = agent_growth - bh_growth;*/
}
void tradingenv::update_state(double* action){
    for(int i=0;i<num_stocks;i++)
    {
        for(int j=0;j<lookback-1;j++)
        {
            state[i*lookback+j] = state[i*lookback+j+1];
        }
        double today_price=data->stock_prices[i][current_step];
        double prev_price=data->stock_prices[i][current_step-1];
        state[i*lookback+lookback-1]= log(today_price/prev_price)*100;
    }
    for(int i=0;i<portfolio_size;i++)
    {
        state[history_size+i]=action[i];
    }
    compute_features();
}

bool tradingenv::step(double* action) {
    if (done) return true;
    process_portfolio(action);
    current_step++;
    // Update portfolio value at new prices after time advances
    portfolio_value = cash;
    for(int i = 0; i < num_stocks; i++)
        portfolio_value += shares[i] * data->stock_prices[i][current_step];
    compute_reward();
    update_state(action);
    if (current_step >= end_idx - 1)
        done = true;
    return done;
}

int tradingenv::getRandom(int min, int max) {
    thread_local std::random_device rd;      // Only runs once per thread
    thread_local std::mt19937 gen(rd());     // Only runs once per thread
    std::uniform_int_distribution<> dist(min, max);
    return dist(gen);
}
void tradingenv::reset(int ep)
{
  int max_start = end_idx - ep-1; // Ensure room for a full episode
  if (max_start < start_idx + lookback) max_start = start_idx + lookback;
  current_step = getRandom(start_idx + lookback, max_start);
  init_portfolio();
  init_state();
  done = false;
}
void tradingenv::compute_features()
{
    // Compute 10 technical indicators per stock at current_step
    // Uses OHLCV data from data->open_prices, high_prices, low_prices, stock_prices, volume
    int t = current_step;

    for(int s = 0; s < num_stocks; s++)
    {
        int base = s * features_per_stock;
        double* close = data->stock_prices[s];
        double* high  = data->high_prices[s];
        double* low   = data->low_prices[s];
        double* vol   = data->volume[s];

        // --- 1. EMA Distance: (EMA12 - EMA26) / close ---
        {
            double ema12 = close[t];
            double ema26 = close[t];
            double k12 = 2.0 / 13.0;
            double k26 = 2.0 / 27.0;
            int ema_lookback = (t > 50) ? 50 : t;
            // Seed with the price at start of lookback
            ema12 = close[t - ema_lookback];
            ema26 = close[t - ema_lookback];
            for(int j = t - ema_lookback + 1; j <= t; j++)
            {
                ema12 = close[j] * k12 + ema12 * (1.0 - k12);
                ema26 = close[j] * k26 + ema26 * (1.0 - k26);
            }
            double c = close[t];
            features[base + 0] = (c > 0) ? (ema12 - ema26) / c : 0.0;
        }

        // --- 2. MACD Histogram ---
        {
            double ema12 = close[t], ema26 = close[t];
            double k12 = 2.0 / 13.0, k26 = 2.0 / 27.0;
            int ema_lookback = (t > 50) ? 50 : t;
            ema12 = close[t - ema_lookback];
            ema26 = close[t - ema_lookback];

            double macd_line = 0.0;
            double signal = 0.0;
            double k9 = 2.0 / 10.0;
            bool signal_init = false;

            for(int j = t - ema_lookback + 1; j <= t; j++)
            {
                ema12 = close[j] * k12 + ema12 * (1.0 - k12);
                ema26 = close[j] * k26 + ema26 * (1.0 - k26);
                macd_line = ema12 - ema26;
                if(!signal_init) { signal = macd_line; signal_init = true; }
                else { signal = macd_line * k9 + signal * (1.0 - k9); }
            }
            double c = close[t];
            features[base + 1] = (c > 0) ? (macd_line - signal) / c : 0.0;
        }

        // --- 3. RSI (14-period), scaled to [-1, 1] ---
        {
            int period = 14;
            double avg_gain = 0.0, avg_loss = 0.0;
            int start = (t >= period) ? t - period : 0;
            int count = t - start;
            if(count > 0)
            {
                for(int j = start + 1; j <= t; j++)
                {
                    double change = close[j] - close[j-1];
                    if(change > 0) avg_gain += change;
                    else avg_loss -= change; // make positive
                }
                avg_gain /= count;
                avg_loss /= count;
            }
            double rsi;
            if(avg_loss < 1e-12) rsi = 100.0;
            else {
                double rs = avg_gain / avg_loss;
                rsi = 100.0 - (100.0 / (1.0 + rs));
            }
            features[base + 2] = (rsi - 50.0) / 50.0; // scale to [-1, 1]
        }

        // --- 4. Bollinger %B (20-period) ---
        // --- 5. Bollinger Bandwidth (20-period) ---
        {
            int period = 20;
            int start = (t >= period) ? t - period + 1 : 0;
            int count = t - start + 1;
            double sum = 0.0;
            for(int j = start; j <= t; j++) sum += close[j];
            double sma = sum / count;

            double sq_sum = 0.0;
            for(int j = start; j <= t; j++) {
                double d = close[j] - sma;
                sq_sum += d * d;
            }
            double stdev = sqrt(sq_sum / count);

            double upper = sma + 2.0 * stdev;
            double lower = sma - 2.0 * stdev;
            double bw = upper - lower;

            // %B = (close - lower) / (upper - lower)
            features[base + 3] = (bw > 1e-12) ? (close[t] - lower) / bw : 0.5;
            // Bandwidth = (upper - lower) / middle
            features[base + 4] = (sma > 1e-12) ? bw / sma : 0.0;
        }

        // --- 6. Normalized ATR (14-period) ---
        {
            int period = 14;
            double atr = 0.0;
            int start = (t >= period) ? t - period + 1 : 1;
            int count = 0;
            for(int j = start; j <= t; j++)
            {
                double tr1 = high[j] - low[j];
                double tr2 = fabs(high[j] - close[j-1]);
                double tr3 = fabs(low[j]  - close[j-1]);
                double tr = tr1;
                if(tr2 > tr) tr = tr2;
                if(tr3 > tr) tr = tr3;
                atr += tr;
                count++;
            }
            if(count > 0) atr /= count;
            features[base + 5] = (close[t] > 1e-12) ? atr / close[t] : 0.0;
        }

        // --- 7. VROC (Volume Rate of Change, 14-period) ---
        {
            int period = 14;
            double prev_vol = (t >= period) ? vol[t - period] : vol[0];
            features[base + 6] = (prev_vol > 1e-6) ? (vol[t] - prev_vol) / prev_vol : 0.0;
            // Clamp VROC to reasonable range
            if(features[base + 6] > 5.0) features[base + 6] = 5.0;
            if(features[base + 6] < -1.0) features[base + 6] = -1.0;
        }

        // --- 8. Chaikin Money Flow (20-period) ---
        {
            int period = 20;
            int start = (t >= period) ? t - period + 1 : 0;
            double mfv_sum = 0.0;
            double vol_sum = 0.0;
            for(int j = start; j <= t; j++)
            {
                double hl = high[j] - low[j];
                double mf_mult = (hl > 1e-12) ? ((close[j] - low[j]) - (high[j] - close[j])) / hl : 0.0;
                mfv_sum += mf_mult * vol[j];
                vol_sum += vol[j];
            }
            features[base + 7] = (vol_sum > 1e-6) ? mfv_sum / vol_sum : 0.0;
        }

        // --- 9. Rolling Skewness (20-period of log returns) ---
        // --- 10. Rolling Kurtosis (20-period of log returns) ---
        {
            int period = 20;
            int start = (t >= period) ? t - period + 1 : 1;
            int n = t - start + 1;

            // Compute log returns and mean
            double mean = 0.0;
            double* rets = new double[n];
            for(int j = 0; j < n; j++)
            {
                int idx = start + j;
                rets[j] = (close[idx-1] > 1e-12) ? log(close[idx] / close[idx-1]) : 0.0;
                mean += rets[j];
            }
            if(n > 0) mean /= n;

            // Compute moments
            double m2 = 0.0, m3 = 0.0, m4 = 0.0;
            for(int j = 0; j < n; j++)
            {
                double d = rets[j] - mean;
                double d2 = d * d;
                m2 += d2;
                m3 += d2 * d;
                m4 += d2 * d2;
            }
            if(n > 0) { m2 /= n; m3 /= n; m4 /= n; }

            double stdev = sqrt(m2);
            double skew = (stdev > 1e-12) ? m3 / (stdev * stdev * stdev) : 0.0;
            double kurt = (m2 > 1e-12) ? (m4 / (m2 * m2)) - 3.0 : 0.0; // excess kurtosis

            // Clamp to avoid extreme values
            if(skew > 3.0) skew = 3.0;
            if(skew < -3.0) skew = -3.0;
            if(kurt > 10.0) kurt = 10.0;
            if(kurt < -3.0) kurt = -3.0;

            features[base + 8] = skew;
            features[base + 9] = kurt;

            delete[] rets;
        }
    }
}
tradingenv::~tradingenv(){
 delete [] state;
 delete [] portfolio;
 delete [] features;
 delete [] shares;
 delete [] prev_shares;
}
