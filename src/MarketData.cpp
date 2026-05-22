#include "MarketData.h"

const int WINDOW_NEED = 320;


MarketDataLoader::MarketDataLoader() : numStocks(0), numDates(0), numFeatures(0),
    open_prices(nullptr), high_prices(nullptr), low_prices(nullptr),
    stock_prices(nullptr), volume(nullptr) {}

MarketDataLoader::~MarketDataLoader() {
    if (stock_prices) {
        for (int i = 0; i < numStocks; i++) {
            delete[] open_prices[i];
            delete[] high_prices[i];
            delete[] low_prices[i];
            delete[] stock_prices[i];
            delete[] volume[i];
        }
        delete[] open_prices;
        delete[] high_prices;
        delete[] low_prices;
        delete[] stock_prices;
        delete[] volume;
    }
}

bool MarketDataLoader::load(const std::string& folder, const std::vector<std::string>& fileNames) {
    numStocks = (int)fileNames.size();
    tickers = fileNames;
    numFeatures = 1; // currently just close prices

    if (numStocks == 0) return false;

    // 1. Read first file to establish common dates and row count
    std::string firstPath = folder + "/" + fileNames[0];
    std::ifstream f1(firstPath);
    if (!f1.is_open()) {
        std::cout << "ERROR: Cannot open " << firstPath << std::endl;
        return false;
    }
    std::string line;
    std::getline(f1, line); // skip header
    while (std::getline(f1, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date;
        std::getline(ss, date, ',');
        commonDates.push_back(date);
    }
    f1.close();

    numDates = (int)commonDates.size();
    if (numDates < WINDOW_NEED) {
        std::cout << "ERROR: Only " << numDates << " dates, need " << WINDOW_NEED << std::endl;
        return false;
    }

    // 3. Allocate OHLCV
    open_prices  = new double*[numStocks];
    high_prices  = new double*[numStocks];
    low_prices   = new double*[numStocks];
    stock_prices = new double*[numStocks];
    volume       = new double*[numStocks];
    for (int i = 0; i < numStocks; i++) {
        open_prices[i]  = new double[numDates];
        high_prices[i]  = new double[numDates];
        low_prices[i]   = new double[numDates];
        stock_prices[i] = new double[numDates];
        volume[i]       = new double[numDates];
        for (int d = 0; d < numDates; d++) {
            open_prices[i][d] = 0;
            high_prices[i][d] = 0;
            low_prices[i][d]  = 0;
            stock_prices[i][d] = 0;
            volume[i][d]       = 0;
        }
    }

    // 4. Parse CSVs directly, fill matrix, and clamp returns simultaneously
    int clampCount = 0;
    for (int i = 0; i < numStocks; i++) {
        std::string path = folder + "/" + fileNames[i];
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cout << "Warning: Failed to load " << fileNames[i] << std::endl;
            continue;
        }

        std::getline(f, line); // skip header
        int d = 0;
        double prevClose = 0.0;

        while (std::getline(f, line) && d < numDates) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string token;
            
            // CSV columns: Date,Open,High,Low,Close,Volume
            std::getline(ss, token, ','); // skip Date
            std::getline(ss, token, ','); // Open
            double openPrice = atof(token.c_str());
            std::getline(ss, token, ','); // High
            double highPrice = atof(token.c_str());
            std::getline(ss, token, ','); // Low
            double lowPrice = atof(token.c_str());
            std::getline(ss, token, ','); // Close
            double closePrice = atof(token.c_str());
            std::getline(ss, token, ','); // Volume
            double vol = atof(token.c_str());

            // Clamp daily return to +/- 20%
            if (d > 0 && prevClose > 0) {
                double ret = (closePrice - prevClose) / prevClose;
                if (ret > 0.20) {
                    double scale = (prevClose * 1.20) / closePrice;
                    openPrice *= scale; highPrice *= scale;
                    lowPrice *= scale; closePrice = prevClose * 1.20;
                    clampCount++;
                } else if (ret < -0.20) {
                    double scale = (prevClose * 0.80) / closePrice;
                    openPrice *= scale; highPrice *= scale;
                    lowPrice *= scale; closePrice = prevClose * 0.80;
                    clampCount++;
                }
            }
            
            open_prices[i][d]  = openPrice;
            high_prices[i][d]  = highPrice;
            low_prices[i][d]   = lowPrice;
            stock_prices[i][d] = closePrice;
            volume[i][d]       = vol;
            prevClose = closePrice;
            d++;
        }
        f.close();
    }

    std::cout << "Loaded " << numStocks << " stocks directly into 2D matrix, "
         << numDates << " dates each" << std::endl;

    if (clampCount > 0) {
        std::cout << "Sanitised " << clampCount << " anomalous daily price jumps (clamped to +/-20%)" << std::endl;
    }

    return true;
}

std::map<std::string, double> MarketDataLoader::loadBenchmark(const std::string& path) {
    std::map<std::string, double> benchClose;
    std::ifstream f(path);
    if (!f.is_open()) return benchClose;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date, token;
        std::getline(ss, date, ',');
        std::getline(ss, token, ','); // skip Open
        std::getline(ss, token, ','); // skip High
        std::getline(ss, token, ','); // skip Low
        std::getline(ss, token, ','); // Close
        benchClose[date] = atof(token.c_str());
    }
    return benchClose;
}
