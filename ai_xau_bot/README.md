# XAU/USD Price Prediction System

Advanced price forecasting system for gold (XAU/USD) with web interface and local MetaTrader5 integration.

## 🌐 Live Demo
**Web App (Cloud)**: https://xauusd-bot.streamlit.app

The cloud version uses Yahoo Finance for real-time gold price data and runs completely in the browser with interactive charts and predictions.

## Features

- **Data Collection**: Fetches 3 years of historical XAU/USD data from MetaTrader5
- **Technical Analysis**: 11 core technical indicators + derived features
- **Machine Learning**: XGBoost regression model with 80/20 train-test split
- **Monte Carlo Simulation**: 10,000 simulations for probabilistic forecasting
- **Comprehensive Evaluation**: MSE, RMSE, MAE, R² metrics
- **Visualization**: Multiple charts for analysis and insights

## Technical Indicators (11 Core)

1. **RSI** - Relative Strength Index
2. **MACD** - Moving Average Convergence Divergence
3. **MACD Signal** - MACD Signal Line
4. **Bollinger Bands** - Upper and Lower bands
5. **ATR** - Average True Range
6. **SMA** - Simple Moving Average (20, 50 periods)
7. **EMA** - Exponential Moving Average (20, 50 periods)
8. **Stochastic** - Stochastic Oscillator
9. **CCI** - Commodity Channel Index
10. **ADX** - Average Directional Index
11. **Williams %R** - Williams Percent Range

Plus additional features: OBV, ROC, price changes, rolling statistics, and lag features.

## Project Structure

```
20260107_ai_xauusd_bot/
├── config.py              # Configuration settings
├── data_loader.py         # MetaTrader5 data fetching
├── indicators.py          # Technical indicators calculation
├── model.py              # XGBoost model training & evaluation
├── monte_carlo.py        # Monte Carlo simulation
├── main.py               # Main application orchestrator
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── models/               # Trained models directory
├── plots/                # Visualization outputs
└── results/              # Analysis results (CSV, reports)
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment** (already done)

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure MetaTrader5 credentials**:
   - Copy `.env.example` to `.env`
   - Edit `.env` with your MetaTrader5 credentials:
   ```
   MT5_LOGIN=your_account_number
   MT5_PASSWORD=your_password
   MT5_SERVER=your_broker_server
   MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
   ```

## Usage

### 🌐 Web Application (Recommended)

**Cloud Deployment**: Visit https://xauusd-bot.streamlit.app

**Local Streamlit App**:
```bash
streamlit run streamlit_app.py
```

Features:
- Real-time gold price data from Yahoo Finance
- Interactive candlestick charts
- Technical indicators (RSI, MACD, Bollinger Bands)
- AI-powered next-day price predictions
- No MetaTrader5 required

### 🖥️ Local Analysis (Windows + MetaTrader5)

Run complete analysis with MT5 data:
```bash
python main_mt5.py
```

This will:
1. Load 3 years of XAU/USD historical data
2. Calculate 11 technical indicators
3. Train XGBoost model (80/20 split)
4. Evaluate with MSE, RMSE, MAE, R²
5. Run Monte Carlo simulation (10,000 paths)
6. Generate comprehensive reports and visualizations

### Run Individual Modules

**Test Data Loading**:
```bash
python data_loader.py
```

**Test Indicators**:
```bash
python indicators.py
```

**Test Model Training**:
```bash
python model.py
```

**Test Monte Carlo**:
```bash
python monte_carlo.py
```

## Configuration

Edit [config.py](config.py) to customize:

- **Symbol**: Trading pair (default: XAUUSD)
- **Timeframe**: H1, H4, D1, etc. (default: H1)
- **Historical Period**: Years of data (default: 3)
- **Train/Test Split**: (default: 80/20)
- **XGBoost Parameters**: Learning rate, max_depth, etc.
- **Monte Carlo Settings**: Number of simulations, forecast horizon

## Output Files

### Models
- `models/xauusd_xgboost_model.pkl` - Trained XGBoost model
- `models/scaler.pkl` - Feature scaler

### Plots
- `plots/predictions.png` - Actual vs predicted prices
- `plots/feature_importance.png` - Feature importance chart
- `plots/monte_carlo_paths.png` - Simulation paths
- `plots/monte_carlo_distribution.png` - Price distribution

### Results
- `results/model_metrics.csv` - MSE, RMSE, MAE, R²
- `results/feature_importance.csv` - Feature rankings
- `results/monte_carlo_forecast.csv` - Forecast statistics
- `results/price_probabilities.csv` - Target price probabilities
- `results/summary_report.txt` - Comprehensive analysis report

## Model Performance Metrics

The system evaluates model performance using:

- **MSE (Mean Squared Error)**: Lower is better
- **RMSE (Root Mean Squared Error)**: Lower is better, in same units as price
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² (R-squared)**: Proportion of variance explained (closer to 1 is better)

## Monte Carlo Simulation

Provides probabilistic forecasting with:
- Mean, median, and percentile forecasts
- Confidence intervals (5th to 95th percentile)
- Value at Risk (VaR) calculations
- Probability analysis for target prices

## Requirements

- Python 3.8+
- MetaTrader5 terminal installed
- Active MetaTrader5 account (or demo account)
- See [requirements.txt](requirements.txt) for package dependencies

## Troubleshooting

**MetaTrader5 Connection Issues**:
- Ensure MT5 terminal is installed and running
- Check credentials in `.env` file
- Verify server name is correct
- Try demo account if live connection fails

**Data Loading Errors**:
- Ensure XAUUSD symbol is available on your broker
- Check internet connection
- Verify MT5 terminal has historical data

**Import Errors**:
- Ensure all packages are installed: `pip install -r requirements.txt`
- Activate virtual environment before running

## Notes

- **Time Series Data**: The system uses time-series aware splitting (no shuffling)
- **Feature Scaling**: StandardScaler is applied to all features
- **Early Stopping**: XGBoost uses early stopping to prevent overfitting
- **Geometric Brownian Motion**: Monte Carlo uses GBM for realistic price paths

## License

This project is for educational and research purposes.

## Disclaimer

⚠️ **Trading Risk Warning**: This system is for educational purposes only. Past performance does not guarantee future results. Always conduct your own research and risk assessment before trading.
