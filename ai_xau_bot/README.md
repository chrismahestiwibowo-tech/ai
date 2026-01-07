# XAU/USD Price Prediction System

AI-powered gold price forecasting system with TradingView technical analysis.

## 🌐 Live Demo
**Web App (Cloud)**: https://xauusd-bot.streamlit.app

Uses Yahoo Finance for real-time gold price data and TradingView for technical analysis signals. Runs completely in the browser with interactive charts and AI predictions.

## Features

- **Real-Time Data**: Yahoo Finance XAU/USD price data
- **TradingView Analysis**: Live technical analysis signals (BUY/SELL/NEUTRAL)
- **AI Predictions**: Machine learning price forecasting
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Interactive Charts**: Candlestick charts with technical overlays
- **Cloud Deployment**: Accessible from any device

## Technical Indicators

1. **RSI** - Relative Strength Index
2. **MACD** - Moving Average Convergence Divergence
3. **Bollinger Bands** - Upper and Lower bands
4. **SMA** - Simple Moving Average (20, 50 periods)
5. **EMA** - Exponential Moving Average (20 period)
6. **TradingView Signals** - Oscillators and Moving Averages analysis

## Project Structure## Output

- **Live Price Charts**: Interactive candlestick visualizations
- **Technical Analysis**: RSI and MACD indicators
- **AI Predictions**: Next-day price forecasts
- **TradingView Signals**: Real-time trading recommendations

## Technologies

- **Python** - Core language
- **Streamlit** - Web interface
- **Yahoo Finance (yfinance)** - Real-time market data
- **TradingView-TA** - Technical analysis signals
- **Scikit-learn** - Machine learning
- **Plotly** - Interactive charts
- **Pandas/NumPy** - Data processing

## Deployment

Deployed on **Streamlit Cloud**:
- Auto-deploys from GitHub repository
- No server maintenance required
- Free tier available
- Accessible from any device

## Contributing

Feel free to fork and submit pull requests!

## Disclaimer

**Educational purposes only. Not financial advice. Always do your own research before trading.**

## License

MIT License

## Author

**Chris Mahestiwibowo**
- GitHub: [@chrismahestiwibowo-tech](https://github.com/chrismahestiwibowo-tech)
- Email: chrismahestiwibowo.ae@gmail.com

ai_xau_bot/
├── streamlit_app.py       # Cloud web app (yfinance)
├── streamlit_tv_app.py    # Local app with TradingView
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/chrismahestiwibowo-tech/ai.git
cd ai/ai_xau_bot
```

2. **Create virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### 🌐 Cloud Version (Recommended)

Visit: **https://xauusd-bot.streamlit.app**

### 💻 Run Locally

**Option 1: Basic Version (Yahoo Finance only)**
```bash
streamlit run streamlit_app.py
```

**Option 2: With TradingView Signals**
```bash
streamlit run streamlit_tv_app.py
```

Features:
- Real-time gold price data from Yahoo Finance
- Interactive candlestick charts
- Technical indicators (RSI, MACD, Bollinger Bands)
- AI-powered next-day price predictions
- TradingView trading signals (streamlit_tv_app.py only)

## Output

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
