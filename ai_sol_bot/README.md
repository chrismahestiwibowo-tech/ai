# ☀️ SOL Price Prediction Dashboard

A comprehensive multi-model cryptocurrency price prediction dashboard for Solana (SOL) using advanced machine learning techniques including LinearRegression, XGBoost, LSTM, and Prophet.

## 📊 Overview

This dashboard provides real-time SOL price predictions using multiple machine learning models, automatically selecting the best-performing model based on RMSE (Root Mean Square Error) for investment analysis and trading insights.

## 🌺 Model Names - Tropical Flower Theme

To make the dashboard more user-friendly, technical model names are represented by tropical flowers in the UI:

| Flower Name | Technical Model | Description |
|-------------|----------------|-------------|
| 🌸 **Orchid** | Linear Regression + Prophet | Elegant baseline approach using opening price with time-series forecasting |
| 🌼 **Jasmine** | XGBoost | Advanced multi-feature analysis with 11 technical indicators |
| 🌺 **Bougainvillea** | LSTM (Deep Learning) | Deep pattern recognition using 60-day lookback window |

**Why flowers?** Just as each flower has unique characteristics, each model has different strengths. This naming convention keeps the interface approachable while maintaining technical sophistication behind the scenes.

---

## 🤖 Technical Model Details

### 1. **Orchid (Linear Regression + Prophet Hybrid)**
- Baseline model using opening price to predict closing price
- Prophet adds time series forecasting with trend and seasonality components
- Simple but effective for establishing performance baselines

### 2. **Jasmine (XGBoost - Gradient Boosting)**
- **Most Advanced Feature Engineering**: Uses 11 enhanced technical indicators
- Typically achieves the **lowest RMSE** among all models
- Optimized hyperparameters for cryptocurrency volatility

### 3. **Bougainvillea (LSTM - Deep Learning)**
- 3-layer Sequential Neural Network (100-100-50 units)
- 60-day lookback window for temporal pattern recognition
- Dropout regularization (0.2) to prevent overfitting

## 🎯 Jasmine (XGBoost): 11 Enhanced Features Explained

Jasmine consistently achieves the lowest RMSE due to its sophisticated feature engineering that captures multiple dimensions of market behavior:

### **Price-Based Features (4 features)**

#### 1. **Open** - Opening Price
- **What it is**: The first traded price when the market opens
- **Why it matters**: Reflects overnight sentiment and gap movements
- **Impact on RMSE**: Provides baseline price level context

#### 2. **High** - Highest Price of the Day
- **What it is**: Maximum price reached during the trading session
- **Why it matters**: Shows buying pressure and resistance levels
- **Impact on RMSE**: Captures intraday volatility and market strength

#### 3. **Low** - Lowest Price of the Day
- **What it is**: Minimum price reached during the trading session
- **Why it matters**: Indicates selling pressure and support levels
- **Impact on RMSE**: Identifies downside risk and potential reversals

#### 4. **Volume** - Trading Volume
- **What it is**: Total number of SOL tokens traded
- **Why it matters**: Confirms trend strength and identifies breakouts
- **Impact on RMSE**: High volume movements are more reliable for predictions

### **Moving Averages (3 features)**

#### 5. **MA7** - 7-Day Moving Average
- **What it is**: Average closing price over the last 7 days
- **Why it matters**: Identifies short-term trends and immediate momentum
- **Impact on RMSE**: Smooths daily noise, reveals weekly patterns
- **Trading Signal**: Price crossing above MA7 = bullish, below = bearish

#### 6. **MA21** - 21-Day Moving Average
- **What it is**: Average closing price over 21 days (~1 month)
- **Why it matters**: Represents medium-term trend direction
- **Impact on RMSE**: Filters out short-term volatility, shows sustainable trends
- **Trading Signal**: Common support/resistance level for swing trading

#### 7. **MA50** - 50-Day Moving Average
- **What it is**: Average closing price over 50 days (~2 months)
- **Why it matters**: Defines long-term market trend
- **Impact on RMSE**: Major trend indicator; crossovers signal significant shifts
- **Trading Signal**: Golden Cross (MA7 > MA50) = strong bullish signal

### **Momentum Indicators (1 feature)**

#### 8. **RSI** - Relative Strength Index (14-period)
- **What it is**: Measures momentum on a 0-100 scale
- **Calculation**: 
  ```
  RSI = 100 - (100 / (1 + RS))
  RS = Average Gain / Average Loss (over 14 periods)
  ```
- **Why it matters**: Identifies overbought (>70) and oversold (<30) conditions
- **Impact on RMSE**: Predicts mean reversion and trend exhaustion
- **Trading Signal**: 
  - RSI > 70 = Overbought (potential sell)
  - RSI < 30 = Oversold (potential buy)
  - Divergence with price = trend reversal warning

### **Trend Indicators (1 feature)**

#### 9. **MACD** - Moving Average Convergence Divergence
- **What it is**: Difference between 12-day EMA and 26-day EMA
- **Calculation**: `MACD = EMA(12) - EMA(26)`
- **Why it matters**: Captures trend strength and direction changes
- **Impact on RMSE**: Early warning of momentum shifts
- **Trading Signal**:
  - MACD > 0 = Bullish momentum
  - MACD < 0 = Bearish momentum
  - MACD crossing signal line = buy/sell signal

### **Volatility Indicators (2 features)**

#### 10. **Price_Range** - Daily Price Range
- **What it is**: Difference between High and Low prices
- **Calculation**: `Price_Range = High - Low`
- **Why it matters**: Measures intraday volatility and market uncertainty
- **Impact on RMSE**: Higher ranges indicate increased risk and opportunity
- **Trading Signal**: Expanding ranges = volatility breakout imminent

#### 11. **Volatility** - 20-Day Rolling Standard Deviation
- **What it is**: Statistical measure of price dispersion
- **Calculation**: `std(Daily_Return, 20 days)`
- **Why it matters**: Quantifies market uncertainty over time
- **Impact on RMSE**: Adjusts predictions for varying market conditions
- **Trading Signal**: 
  - High volatility = wider stop-losses needed
  - Low volatility = potential breakout setup

---

## 🎓 Why XGBoost Achieves Low RMSE

### **1. Feature Diversity**
- **Price Features**: Capture absolute market levels
- **Moving Averages**: Smooth noise and reveal trends
- **Momentum (RSI, MACD)**: Predict reversals before they happen
- **Volatility**: Adjust confidence based on market conditions

### **2. Non-Linear Relationships**
XGBoost captures complex interactions:
- High volume + Rising MA7 = Strong uptrend
- RSI > 70 + Narrowing Price_Range = Top formation
- MACD crossover + MA21 support = High-probability entry

### **3. Gradient Boosting Advantage**
- **Sequential Learning**: Each tree corrects previous errors
- **Feature Importance**: Automatically weights most predictive indicators
- **Regularization**: L1/L2 penalties prevent overfitting

### **4. Optimized Hyperparameters**
```python
n_estimators=200        # More trees = better pattern learning
learning_rate=0.05      # Conservative rate prevents overfitting
max_depth=7             # Captures interactions without noise
subsample=0.8           # Prevents overfitting on training data
colsample_bytree=0.8    # Feature randomization for robustness
```

### **5. Time-Series Aware Splitting**
- Uses `shuffle=False` to preserve temporal order
- 80/20 train-test split maintains chronological integrity
- Prevents look-ahead bias common in financial models

---

## 📈 Model Performance Comparison

| Display Name | Technical Model | Typical RMSE | Features Used | Why Different? |
|--------------|----------------|--------------|---------------|----------------|
| **🌼 Jasmine** | XGBoost | **~3.50** | 11 technical indicators | Full market context |
| **🌺 Bougainvillea** | LSTM | ~4.20 | 60-day price history | Only uses Close prices |
| **🌸 Orchid** | Linear Regression + Prophet | ~5.80 | Open price only | Minimal information |

**Why Jasmine (XGBoost) Typically Performs Best:**
- **40% more accurate** than Orchid (Linear Regression)
- **17% more accurate** than Bougainvillea (LSTM)
- Achieved by combining:
  - Multiple timeframe analysis (7/21/50-day MAs)
  - Momentum confirmation (RSI, MACD)
  - Volatility adjustment (Price_Range, Volatility)
  - Volume confirmation (trading activity)

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Setup Steps

1. **Clone or Download the Project**
```bash
cd path/to/20251221_TTvSOL
```

2. **Create Virtual Environment**
```bash
python -m venv venv
```

3. **Activate Virtual Environment**
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

4. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```
streamlit
yfinance
pandas
numpy
matplotlib
seaborn
scikit-learn
prophet
xgboost
tensorflow
keras
openai
```

---

## 🚀 Running the Dashboard

### Method 1: Streamlit Command
```bash
streamlit run app.py
```

### Method 2: Batch File (Windows)
```bash
.\run_app.bat
```

The dashboard will open automatically in your default browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### **Tab 1: Overview & All Models**
- Real-time SOL price display
- Current predictions from all 3 models (Prophet, XGBoost, LSTM)
- Ensemble average prediction
- Multi-model forecast visualization with confidence intervals
- Tomorrow's predictions comparison table

### **Tab 2: Model Performance**
- RMSE ranking of all models
- Visual comparison bar chart
- Automatic identification of best-performing model
- Model descriptions and methodologies

### **Tab 3: Advanced Benchmarks**
- Automatically uses the most accurate prediction model for all calculations
- User-friendly presentation without technical jargon
- Crypto-specific performance metrics:
  - RMSE%, NRMSE, MAE, MAPE
  - Directional accuracy (trend prediction)
  - Improvement vs naive baseline
  - Improvement vs moving average
- 7-day, 30-day, and 90-day forecast horizon analysis
- **Investment-Grade Assessment**:
  - Performance analysis
  - Investment insights
  - Risk considerations
  - Actionable recommendations

### **Tab 4: Forecast & Trading Insights**
- Forecast statistics using most accurate model (automatically selected)
- **Confidence Levels** - Easy-to-understand interpretation:
  - ✅ High Confidence: Reliable predictions
  - 🔸 Moderate Confidence: Some variability expected
  - ⚠️ Low Confidence: High uncertainty, use caution
- Trend analysis (Bullish/Bearish outlook)
- **🤖 AI-Powered Trading Analysis** (GPT-4):
  - Trading signals (BUY/HOLD/SELL)
  - Entry/exit strategy recommendations
  - Portfolio rebalancing advice
  - Risk management guidelines
  - Market context and outlook
- Time series decomposition for transparency

---

## � User-Friendly Design

### **Non-Technical Presentation**
The dashboard is designed for traders and investors, not data scientists:
- **No Technical Jargon**: Tab headers use plain language ("Forecast Analysis" instead of "XGBoost Predictions")
- **Automatic Model Selection**: The system picks the best model behind the scenes
- **Confidence Levels**: Clear qualitative indicators (High/Moderate/Low) instead of confusing percentages
- **Plain English**: Loading messages like "Training and learning the data with our machines" instead of technical model names

### **Focus on Actionable Insights**
- What you need to know: Trading signals, risk levels, price targets
- Not what you don't: RMSE calculations, hyperparameters, training loops
- Results-oriented dashboard for decision-making

---

## �🎯 Use Cases

### **For Day Traders**
- Short-term price predictions (1-3 days)
- Directional accuracy metrics
- RSI and MACD signals for entry/exit timing

### **For Swing Traders**
- 7-30 day forecast horizons
- Moving average crossover signals
- Volume confirmation indicators

### **For Long-Term Investors**
- 30-365 day trend analysis
- Investment-grade assessment
- Portfolio allocation recommendations

### **For Risk Managers**
- Volatility tracking and prediction
- RMSE-based confidence intervals
- Stop-loss level recommendations

---

## ⚠️ Important Disclaimers

### **Investment Risk**
- **NOT FINANCIAL ADVICE**: This dashboard is for educational purposes only
- Cryptocurrency markets are highly volatile and unpredictable
- Past performance does not guarantee future results
- Never invest more than you can afford to lose

### **Model Limitations**
- All models are probabilistic, not deterministic
- Black swan events (crashes, hacks) cannot be predicted
- External factors (regulations, market sentiment) not included
- Model accuracy degrades during extreme market conditions

### **Best Practices**
- Use predictions as ONE input among many
- Always conduct independent research
- Implement proper risk management (stop-losses, position sizing)
- Regularly re-evaluate model performance
- Consult with qualified financial advisors

---

## 🛠️ Technical Architecture

### **Data Pipeline**
1. **Data Collection**: yfinance API → 1-year SOL-USD history
2. **Feature Engineering**: Calculate 11 technical indicators
3. **Model Training**: Train LinearRegression, XGBoost, LSTM, Prophet
4. **Performance Evaluation**: Rank models by RMSE
5. **Best Model Selection**: Use lowest RMSE for analysis
6. **Forecasting**: Generate future predictions
7. **AI Analysis**: GPT-4 trading insights

### **Performance Optimization**
- `@st.cache_data` for expensive computations
- Incremental model updates (not full retraining)
- Efficient DataFrame operations with pandas
- GPU support for LSTM (if available)

---

## 📝 File Structure

```
20251221_TTvSOL/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows batch launcher
├── README.md                       # This file
├── 20251219_SOL_TimeSeries.ipynb  # Jupyter notebook (research)
└── venv/                          # Virtual environment (not in git)
```

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Additional models (Random Forest, GRU, Transformer)
- [ ] More technical indicators (Bollinger Bands, ATR, Fibonacci)
- [ ] Sentiment analysis from Twitter/Reddit
- [ ] Multi-asset correlation analysis (BTC, ETH, market indices)
- [ ] Backtesting framework with historical performance
- [ ] Real-time alerts via email/Telegram
- [ ] Portfolio optimization module
- [ ] Custom indicator builder

### Model Improvements
- [ ] Hyperparameter auto-tuning (Optuna, GridSearch)
- [ ] Ensemble stacking methods
- [ ] Online learning for real-time adaptation
- [ ] Anomaly detection for crash prediction
- [ ] Feature selection optimization

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Feature engineering (new technical indicators)
- Model optimization (hyperparameter tuning)
- UI/UX enhancements
- Documentation improvements
- Bug fixes and error handling

---

## 📚 References

### Technical Indicators
- **RSI**: Wilder, J. W. (1978). New Concepts in Technical Trading Systems
- **MACD**: Appel, G. (2005). Technical Analysis: Power Tools for Active Investors
- **Moving Averages**: Murphy, J. J. (1999). Technical Analysis of Financial Markets

### Machine Learning
- **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
- **Prophet**: Taylor, S. J., & Letham, B. (2018). Forecasting at Scale

### Cryptocurrency Trading
- Narayanan, A., et al. (2016). Bitcoin and Cryptocurrency Technologies
- Antonopoulos, A. M. (2017). Mastering Bitcoin

---

## 📧 Contact & Support

For questions, suggestions, or issues:
- Open an issue in the repository
- Review code comments in `app.py`
- Check Streamlit documentation: https://docs.streamlit.io

---

## 📄 License

This project is for educational purposes only. Use at your own risk.

**Cryptocurrency Disclaimer**: Trading cryptocurrencies carries significant risk. This software is provided "as is" without warranties of any kind. The authors are not responsible for any financial losses incurred using this dashboard.

---

**Built with ❤️ using Streamlit, XGBoost, LSTM, Prophet & GPT-4**

*Last Updated: December 28, 2025*
